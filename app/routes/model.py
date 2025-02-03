import io
import json
import logging
import pickle
from datetime import datetime
from importlib import import_module
from typing import List, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sklearn import datasets
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.db.session import get_db
from app.db import models
from app.models.pydantic_models import (
    TrainRequest,
    TrainResponse,
    ModelResponse,
    ModelDetailResponse,
    PromoteRequest,
    PredictRequest,
    RemoveRequest,
    PredictResponse,
)
from app.utils.constants import EnvironmentEnum
from app.utils.object_storage import (
    delete_object_from_bucket,
    put_object_to_bucket,
    get_object_from_bucket,
    load_dataset_from_storage,
)
from app.utils.preprocessing import create_preprocessor, detect_target_column
from app.config import TRAINED_MODELS_BUCKET
from app.utils.string_utils import extract_model_name

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /train endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/train",
    summary="Train a model on a dataset",
    description=(
        "Trains a new machine learning model using the specified dataset, "
        "target column, and model configuration. The trained model is then "
        "stored in object storage and registered in the database. "
        "If no target column is provided, the latest column of the dataset "
        "is used."
    ),
    tags=["models"],
    response_model=TrainResponse,
    responses={
        400: {
            "description": "Invalid request or model configuration",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid model configuration"}
                }
            },
        },
        404: {
            "description": "Dataset not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found"}
                }
            },
        },
        200: {
            "description": "Model trained and registered successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model trained and registered successfully",
                        "dataset": "co2_emission",
                        "version": 1,
                        "model_file": "co2_emission_model_dev_v1.pkl",
                        "metrics": {
                            "accuracy": 0.95,
                            "f1_score": 0.93,
                            "precision": {"0": 0.94, "1": 0.96},
                            "recall": {"0": 0.92, "1": 0.97},
                        },
                    }
                }
            },
        },
    },
)
def train_model(
    request_data: TrainRequest,
    db: Session = Depends(get_db),
) -> Any:
    dataset_name = request_data.dataset_name
    use_example = request_data.use_example
    target_column = request_data.target_column
    test_size = request_data.test_size
    random_state = request_data.random_state

    try:
        model_module = request_data.model["model_module"]
        model_class = request_data.model["model_class"]
        model_params = request_data.model.get("model_params", {})
    except KeyError as e:
        raise HTTPException(
            status_code=400, detail=f"Missing key in model configuration: {e}"
        )

    if use_example:
        dataset_name = "co2_emission"
        iris = datasets.load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["target"] = iris.target
    else:
        dataset_entry = (
            db.query(models.DatasetCatalog)
            .filter(models.DatasetCatalog.name == dataset_name)
            .first()
        )
        if not dataset_entry:
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = load_dataset_from_storage(dataset_entry.location)

    target_column = target_column or detect_target_column(df)
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset.",
        )

    X_df = df.drop(columns=[target_column])
    y = df[target_column]
    feature_names = X_df.columns.tolist()
    preprocessor = create_preprocessor(X_df)

    try:
        module = import_module(model_module)
        ModelClass = getattr(module, model_class)
        model_instance = ModelClass(**model_params)
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid model configuration"
        )

    full_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model_instance)]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    metrics = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")

    if not use_example:
        existing_versions = (
            db.query(models.ModelRegistry)
            .filter(models.ModelRegistry.dataset_id == dataset_entry.id)
            .all()
        )
        version = (
            max([entry.version for entry in existing_versions], default=0) + 1
        )
    else:
        version = 1

    environment = "dev"
    model_file_name = f"{dataset_name}_model_{environment}_v{version}.pkl"
    model_data = io.BytesIO()
    pickle.dump(full_pipeline, model_data)
    model_data.seek(0)
    put_object_to_bucket(
        TRAINED_MODELS_BUCKET, model_file_name, model_data.getvalue()
    )

    if not use_example:
        model_params["feature_names"] = feature_names
        new_registry_entry = models.ModelRegistry(
            dataset_id=dataset_entry.id,
            version=version,
            environment=environment,
            artifact_path=model_file_name,
            metrics=json.dumps(metrics),
            parameters=json.dumps(model_params),
            description="Model trained automatically",
            promotion_timestamp=None,
        )
        db.add(new_registry_entry)
        db.commit()
        db.refresh(new_registry_entry)

    return {
        "message": "Model trained and registered successfully",
        "dataset": dataset_name,
        "version": version,
        "model_file": model_file_name,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# /promote endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/promote",
    summary="Promote a model to a new environment",
    description=(
        "Promote a model from its current environment to a higher environment "
        "(e.g., dev to staging or production). "
        "Demotion or re-promotion to the same environment is not allowed."
    ),
    tags=["models"],
    responses={
        400: {
            "description": "Cannot demote or promote to the same environment",
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "Cannot demote or promote to the same environment"
                        )
                    }
                }
            },
        },
        404: {
            "description": "Dataset or model version not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found"}
                }
            },
        },
        200: {
            "description": "Model promoted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model promoted to production successfully",
                        "dataset": "co2_emission",
                        "version": 1,
                        "model_file": "co2_emission_model_production_v1.pkl",
                    }
                }
            },
        },
    },
)
def promote_model(
    request_data: PromoteRequest,
    db: Session = Depends(get_db),
):
    dataset_name = request_data.dataset_name
    version = request_data.version
    environment = request_data.environment.lower()

    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        raise HTTPException(status_code=404, detail="Dataset not found")

    entry = (
        db.query(models.ModelRegistry)
        .filter(
            models.ModelRegistry.dataset_id == dataset_entry.id,
            models.ModelRegistry.version == version,
        )
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Model version not found")

    promotion_order = {
        EnvironmentEnum.dev.value: 1,
        EnvironmentEnum.staging.value: 2,
        EnvironmentEnum.production.value: 3,
    }
    # Use the updated "environment" property from the registry
    if promotion_order.get(environment, 0) <= promotion_order.get(
        entry.environment, 0
    ):
        raise HTTPException(
            status_code=400,
            detail="Cannot demote or promote to the same environment",
        )

    new_model_file = f"{dataset_name}_model_{environment}_v{version}.pkl"
    artifact_bytes = get_object_from_bucket(
        TRAINED_MODELS_BUCKET, entry.artifact_path
    )["Body"].read()
    put_object_to_bucket(TRAINED_MODELS_BUCKET, new_model_file, artifact_bytes)
    entry.environment = environment  # Updated property name
    entry.artifact_path = new_model_file
    entry.promotion_timestamp = datetime.now()
    db.commit()
    db.refresh(entry)
    return {
        "message": f"Model promoted to {environment} successfully",
        "dataset": dataset_name,
        "version": version,
        "model_file": new_model_file,
    }


# ---------------------------------------------------------------------------
# /predict endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/predict",
    summary="Make predictions using a deployed model",
    description=(
        "Perform inference using a model deployed in a specific environment "
        "(dev, staging, or production)."
    ),
    tags=["models"],
    response_model=PredictResponse,
    responses={
        404: {
            "description": "Dataset or model not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found"}
                }
            },
        },
        200: {
            "description": "Predictions generated successfully",
            "content": {
                "application/json": {"example": {"predictions": [0, 1, 1]}}
            },
        },
    },
)
def predict(
    request_data: PredictRequest,
    db: Session = Depends(get_db),
):
    dataset_name = request_data.dataset_name
    environment = request_data.environment.lower()
    input_features = request_data.features

    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        raise HTTPException(status_code=404, detail="Dataset not found")

    entry = (
        db.query(models.ModelRegistry)
        .filter(
            models.ModelRegistry.dataset_id == dataset_entry.id,
            models.ModelRegistry.environment
            == environment,  # Updated property name
        )
        .order_by(models.ModelRegistry.version.desc())
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=404, detail="No model found for the given environment"
        )

    model_response = get_object_from_bucket(
        TRAINED_MODELS_BUCKET, entry.artifact_path
    )
    model_pipeline = pickle.loads(model_response["Body"].read())
    input_df = pd.DataFrame(input_features)
    params = json.loads(entry.parameters)
    if "feature_names" in params:
        input_df.columns = params["feature_names"]
    predictions = model_pipeline.predict(input_df)
    return {"predictions": predictions.tolist()}


# ---------------------------------------------------------------------------
# /models endpoint (List all models)
# ---------------------------------------------------------------------------
@router.get(
    "/models",
    summary="List all models in the registry",
    description="Retrieve a list of all registered models.",
    tags=["models"],
    response_model=List[ModelResponse],
    responses={
        200: {
            "description": "List of all registered models",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "dataset_id": 1,
                            "version": 1,
                            "environment": "dev",
                            "artifact_path": "co2_emission_model_dev_v1.pkl",
                            "metrics": {"accuracy": 0.95, "f1_score": 0.93},
                            "parameters": {"n_estimators": 100},
                            "description": "Trained model",
                            "timestamp": "2024-01-01T12:00:00",
                            "promotion_timestamp": None,
                        }
                    ]
                }
            },
        }
    },
)
def list_models(db: Session = Depends(get_db)):
    return db.query(models.ModelRegistry).all()


# ---------------------------------------------------------------------------
# /models/by-dataset endpoint
# ---------------------------------------------------------------------------
@router.get(
    "/models/by-dataset",
    summary="List models for a specific dataset",
    description="Retrieve all models associated with a specific dataset.",
    tags=["models"],
    response_model=List[ModelResponse],
    responses={
        404: {
            "description": "Dataset not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found"}
                }
            },
        },
        200: {
            "description": "List of models for the dataset",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "name": "co2_emission_model_dev_v1",
                            "version": 1,
                            "environment": "dev",
                            "dataset_name": "co2_emission",
                            "f1_score": 0.93,
                            "accuracy": 0.95,
                            "trained_at": "2024-01-01 12:00:00",
                            "promoted_at": None,
                        }
                    ]
                }
            },
        },
    },
)
def list_models_by_dataset(
    dataset_name: str,
    db: Session = Depends(get_db),
):
    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    models_list = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.dataset_id == dataset_entry.id)
        .all()
    )
    return [
        ModelResponse(
            id=model.id,
            name=extract_model_name(model.artifact_path),
            version=model.version,
            environment=model.environment,  # Updated property name
            dataset_name=dataset_name,
            f1_score=(
                json.loads(model.metrics).get("f1_score")
                if isinstance(model.metrics, str)
                else model.metrics.get("f1_score")
            ),
            accuracy=(
                json.loads(model.metrics).get("accuracy")
                if isinstance(model.metrics, str)
                else model.metrics.get("accuracy")
            ),
            trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            promoted_at=(
                model.promotion_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if model.promotion_timestamp
                else None
            ),
        )
        for model in models_list
    ]


# ---------------------------------------------------------------------------
# /models/by-environment endpoint
# ---------------------------------------------------------------------------
@router.get(
    "/models/by-environment",
    summary="List models by environment",
    description="Retrieve all models in a specific environment.",
    tags=["models"],
    response_model=List[ModelResponse],
    responses={
        400: {
            "description": "Invalid environment specified",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid environment specified"}
                }
            },
        },
        200: {
            "description": "List of models for the specified environment",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "name": "co2_emission_model_dev_v1",
                            "version": 1,
                            "environment": "dev",
                            "dataset_name": "co2_emission",
                            "f1_score": 0.93,
                            "accuracy": 0.95,
                            "trained_at": "2024-01-01 12:00:00",
                            "promoted_at": None,
                        }
                    ]
                }
            },
        },
    },
)
def list_models_by_environment(
    environment: EnvironmentEnum,
    db: Session = Depends(get_db),
):
    models_list = (
        db.query(models.ModelRegistry)
        .filter(
            models.ModelRegistry.environment
            == environment.value  # Updated property name
        )
        .all()
    )
    return [
        ModelResponse(
            id=model.id,
            name=extract_model_name(model.artifact_path),
            version=model.version,
            environment=model.environment,  # Updated property name
            dataset_name=db.query(models.DatasetCatalog)
            .filter(models.DatasetCatalog.id == model.dataset_id)
            .first()
            .name,
            f1_score=(
                json.loads(model.metrics).get("f1_score")
                if isinstance(model.metrics, str)
                else model.metrics.get("f1_score")
            ),
            accuracy=(
                json.loads(model.metrics).get("accuracy")
                if isinstance(model.metrics, str)
                else model.metrics.get("accuracy")
            ),
            trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for model in models_list
    ]


# ---------------------------------------------------------------------------
# /models/{model_id} endpoint
# ---------------------------------------------------------------------------
@router.get(
    "/models/{model_id}",
    summary="Get detailed model information",
    description="Retrieve detailed information of a specific model.",
    tags=["models"],
    response_model=ModelDetailResponse,
    responses={
        404: {
            "description": "Model not found",
            "content": {
                "application/json": {"example": {"detail": "Model not found"}}
            },
        },
        200: {
            "description": "Detailed model information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "name": "co2_emission_model_dev_v1",
                        "version": 1,
                        "environment": "dev",
                        "dataset_name": "co2_emission",
                        "artifact_path": "co2_emission_model_dev_v1.pkl",
                        "metrics": {
                            "accuracy": 0.95,
                            "f1_score": 0.93,
                            "precision": {"0": 0.94, "1": 0.96},
                            "recall": {"0": 0.92, "1": 0.97},
                        },
                        "parameters": {"n_estimators": 100},
                        "description": "Trained model",
                        "trained_at": "2024-01-01 12:00:00",
                        "promoted_at": None,
                    }
                }
            },
        },
    },
)
def get_model_by_id(
    model_id: int,
    db: Session = Depends(get_db),
):
    model = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.id == model_id)
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.id == model.dataset_id)
        .first()
    )
    dataset_name = dataset_entry.name if dataset_entry else "Unknown"
    return ModelDetailResponse(
        id=model.id,
        name=extract_model_name(model.artifact_path),
        version=model.version,
        environment=model.environment,  # Updated property name
        dataset_name=dataset_name,
        artifact_path=model.artifact_path,
        metrics=(
            json.loads(model.metrics)
            if isinstance(model.metrics, str)
            else model.metrics
        ),
        parameters=(
            json.loads(model.parameters)
            if isinstance(model.parameters, str)
            else model.parameters
        ),
        description=model.description,
        trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        promoted_at=(
            model.promotion_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if model.promotion_timestamp
            else None
        ),
    )


# ---------------------------------------------------------------------------
# /models/remove endpoint
# ---------------------------------------------------------------------------
@router.delete(
    "/models/remove",
    summary="Remove a model from the registry",
    description=(
        "Delete a specific model version from the registry and remove its "
        "associated model file from storage. "
        "The environment (dev, staging, production) must be specified."
    ),
    tags=["models"],
    responses={
        404: {
            "description": "Dataset or model version not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found"}
                }
            },
        },
        500: {
            "description": "Error deleting the model file from storage",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to delete object from storage"
                    }
                }
            },
        },
        200: {
            "description": "Model removed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": (
                            "Model version 1 removed successfully from "
                            "staging"
                        ),
                        "dataset": "co2_emission",
                        "version": 1,
                        "environment": "staging",
                    }
                }
            },
        },
    },
)
def remove_model(
    request_data: RemoveRequest,
    db: Session = Depends(get_db),
):
    dataset_name = request_data.dataset_name
    version = request_data.version
    environment = request_data.environment.lower()

    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        raise HTTPException(status_code=404, detail="Dataset not found")

    entry = (
        db.query(models.ModelRegistry)
        .filter(
            models.ModelRegistry.dataset_id == dataset_entry.id,
            models.ModelRegistry.version == version,
            models.ModelRegistry.environment == environment,
        )
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Model version not found in specified environment",
        )

    try:
        delete_object_from_bucket(TRAINED_MODELS_BUCKET, entry.artifact_path)
    except HTTPException:
        raise HTTPException(
            status_code=500, detail="Failed to delete object from storage"
        )

    db.delete(entry)
    db.commit()

    return {
        "message": (
            f"Model version {version} removed successfully from "
            f"{environment}"
        ),
        "dataset": dataset_name,
        "version": version,
        "environment": environment,
    }

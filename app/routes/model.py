import io
import json
import logging
import pickle
from typing import List, Optional
from datetime import datetime
from importlib import import_module

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from sqlalchemy.orm import Session
from sklearn import datasets
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from app.db.session import get_db
from app.db import models
from app.models.pydantic_models import (
    ModelDetailResponse,
    ModelListResponse,
    ModelResponse,
    InputData,
    TrainResponse,
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


@router.post(
    "/train",
    summary="Train a model on a dataset",
    description=(
        "Trains a new machine learning model using the specified dataset, "
        "target column, and model configuration. "
        "The trained model is then stored and registered."
    ),
    tags=["models"],
    response_model=TrainResponse,
    responses={
        400: {"description": "Invalid request or model configuration"},
        404: {"description": "Dataset not found"},
        200: {
            "description": "Returns details about the trained model",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model trained and registered successfully",
                        "dataset": "CO2_emission",
                        "version": 1,
                        "model_file": "CO2_emission_model_dev_v1.pkl",
                        "metrics": {
                            "f1_score": 0.95,
                            "accuracy": 0.97,
                            "precision": {"0": 0.96, "1": 0.94},
                            "recall": {"0": 0.98, "1": 0.93},
                        },
                    }
                }
            },
        },
    },
)
def train_model(
    dataset_name: str = Query(
        ...,
        example="CO2_emission",
        description="Name of the dataset to use for training",
    ),
    use_example: bool = Query(
        False,
        example=False,
        description="Whether to use an example dataset (Iris)",
    ),
    target_column: Optional[str] = Query(
        None,
        example="Smog_Level",
        description="Column to be used as the target variable",
    ),
    model_module: str = Query(
        ...,
        example="sklearn.ensemble",
        description="Module path of the model (e.g., sklearn.ensemble)",
    ),
    model_class: str = Query(
        ...,
        example="RandomForestClassifier",
        description="Class name of the model (e.g., RandomForestClassifier)",
    ),
    model_params: Optional[str] = Query(
        "{}",
        example='{"n_estimators": 100}',
        description="JSON string of model hyperparameters",
    ),
    test_size: Optional[float] = Query(
        0.2,
        example=0.2,
        description="Proportion of data used for testing (between 0 and 1)",
    ),
    random_state: Optional[int] = Query(
        42, example=42, description="Random seed for reproducibility"
    ),
    db: Session = Depends(get_db),
):
    """
    Trains a machine learning model on a dataset.

    **Args**:
    - dataset_name (str): Name of the dataset to use for training.
    - use_example (bool): Whether to use an example dataset (Iris).
    - target_column (Optional[str]): Column in the dataset to be used as the target variable.
    - model_module (str): Full module path of the ML model (e.g., sklearn.ensemble).
    - model_class (str): Class name of the ML model (e.g., RandomForestClassifier).
    - model_params (Optional[str]): JSON string of model hyperparameters.
    - test_size (Optional[float]): Fraction of data to use for testing.
    - random_state (Optional[int]): Random seed for reproducibility.
    - db (Session): SQLAlchemy database session.

    **Returns**:
    - (TrainResponse): A response containing the model name, version, and evaluation metrics.
    """

    if use_example:
        dataset_name = "iris-sample"
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

    # Load model dynamically
    try:
        module = import_module(model_module)
        ModelClass = getattr(module, model_class)
        parsed_params = (
            json.loads(model_params) if model_params else {}
        )  # Convert JSON string to dictionary
        model_instance = ModelClass(**parsed_params)
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid model configuration"
        )

    # Train model
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

    # Register model
    existing_versions = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.dataset_id == dataset_entry.id)
        .all()
    )
    version = (
        max([entry.version for entry in existing_versions], default=0) + 1
    )
    environment = "dev"
    model_file_name = f"{dataset_name}_model_{environment}_v{version}.pkl"

    # Save model to storage
    model_data = io.BytesIO()
    pickle.dump(full_pipeline, model_data)
    model_data.seek(0)
    put_object_to_bucket(
        TRAINED_MODELS_BUCKET, model_file_name, model_data.getvalue()
    )

    # Store metadata
    model_params_dict = parsed_params
    model_params_dict["feature_names"] = feature_names

    new_registry_entry = models.ModelRegistry(
        dataset_id=dataset_entry.id,
        version=version,
        stage=environment,
        artifact_path=model_file_name,
        metrics=json.dumps(metrics),
        parameters=json.dumps(model_params_dict),
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


@router.post(
    "/promote",
    summary="Promote a model to a new stage",
    description=(
        "Promote a model (identified by dataset name and version) from its"
        "current stage (e.g., dev) to a "
        "higher stage (e.g., staging or production). "
        "Demotion or re-promotion to the same stage is not allowed."
    ),
    tags=["models"],
    responses={
        400: {"description": "Cannot demote or promote to the same stage"},
        404: {"description": "Dataset or model version not found"},
        200: {
            "description": (
                "Returns a message confirming the successful promotion."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model promoted to production successfully",
                        "dataset": "CO2_emission",
                        "version": 1,
                        "model_file": "CO2_emission_model_production_v1.pkl",
                    }
                }
            },
        },
    },
)
def promote_model(
    dataset_name: str = Body(..., embed=True, example={"co2_emission"}),
    version: int = Body(..., embed=True, example={1}),
    environment: str = Body(..., embed=True, example={"production"}),
    db: Session = Depends(get_db),
):
    """
    Promote a trained model to a higher stage
    (e.g., dev -> staging -> production).

    **Args**:
    - dataset_name (str): Name of the dataset/model family.
    - version (int): Model version number to promote.
    - target_stage (str): The new stage to promote the model to
      (e.g., staging or production).

    **Returns**:
    - (dict): A message confirming the promotion, along with metadata.
    """
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
    if promotion_order[environment] <= promotion_order[entry.stage]:
        raise HTTPException(
            status_code=400,
            detail="Cannot demote or promote to the same stage",
        )
    new_model_file = f"{dataset_name}_model_{environment}_v{version}.pkl"
    artifact_bytes = get_object_from_bucket(
        TRAINED_MODELS_BUCKET, entry.artifact_path
    )["Body"].read()
    put_object_to_bucket(TRAINED_MODELS_BUCKET, new_model_file, artifact_bytes)
    entry.stage = environment
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


@router.post(
    "/predict",
    summary="Make predictions using a deployed model",
    description=(
        "Use this endpoint to send data for inference using a model in a "
        "specific environment (dev, staging, or production). "
        "The endpoint will load the appropriate model pipeline and "
        "run prediction on the input data."
    ),
    tags=["models"],
    responses={
        404: {
            "description": (
                "Dataset or model stage not found (no model in that stage)."
            )
        },
        200: {
            "description": (
                "Returns a list of predictions for each provided input row."
            ),
            "content": {
                "application/json": {"example": {"predictions": [0, 1, 1]}}
            },
        },
    },
)
def predict(
    dataset_name: str = Body(..., embed=True, example={"co2_emission"}),
    environment: str = Body(..., embed=True, example={"production"}),
    input_data: InputData = Body(
        ...,
        example={
            "features": [
                [120.5, 75.3, 1],  # Example row
                [145.2, 80.1, 0],  # Another example row
            ]
        },
    ),
    db: Session = Depends(get_db),
):
    """
    Perform prediction using the latest model in the given environment.

    **Args**:
    - dataset_name (str): Name of the dataset/model family.
    - environment (str): The environment from which to load the model
     (dev, staging, or production).
    - input_data (InputData): The input features for inference.

    **Returns**:
    - (dict): A dictionary containing the model predictions.
    """
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
            models.ModelRegistry.stage == environment,
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
    input_df = pd.DataFrame(input_data.features)
    params = json.loads(entry.parameters)
    if "feature_names" in params:
        input_df.columns = params["feature_names"]
    predictions = model_pipeline.predict(input_df)
    return {"predictions": predictions.tolist()}


@router.get(
    "/models",
    summary="List all models in the registry",
    description=(
        "Retrieve a list of all models registered in the system, "
        "including their dataset name, version, stage, and artifact path."
    ),
    tags=["models"],
    response_model=List[ModelResponse],
)
def list_models(db: Session = Depends(get_db)):
    """
    List all models registered in the system.

    **Args**:
    - db (Session): SQLAlchemy session dependency.

    **Returns**:
    - (List[ModelRegistry]): A list of all registered models.
    """
    return db.query(models.ModelRegistry).all()


@router.get(
    "/models/by-dataset",
    summary="List models for a specific dataset",
    description="Retrieve all models associated with a dataset in a structured response.",
    tags=["models"],
    response_model=List[ModelListResponse],  # ✅ Now using response model
    responses={
        404: {"description": "Dataset not found"},
        200: {
            "description": "Returns a list of models for the dataset",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 3,
                            "name": "co2_emission_model_staging_v1",
                            "version": 1,
                            "environment": "staging",
                            "dataset_name": "CO2_emission",
                            "f1_score": 0.93,
                            "accuracy": 0.95,
                            "trained_at": "2024-01-30 12:45:00",
                            "promoted_at": "2024-02-02 14:10:00",
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
    """List models by dataset with structured response."""
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
        ModelListResponse(
            id=model.id,
            name=extract_model_name(model.artifact_path),
            version=model.version,
            environment=model.stage,
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


@router.get(
    "/models/by-environment",
    summary="List models by environment",
    description="Retrieve all models in a specific environment with structured response.",
    tags=["models"],
    response_model=List[
        ModelListResponse
    ],  # ✅ Using structured response model
    responses={
        400: {"description": "Invalid environment specified"},
        200: {
            "description": "Returns a list of models for the specified environment",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 5,
                            "name": "fraud_detection_model_production_v2",
                            "version": 2,
                            "environment": "production",
                            "dataset_name": "Fraud_Detection",
                            "f1_score": 0.92,
                            "accuracy": 0.97,
                            "trained_at": "2024-02-05 08:30:00",
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
    """List models by environment with structured response."""
    models_list = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.stage == environment.value)
        .all()
    )

    return [
        ModelListResponse(
            id=model.id,
            name=extract_model_name(model.artifact_path),
            version=model.version,
            environment=model.stage,
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


@router.get(
    "/models/{model_id}",
    summary="Get detailed model information",
    description="Retrieve all details of a specific model, including parameters, full metrics, and storage path.",
    tags=["models"],
    response_model=ModelDetailResponse,  # ✅ Using detailed response model
    responses={
        404: {"description": "Model not found"},
        200: {
            "description": "Returns detailed information about a model",
            "content": {
                "application/json": {
                    "example": {
                        "id": 3,
                        "name": "co2_emission_model_staging_v1",
                        "version": 1,
                        "environment": "staging",
                        "dataset_name": "CO2_emission",
                        "artifact_path": "co2_emission_model_staging_v1.pkl",
                        "metrics": {
                            "accuracy": 0.95,
                            "f1_score": 0.93,
                            "precision": {"0": 0.96, "1": 0.94},
                            "recall": {"0": 0.98, "1": 0.93},
                        },
                        "parameters": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "feature_names": [
                                "engine_size",
                                "fuel_consumption",
                                "CO2_emission",
                            ],
                        },
                        "description": "CO2 emissions prediction model",
                        "trained_at": "2024-01-30 12:45:00",
                        "promoted_at": "2024-02-02 14:10:00",
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
    """Get full details of a specific model."""
    model = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.id == model_id)
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_name = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.id == model.dataset_id)
        .first()
        .name
    )

    return ModelDetailResponse(
        id=model.id,
        name=extract_model_name(model.artifact_path),
        version=model.version,
        environment=model.stage,
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


@router.delete(
    "/models/remove",
    summary="Remove a model from the registry",
    description=(
        "Deletes a specific model version from the registry. "
        "This also removes the associated model file from object storage. "
        "The environment (dev, staging, production) must be specified."
    ),
    tags=["models"],
    responses={
        404: {"description": "Dataset or model version not found"},
        200: {
            "description": "Model removed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": (
                            "Model version 1 removed successfully from staging"
                        ),
                        "dataset": "CO2_emission",
                        "version": 1,
                        "environment": "staging",
                    }
                }
            },
        },
        500: {
            "description": "Error deleting the model file from object storage",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to delete object from storage"
                    }
                }
            },
        },
    },
)
def remove_model(
    dataset_name: str = Body(..., embed=True, example="CO2_emission"),
    version: int = Body(..., embed=True, example=1),
    environment: EnvironmentEnum = Body(
        ..., embed=True, example=EnvironmentEnum.staging
    ),  # ✅ Use Enum for validation
    db: Session = Depends(get_db),
):
    """
    Remove a model from the registry.

    **Args**:
    - dataset_name (str): Name of the dataset/model family.
    - version (int): Model version number to delete.
    - environment (EnvironmentEnum): The environment where the model is stored
      (dev, staging, production).

    **Returns**:
    - (dict): A message confirming the model deletion.
    """
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
            models.ModelRegistry.stage
            == environment.value,  # ✅ Use .value to extract string from Enum
        )
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Model version not found in specified environment",
        )

    # Delete model file from object storage
    try:
        delete_object_from_bucket(TRAINED_MODELS_BUCKET, entry.artifact_path)
    except HTTPException:
        raise HTTPException(
            status_code=500, detail="Failed to delete object from storage"
        )

    # Remove model entry from database
    db.delete(entry)
    db.commit()

    return {
        "message": (
            f"Model version {version} removed successfully from "
            f"{environment.value}"
        ),
        "dataset": dataset_name,
        "version": version,
        "environment": environment.value,
    }

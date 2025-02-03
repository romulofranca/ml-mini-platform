from importlib import import_module
import io
import json
import logging
from datetime import datetime
from typing import List, Any

import joblib
import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
)

from sqlalchemy.orm import Session
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.db.session import get_db
from app.db import models
from app.models.pydantic_models import (
    PromoteResponse,
    RemoveResponse,
    TrainRequest,
    TrainResponse,
    ModelResponse,
    ModelDetailResponse,
    PromoteRequest,
    PredictRequest,
    RemoveRequest,
    PredictResponse,
)
from app.utils.constants import MODEL_MAPPING, EnvironmentEnum
from app.utils.object_storage import (
    delete_object_from_bucket,
    put_object_to_bucket,
    get_object_from_bucket,
    load_dataset_from_storage,
)
from app.utils.preprocessing import create_preprocessor
from app.config import TRAINED_MODELS_BUCKET
from app.utils.problem_type import detect_problem_type
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
                        "environment": "dev",
                        "version": 1,
                        "features": [
                            "Model_Year",
                            "Make",
                            "Model",
                            "Vehicle_Class",
                            "Engine_Size",
                            "Cylinders",
                            "Transmission",
                            "Fuel_Consumption_in_City(L/100 km)",
                            "Fuel_Consumption_in_City_Hwy(L/100 km)",
                            "Fuel_Consumption_comb(L/100km)",
                            "CO2_Emissions",
                        ],
                        "model_file": "co2_emission_model_dev_v1.joblib",
                        "metrics": {
                            "accuracy": 0.5240641711229946,
                            "precision": 0.6390650057529853,
                            "recall": 0.5240641711229946,
                            "f1_score": 0.4454557503075632,
                            "classification_report": {
                                "1": {
                                    "precision": 1,
                                    "recall": 0,
                                    "f1-score": 0,
                                    "support": 17,
                                },
                            },
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
    # Input validation and preprocessing
    dataset_name = request_data.dataset_name
    target_column = request_data.target_column
    test_size = request_data.test_size or 0.2
    random_state = request_data.random_state or 42

    try:
        model_class = request_data.model["model_class"]
        model_params = request_data.model.get("model_params", {})
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")

    # Load dataset
    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_from_storage(dataset_entry.location)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

    # Determine problem type
    if target_column is None:
        problem_type = "unsupervised"
        X_df = df
        logger.warning(
            "⚠️ No target_column provided. Treating as unsupervised learning."
        )
    else:
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' not found in dataset."
                ),
            )
        y = df[target_column]
        X_df = df.drop(columns=[target_column])
        problem_type = detect_problem_type(y)

    feature_names = list(X_df.columns)

    # Dynamically import model class
    module_name = MODEL_MAPPING.get(model_class, "sklearn.ensemble")
    try:
        module = import_module(module_name)
        ModelClass = getattr(module, model_class)
        model_instance = ModelClass(**model_params)
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid model class: {model_class}"
        )

    # Create ML pipeline
    preprocessor = create_preprocessor(X_df)
    full_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model_instance)]
    )

    # Train model and calculate metrics
    metrics = {}
    if problem_type in ["classification", "regression"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=random_state
        )
        try:
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise HTTPException(
                status_code=500, detail="Model training failed"
            )

        if problem_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=1
            )
            metrics["recall"] = recall_score(
                y_test, y_pred, average="weighted", zero_division=1
            )
            metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
            metrics["classification_report"] = classification_report(
                y_test, y_pred, output_dict=True, zero_division=1
            )
        elif problem_type == "regression":
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["RMSE"] = root_mean_squared_error(y_test, y_pred)
            metrics["R2_Score"] = r2_score(y_test, y_pred)
    else:
        try:
            full_pipeline.fit(X_df)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise HTTPException(
                status_code=500, detail="Model training failed"
            )

        if model_class == "KMeans":
            y_pred = full_pipeline.named_steps["model"].labels_
            metrics["silhouette_score"] = silhouette_score(X_df, y_pred)
            metrics["davies_bouldin_score"] = davies_bouldin_score(
                X_df, y_pred
            )
        elif model_class == "IsolationForest":
            y_pred = full_pipeline.named_steps["model"].predict(X_df)
            metrics["anomalies_detected"] = int((y_pred == -1).sum())

    # Save and register the model
    existing_versions = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.dataset_id == dataset_entry.id)
        .all()
    )
    version = (
        max([entry.version for entry in existing_versions], default=0) + 1
    )

    environment = "dev"
    model_file_name = f"{dataset_name}_model_{environment}_v{version}.joblib"
    model_data = io.BytesIO()
    joblib.dump(full_pipeline, model_data)
    model_data.seek(0)
    put_object_to_bucket(
        TRAINED_MODELS_BUCKET, model_file_name, model_data.getvalue()
    )

    new_registry_entry = models.ModelRegistry(
        dataset_id=dataset_entry.id,
        version=version,
        environment=environment,
        artifact_path=model_file_name,
        metrics=json.dumps(metrics),
        parameters=json.dumps(model_params),
        target_column=target_column,
        feature_names=json.dumps(feature_names),
        description="Model trained automatically",
        promotion_timestamp=None,
    )

    db.add(new_registry_entry)
    db.commit()
    db.refresh(new_registry_entry)

    return TrainResponse(
        message="Model trained and registered successfully",
        model_file=model_file_name,
        dataset=dataset_name,
        environment=environment,
        version=version,
        target=target_column,
        features=feature_names,
        metrics=metrics,
    )


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
    return PromoteResponse(
        message=f"Model promoted to {environment} successfully",
        model_name=extract_model_name(new_model_file),
        dataset=dataset_name,
        version=version,
        environment=environment,
        promoted_at=entry.promotion_timestamp,
    )


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
@router.post(
    "/predict",
    summary="Make predictions using a deployed model",
    tags=["models"],
)
def predict(request_data: PredictRequest, db: Session = Depends(get_db)):
    dataset_name = request_data.dataset_name
    environment = request_data.environment.lower()
    input_features = request_data.features  # List of dictionaries
    version = request_data.version  # Optional: Specify version

    # Load dataset entry from DB
    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Query model registry for the specific version (if provided) or latest
    query = db.query(models.ModelRegistry).filter(
        models.ModelRegistry.dataset_id == dataset_entry.id,
        models.ModelRegistry.environment == environment,
    )

    if version:
        query = query.filter(models.ModelRegistry.version == version)

    query = query.order_by(models.ModelRegistry.version.desc())
    entry = query.first()

    if not entry:
        raise HTTPException(
            status_code=404,
            detail="No model found for the given environment and version",
        )

    # Load model from object storage
    try:
        model_response = get_object_from_bucket(
            TRAINED_MODELS_BUCKET, entry.artifact_path
        )
        model_pipeline = joblib.load(io.BytesIO(model_response["Body"].read()))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

    # Load feature names from the model registry
    feature_names = json.loads(entry.feature_names)

    # Convert input features to DataFrame
    input_df = pd.DataFrame(input_features)

    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_df.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features in input: {missing_features}",
        )

    # Ensure no extra features are provided
    extra_features = set(input_df.columns) - set(feature_names)
    if extra_features:
        raise HTTPException(
            status_code=400,
            detail=f"Extra features provided: {extra_features}",
        )

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Ensure consistent data types
    for column in input_df.columns:
        if (
            input_df[column].dtype == object
        ):  # Convert object (string) columns to categorical
            input_df[column] = input_df[column].astype("category")

    # Get predictions
    try:
        predictions = model_pipeline.predict(input_df).tolist()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictResponse(
        dataset=dataset_name,
        environment=environment,
        version=entry.version,
        predictions=predictions,
    )


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
                            "name": "co2_emission_model_production_v1",
                            "version": 1,
                            "environment": "production",
                            "dataset_name": "co2_emission",
                            "features": [
                                "Model_Year",
                                "Make",
                                "Model",
                                "Vehicle_Class",
                                "Engine_Size",
                                "Cylinders",
                                "Transmission",
                                "Fuel_Consumption_in_City(L/100 km)",
                                "Fuel_Consumption_in_City_Hwy(L/100 km)",
                                "Fuel_Consumption_comb(L/100km)",
                                "CO2_Emissions",
                            ],
                            "trained_at": "2025-02-03T22:17:26",
                            "promoted_at": "2025-02-03T22:18:05",
                        },
                    ]
                }
            },
        }
    },
)
def list_models(db: Session = Depends(get_db)):
    models_list = db.query(models.ModelRegistry).all()

    return [
        ModelResponse(
            id=model.id,
            name=extract_model_name(model.artifact_path),
            version=model.version,
            environment=model.environment,
            dataset_name=db.query(models.DatasetCatalog)
            .filter(models.DatasetCatalog.id == model.dataset_id)
            .first()
            .name,
            features=json.loads(model.feature_names),
            trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            promoted_at=(
                model.promotion_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if model.promotion_timestamp
                else model.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ),
        )
        for model in models_list
    ]


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
                            "name": "co2_emission_model_production_v1",
                            "version": 1,
                            "environment": "production",
                            "dataset_name": "co2_emission",
                            "features": [
                                "Model_Year",
                                "Make",
                                "Model",
                                "Vehicle_Class",
                                "Engine_Size",
                                "Cylinders",
                                "Transmission",
                                "Fuel_Consumption_in_City(L/100 km)",
                                "Fuel_Consumption_in_City_Hwy(L/100 km)",
                                "Fuel_Consumption_comb(L/100km)",
                                "CO2_Emissions",
                            ],
                            "trained_at": "2025-02-03T22:17:26",
                            "promoted_at": "2025-02-03T22:18:05",
                        },
                    ]
                }
            },
        },
    },
)
def list_models_by_dataset(
    dataset_name: str = Query(..., example="co2_emission"),
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
            environment=model.environment,
            dataset_name=dataset_name,
            features=json.loads(model.feature_names),
            trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            promoted_at=(
                model.promotion_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if model.promotion_timestamp
                else model.timestamp.strftime("%Y-%m-%d %H:%M:%S")
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
                            "id": 2,
                            "name": "co2_emission_model_staging_v2",
                            "version": 2,
                            "environment": "staging",
                            "dataset_name": "co2_emission",
                            "features": [
                                "Model_Year",
                                "Make",
                                "Model",
                                "Vehicle_Class",
                                "Engine_Size",
                                "Cylinders",
                                "Transmission",
                                "Fuel_Consumption_in_City(L/100 km)",
                                "Fuel_Consumption_in_City_Hwy(L/100 km)",
                                "Fuel_Consumption_comb(L/100km)",
                                "CO2_Emissions",
                            ],
                            "trained_at": "2025-02-03T22:47:36",
                            "promoted_at": "2025-02-03T22:51:05",
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
        .filter(models.ModelRegistry.environment == environment.value)
        .all()
    )
    return [
        ModelResponse(
            id=model.id,
            name=extract_model_name(model.artifact_path),
            version=model.version,
            environment=model.environment,
            dataset_name=db.query(models.DatasetCatalog)
            .filter(models.DatasetCatalog.id == model.dataset_id)
            .first()
            .name,
            features=json.loads(model.feature_names),
            trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            promoted_at=(
                model.promotion_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if model.promotion_timestamp
                else model.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ),
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
                        "features": [
                            "Model_Year",
                            "Make",
                            "Model",
                            "Vehicle_Class",
                            "Engine_Size",
                            "Cylinders",
                            "Transmission",
                            "Fuel_Consumption_in_City(L/100 km)",
                            "Fuel_Consumption_in_City_Hwy(L/100 km)",
                            "Fuel_Consumption_comb(L/100km)",
                            "CO2_Emissions",
                        ],
                        "metrics": {
                            "accuracy": 0.5240641711229946,
                            "precision": 0.6390650057529853,
                            "recall": 0.5240641711229946,
                            "f1_score": 0.4454557503075632,
                            "classification_report": {
                                "1": {
                                    "precision": 1,
                                    "recall": 0,
                                    "f1-score": 0,
                                    "support": 17,
                                },
                                "3": {
                                    "precision": 0.5416666666666666,
                                    "recall": 0.3333333333333333,
                                    "f1-score": 0.4126984126984127,
                                    "support": 39,
                                },
                                "5": {
                                    "precision": 0.48717948717948717,
                                    "recall": 0.8507462686567164,
                                    "f1-score": 0.6195652173913043,
                                    "support": 67,
                                },
                                "6": {
                                    "precision": 1,
                                    "recall": 0,
                                    "f1-score": 0,
                                    "support": 25,
                                },
                                "7": {
                                    "precision": 0.6086956521739131,
                                    "recall": 0.717948717948718,
                                    "f1-score": 0.6588235294117647,
                                    "support": 39,
                                },
                                "accuracy": 0.5240641711229946,
                                "macro avg": {
                                    "precision": 0.7275083612040134,
                                    "recall": 0.3804056639877535,
                                    "f1-score": 0.33821743190029635,
                                    "support": 187,
                                },
                                "weighted avg": {
                                    "precision": 0.6390650057529853,
                                    "recall": 0.5240641711229946,
                                    "f1-score": 0.4454557503075632,
                                    "support": 187,
                                },
                            },
                        },
                        "parameters": {"max_depth": 5, "n_estimators": 100},
                        "description": "Model trained automatically",
                        "trained_at": "2025-02-03T22:17:26",
                        "promoted_at": "2025-02-03T22:18:05",
                    }
                }
            },
        },
    },
)
def get_model_by_id(
    model_id: int = Path(..., example=1),
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
        features=json.loads(model.feature_names),
        trained_at=model.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        promoted_at=(
            model.promotion_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if model.promotion_timestamp
            else model.timestamp.strftime("%Y-%m-%d %H:%M:%S")
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
                            "Model version 2 removed successfully from staging"
                        ),
                        "dataset": "co2_emission",
                        "version": 2,
                        "environment": "staging",
                        "removed_at": "2025-02-03T22:55:03.020978",
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

    return RemoveResponse(
        message=(
            f"Model version {version} removed successfully from {environment}"
        ),
        dataset=dataset_name,
        version=version,
        environment=environment,
        removed_at=datetime.now(),
    )

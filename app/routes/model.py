import io
import json
import logging
import os
import pickle
import time
import uuid
from datetime import datetime
from importlib import import_module

import pandas as pd
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from sklearn import datasets
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.models.pydantic_models import TrainRequest, InputData
from app.utils.object_storage import (
    object_storage_client,
    load_dataset_from_storage,
    list_objects,
)
from app.config import TRAINED_MODELS_BUCKET
from app.utils.preprocessing import create_preprocessor, detect_target_column

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/list-models", summary="List trained models", tags=["Model"])
def list_models():
    try:
        models_list = list_objects(bucket=TRAINED_MODELS_BUCKET)
        return {"models": models_list}
    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post(
    "/train",
    summary="Train a model",
    tags=["Model"],
    response_description="Returns training metrics and the model storage path",
)
def train_model(
    background_tasks: BackgroundTasks,
    request: TrainRequest = Body(...),
):
    try:
        if request.use_example:
            iris = datasets.load_iris()
            df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            df["target"] = iris.target
            dataset_identifier = "iris-sample"
            logger.info("Using example Iris dataset for training.")
        else:
            if not request.dataset_path:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset path is required when not using example data."
                    ),
                )
            df = load_dataset_from_storage(request.dataset_path)
            dataset_identifier = request.dataset_path

        # Determine the target column
        target_column = request.target_column or detect_target_column(df)
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' not found in dataset."
                ),
            )
        logger.info(f"Using target column: {target_column}")

        # Separate features and target
        X_df = df.drop(columns=[target_column])
        y = df[target_column]
        feature_names = X_df.columns.tolist()

        # Create the preprocessor
        preprocessor = create_preprocessor(X_df)

        # Dynamically load the ML model class
        try:
            module_name = request.model.get("module")
            class_name = request.model.get("class")
            if not module_name or not class_name:
                raise ValueError(
                    "Model configuration must include 'module' and 'class'."
                )
            if not module_name.startswith("sklearn."):
                module_name = "sklearn." + module_name
            module = import_module(module_name)
            ModelClass = getattr(module, class_name)
            model_instance = ModelClass(**request.model.get("params", {}))
            logger.info(
                f"Model {class_name} loaded successfully from {module_name}."
            )
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=400, detail="Invalid model configuration"
            )

        # Combine preprocessor and model into a single pipeline
        full_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model_instance)]
        )

        test_size = request.test_size or 0.2
        random_state = request.random_state or 42
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=random_state
        )
        logger.info("Data split into training and testing sets.")

        start_time = time.time()
        full_pipeline.fit(X_train, y_train)
        logger.info(
            f"Model training completed in "
            f"{time.time() - start_time:.2f} seconds."
        )

        y_pred = full_pipeline.predict(X_test)
        metrics = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
        logger.info("Model evaluation completed.")

        # Generate version and prepare storage path
        model_version = str(uuid.uuid4())
        model_dir = f"{dataset_identifier}/version-{model_version}"
        model_file_path = f"{model_dir}/model.pkl"
        metadata = {
            "version": model_version,
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": request.model.get("params", {}),
            "feature_names": feature_names,
        }

        # Serialize the pipeline and save it to object storage
        model_data = io.BytesIO()
        pickle.dump(full_pipeline, model_data)
        model_data.seek(0)
        object_storage_client.put_object(
            Bucket=TRAINED_MODELS_BUCKET,
            Key=model_file_path,
            Body=model_data.getvalue(),
        )
        object_storage_client.put_object(
            Bucket=TRAINED_MODELS_BUCKET,
            Key=f"{model_dir}/metadata.json",
            Body=json.dumps(metadata),
        )
        logger.info(f"Model and metadata saved at '{model_dir}'.")

        background_tasks.add_task(monitor_model_performance, model_file_path)

        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "model_path": model_file_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/predict",
    summary="Make predictions using a trained model",
    tags=["Model"],
    response_description="Returns predictions for the input data",
)
def predict(
    model_path: str = Body(..., embed=True),
    input_data: InputData = Body(...),
):
    try:
        if not input_data.features:
            raise HTTPException(
                status_code=400, detail="No input data provided."
            )

        response = object_storage_client.get_object(
            Bucket=TRAINED_MODELS_BUCKET, Key=model_path
        )
        model_pipeline = pickle.loads(response["Body"].read())

        # Retrieve metadata to obtain the feature names
        model_dir = os.path.dirname(model_path)
        metadata_key = f"{model_dir}/metadata.json"
        meta_response = object_storage_client.get_object(
            Bucket=TRAINED_MODELS_BUCKET, Key=metadata_key
        )
        metadata = json.loads(meta_response["Body"].read())
        feature_names = metadata.get("feature_names")
        if not feature_names:
            raise HTTPException(
                status_code=500,
                detail="Model metadata does not include feature names.",
            )

        expected_features = len(feature_names)
        for idx, sample in enumerate(input_data.features):
            if len(sample) != expected_features:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Sample {idx} has {len(sample)} features; "
                        f"expected {expected_features}."
                    ),
                )

        input_df = pd.DataFrame(input_data.features, columns=feature_names)
        predictions = model_pipeline.predict(input_df)
        logger.info("Predictions made successfully.")
        return {"predictions": predictions.tolist()}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to make predictions: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to make predictions"
        )


def monitor_model_performance(model_path: str):
    logger.info(f"Monitoring model performance for {model_path}...")
    # TODO: Implement model performance monitoring logic

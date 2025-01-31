import io
import json
import logging
import os
import pickle
import traceback
import uuid
from datetime import datetime
from importlib import import_module
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd
import uvicorn
from botocore.client import Config
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for S3/MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
DATASETS_BUCKET = os.getenv("DATASETS_BUCKET")
TRAINED_MODELS_BUCKET = os.getenv("TRAINED_MODELS_BUCKET")

# Initialize boto3 S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

# Initialize FastAPI app
app = FastAPI(
    title="ML Mini Platform",
    description="A platform for training and serving ML models",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic models
# ---------------------------
class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    use_example: Optional[bool] = False
    model: dict
    target_column: Optional[str] = None
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42

class InputData(BaseModel):
    features: List[List[float]]

# ---------------------------
# Helper Functions
# ---------------------------
def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detects the target column from a DataFrame using a list of possible names.
    """
    possible_target_columns = ["target", "label", "class", "y"]
    for col in possible_target_columns:
        if col in df.columns:
            return col
    return df.columns[-1]

def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a preprocessor for numeric and categorical columns.
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    logger.info("Preprocessor created successfully.")
    return preprocessor

def load_dataset_from_s3(dataset_path: str) -> pd.DataFrame:
    """
    Loads a dataset from S3 given its path.
    """
    try:
        response = s3_client.get_object(Bucket=DATASETS_BUCKET, Key=dataset_path)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info(f"Dataset '{dataset_path}' loaded successfully.")
        return df
    except Exception as e:
        logger.exception(f"Failed to load dataset '{dataset_path}': {e}")
        raise HTTPException(status_code=404, detail="Dataset not found")

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/", summary="Check API status", tags=["General"])
def home():
    """
    API status endpoint.
    """
    return {"message": "ML Mini Platform is running!"}

@app.post("/upload", summary="Upload dataset", tags=["Dataset"])
def upload_file(file: UploadFile = File(...)):
    """
    Uploads a dataset file to the storage.
    """
    try:
        file_content = file.file.read()
        s3_client.put_object(Bucket=DATASETS_BUCKET, Key=file.filename, Body=file_content)
        logger.info(f"File '{file.filename}' uploaded successfully.")
        return {"message": f"File '{file.filename}' uploaded successfully!"}
    except Exception as e:
        logger.exception(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@app.get("/list-datasets", summary="List available datasets", tags=["Dataset"])
def list_datasets():
    """
    Lists datasets available in the storage.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=DATASETS_BUCKET)
        datasets_list = [obj["Key"] for obj in response.get("Contents", [])] if response.get("Contents") else []
        logger.info("Datasets listed successfully.")
        return {"datasets": datasets_list}
    except Exception as e:
        logger.exception(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail="Failed to list datasets")

@app.get("/list-models", summary="List trained models", tags=["Model"])
def list_models():
    """
    Lists trained models available in the storage.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=TRAINED_MODELS_BUCKET)
        models_list = [obj["Key"] for obj in response.get("Contents", [])] if response.get("Contents") else []
        logger.info("Models listed successfully.")
        return {"models": models_list}
    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.post(
    "/train",
    summary="Train a model",
    tags=["Model"],
    response_description="Returns training metrics and the model storage path",
)
def train_model(
    background_tasks: BackgroundTasks,
    request: TrainRequest = Body(...),
):
    """
    Trains a model based on the provided configuration.
    It uses an example dataset (Iris) if requested or a custom dataset from S3.
    The training process automatically stores the feature names for later use.
    """
    try:
        # Load dataset: use the example Iris dataset or load from S3
        if request.use_example:
            iris = datasets.load_iris()
            df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            df["target"] = iris.target
            dataset_identifier = "iris-sample"
            logger.info("Using example Iris dataset for training.")
        else:
            if not request.dataset_path:
                raise HTTPException(
                    status_code=400, detail="Dataset path is required when not using example data."
                )
            df = load_dataset_from_s3(request.dataset_path)
            dataset_identifier = request.dataset_path

        # Determine target column
        target_column = request.target_column or detect_target_column(df)
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")
        logger.info(f"Using target column: {target_column}")

        # Separate features and target
        X_df = df.drop(columns=[target_column])
        y = df[target_column]

        # Capture the feature names to store in the model metadata.
        feature_names = X_df.columns.tolist()

        # Create preprocessing step
        preprocessor = create_preprocessor(X_df)

        # Dynamically load the ML model class
        try:
            module_name = request.model.get("module")
            class_name = request.model.get("class")
            if not module_name or not class_name:
                raise ValueError("Model configuration must include 'module' and 'class'.")
            # Optionally prefix with "sklearn." if not provided
            if not module_name.startswith("sklearn."):
                module_name = "sklearn." + module_name
            module = import_module(module_name)
            ModelClass = getattr(module, class_name)
            model_instance = ModelClass(**request.model.get("params", {}))
            logger.info(f"Model {class_name} loaded successfully from {module_name}.")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise HTTPException(status_code=400, detail="Invalid model configuration")

        # Combine preprocessing and model into a single pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_instance)
        ])

        # Split data into training and testing sets
        test_size = request.test_size if request.test_size is not None else 0.2
        random_state = request.random_state if request.random_state is not None else 42
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=random_state)
        logger.info("Data split into training and testing sets.")

        # Train the model pipeline
        full_pipeline.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Evaluate the model
        y_pred = full_pipeline.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
        logger.info("Model evaluation completed.")

        # Prepare model saving details
        model_version = str(uuid.uuid4())
        model_dir = f"{dataset_identifier}/version-{model_version}"
        model_file_path = f"{model_dir}/model.pkl"
        metadata = {
            "version": model_version,
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": request.model.get("params", {}),
            "feature_names": feature_names  # Save feature names for later use during prediction.
        }

        # Serialize the pipeline and save to S3
        model_data = io.BytesIO()
        pickle.dump(full_pipeline, model_data)
        model_data.seek(0)
        s3_client.put_object(Bucket=TRAINED_MODELS_BUCKET, Key=model_file_path, Body=model_data.getvalue())
        s3_client.put_object(Bucket=TRAINED_MODELS_BUCKET, Key=f"{model_dir}/metadata.json", Body=json.dumps(metadata))
        logger.info(f"Model and metadata saved at '{model_dir}'.")

        # Launch background task for future model performance monitoring
        background_tasks.add_task(monitor_model_performance, model_file_path)

        return {"message": "Model trained successfully", "metrics": metrics, "model_path": model_file_path}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during model training: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", summary="Make predictions using a trained model", tags=["Model"])
def predict(
    model_path: str = Body(..., embed=True),
    input_data: InputData = Body(...),
):
    """
    Loads a trained model pipeline from storage and makes predictions.
    The input features are automatically converted into a DataFrame using the
    feature names stored in the model metadata.
    """
    try:
        if not input_data.features:
            raise HTTPException(status_code=400, detail="No input data provided.")

        # Retrieve the model pipeline from storage
        response = s3_client.get_object(Bucket=TRAINED_MODELS_BUCKET, Key=model_path)
        model_pipeline = pickle.loads(response["Body"].read())

        # Derive the metadata file path (assumed to be in the same folder as the model)
        model_dir = os.path.dirname(model_path)
        metadata_key = f"{model_dir}/metadata.json"
        meta_response = s3_client.get_object(Bucket=TRAINED_MODELS_BUCKET, Key=metadata_key)
        metadata = json.loads(meta_response["Body"].read())
        feature_names = metadata.get("feature_names")
        if not feature_names:
            raise HTTPException(status_code=500, detail="Model metadata does not include feature names.")

        # Validate that each sample contains the expected number of features.
        expected_features = len(feature_names)
        for idx, sample in enumerate(input_data.features):
            if len(sample) != expected_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample {idx} has {len(sample)} features; expected {expected_features}."
                )

        # Convert the input features into a DataFrame with the correct column names.
        input_df = pd.DataFrame(input_data.features, columns=feature_names)

        # Make predictions using the model pipeline.
        predictions = model_pipeline.predict(input_df)
        logger.info("Predictions made successfully.")
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.exception(f"Failed to make predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to make predictions")

@app.get("/openapi.json", summary="Get OpenAPI schema", tags=["General"])
def get_openapi_schema():
    """
    Returns the OpenAPI schema for this API.
    """
    return get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)

def monitor_model_performance(model_path: str):
    """
    Placeholder for background model performance monitoring.
    """
    logger.info(f"Monitoring model performance for {model_path}...")
    # TODO: Implement model performance monitoring logic here.
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

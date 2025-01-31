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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
DATASETS_BUCKET = os.getenv("DATASETS_BUCKET")
TRAINED_MODELS_BUCKET = os.getenv("TRAINED_MODELS_BUCKET")

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

app = FastAPI(
    title="ML Mini Platform",
    description="A platform for training and serving ML models",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    use_example: bool = False
    model: dict
    target_column: Optional[str] = None

class InputData(BaseModel):
    features: List[List[float]]

def detect_target_column(df):
    possible_target_columns = ["target", "label", "class", "y"]
    for col in possible_target_columns:
        if col in df.columns:
            return col
    return df.columns[-1]

def preprocess_data(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

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
        ])

    logger.info("Preprocessing completed successfully.")
    return preprocessor.fit_transform(df)

@app.get("/", summary="Check API status", tags=["General"])
def home():
    return {"message": "ML Mini Platform is running!"}

@app.post("/upload", summary="Upload dataset", tags=["Dataset"])
def upload_file(file: UploadFile = File(...)):
    try:
        s3_client.put_object(Bucket=DATASETS_BUCKET, Key=file.filename, Body=file.file.read())
        return {"message": f"File {file.filename} uploaded successfully!"}
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@app.get("/list-datasets", summary="List available datasets", tags=["Dataset"])
def list_datasets():
    try:
        response = s3_client.list_objects_v2(Bucket=DATASETS_BUCKET)
        datasets = [obj["Key"] for obj in response.get("Contents", [])]
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail="Failed to list datasets")

@app.get("/list-models", summary="List trained models", tags=["Model"])
def list_models():
    try:
        response = s3_client.list_objects_v2(Bucket=TRAINED_MODELS_BUCKET)
        models = [obj["Key"] for obj in response.get("Contents", [])]
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.post(
    "/train",
    summary="Train a model",
    tags=["Model"],
    response_description="Returns the accuracy and path of the trained model",
    responses={
        200: {
            "description": "Model trained successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "example1": {
                            "summary": "Train a RandomForestClassifier on the Iris dataset",
                            "value": {
                                "message": "Model trained successfully",
                                "accuracy": 0.95,
                                "model_path": "iris-sample/version-1/model.pkl"
                            }
                        },
                        "example2": {
                            "summary": "Train a LogisticRegression on a custom dataset",
                            "value": {
                                "message": "Model trained successfully",
                                "accuracy": 0.92,
                                "model_path": "custom/version-1/model.pkl"
                            }
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid input or model configuration"},
        404: {"description": "Dataset not found"},
        500: {"description": "Internal server error"},
    },
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
            dataset_path = "iris-sample"
        else:
            if not request.dataset_path:
                raise HTTPException(status_code=400, detail="Dataset path is required")
            try:
                response = s3_client.get_object(Bucket=DATASETS_BUCKET, Key=request.dataset_path)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
                dataset_path = request.dataset_path
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise HTTPException(status_code=404, detail="Dataset not found")

        if request.target_column:
            target_column = request.target_column
        else:
            target_column = detect_target_column(df)
            logger.info(f"Target column detected automatically: {target_column}")

        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        num_classes = len(y.unique())
        logger.info(f"Number of classes in the dataset: {num_classes}")

        X = preprocess_data(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            module = import_module("sklearn." + request.model["module"])
            ModelClass = getattr(module, request.model["class"])
            model = ModelClass(**request.model.get("params", {}))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=400, detail="Invalid model configuration")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")

        model_version = str(uuid.uuid4())
        model_dir = f"{dataset_path}/version-{model_version}"
        model_file_path = f"{model_dir}/model.pkl"
        metadata = {
            "version": model_version,
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": request.model.get("params", {}),
        }

        model_data = io.BytesIO()
        pickle.dump(model, model_data)
        model_data.seek(0)
        s3_client.put_object(Bucket=TRAINED_MODELS_BUCKET, Key=model_file_path, Body=model_data.getvalue())
        s3_client.put_object(Bucket=TRAINED_MODELS_BUCKET, Key=f"{model_dir}/metadata.json", Body=json.dumps(metadata))

        background_tasks.add_task(monitor_model_performance, model_file_path)

        return {"message": "Model trained successfully", "metrics": metrics, "model_path": model_file_path}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", summary="Make predictions using a trained model", tags=["Model"])
def predict(model_path: str, input_data: InputData):
    try:
        if not input_data.features:
            raise HTTPException(status_code=400, detail="No input data provided.")

        response = s3_client.get_object(Bucket=TRAINED_MODELS_BUCKET, Key=model_path)
        model = pickle.loads(response["Body"].read())
        predictions = model.predict(np.array(input_data.features))
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to make predictions")

@app.get("/openapi.json", summary="Get OpenAPI schema", tags=["General"])
def get_openapi_schema():
    return get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)

def monitor_model_performance(model_path: str):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

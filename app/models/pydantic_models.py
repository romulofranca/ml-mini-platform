from typing import Any, List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import json


class DatasetResponse(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="iris_dataset")
    description: str = Field(..., example="Iris dataset")
    location: str = Field(..., example="s3://bucket/iris_dataset.csv")
    created_at: datetime = Field(..., example="2024-01-01T12:00:00")


class RemoveDatasetResponse(BaseModel):
    message: str = Field(..., example="Dataset deleted successfully")
    dataset: str = Field(..., example="iris_dataset")
    deleted_at: datetime = Field(..., example="2024-01-01T12:00:00")


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., example="iris_dataset")
    model: Dict[str, Any] = Field(
        ...,
        example={
            "model_class": "RandomForestClassifier",
            "model_params": {"n_estimators": 100, "max_depth": 5},
        },
    )
    target_column: Optional[str] = Field(None, example="Species")
    test_size: Optional[float] = Field(0.2, example=0.2)
    random_state: Optional[int] = Field(42, example=42)


class TrainResponse(BaseModel):
    message: str = Field(
        ..., example="Model trained and registered successfully"
    )
    dataset: str = Field(..., example="iris_dataset")
    environment: str = Field(..., example="dev")
    version: int = Field(..., example=1)
    target: str = Field(..., example="Smog_Level")
    features: List[str] = Field(..., example=["CO2", "Temperature", "Traffic"])
    model_file: str = Field(..., example="iris_dataset_model_dev_v1.pkl")
    metrics: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("metrics", mode="before")
    def parse_metrics(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class ModelResponse(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="iris_dataset_model_dev_v1")
    version: int = Field(..., example=1)
    environment: str = Field(..., example="dev")
    dataset_name: str = Field(..., example="iris_dataset")
    features: List[str] = Field(..., example=["CO2", "Temperature", "Traffic"])
    trained_at: datetime = Field(..., example="2024-01-01T12:00:00")
    promoted_at: Optional[datetime] = Field(
        None, example="2024-01-02T12:00:00"
    )


class ModelDetailResponse(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="iris_dataset_model_dev_v1")
    version: int = Field(..., example=1)
    environment: str = Field(..., example="dev")
    dataset_name: str = Field(..., example="iris_dataset")
    artifact_path: str = Field(..., example="iris_dataset_model_dev_v1.pkl")
    features: List[str] = Field(..., example=["CO2", "Temperature", "Traffic"])
    metrics: Dict[str, Any] = Field(
        ...,
        example={
            "accuracy": 0.95,
            "f1_score": 0.93,
            "precision": {"0": 0.94, "1": 0.96},
            "recall": {"0": 0.92, "1": 0.97},
        },
    )
    parameters: Dict[str, Any] = Field(..., example={"n_estimators": 100})
    description: str = Field(..., example="Trained model")
    trained_at: datetime = Field(..., example="2024-01-01T12:00:00")
    promoted_at: Optional[datetime] = Field(
        None, example="2024-01-02T12:00:00"
    )

    @field_validator("metrics", "parameters", mode="before")
    def parse_json_fields(cls, v):
        if isinstance(v, str):  # Handle stringified JSON
            return json.loads(v)
        return v


class PromoteRequest(BaseModel):
    dataset_name: str = Field(
        ..., example="iris_dataset", description="Dataset/model family name"
    )
    version: int = Field(
        ..., example=1, description="Model version to promote"
    )
    environment: str = Field(
        ...,
        example="production",
        description="Target environment for promotion (e.g., production)",
    )


class PromoteResponse(BaseModel):
    message: str = Field(..., example="Model promoted successfully")
    model_name: str = Field(..., example="iris_dataset_model_production_v1")
    dataset: str = Field(..., example="iris_dataset")
    version: int = Field(..., example=1)
    environment: str = Field(..., example="production")
    promoted_at: datetime = Field(..., example="2024-01-01T12:00:00")


class PredictRequest(BaseModel):
    dataset_name: str = Field(
        ..., example="iris_dataset", description="Dataset/model family name"
    )
    environment: str = Field(
        ...,
        example="production",
        description="Environment from which to load the model",
    )
    version: int = Field(..., example=1, description="Model version to use")
    features: List[Dict[str, float | int | str]] = Field(
        ...,
        example=[
            {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
            },
            {
                "SepalLengthCm": 6.2,
                "SepalWidthCm": 2.9,
                "PetalLengthCm": 4.3,
                "PetalWidthCm": 1.3,
            },
            {
                "SepalLengthCm": 7.3,
                "SepalWidthCm": 2.9,
                "PetalLengthCm": 6.3,
                "PetalWidthCm": 1.8,
            },
        ],
    )


class PredictResponse(BaseModel):
    dataset: str = Field(..., example="iris_dataset")
    environment: str = Field(..., example="production")
    version: int = Field(..., example=1)
    predictions: List[str | int] = Field(
        ...,
        example=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        description="Predicted target values",
    )


class RemoveRequest(BaseModel):
    dataset_name: str = Field(
        ..., example="iris_dataset", description="Dataset/model family name"
    )
    version: int = Field(..., example=1, description="Model version to remove")
    environment: str = Field(
        ...,
        example="staging",
        description="Environment where the model is stored",
    )


class RemoveResponse(BaseModel):
    message: str = Field(..., example="Model removed successfully")
    dataset: str = Field(..., example="iris_dataset")
    version: int = Field(..., example=1)
    environment: str = Field(..., example="staging")
    removed_at: datetime = Field(..., example="2024-01-01T12:00:00")

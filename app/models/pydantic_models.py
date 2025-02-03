from typing import Any, List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TrainRequest(BaseModel):
    dataset_name: str = Field(
        ...,
        example="co2_emission",
        description="Name of the dataset to use for training",
    )
    use_example: bool = Field(
        False,
        example=False,
        description=(
            "Whether to use the built-in example dataset "
            "(if True, a sample dataset will be used)"
        ),
    )
    model: Dict = Field(
        ...,
        example={
            "model_module": "sklearn.ensemble",
            "model_class": "RandomForestClassifier",
            "model_params": {"n_estimators": 100, "max_depth": 5},
        },
        description=(
            "Dictionary with model configuration keys: "
            "model_module, model_class, model_params"
        ),
    )
    target_column: Optional[str] = Field(
        None,
        example="target",
        description=(
            "Optional target column name. If not provided, the latest column "
            "of the dataset is used."
        ),
    )
    test_size: Optional[float] = Field(
        0.2, example=0.2, description="Fraction of data to use for testing"
    )
    random_state: Optional[int] = Field(
        42, example=42, description="Random seed for reproducibility"
    )


class InputData(BaseModel):
    features: List[List[float]] = Field(
        ...,
        example=[[120.5, 75.3, 1], [145.2, 80.1, 0]],
        description="2D list (matrix) of input feature values",
    )

    @field_validator("features")
    def check_feature_length(cls, v):
        if not v:
            raise ValueError("Input data cannot be empty")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("All feature rows must have the same length")
        return v


class DatasetResponse(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="co2_emission")
    description: str = Field(..., example="CO2 Emission dataset")
    location: str = Field(..., example="s3://bucket/co2_emission.csv")
    created_at: datetime = Field(..., example="2024-01-01T12:00:00")
    model_config = ConfigDict(from_attributes=True)


class TrainResponse(BaseModel):
    message: str = Field(
        ..., example="Model trained and registered successfully"
    )
    dataset: str = Field(..., example="co2_emission")
    version: int = Field(..., example=1)
    model_file: str = Field(..., example="co2_emission_model_dev_v1.pkl")
    metrics: Dict[str, Any] = Field(
        ...,
        example={
            "accuracy": 0.95,
            "f1_score": 0.93,
            "precision": {"0": 0.94, "1": 0.96},
            "recall": {"0": 0.92, "1": 0.97},
        },
    )


class ModelResponse(BaseModel):
    id: int = Field(..., example=1)
    dataset_id: int = Field(..., example=1)
    version: int = Field(..., example=1)
    environment: str = Field(
        ..., example="dev"
    )  # Renamed from stage to environment
    artifact_path: str = Field(..., example="co2_emission_model_dev_v1.pkl")
    metrics: Dict[str, Any] = Field(
        default_factory=dict, example={"accuracy": 0.95, "f1_score": 0.93}
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, example={"n_estimators": 100}
    )
    description: str = Field(..., example="Trained model")
    timestamp: datetime = Field(..., example="2024-01-01T12:00:00")
    promotion_timestamp: Optional[datetime] = Field(
        None, example="2024-01-02T12:00:00"
    )
    model_config = ConfigDict(from_attributes=True)


class ModelListResponse(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="co2_emission_model_dev_v1")
    version: int = Field(..., example=1)
    environment: str = Field(..., example="dev")
    dataset_name: str = Field(..., example="co2_emission")
    f1_score: Optional[float] = Field(None, example=0.93)
    accuracy: Optional[float] = Field(None, example=0.95)
    trained_at: str = Field(..., example="2024-01-01 12:00:00")
    promoted_at: Optional[str] = Field(None, example="2024-01-02 12:00:00")


class ModelDetailResponse(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="co2_emission_model_dev_v1")
    version: int = Field(..., example=1)
    environment: str = Field(..., example="dev")
    dataset_name: str = Field(..., example="co2_emission")
    artifact_path: str = Field(..., example="co2_emission_model_dev_v1.pkl")
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
    trained_at: str = Field(..., example="2024-01-01 12:00:00")
    promoted_at: Optional[str] = Field(None, example="2024-01-02 12:00:00")


# Additional Request Models


class PromoteRequest(BaseModel):
    dataset_name: str = Field(
        ..., example="co2_emission", description="Dataset/model family name"
    )
    version: int = Field(
        ..., example=1, description="Model version to promote"
    )
    environment: str = Field(
        ...,
        example="production",
        description="Target environment for promotion (e.g., production)",
    )


class PredictRequest(BaseModel):
    dataset_name: str = Field(
        ..., example="co2_emission", description="Dataset/model family name"
    )
    environment: str = Field(
        ...,
        example="production",
        description="Environment from which to load the model",
    )
    features: List[List[float]] = Field(
        ...,
        example=[[120.5, 75.3, 1], [145.2, 80.1, 0]],
        description="2D list of feature values for prediction",
    )


class RemoveRequest(BaseModel):
    dataset_name: str = Field(
        ..., example="co2_emission", description="Dataset/model family name"
    )
    version: int = Field(..., example=1, description="Model version to remove")
    environment: str = Field(
        ...,
        example="staging",
        description="Environment where the model is stored",
    )

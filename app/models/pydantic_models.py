from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, field_validator, ConfigDict


class TrainRequest(BaseModel):
    dataset_name: str
    use_example: bool = False
    model: Dict
    target_column: Optional[str] = None
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42


class InputData(BaseModel):
    features: List[List[float]]

    @field_validator("features")
    def check_feature_length(cls, v):
        if not v:
            raise ValueError("Input data cannot be empty")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("All feature rows must have the same length")
        return v


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str
    location: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

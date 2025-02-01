from typing import List, Optional
from pydantic import BaseModel, validator


class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    use_example: Optional[bool] = False
    model: dict
    target_column: Optional[str] = None
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42


class InputData(BaseModel):
    features: List[List[float]]

    @validator("features")
    def check_feature_length(cls, v):
        if not v:
            raise ValueError("Input data cannot be empty")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("All feature rows must have the same length")
        return v

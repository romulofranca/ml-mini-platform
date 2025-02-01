import io
import json
import logging
import pickle
from datetime import datetime
from importlib import import_module

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sklearn import datasets
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.db.session import get_db
from app.db import models
from app.models.pydantic_models import TrainRequest, InputData
from app.utils.object_storage import (
    object_storage_client,
    load_dataset_from_storage,
)
from app.utils.preprocessing import create_preprocessor, detect_target_column
from app.config import TRAINED_MODELS_BUCKET

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/train", summary="Train a model")
def train_model(
    request: TrainRequest = Body(...), db: Session = Depends(get_db)
):
    # Use example dataset if specified; otherwise, load from catalog.
    if request.use_example:
        dataset_name = "iris-sample"
        iris = datasets.load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["target"] = iris.target
    else:
        dataset_name = request.dataset_name
        dataset_entry = (
            db.query(models.DatasetCatalog)
            .filter(models.DatasetCatalog.name == dataset_name)
            .first()
        )
        if not dataset_entry:
            raise HTTPException(
                status_code=404, detail="Dataset not found in catalog"
            )
        df = load_dataset_from_storage(dataset_entry.location)

    target_column = request.target_column or detect_target_column(df)
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
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid model configuration"
        )

    full_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model_instance)]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=request.test_size, random_state=request.random_state
    )
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    metrics = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")

    # Lookup dataset; if not found (for example use), create a dummy entry.
    dataset_entry = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == dataset_name)
        .first()
    )
    if not dataset_entry:
        dataset_entry = models.DatasetCatalog(
            name=dataset_name,
            description="Example dataset",
            location="example",
        )
        db.add(dataset_entry)
        db.commit()
        db.refresh(dataset_entry)

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

    model_data = io.BytesIO()
    pickle.dump(full_pipeline, model_data)
    model_data.seek(0)
    object_storage_client.put_object(
        Bucket=TRAINED_MODELS_BUCKET,
        Key=model_file_name,
        Body=model_data.getvalue(),
    )

    new_registry_entry = models.ModelRegistry(
        dataset_id=dataset_entry.id,
        version=version,
        stage=environment,
        artifact_path=model_file_name,
        metrics=json.dumps(metrics),
        parameters=json.dumps(request.model.get("params", {})),
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


@router.post("/promote", summary="Promote a model")
def promote_model(
    dataset_name: str = Body(..., embed=True),
    version: int = Body(..., embed=True),
    target_stage: str = Body(..., embed=True),
    db: Session = Depends(get_db),
):
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
    promotion_order = {"dev": 1, "staging": 2, "production": 3}
    if promotion_order[target_stage] <= promotion_order[entry.stage]:
        raise HTTPException(
            status_code=400,
            detail="Cannot demote or promote to the same stage",
        )
    new_model_file = f"{dataset_name}_model_{target_stage}_v{version}.pkl"
    artifact_bytes = object_storage_client.get_object(
        Bucket=TRAINED_MODELS_BUCKET, Key=entry.artifact_path
    )["Body"].read()
    object_storage_client.put_object(
        Bucket=TRAINED_MODELS_BUCKET, Key=new_model_file, Body=artifact_bytes
    )
    entry.stage = target_stage
    entry.artifact_path = new_model_file
    entry.promotion_timestamp = datetime.now()
    db.commit()
    db.refresh(entry)
    return {
        "message": f"Model promoted to {target_stage} successfully",
        "dataset": dataset_name,
        "version": version,
        "model_file": new_model_file,
    }


@router.post("/predict", summary="Make predictions with a model")
def predict(
    dataset_name: str = Body(..., embed=True),
    environment: str = Body(..., embed=True),
    input_data: InputData = Body(...),
    db: Session = Depends(get_db),
):
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

    model_response = object_storage_client.get_object(
        Bucket=TRAINED_MODELS_BUCKET, Key=entry.artifact_path
    )
    model_pipeline = pickle.loads(model_response["Body"].read())
    input_df = pd.DataFrame(input_data.features)
    params = json.loads(entry.parameters)
    if "feature_names" in params:
        input_df.columns = params["feature_names"]
    predictions = model_pipeline.predict(input_df)
    return {"predictions": predictions.tolist()}

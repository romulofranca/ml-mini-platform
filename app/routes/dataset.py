from datetime import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db import models
from app.utils.string_utils import normalize_dataset_name
from app.config import DATASETS_BUCKET
from app.utils.object_storage import object_storage_client
from app.models.pydantic_models import (
    DatasetResponse,
    RemoveDatasetResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/datasets/upload",
    response_model=DatasetResponse,
    summary="Upload a new dataset",
    description=(
        "Upload a file to create a new dataset entry. The file is stored in "
        "the configured object storage, and a corresponding record "
        "is added to the dataset catalog. "
        "You can optionally provide a `description` for the dataset."
    ),
    tags=["datasets"],
    responses={
        400: {"description": "A dataset with this name already exists."},
        200: {
            "description": "Details about the newly uploaded dataset",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "name": "my_dataset",
                        "description": (
                            "A sample dataset containing climate data"
                        ),
                        "location": "my_dataset",
                        "created_at": "2025-02-02T15:42:57",
                    }
                }
            },
        },
    },
)
def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form("iris_dataset"),
    description: Optional[str] = Form(
        "A sample dataset", description="Dataset description"
    ),
    db: Session = Depends(get_db),
) -> DatasetResponse:

    if not name:
        name = normalize_dataset_name(file.filename)

    existing = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == name)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400, detail="A dataset with this name already exists."
        )

    file.filename = name

    file_content = file.file.read()
    object_storage_client.put_object(
        Bucket=DATASETS_BUCKET, Key=file.filename, Body=file_content
    )

    new_dataset = models.DatasetCatalog(
        name=name,
        description=description or "",
        location=file.filename,
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    logger.info(f"Dataset '{name}' has been uploaded and cataloged.")
    return DatasetResponse(
        id=new_dataset.id,
        name=new_dataset.name,
        description=description,
        location=new_dataset.location,
        created_at=new_dataset.created_at,
    )


@router.get(
    "/datasets",
    response_model=List[DatasetResponse],
    summary="List all datasets",
    description="Retrieve a list of all dataset entries from the catalog.",
    tags=["datasets"],
    responses={
        200: {
            "description": "A list of datasets in the catalog",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "name": "my_dataset",
                            "description": "",
                            "location": "my_dataset",
                            "created_at": "2025-02-02T15:42:57",
                        },
                        {
                            "id": 2,
                            "name": "another_dataset",
                            "description": "Additional info",
                            "location": "another_dataset",
                            "created_at": "2025-02-02T15:42:57",
                        },
                    ]
                }
            },
        }
    },
)
def list_datasets(db: Session = Depends(get_db)) -> List[DatasetResponse]:
    datasets = db.query(models.DatasetCatalog).all()
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            location=dataset.location,
            created_at=dataset.created_at,
        )
        for dataset in datasets
    ]


@router.delete(
    "/datasets/{dataset_id}",
    summary="Delete a dataset",
    description=(
        "Delete a dataset entry from the catalog if no related models exist."
    ),
    tags=["datasets"],
    response_model=RemoveDatasetResponse,
    responses={
        200: {
            "description": "Details about the deleted dataset",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Dataset deleted successfully",
                        "dataset": "my_dataset",
                        "deleted_at": "2025-02-02T15:42:57",
                    }
                }
            },
        },
        204: {"description": "Dataset deleted successfully"},
        404: {"description": "Dataset not found"},
        400: {"description": "Cannot delete dataset with existing models"},
    },
)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.id == dataset_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    related_models_count = (
        db.query(models.ModelRegistry)
        .filter(models.ModelRegistry.dataset_id == dataset_id)
        .count()
    )

    print(related_models_count)
    if related_models_count > 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete dataset with existing models",
        )

    db.delete(dataset)
    db.commit()
    logger.info(f"Dataset '{dataset.name}' has been deleted.")
    response_data = RemoveDatasetResponse(
        message="Dataset deleted successfully",
        dataset=dataset.name,
        deleted_at=datetime.now(),
    )

    return response_data.model_dump()

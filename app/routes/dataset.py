import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db import models
from app.utils.string_utils import normalize_dataset_name
from app.config import DATASETS_BUCKET
from app.utils.object_storage import object_storage_client
from app.models.pydantic_models import DatasetResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/datasets/upload",
    response_model=DatasetResponse,
    summary="Upload and catalog a dataset",
)
def upload_dataset(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    normalized_name = normalize_dataset_name(file.filename)
    existing = (
        db.query(models.DatasetCatalog)
        .filter(models.DatasetCatalog.name == normalized_name)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400, detail="A dataset with this name already exists."
        )

    file_content = file.file.read()
    object_storage_client.put_object(
        Bucket=DATASETS_BUCKET, Key=file.filename, Body=file_content
    )

    new_dataset = models.DatasetCatalog(
        name=normalized_name, description="", location=file.filename
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    logger.info(
        f"Dataset '{normalized_name}' has been uploaded and cataloged."
    )
    return new_dataset


@router.get(
    "/datasets",
    response_model=List[DatasetResponse],
    summary="List all datasets",
)
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(models.DatasetCatalog).all()
    return datasets

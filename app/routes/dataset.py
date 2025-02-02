import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
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
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> DatasetResponse:
    """
    Uploads a new dataset file and catalogs it in the database.

    **Args**:
    - file (UploadFile): The dataset file to be uploaded.
    - description (str, optional): A brief description of the dataset.

    **Raises**:
    - HTTPException(400): If a dataset with the same name already exists.

    **Returns**:
    - (DatasetResponse): The newly created dataset entry, including the
      optional description.
    """

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

    file.filename = normalized_name

    file_content = file.file.read()
    object_storage_client.put_object(
        Bucket=DATASETS_BUCKET, Key=file.filename, Body=file_content
    )

    new_dataset = models.DatasetCatalog(
        name=normalized_name,
        description=description
        or "",
        location=file.filename,
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
    """
    Retrieves all datasets currently stored in the catalog.

    **Args**:
    - db (Session): The database session.

    **Returns**:
    - (List[DatasetResponse]): A list of dataset entries.
    """
    datasets = db.query(models.DatasetCatalog).all()
    return datasets


@router.delete(
    "/datasets/{dataset_id}",
    summary="Delete a dataset",
    description="Delete a dataset entry from the catalog.",
    tags=["datasets"],
    response_model=dict,
    responses={
        204: {"description": "Dataset deleted successfully"},
        404: {"description": "Dataset not found"},
    },
)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Deletes a dataset entry from the catalog.

    **Args**:
    - dataset_id (int): The ID of the dataset to delete.
    - db (Session): The database session.

    **Raises**:
    - HTTPException(404): If the dataset ID is not found.

    **Returns**:
    - (dict): A message confirming the deletion.
    """
    dataset = db.query(models.DatasetCatalog).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(dataset)
    db.commit()
    logger.info(f"Dataset '{dataset.name}' has been deleted.")
    return {"message": f"Dataset '{dataset.name}' has been deleted."}

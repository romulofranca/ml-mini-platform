import json
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db import models
from app.utils.string_utils import normalize_dataset_name
from app.config import DATASETS_BUCKET
from app.utils.object_storage import object_storage_client
from app.models.pydantic_models import DatasetResponse, ModelResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/datasets/upload",
    response_model=DatasetResponse,
    summary="Upload a new dataset",
    description=(
        "Upload a file to create a new dataset entry. The file is stored in "
        "the configured object storage, and a corresponding record "
        "is added to the dataset catalog."
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
                        "name": "my_dataset_csv",
                        "description": "",
                        "location": "my_dataset_csv",
                    }
                }
            },
        },
    },
)
def upload_dataset(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> DatasetResponse:
    """
    Uploads a new dataset file and catalogs it in the database.

    **Args**:
    - file (UploadFile): The dataset file to be uploaded.

    **Raises**:
    - HTTPException(400): If a dataset filename already exists.

    **Returns**:
    - (DatasetResponse): The newly created dataset entry.
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
                            "name": "my_dataset_csv",
                            "description": "",
                            "location": "my_dataset_csv",
                        },
                        {
                            "id": 2,
                            "name": "another_dataset_parquet",
                            "description": "Additional info",
                            "location": "another_dataset_parquet",
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


@router.get(
    "/models",
    summary="List all models in the registry",
    description=(
        "Retrieve a list of all models registered in the system, "
        "including their dataset name, version, stage, artifact path, "
        "metrics, and parameters."
    ),
    tags=["models"],
    response_model=List[ModelResponse],
)
def list_models(db: Session = Depends(get_db)) -> List[ModelResponse]:
    """
    List all models registered in the system.

    **Args**:
    - db (Session): SQLAlchemy session dependency.

    **Returns**:
    - (List[ModelResponse]): A list of all registered models, with `metrics`
      and `parameters` parsed into dictionaries.
    """

    model_records = db.query(models.ModelRegistry).all()

    model_responses = []
    for record in model_records:
        if isinstance(record.metrics, str):
            try:
                metrics_dict = json.loads(record.metrics)
            except json.JSONDecodeError:
                metrics_dict = {}
        else:
            metrics_dict = record.metrics or {}

        if isinstance(record.parameters, str):
            try:
                parameters_dict = json.loads(record.parameters)
            except json.JSONDecodeError:
                parameters_dict = {}
        else:
            parameters_dict = record.parameters or {}

        model_responses.append(
            ModelResponse(
                id=record.id,
                dataset_id=record.dataset_id,
                version=record.version,
                stage=record.stage,
                artifact_path=record.artifact_path,
                metrics=metrics_dict,
                parameters=parameters_dict,
                description=record.description,
                timestamp=record.timestamp,
                promotion_timestamp=record.promotion_timestamp,
            )
        )

    return model_responses

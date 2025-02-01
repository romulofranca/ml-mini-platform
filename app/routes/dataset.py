import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.object_storage import upload_file_to_storage, list_objects
from app.config import DATASETS_BUCKET

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", summary="Upload dataset", tags=["Dataset"])
def upload_file(file: UploadFile = File(...)):
    try:
        file_content = file.file.read()
        upload_file_to_storage(
            file_content, file.filename, bucket=DATASETS_BUCKET
        )
        return {"message": f"File '{file.filename}' uploaded successfully!"}
    except Exception as e:
        logger.exception(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")


@router.get(
    "/list-datasets", summary="List available datasets", tags=["Dataset"]
)
def list_datasets():
    try:
        datasets_list = list_objects(bucket=DATASETS_BUCKET)
        return {"datasets": datasets_list}
    except Exception as e:
        logger.exception(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail="Failed to list datasets")

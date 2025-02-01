import io
import logging
import boto3
from botocore.client import Config
from fastapi import HTTPException
import pandas as pd
from app.config import (
    OBJECT_STORAGE_ENDPOINT,
    OBJECT_STORAGE_ACCESS_KEY,
    OBJECT_STORAGE_SECRET_KEY,
    DATASETS_BUCKET,
    TRAINED_MODELS_BUCKET,
)

logger = logging.getLogger(__name__)

object_storage_client = boto3.client(
    "s3",
    endpoint_url=OBJECT_STORAGE_ENDPOINT,
    aws_access_key_id=OBJECT_STORAGE_ACCESS_KEY,
    aws_secret_access_key=OBJECT_STORAGE_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)


def load_dataset_from_storage(dataset_key: str) -> pd.DataFrame:
    try:
        response = object_storage_client.get_object(
            Bucket=DATASETS_BUCKET, Key=dataset_key
        )
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info(
            f"Dataset '{dataset_key}' loaded successfully from storage."
        )
        return df
    except Exception as e:
        logger.exception(f"Failed to load dataset '{dataset_key}': {e}")
        raise HTTPException(
            status_code=404, detail="Dataset not found in storage"
        )

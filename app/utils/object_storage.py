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
)

logger = logging.getLogger(__name__)

# Initialize the Object Storage client (e.g., MinIO, S3)
object_storage_client = boto3.client(
    "s3",
    endpoint_url=OBJECT_STORAGE_ENDPOINT,
    aws_access_key_id=OBJECT_STORAGE_ACCESS_KEY,
    aws_secret_access_key=OBJECT_STORAGE_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)


def load_dataset_from_storage(dataset_path: str) -> pd.DataFrame:
    """
    Load a dataset from object storage given its path.
    """
    try:
        response = object_storage_client.get_object(
            Bucket=DATASETS_BUCKET, Key=dataset_path
        )
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info(f"Dataset '{dataset_path}' loaded successfully.")
        return df
    except Exception as e:
        logger.exception(f"Failed to load dataset '{dataset_path}': {e}")
        raise HTTPException(status_code=404, detail="Dataset not found")


def upload_file_to_storage(file_content: bytes, filename: str, bucket: str):
    """
    Upload a file to the specified bucket in object storage.
    """
    try:
        object_storage_client.put_object(
            Bucket=bucket, Key=filename, Body=file_content
        )
        logger.info(
            f"File '{filename}' uploaded successfully to bucket '{bucket}'."
        )
    except Exception as e:
        logger.exception(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")


def list_objects(bucket: str):
    """
    List objects in the specified bucket.
    """
    try:
        response = object_storage_client.list_objects_v2(Bucket=bucket)
        objects_list = (
            [obj["Key"] for obj in response.get("Contents", [])]
            if response.get("Contents")
            else []
        )
        logger.info(f"Objects listed successfully in bucket '{bucket}'.")
        return objects_list
    except Exception as e:
        logger.exception(f"Failed to list objects in bucket '{bucket}': {e}")
        raise HTTPException(status_code=500, detail="Failed to list objects")

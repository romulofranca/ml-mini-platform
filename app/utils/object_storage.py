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


def put_object_to_bucket(bucket: str, key: str, body) -> None:
    try:
        object_storage_client.put_object(Bucket=bucket, Key=key, Body=body)
        logger.info(
            f"Object '{key}' stored successfully in bucket '{bucket}'."
        )
    except Exception as e:
        logger.exception(
            f"Failed to put object '{key}' in bucket '{bucket}': {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to upload object")


def get_object_from_bucket(bucket: str, key: str):
    try:
        return object_storage_client.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        logger.exception(
            f"Failed to get object '{key}' from bucket '{bucket}': {e}"
        )
        raise HTTPException(status_code=404, detail="Object not found")


def list_objects_in_bucket(bucket: str):
    try:
        response = object_storage_client.list_objects_v2(Bucket=bucket)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"]]
        return []
    except Exception as e:
        logger.exception(f"Failed to list objects in bucket '{bucket}': {e}")
        raise HTTPException(status_code=500, detail="Failed to list objects")


def delete_object_from_bucket(bucket: str, key: str):
    try:
        object_storage_client.delete_object(Bucket=bucket, Key=key)
        logger.info(
            f"Object '{key}' deleted successfully from bucket '{bucket}'."
        )
    except Exception as e:
        logger.exception(
            f"Failed to delete object '{key}' from bucket '{bucket}': {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to delete object")

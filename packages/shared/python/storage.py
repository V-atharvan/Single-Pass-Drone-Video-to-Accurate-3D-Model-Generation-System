"""
S3StorageClient – thin async wrapper around boto3 (synchronous) / aiobotocore.
Enforces the canonical key structure from the tech stack document.

Key paths:
    raw:     projects/{project_id}/flights/{flight_id}/
    interim: jobs/{job_id}/interim/{stage}/
    output:  projects/{project_id}/models/{model_id}/
    exports: projects/{project_id}/exports/{export_id}/
"""
from __future__ import annotations

import os
from typing import Any, Optional
from uuid import UUID

import boto3
from botocore.client import Config


class S3StorageClient:
    """
    Typed, key-enforcing wrapper around boto3 S3 client.
    Defaults to MinIO endpoint for local development; point at AWS in production.
    """

    RAW_BUCKET = os.environ.get("S3_RAW_BUCKET", "single-pass-3d-raw")
    INTERIM_BUCKET = os.environ.get("S3_INTERIM_BUCKET", "single-pass-3d-interim")
    MODELS_BUCKET = os.environ.get("S3_MODELS_BUCKET", "single-pass-3d-models")

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT_URL"),
            region_name=region_name,
            aws_access_key_id=access_key or os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            config=Config(signature_version="s3v4"),
        )

    # ------------------------------------------------------------------
    # Key generators
    # ------------------------------------------------------------------

    @staticmethod
    def get_flight_raw_prefix(project_id: UUID | str, flight_id: UUID | str) -> str:
        """projects/{project_id}/flights/{flight_id}/"""
        return f"projects/{project_id}/flights/{flight_id}/"

    @staticmethod
    def get_reconstruction_interim_prefix(job_id: UUID | str, stage: str) -> str:
        """jobs/{job_id}/interim/{stage}/"""
        return f"jobs/{job_id}/interim/{stage}/"

    @staticmethod
    def get_model_output_prefix(project_id: UUID | str, model_id: UUID | str) -> str:
        """projects/{project_id}/models/{model_id}/"""
        return f"projects/{project_id}/models/{model_id}/"

    @staticmethod
    def get_export_prefix(project_id: UUID | str, export_id: UUID | str) -> str:
        """projects/{project_id}/exports/{export_id}/"""
        return f"projects/{project_id}/exports/{export_id}/"

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    def generate_presigned_upload_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
        content_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate a presigned POST URL for direct client-to-S3 upload.
        Returns a dict with 'url' and 'fields'.
        """
        conditions: list[Any] = []
        if content_type:
            conditions.append({"Content-Type": content_type})

        response = self._client.generate_presigned_post(
            Bucket=bucket,
            Key=key,
            Conditions=conditions or None,
            ExpiresIn=expires_in,
        )
        return response

    def generate_presigned_put_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
        content_type: Optional[str] = None,
    ) -> str:
        """Generate a presigned PUT URL for direct binary upload."""
        params: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return self._client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    def generate_presigned_download_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
        filename: Optional[str] = None,
    ) -> str:
        """Generate a presigned GET URL for secure object download."""
        params: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        return self._client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    def initiate_multipart_upload(self, bucket: str, key: str) -> str:
        """Start a multipart upload and return the UploadId."""
        response = self._client.create_multipart_upload(Bucket=bucket, Key=key)
        return response["UploadId"]

    def generate_presigned_part_upload_url(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        part_number: int,
        expires_in: int = 3600,
    ) -> str:
        """Generate a presigned PUT URL for uploading a specific part of a multipart upload."""
        return self._client.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": bucket,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=expires_in,
        )

    def abort_multipart_upload(self, bucket: str, key: str, upload_id: str) -> None:
        """Abort an in-progress multipart upload and discard uploaded parts."""
        self._client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)

    def complete_multipart_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Complete a multipart upload.
        `parts` must be a list of {'ETag': str, 'PartNumber': int}.
        """
        return self._client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

    def put_object(self, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
        """Synchronous single-shot object upload (for small payloads / tests)."""
        self._client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)

    def upload_bytes(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Semantic alias for put_object using `data` keyword; preferred for non-multipart binary uploads."""
        self.put_object(bucket=bucket, key=key, body=data, content_type=content_type)

    def get_object(self, bucket: str, key: str) -> bytes:
        """Download object body as bytes."""
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        """Return object metadata without downloading the body."""
        return self._client.head_object(Bucket=bucket, Key=key)

"""
S3 Pre-Signed Direct Upload URL Service – TASK-014
Facilitates direct-to-S3 multi-gigabyte video and GPS file uploads.
"""
from __future__ import annotations

import math
import os
import re
import uuid
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field

from src.auth.rbac import require_operator_project_access
from src.db.models import Project, UserRole

# Try import S3StorageClient
try:
    from packages.shared.python.storage import S3StorageClient
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    from packages.shared.python.storage import S3StorageClient


router = APIRouter(prefix="/projects/{project_id}/uploads", tags=["Uploads"])


def get_storage_client() -> S3StorageClient:
    """Dependency provider for S3 storage client (can be overridden in tests)."""
    return S3StorageClient()


# ---------------------------------------------------------------------------
# Enums and Schemas
# ---------------------------------------------------------------------------

class FileType(str, Enum):
    VIDEO = "VIDEO"
    TELEMETRY = "TELEMETRY"


_ALLOWED_EXTENSIONS = {
    FileType.VIDEO: {".mp4", ".mov", ".m4v", ".avi"},
    FileType.TELEMETRY: {".csv", ".json", ".srt", ".txt", ".gpx"},
}

_MAX_FILE_SIZES = {
    FileType.VIDEO: 50 * 1024 * 1024 * 1024,      # 50 GB
    FileType.TELEMETRY: 500 * 1024 * 1024,        # 500 MB
}

# 100 MB threshold for initiating multipart upload
MULTIPART_THRESHOLD_BYTES = 100 * 1024 * 1024
DEFAULT_PART_SIZE_BYTES = 50 * 1024 * 1024        # 50 MB


class UploadInitiateIn(BaseModel):
    filename: str = Field(..., min_length=1, max_length=512)
    content_type: str = Field(..., min_length=1)
    file_size_bytes: int = Field(..., gt=0)
    file_type: FileType


class PartUploadInfo(BaseModel):
    part_number: int
    upload_url: str


class UploadInitiateOut(BaseModel):
    upload_id: str
    s3_key: str
    bucket: str
    upload_url: str
    fields: Optional[dict[str, Any]] = None
    is_multipart: bool = False
    part_size_bytes: Optional[int] = None
    parts: Optional[list[PartUploadInfo]] = None


class CompletedPartIn(BaseModel):
    PartNumber: int
    ETag: str


class UploadCompleteIn(BaseModel):
    s3_key: str
    upload_id: Optional[str] = None
    parts: Optional[list[CompletedPartIn]] = None


class UploadCompleteOut(BaseModel):
    s3_key: str
    file_size_bytes: int
    content_type: Optional[str] = None
    etag: Optional[str] = None
    status: str = "COMPLETED"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/initiate",
    response_model=UploadInitiateOut,
    status_code=status.HTTP_200_OK,
    summary="Initiate direct-to-S3 upload",
)
async def initiate_upload(
    body: UploadInitiateIn,
    project_id: UUID = Path(...),
    project: Project = Depends(require_operator_project_access),
    storage: S3StorageClient = Depends(get_storage_client),
) -> UploadInitiateOut:
    """
    Initiate a direct-to-S3 upload for a drone video or telemetry file.
    Returns a pre-signed PUT/POST upload URL (or multipart URLs for files > 100MB).
    """
    ext = os.path.splitext(body.filename)[1].lower()
    allowed_exts = _ALLOWED_EXTENSIONS.get(body.file_type, set())
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}' for file type {body.file_type.value}. Allowed: {sorted(allowed_exts)}",
        )

    max_size = _MAX_FILE_SIZES.get(body.file_type, MULTIPART_THRESHOLD_BYTES)
    if body.file_size_bytes > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size {body.file_size_bytes} exceeds maximum allowed size {max_size} bytes.",
        )

    # Sanitize filename and create deterministic prefix
    clean_filename = re.sub(r"[^a-zA-Z0-9._-]", "_", body.filename)
    upload_session_id = str(uuid.uuid4())
    bucket = storage.RAW_BUCKET
    s3_key = f"projects/{project_id}/uploads/{upload_session_id}/{clean_filename}"

    if body.file_size_bytes > MULTIPART_THRESHOLD_BYTES:
        # Multipart upload flow
        try:
            s3_upload_id = storage.initiate_multipart_upload(bucket=bucket, key=s3_key)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initiate S3 multipart upload: {e}",
            )

        num_parts = math.ceil(body.file_size_bytes / DEFAULT_PART_SIZE_BYTES)
        parts_list: list[PartUploadInfo] = []
        for part_num in range(1, num_parts + 1):
            part_url = storage.generate_presigned_part_upload_url(
                bucket=bucket,
                key=s3_key,
                upload_id=s3_upload_id,
                part_number=part_num,
                expires_in=3600,
            )
            parts_list.append(PartUploadInfo(part_number=part_num, upload_url=part_url))

        # Single presigned PUT as a fallback URL
        fallback_put_url = storage.generate_presigned_put_url(
            bucket=bucket,
            key=s3_key,
            expires_in=3600,
            content_type=body.content_type,
        )

        return UploadInitiateOut(
            upload_id=s3_upload_id,
            s3_key=s3_key,
            bucket=bucket,
            upload_url=fallback_put_url,
            is_multipart=True,
            part_size_bytes=DEFAULT_PART_SIZE_BYTES,
            parts=parts_list,
        )

    # Standard single-shot presigned upload flow
    presigned_put = storage.generate_presigned_put_url(
        bucket=bucket,
        key=s3_key,
        expires_in=3600,
        content_type=body.content_type,
    )

    try:
        presigned_post = storage.generate_presigned_upload_url(
            bucket=bucket,
            key=s3_key,
            expires_in=3600,
            content_type=body.content_type,
        )
        fields = presigned_post.get("fields")
    except Exception:
        fields = None

    return UploadInitiateOut(
        upload_id=upload_session_id,
        s3_key=s3_key,
        bucket=bucket,
        upload_url=presigned_put,
        fields=fields,
        is_multipart=False,
    )


@router.post(
    "/complete",
    response_model=UploadCompleteOut,
    status_code=status.HTTP_200_OK,
    summary="Complete direct-to-S3 upload and verify object",
)
async def complete_upload(
    body: UploadCompleteIn,
    project_id: UUID = Path(...),
    project: Project = Depends(require_operator_project_access),
    storage: S3StorageClient = Depends(get_storage_client),
) -> UploadCompleteOut:
    """
    Finalize an S3 upload, completing multipart assembly if applicable,
    and verifying file presence and metadata in S3 storage.
    """
    expected_prefix = f"projects/{project_id}/"
    if not body.s3_key.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid S3 key: must start with '{expected_prefix}'",
        )

    bucket = storage.RAW_BUCKET

    # If multipart upload, assemble the parts
    if body.upload_id and body.parts:
        try:
            parts_dict_list = [{"PartNumber": p.PartNumber, "ETag": p.ETag} for p in body.parts]
            storage.complete_multipart_upload(
                bucket=bucket,
                key=body.s3_key,
                upload_id=body.upload_id,
                parts=parts_dict_list,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to complete S3 multipart upload: {e}",
            )

    # Verify object existence and retrieve metadata
    try:
        head_meta = storage.head_object(bucket=bucket, key=body.s3_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded object not found in S3 bucket '{bucket}' at key '{body.s3_key}': {e}",
        )

    file_size = head_meta.get("ContentLength", 0)
    content_type = head_meta.get("ContentType")
    etag = head_meta.get("ETag", "").strip('"')

    return UploadCompleteOut(
        s3_key=body.s3_key,
        file_size_bytes=file_size,
        content_type=content_type,
        etag=etag,
        status="COMPLETED",
    )

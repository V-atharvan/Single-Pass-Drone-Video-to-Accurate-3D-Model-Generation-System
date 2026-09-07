"""
Pytest configuration and test fixtures for control plane API.
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure python path includes workspace root and apps/api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.models import Flight, FlightStatus, Organization, Project, User, UserRole
from src.main import app
from src.auth.rbac import resolve_db_user, require_project_access, require_flight_access
from src.db.base import get_session
from src.routers.uploads import get_storage_client


@pytest.fixture
def test_org_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def test_project_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def test_flight_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def mock_user(test_org_id: uuid.UUID) -> User:
    return User(
        id=uuid.uuid4(),
        org_id=test_org_id,
        email="operator@example.com",
        full_name="Drone Operator",
        role=UserRole.OPERATOR,
        auth0_sub="auth0|test_operator",
    )


@pytest.fixture
def mock_project(test_project_id: uuid.UUID, test_org_id: uuid.UUID) -> Project:
    return Project(
        id=test_project_id,
        org_id=test_org_id,
        name="Test Drone Survey",
        description="Survey project for testing",
        location_name="Zone A",
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_storage():
    """Mocked S3StorageClient."""
    storage = MagicMock()
    storage.RAW_BUCKET = "single-pass-3d-raw"
    storage.generate_presigned_put_url.return_value = "https://s3.local/single-pass-3d-raw/test-upload?token=abc"
    storage.generate_presigned_upload_url.return_value = {
        "url": "https://s3.local/single-pass-3d-raw",
        "fields": {"key": "test", "policy": "xyz"},
    }
    storage.initiate_multipart_upload.return_value = "upload_id_12345"
    storage.generate_presigned_part_upload_url.side_effect = (
        lambda bucket, key, upload_id, part_number, expires_in=3600: f"https://s3.local/{bucket}/{key}?partNumber={part_number}&uploadId={upload_id}"
    )
    storage.complete_multipart_upload.return_value = {"status": "success"}
    storage.head_object.return_value = {
        "ContentLength": 52428800,
        "ContentType": "video/mp4",
        "ETag": '"test-etag-12345"',
    }
    return storage


@pytest.fixture
def mock_db_session():
    """Mocked SQLAlchemy AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    return session

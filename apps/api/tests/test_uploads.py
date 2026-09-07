"""
Integration tests for S3 direct upload service – TASK-014.
"""
from __future__ import annotations

import uuid
import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.rbac import require_operator_project_access
from src.db.models import Project, UserRole
from src.main import app
from src.routers.uploads import get_storage_client


@pytest.mark.asyncio
async def test_initiate_upload_single_part_success(mock_project: Project, mock_storage):
    """Verify single-part presigned upload initiation for video under 100MB."""
    project_id = mock_project.id

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "filename": "flight_video_01.mp4",
            "content_type": "video/mp4",
            "file_size_bytes": 50 * 1024 * 1024,  # 50MB
            "file_type": "VIDEO",
        }
        resp = await client.post(f"/projects/{project_id}/uploads/initiate", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "upload_id" in data
        assert data["is_multipart"] is False
        assert data["s3_key"].startswith(f"projects/{project_id}/uploads/")
        assert data["s3_key"].endswith("flight_video_01.mp4")
        assert "upload_url" in data
        assert "fields" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_initiate_upload_multipart_for_large_file(mock_project: Project, mock_storage):
    """Verify multipart upload initiation and part URL generation for files over 100MB."""
    project_id = mock_project.id

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "filename": "large_survey.mov",
            "content_type": "video/quicktime",
            "file_size_bytes": 150 * 1024 * 1024,  # 150MB -> 3 parts of 50MB
            "file_type": "VIDEO",
        }
        resp = await client.post(f"/projects/{project_id}/uploads/initiate", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["is_multipart"] is True
        assert data["upload_id"] == "upload_id_12345"
        assert len(data["parts"]) == 3
        assert data["parts"][0]["part_number"] == 1
        assert "partNumber=1" in data["parts"][0]["upload_url"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_initiate_upload_invalid_extension(mock_project: Project, mock_storage):
    """Verify rejection of invalid file extensions for declared file type."""
    project_id = mock_project.id

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "filename": "malicious_script.exe",
            "content_type": "application/x-msdownload",
            "file_size_bytes": 1024 * 1024,
            "file_type": "VIDEO",
        }
        resp = await client.post(f"/projects/{project_id}/uploads/initiate", json=payload)
        assert resp.status_code == 400
        assert "Invalid file extension" in resp.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_initiate_upload_telemetry_valid(mock_project: Project, mock_storage):
    """Verify telemetry CSV upload initiation."""
    project_id = mock_project.id

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "filename": "flight_gps_log.csv",
            "content_type": "text/csv",
            "file_size_bytes": 2 * 1024 * 1024,
            "file_type": "TELEMETRY",
        }
        resp = await client.post(f"/projects/{project_id}/uploads/initiate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["s3_key"].endswith("flight_gps_log.csv")

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_complete_upload_success(mock_project: Project, mock_storage):
    """Verify upload completion and S3 metadata verification."""
    project_id = mock_project.id
    valid_key = f"projects/{project_id}/uploads/session-123/flight.mp4"

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "s3_key": valid_key,
        }
        resp = await client.post(f"/projects/{project_id}/uploads/complete", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["status"] == "COMPLETED"
        assert data["file_size_bytes"] == 52428800
        assert data["content_type"] == "video/mp4"
        assert data["etag"] == "test-etag-12345"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_complete_upload_invalid_key_prefix(mock_project: Project, mock_storage):
    """Verify rejection of S3 keys outside project namespace."""
    project_id = mock_project.id
    foreign_key = f"projects/{uuid.uuid4()}/uploads/file.mp4"

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "s3_key": foreign_key,
        }
        resp = await client.post(f"/projects/{project_id}/uploads/complete", json=payload)
        assert resp.status_code == 400
        assert "Invalid S3 key" in resp.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_complete_upload_missing_s3_file(mock_project: Project, mock_storage):
    """Verify 404 response when S3 object does not exist."""
    project_id = mock_project.id
    valid_key = f"projects/{project_id}/uploads/session-123/nonexistent.mp4"

    mock_storage.head_object.side_effect = Exception("NoSuchKey: Object does not exist")

    async def _mock_require_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_project_access
    app.dependency_overrides[get_storage_client] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "s3_key": valid_key,
        }
        resp = await client.post(f"/projects/{project_id}/uploads/complete", json=payload)
        assert resp.status_code == 404
        assert "not found in S3 bucket" in resp.json()["detail"]

    app.dependency_overrides.clear()

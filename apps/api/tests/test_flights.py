"""
Integration tests for Flights registration and metadata linkage – TASK-015.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.rbac import (
    require_flight_access,
    require_operator_project_access,
    require_viewer_flight_access,
    require_viewer_project_access,
)
from src.db.base import get_session
from src.db.models import Flight, FlightStatus, Project, UserRole
from src.main import app


@pytest.mark.asyncio
async def test_create_flight_success(mock_project: Project, mock_db_session):
    """Verify registering an uploaded video and GPS file as a cohesive flight asset."""
    project_id = mock_project.id

    async def _mock_require_operator_project_access():
        return mock_project

    # Mock DB session refresh to populate generated fields
    async def _mock_refresh(obj):
        if hasattr(obj, "id") and obj.id is None:
            obj.id = uuid.uuid4()
        if hasattr(obj, "created_at") and obj.created_at is None:
            obj.created_at = datetime.now(timezone.utc)

    mock_db_session.refresh = _mock_refresh

    app.dependency_overrides[require_operator_project_access] = _mock_require_operator_project_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "original_filename": "flight_survey_01.mp4",
            "s3_video_key": f"projects/{project_id}/uploads/123/flight_survey_01.mp4",
            "s3_telemetry_key": f"projects/{project_id}/uploads/123/flight_telemetry.csv",
            "duration_seconds": 185.5,
            "fps": 30.0,
            "resolution_width": 3840,
            "resolution_height": 2160,
            "total_frames": 5565,
        }
        resp = await client.post(f"/projects/{project_id}/flights", json=payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["original_filename"] == "flight_survey_01.mp4"
        assert data["status"] == "PENDING"
        assert data["s3_video_key"] == f"projects/{project_id}/uploads/123/flight_survey_01.mp4"
        assert data["s3_telemetry_key"] == f"projects/{project_id}/uploads/123/flight_telemetry.csv"
        assert data["duration_seconds"] == 185.5
        assert data["fps"] == 30.0
        assert data["resolution_width"] == 3840
        assert data["resolution_height"] == 2160
        assert data["total_frames"] == 5565

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_flight_invalid_video_prefix(mock_project: Project, mock_db_session):
    """Verify rejection when S3 video key belongs to a different project."""
    project_id = mock_project.id
    other_project_id = uuid.uuid4()

    async def _mock_require_operator_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_operator_project_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "original_filename": "flight_video.mp4",
            "s3_video_key": f"projects/{other_project_id}/uploads/video.mp4",
        }
        resp = await client.post(f"/projects/{project_id}/flights", json=payload)
        assert resp.status_code == 400
        assert "S3 video key must belong to project path" in resp.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_flight_invalid_telemetry_prefix(mock_project: Project, mock_db_session):
    """Verify rejection when S3 telemetry key belongs to a different project."""
    project_id = mock_project.id
    other_project_id = uuid.uuid4()

    async def _mock_require_operator_project_access():
        return mock_project

    app.dependency_overrides[require_operator_project_access] = _mock_require_operator_project_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "original_filename": "flight_video.mp4",
            "s3_video_key": f"projects/{project_id}/uploads/video.mp4",
            "s3_telemetry_key": f"projects/{other_project_id}/uploads/gps.csv",
        }
        resp = await client.post(f"/projects/{project_id}/flights", json=payload)
        assert resp.status_code == 400
        assert "S3 telemetry key must belong to project path" in resp.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_project_flights(mock_project: Project, mock_db_session):
    """Verify listing all flights in a project."""
    project_id = mock_project.id
    flight_id = uuid.uuid4()

    mock_flight = Flight(
        id=flight_id,
        project_id=project_id,
        original_filename="flight_01.mp4",
        s3_video_key=f"projects/{project_id}/uploads/flight_01.mp4",
        status=FlightStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_flight]
    mock_db_session.execute.return_value = mock_result

    async def _mock_require_viewer_project_access():
        return mock_project

    app.dependency_overrides[require_viewer_project_access] = _mock_require_viewer_project_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/projects/{project_id}/flights")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == str(flight_id)
        assert data[0]["original_filename"] == "flight_01.mp4"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_flight_details(test_flight_id: uuid.UUID, test_project_id: uuid.UUID, mock_db_session):
    """Verify fetching detailed flight metadata with telemetry and job counts."""
    flight_id = test_flight_id

    mock_flight = Flight(
        id=flight_id,
        project_id=test_project_id,
        original_filename="flight_detailed.mp4",
        s3_video_key=f"projects/{test_project_id}/uploads/flight_detailed.mp4",
        status=FlightStatus.PENDING,
        duration_seconds=90.0,
        fps=60.0,
        resolution_width=1920,
        resolution_height=1080,
        total_frames=5400,
        created_at=datetime.now(timezone.utc),
    )

    # Mock count queries
    count_result = MagicMock()
    count_result.scalar_one.side_effect = [1500, 2]  # 1500 telemetry points, 2 reconstruction jobs
    mock_db_session.execute.return_value = count_result

    async def _mock_require_viewer_flight_access():
        return mock_flight

    app.dependency_overrides[require_viewer_flight_access] = _mock_require_viewer_flight_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/flights/{flight_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["id"] == str(flight_id)
        assert data["original_filename"] == "flight_detailed.mp4"
        assert data["telemetry_record_count"] == 1500
        assert data["reconstruction_job_count"] == 2
        assert data["duration_seconds"] == 90.0
        assert data["status"] == "PENDING"

    app.dependency_overrides.clear()

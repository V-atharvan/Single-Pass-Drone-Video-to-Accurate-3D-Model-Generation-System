"""
Integration tests for Reconstruction Job Submission & State Machine API – TASK-016.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.rbac import (
    require_operator_flight_access,
    require_operator_job_access,
    require_viewer_job_access,
)
from src.db.base import get_session
from src.db.models import Flight, FlightStatus, JobStatus, Project, QualityPreset, ReconstructionJob
from src.main import app
from src.routers.reconstruction_jobs import get_queue_client


@pytest.fixture
def mock_queue():
    queue = MagicMock()
    queue.publish_job.return_value = "msg_id_12345"
    return queue


@pytest.mark.asyncio
async def test_submit_reconstruction_job_success(
    test_flight_id: uuid.UUID,
    test_project_id: uuid.UUID,
    test_org_id: uuid.UUID,
    mock_db_session,
    mock_queue,
):
    """Verify job submission creates DB record with state QUEUED and enqueues payload to SQS."""
    mock_project = Project(id=test_project_id, org_id=test_org_id, name="Survey Project")
    mock_flight = Flight(
        id=test_flight_id,
        project_id=test_project_id,
        original_filename="flight_video.mp4",
        s3_video_key=f"projects/{test_project_id}/uploads/vid.mp4",
        s3_telemetry_key=f"projects/{test_project_id}/uploads/gps.csv",
        status=FlightStatus.READY,
        project=mock_project,
        total_frames=3600,
    )

    async def _mock_require_operator_flight_access():
        return mock_flight

    async def _mock_refresh(obj):
        if hasattr(obj, "id") and obj.id is None:
            obj.id = uuid.uuid4()
        if hasattr(obj, "created_at") and obj.created_at is None:
            obj.created_at = datetime.now(timezone.utc)

    mock_db_session.refresh = _mock_refresh

    app.dependency_overrides[require_operator_flight_access] = _mock_require_operator_flight_access
    app.dependency_overrides[get_session] = lambda: mock_db_session
    app.dependency_overrides[get_queue_client] = lambda: mock_queue

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "quality_preset": "BALANCED",
            "target_crs": "EPSG:4326",
            "enable_dynamic_removal": True,
            "enable_semantics": True,
        }
        resp = await client.post(f"/flights/{test_flight_id}/reconstruction-jobs", json=payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["flight_id"] == str(test_flight_id)
        assert data["status"] == "QUEUED"
        assert data["quality_preset"] == "BALANCED"
        assert data["current_stage"] == "QUEUED"
        assert data["progress"] == 0

        # Verify SQS publish call
        mock_queue.publish_job.assert_called_once()
        published_payload = mock_queue.publish_job.call_args[0][0]
        assert published_payload["flight_id"] == str(test_flight_id)
        assert published_payload["quality_preset"] == "BALANCED"
        assert published_payload["s3_video_key"] == f"projects/{test_project_id}/uploads/vid.mp4"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_submit_job_rejected_for_failed_flight(
    test_flight_id: uuid.UUID,
    test_project_id: uuid.UUID,
    mock_db_session,
    mock_queue,
):
    """Verify rejection when flight is in FAILED state."""
    mock_flight = Flight(
        id=test_flight_id,
        project_id=test_project_id,
        original_filename="corrupted.mp4",
        status=FlightStatus.FAILED,
    )

    async def _mock_require_operator_flight_access():
        return mock_flight

    app.dependency_overrides[require_operator_flight_access] = _mock_require_operator_flight_access
    app.dependency_overrides[get_session] = lambda: mock_db_session
    app.dependency_overrides[get_queue_client] = lambda: mock_queue

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"quality_preset": "HIGH"}
        resp = await client.post(f"/flights/{test_flight_id}/reconstruction-jobs", json=payload)
        assert resp.status_code == 400
        assert "marked as FAILED" in resp.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_reconstruction_job_status(test_flight_id: uuid.UUID, test_org_id: uuid.UUID, mock_db_session):
    """Verify fetching job progress, active stage, and frame counts."""
    job_id = uuid.uuid4()
    mock_flight = Flight(id=test_flight_id, total_frames=4000)
    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=test_flight_id,
        org_id=test_org_id,
        status=JobStatus.ESTIMATING_DEPTH,
        quality_preset=QualityPreset.HIGH,
        current_stage="Depth Estimation",
        progress=45,
        flight=mock_flight,
    )

    async def _mock_require_viewer_job_access():
        return mock_job

    app.dependency_overrides[require_viewer_job_access] = _mock_require_viewer_job_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/reconstruction-jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["id"] == str(job_id)
        assert data["status"] == "ESTIMATING_DEPTH"
        assert data["progress"] == 45
        assert data["current_stage"] == "Depth Estimation"
        assert data["frames_total"] == 4000
        assert data["frames_processed"] == 1800  # 45% of 4000

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cancel_reconstruction_job_success(test_flight_id: uuid.UUID, test_org_id: uuid.UUID, mock_db_session):
    """Verify cancelling an in-progress reconstruction job."""
    job_id = uuid.uuid4()
    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=test_flight_id,
        org_id=test_org_id,
        status=JobStatus.ESTIMATING_POSE,
        quality_preset=QualityPreset.BALANCED,
        current_stage="Pose Estimation",
        progress=20,
    )

    async def _mock_require_operator_job_access():
        return mock_job

    app.dependency_overrides[require_operator_job_access] = _mock_require_operator_job_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/reconstruction-jobs/{job_id}/cancel")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["status"] == "CANCELLED"
        assert data["current_stage"] == "CANCELLED"
        mock_db_session.commit.assert_called()

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cancel_reconstruction_job_already_terminal(test_flight_id: uuid.UUID, test_org_id: uuid.UUID, mock_db_session):
    """Verify error when attempting to cancel a job that has already completed."""
    job_id = uuid.uuid4()
    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=test_flight_id,
        org_id=test_org_id,
        status=JobStatus.COMPLETED,
        current_stage="Completed",
        progress=100,
    )

    async def _mock_require_operator_job_access():
        return mock_job

    app.dependency_overrides[require_operator_job_access] = _mock_require_operator_job_access
    app.dependency_overrides[get_session] = lambda: mock_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/reconstruction-jobs/{job_id}/cancel")
        assert resp.status_code == 400
        assert "terminal state" in resp.json()["detail"]

    app.dependency_overrides.clear()

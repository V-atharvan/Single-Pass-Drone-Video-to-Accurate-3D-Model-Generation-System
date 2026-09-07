"""
Integration and unit tests for Reconstruction Worker Dispatcher & SQS Consumer Loop – TASK-018.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.api.src.db.models import Flight, FlightStatus, JobStatus, QualityPreset, ReconstructionJob
from workers.orchestrator.runner import JobWorkerDaemon


@pytest.fixture
def mock_queue():
    queue = MagicMock()
    queue.receive_messages.return_value = []
    queue.delete_message.return_value = None
    queue.publish_job.return_value = "msg-123"
    return queue


@pytest.fixture
def mock_redis():
    redis_mock = AsyncMock()
    redis_mock._client = AsyncMock()
    redis_mock._client.set = AsyncMock(return_value=True)
    redis_mock._client.delete = AsyncMock(return_value=True)
    redis_mock.set_job_progress = AsyncMock()
    redis_mock.record_worker_heartbeat = AsyncMock()
    return redis_mock


class MockAsyncSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_worker_process_one_message_success(mock_queue, mock_redis):
    """
    Verify worker pulls message, transitions job in DB to VALIDATING,
    updates Redis progress, executes stage hook, and deletes SQS message.
    """
    job_id = uuid.uuid4()
    flight_id = uuid.uuid4()
    org_id = uuid.uuid4()

    mock_flight = Flight(
        id=flight_id,
        project_id=uuid.uuid4(),
        original_filename="survey.mp4",
        s3_video_key=f"projects/p1/flights/{flight_id}/survey.mp4",
        status=FlightStatus.READY,
        total_frames=1200,
    )
    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=flight_id,
        org_id=org_id,
        status=JobStatus.QUEUED,
        quality_preset=QualityPreset.BALANCED,
        progress=0,
    )

    # Mock DB session execution results
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    async def _mock_execute(stmt):
        result = MagicMock()
        stmt_str = str(stmt)
        if "reconstruction_jobs" in stmt_str:
            result.scalar_one_or_none.return_value = mock_job
        elif "flights" in stmt_str:
            result.scalar_one_or_none.return_value = mock_flight
        else:
            result.scalar_one_or_none.return_value = None
        return result

    mock_session.execute = AsyncMock(side_effect=_mock_execute)
    session_factory = MagicMock(return_value=MockAsyncSessionContext(mock_session))

    daemon = JobWorkerDaemon(
        worker_id="test-worker-01",
        queue_client=mock_queue,
        redis_tracker=mock_redis,
        session_factory=session_factory,
    )

    sqs_message = {
        "ReceiptHandle": "receipt-token-12345",
        "Body": json.dumps({
            "job_id": str(job_id),
            "flight_id": str(flight_id),
            "quality_preset": "BALANCED",
        }),
    }

    processed = await daemon.process_one_message(sqs_message)

    assert processed is True
    # Verify DB transition
    assert mock_job.status == JobStatus.VALIDATING
    assert mock_job.current_stage == "VALIDATING"
    assert mock_job.progress == 5
    assert mock_job.started_at is not None
    mock_session.commit.assert_called_once()

    # Verify Redis update
    mock_redis.set_job_progress.assert_called_once_with(
        job_id=job_id,
        stage="VALIDATING",
        progress=5,
        frames_processed=0,
        total_frames=1200,
        gpu_stats={"utilization": 0.0},
    )

    # Verify SQS message deletion
    mock_queue.delete_message.assert_called_once_with("receipt-token-12345")


@pytest.mark.asyncio
async def test_worker_skips_cancelled_job(mock_queue, mock_redis):
    """Verify worker acknowledges message but does not process if job was already cancelled."""
    job_id = uuid.uuid4()
    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        status=JobStatus.CANCELLED,
        progress=20,
    )

    mock_session = AsyncMock()
    async def _mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_job
        return result

    mock_session.execute = AsyncMock(side_effect=_mock_execute)
    session_factory = MagicMock(return_value=MockAsyncSessionContext(mock_session))

    daemon = JobWorkerDaemon(
        worker_id="test-worker-02",
        queue_client=mock_queue,
        redis_tracker=mock_redis,
        session_factory=session_factory,
    )

    sqs_message = {
        "ReceiptHandle": "receipt-token-cancelled",
        "Body": json.dumps({"job_id": str(job_id)}),
    }

    processed = await daemon.process_one_message(sqs_message)

    assert processed is False
    assert mock_job.status == JobStatus.CANCELLED  # Not changed
    mock_queue.delete_message.assert_called_once_with("receipt-token-cancelled")
    mock_redis.set_job_progress.assert_not_called()


@pytest.mark.asyncio
async def test_worker_skips_when_lock_unavailable(mock_queue, mock_redis):
    """Verify worker skips message when another worker holds the distributed lock."""
    job_id = uuid.uuid4()
    mock_redis._client.set.return_value = False  # Lock acquisition fails

    daemon = JobWorkerDaemon(
        worker_id="test-worker-03",
        queue_client=mock_queue,
        redis_tracker=mock_redis,
    )

    sqs_message = {
        "ReceiptHandle": "receipt-busy",
        "Body": json.dumps({"job_id": str(job_id)}),
    }

    processed = await daemon.process_one_message(sqs_message)

    assert processed is False
    # Message is not deleted so it can be picked up later when visibility timeout expires
    mock_queue.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_worker_poll_and_process_once(mock_queue, mock_redis):
    """Verify poll_and_process_once retrieves from queue and delegates to process_one_message."""
    job_id = uuid.uuid4()
    mock_queue.receive_messages.return_value = [
        {"ReceiptHandle": "rh-1", "Body": json.dumps({"job_id": str(job_id)})}
    ]

    daemon = JobWorkerDaemon(
        worker_id="test-worker-04",
        queue_client=mock_queue,
        redis_tracker=mock_redis,
        poll_wait_seconds=1,
    )

    with patch.object(daemon, "process_one_message", new=AsyncMock(return_value=True)) as mock_process:
        result = await daemon.poll_and_process_once()
        assert result is True
        mock_process.assert_called_once()

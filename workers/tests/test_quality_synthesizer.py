"""
Unit and integration tests for Input Quality Score Synthesizer – TASK-023.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.api.src.db.models import Flight, JobStatus, ReconstructionJob
from packages.schemas.python.single_pass_schemas.quality import (
    InputQualityScore,
    QualityClassification,
    TrajectoryQualityMetrics,
    VisualQualityMetrics,
)
from workers.preprocessing.quality_synthesizer import (
    process_and_persist_preflight_quality,
    synthesize_input_quality_score,
)


@pytest.fixture
def high_quality_visual_metrics() -> VisualQualityMetrics:
    return VisualQualityMetrics(
        blur_score=92.0,
        exposure_score=88.0,
        texture_score=85.0,
        compression_score=95.0,
        shadow_score=90.0,
        illumination_score=94.0,
        shadow_coverage_pct=4.2,
    )


@pytest.fixture
def high_quality_trajectory_metrics() -> TrajectoryQualityMetrics:
    return TrajectoryQualityMetrics(
        gps_continuity_score=98.0,
        speed_variance_score=95.0,
        baseline_overlap_score=92.0,
    )


def test_high_quality_dataset_synthesis(high_quality_visual_metrics, high_quality_trajectory_metrics):
    """Verify high-grade drone dataset scores >= 80 with HIGH classification and zero warnings."""
    report = synthesize_input_quality_score(
        visual_metrics=high_quality_visual_metrics,
        trajectory_metrics=high_quality_trajectory_metrics,
    )

    assert isinstance(report, InputQualityScore)
    assert report.overall_score >= 80.0
    assert report.classification == QualityClassification.HIGH
    assert report.is_reconstructible is True
    assert len(report.warnings) == 0


def test_high_compression_and_heavy_shadows_subscore_warnings(high_quality_trajectory_metrics):
    """Verify high-compression + heavy shadow input produces specific actionable sub-score warnings."""
    compromised_visual = VisualQualityMetrics(
        blur_score=75.0,
        exposure_score=70.0,
        texture_score=65.0,
        compression_score=35.0,  # Below 50 threshold
        shadow_score=25.0,       # Below 40 threshold
        illumination_score=80.0,
        shadow_coverage_pct=45.0,
    )

    report = synthesize_input_quality_score(
        visual_metrics=compromised_visual,
        trajectory_metrics=high_quality_trajectory_metrics,
    )

    assert any("HIGH_VIDEO_COMPRESSION" in w for w in report.warnings)
    assert any("HEAVY_SHADOWS" in w for w in report.warnings)
    assert report.classification in (QualityClassification.MODERATE, QualityClassification.LOW)


def test_unreconstructible_dataset_below_threshold():
    """Verify severely degraded dataset (< 40) is flagged non-reconstructible with warnings."""
    severely_degraded_visual = VisualQualityMetrics(
        blur_score=18.0,
        exposure_score=25.0,
        texture_score=15.0,
        compression_score=20.0,
        shadow_score=10.0,
        illumination_score=30.0,
    )
    severely_degraded_traj = TrajectoryQualityMetrics(
        gps_continuity_score=20.0,
        speed_variance_score=15.0,
        baseline_overlap_score=10.0,
    )

    report = synthesize_input_quality_score(
        visual_metrics=severely_degraded_visual,
        trajectory_metrics=severely_degraded_traj,
    )

    assert report.overall_score < 40.0
    assert report.is_reconstructible is False
    assert report.classification == QualityClassification.LOW
    assert len(report.warnings) >= 3


@pytest.mark.asyncio
async def test_process_and_persist_quality_success(high_quality_visual_metrics, high_quality_trajectory_metrics):
    """Verify acceptable score transitions job to EXTRACTING_FRAMES and emits Redis event."""
    job_id = uuid.uuid4()
    flight_id = uuid.uuid4()

    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=flight_id,
        status=JobStatus.VALIDATING,
        progress=5,
    )
    mock_flight = Flight(
        id=flight_id,
        project_id=uuid.uuid4(),
        original_filename="flight.mp4",
        total_frames=1500,
    )

    report = synthesize_input_quality_score(
        visual_metrics=high_quality_visual_metrics,
        trajectory_metrics=high_quality_trajectory_metrics,
    )

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    async def _mock_execute(stmt):
        res = MagicMock()
        stmt_str = str(stmt)
        if "reconstruction_jobs" in stmt_str:
            res.scalar_one_or_none.return_value = mock_job
        elif "flights" in stmt_str:
            res.scalar_one_or_none.return_value = mock_flight
        return res

    mock_session.execute = AsyncMock(side_effect=_mock_execute)
    mock_redis = AsyncMock()
    mock_redis.set_job_progress = AsyncMock()

    updated_job = await process_and_persist_preflight_quality(
        job_id=job_id,
        flight_id=flight_id,
        input_score=report,
        session=mock_session,
        redis_tracker=mock_redis,
    )

    assert updated_job.status == JobStatus.EXTRACTING_FRAMES
    assert updated_job.current_stage == "EXTRACTING_FRAMES"
    assert updated_job.progress == 10
    mock_session.commit.assert_called_once()

    # Verify Redis publish
    mock_redis.set_job_progress.assert_called_once_with(
        job_id=job_id,
        stage="EXTRACTING_FRAMES",
        progress=10,
        frames_processed=0,
        total_frames=1500,
        gpu_stats={"utilization": 0.0},
    )


@pytest.mark.asyncio
async def test_process_and_persist_quality_failure():
    """Verify score < 40 transitions job to FAILED with remediation message."""
    job_id = uuid.uuid4()
    flight_id = uuid.uuid4()

    mock_job = ReconstructionJob(
        id=job_id,
        flight_id=flight_id,
        status=JobStatus.VALIDATING,
        progress=5,
    )
    mock_flight = Flight(id=flight_id, project_id=uuid.uuid4(), original_filename="flight.mp4")

    # Score < 40 report
    degraded_visual = VisualQualityMetrics(
        blur_score=15.0, exposure_score=20.0, texture_score=10.0,
        compression_score=15.0, shadow_score=10.0, illumination_score=20.0,
    )
    degraded_traj = TrajectoryQualityMetrics(
        gps_continuity_score=10.0, speed_variance_score=10.0, baseline_overlap_score=10.0,
    )
    report = synthesize_input_quality_score(degraded_visual, degraded_traj)

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    async def _mock_execute(stmt):
        res = MagicMock()
        stmt_str = str(stmt)
        if "reconstruction_jobs" in stmt_str:
            res.scalar_one_or_none.return_value = mock_job
        elif "flights" in stmt_str:
            res.scalar_one_or_none.return_value = mock_flight
        return res

    mock_session.execute = AsyncMock(side_effect=_mock_execute)

    updated_job = await process_and_persist_preflight_quality(
        job_id=job_id,
        flight_id=flight_id,
        input_score=report,
        session=mock_session,
    )

    assert updated_job.status == JobStatus.FAILED
    assert "below minimum threshold" in updated_job.error_message

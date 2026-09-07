"""
Input Quality Score Synthesizer and Pre-Flight Quality Engine – TASK-023.
Aggregates visual, compression, shadow, illumination, and trajectory metrics
into an authoritative 0-100 Input Quality Score matching PRD FR-005.
Persists results to PostgreSQL and emits real-time WebSocket notifications.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional
from uuid import UUID

# Ensure packages are resolvable
from pathlib import Path
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
_APPS_API_DIR = _ROOT_DIR / "apps" / "api"
if str(_APPS_API_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_API_DIR))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.schemas.python.single_pass_schemas.quality import (
    InputQualityScore,
    QualityClassification,
    TrajectoryQualityMetrics,
    VisualQualityMetrics,
)
from packages.shared.python.redis_client import RedisStateTracker
from src.db.models import Flight, JobStatus, ReconstructionJob

logger = logging.getLogger("preprocessing.quality_synthesizer")

DEFAULT_WEIGHTS: dict[str, float] = {
    "blur": 0.28,
    "gps": 0.20,
    "texture": 0.15,
    "compression": 0.12,
    "exposure": 0.10,
    "shadow": 0.08,
    "illumination": 0.07,
}

MIN_RECONSTRUCTIBLE_THRESHOLD = 40.0


def synthesize_input_quality_score(
    visual_metrics: VisualQualityMetrics,
    trajectory_metrics: TrajectoryQualityMetrics,
    camera_metadata_complete: bool = True,
    weights: Optional[dict[str, float]] = None,
    extra_warnings: Optional[list[str]] = None,
) -> InputQualityScore:
    """
    Computes weighted aggregate Input Quality Score in [0..100].
    Formula:
      Score = 0.28*Blur + 0.20*GPS + 0.15*Texture + 0.12*Compression
            + 0.10*Exposure + 0.08*Shadow + 0.07*Illumination
    """
    w = weights or DEFAULT_WEIGHTS

    # Aggregate GPS quality composite
    gps_score = (
        trajectory_metrics.gps_continuity_score * 0.5
        + trajectory_metrics.speed_variance_score * 0.3
        + trajectory_metrics.baseline_overlap_score * 0.2
    )

    overall = (
        w["blur"] * visual_metrics.blur_score
        + w["gps"] * gps_score
        + w["texture"] * visual_metrics.texture_score
        + w["compression"] * visual_metrics.compression_score
        + w["exposure"] * visual_metrics.exposure_score
        + w["shadow"] * visual_metrics.shadow_score
        + w["illumination"] * visual_metrics.illumination_score
    )
    overall_score = round(max(0.0, min(100.0, overall)), 2)

    # Classification
    if overall_score >= 80.0:
        classification = QualityClassification.HIGH
    elif overall_score >= 60.0:
        classification = QualityClassification.MODERATE
    else:
        classification = QualityClassification.LOW

    # Component-level actionable warnings
    warnings: list[str] = list(extra_warnings or [])

    if visual_metrics.blur_score < 60.0:
        warnings.append("HIGH_MOTION_BLUR - camera movement exceeded shutter speed; keyframe recovery enabled")
    if visual_metrics.compression_score < 50.0:
        warnings.append("HIGH_VIDEO_COMPRESSION - recommend higher bitrate source (>= 25 Mbps for 4K)")
    if visual_metrics.shadow_score < 40.0:
        warnings.append("HEAVY_SHADOWS - texture and depth confidence will be reduced in shadow regions")
    if gps_score < 50.0:
        warnings.append("POOR_GPS - georeferencing accuracy reduced due to trajectory gaps or speed variance")
    if visual_metrics.texture_score < 40.0:
        warnings.append("LOW_TEXTURE - low-gradient surfaces detected; dense matching confidence reduced")
    if visual_metrics.exposure_score < 50.0:
        warnings.append("POOR_EXPOSURE - dynamic range clipping detected; feature matching may degrade")

    is_reconstructible = overall_score >= MIN_RECONSTRUCTIBLE_THRESHOLD

    # Video quality composite percent
    video_quality_percent = round(
        (visual_metrics.blur_score * 0.35 + visual_metrics.texture_score * 0.25 + visual_metrics.compression_score * 0.2 + visual_metrics.exposure_score * 0.2),
        2,
    )
    gps_quality_percent = round(gps_score, 2)

    return InputQualityScore(
        overall_score=overall_score,
        classification=classification,
        video_quality_percent=video_quality_percent,
        gps_quality_percent=gps_quality_percent,
        motion_blur_percent=visual_metrics.blur_score,
        scene_texture_percent=visual_metrics.texture_score,
        camera_metadata_complete=camera_metadata_complete,
        expected_quality=classification,
        visual_metrics=visual_metrics,
        trajectory_metrics=trajectory_metrics,
        warnings=warnings,
        is_reconstructible=is_reconstructible,
    )


async def process_and_persist_preflight_quality(
    job_id: UUID,
    flight_id: UUID,
    input_score: InputQualityScore,
    session: AsyncSession,
    redis_tracker: Optional[RedisStateTracker] = None,
) -> ReconstructionJob:
    """
    Persists InputQualityScore into flight record, checks reconstructibility threshold,
    updates job status (fails job with remediation instructions if < 40, otherwise proceeds),
    and emits real-time WebSocket progress packet.
    """
    flight_res = await session.execute(select(Flight).where(Flight.id == flight_id))
    flight = flight_res.scalar_one_or_none()
    if flight is not None:
        # Merge or set video_quality_report_json
        existing_report = flight.video_quality_report_json or {}
        existing_report["input_quality_score"] = input_score.model_dump()
        flight.video_quality_report_json = existing_report

    job_res = await session.execute(select(ReconstructionJob).where(ReconstructionJob.id == job_id))
    job = job_res.scalar_one_or_none()
    if job is None:
        raise ValueError(f"Reconstruction job {job_id} not found.")

    if not input_score.is_reconstructible:
        job.status = JobStatus.FAILED
        remediation = (
            f"Input dataset quality score ({input_score.overall_score}/100) below minimum threshold (40/100). "
            f"Remediation: {'; '.join(input_score.warnings) or 'Provide higher-bitrate video with stable GPS'}"
        )
        job.error_message = remediation
        logger.warning("Job %s marked FAILED during pre-flight quality check: %s", job_id, remediation)
    else:
        # Ready for frame extraction
        job.status = JobStatus.EXTRACTING_FRAMES
        job.current_stage = "EXTRACTING_FRAMES"
        job.progress = 10

    await session.commit()
    await session.refresh(job)

    # Publish WebSocket update via Redis
    if redis_tracker is not None:
        total_frames = (flight.total_frames or 0) if flight else 0
        await redis_tracker.set_job_progress(
            job_id=job_id,
            stage=job.current_stage or "VALIDATING",
            progress=job.progress,
            frames_processed=0,
            total_frames=total_frames,
            gpu_stats={"utilization": 0.0},
        )

    return job

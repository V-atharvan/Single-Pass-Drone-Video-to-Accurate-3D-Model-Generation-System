"""
Reconstruction Job Submission & State Machine API – TASK-016
Manages starting, inspecting, and cancelling asynchronous 3D reconstruction jobs.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.rbac import (
    require_operator_flight_access,
    require_operator_job_access,
    require_viewer_job_access,
)
from src.db.base import get_session
from src.db.models import Flight, FlightStatus, JobStatus, QualityPreset, ReconstructionJob

# Try import SQSQueueClient
try:
    from packages.shared.python.queue import SQSQueueClient
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    from packages.shared.python.queue import SQSQueueClient


flight_jobs_router = APIRouter(prefix="/flights/{flight_id}/reconstruction-jobs", tags=["Reconstruction Jobs"])
direct_jobs_router = APIRouter(prefix="/reconstruction-jobs", tags=["Reconstruction Jobs"])


def get_queue_client() -> SQSQueueClient:
    """Dependency provider for SQS queue client (can be overridden in tests)."""
    return SQSQueueClient()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class JobCreateIn(BaseModel):
    quality_preset: QualityPreset = Field(default=QualityPreset.BALANCED, description="Processing detail preset")
    target_crs: str = Field(default="EPSG:4326", description="Target Coordinate Reference System")
    enable_dynamic_removal: bool = Field(default=True, description="Remove dynamic moving objects")
    enable_semantics: bool = Field(default=True, description="Segment and classify scene elements")


class JobOut(BaseModel):
    id: UUID
    flight_id: UUID
    org_id: UUID
    status: JobStatus
    quality_preset: QualityPreset
    current_stage: Optional[str]
    progress: int
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class JobDetailOut(JobOut):
    frames_processed: int = 0
    frames_total: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@flight_jobs_router.post(
    "",
    response_model=JobOut,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new reconstruction job",
)
async def submit_reconstruction_job(
    body: JobCreateIn,
    flight_id: UUID = Path(...),
    flight: Flight = Depends(require_operator_flight_access),
    session: AsyncSession = Depends(get_session),
    queue: SQSQueueClient = Depends(get_queue_client),
) -> ReconstructionJob:
    """
    Submit a reconstruction job for an uploaded flight.
    Creates a database record with state QUEUED and enqueues payload to SQS.
    """
    if flight.status == FlightStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot submit reconstruction job for a flight marked as FAILED.",
        )

    if not flight.s3_video_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Flight does not have an associated S3 video asset.",
        )

    # Org ID from parent flight project
    org_id = flight.project.org_id if flight.project else None
    if not org_id:
        from sqlalchemy import select
        from src.db.models import Project
        proj_res = await session.execute(select(Project).where(Project.id == flight.project_id))
        proj = proj_res.scalar_one()
        org_id = proj.org_id

    job = ReconstructionJob(
        flight_id=flight_id,
        org_id=org_id,
        status=JobStatus.QUEUED,
        quality_preset=body.quality_preset,
        current_stage="QUEUED",
        progress=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Enqueue payload to SQS
    message_payload = {
        "job_id": str(job.id),
        "flight_id": str(flight_id),
        "project_id": str(flight.project_id),
        "org_id": str(org_id),
        "quality_preset": body.quality_preset.value,
        "target_crs": body.target_crs,
        "enable_dynamic_removal": body.enable_dynamic_removal,
        "enable_semantics": body.enable_semantics,
        "s3_video_key": flight.s3_video_key,
        "s3_telemetry_key": flight.s3_telemetry_key,
    }

    try:
        queue.publish_job(message_payload)
    except Exception as e:
        # If queue fails, mark job failed in DB and raise
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to enqueue job to SQS: {e}"
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reconstruction job created but failed to enqueue to worker queue: {e}",
        )

    return job


@direct_jobs_router.get(
    "/{job_id}",
    response_model=JobDetailOut,
    summary="Get reconstruction job status and progress",
)
async def get_reconstruction_job(
    job_id: UUID = Path(...),
    job: ReconstructionJob = Depends(require_viewer_job_access),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Retrieve the current state, progress percentage, active stage, and frame metrics."""
    frames_total = job.flight.total_frames if job.flight and job.flight.total_frames else 0
    # Approximate frames processed based on progress percentage
    frames_processed = int((job.progress / 100.0) * frames_total) if frames_total > 0 else 0

    return {
        **{col.name: getattr(job, col.name) for col in job.__table__.columns},
        "frames_processed": frames_processed,
        "frames_total": frames_total,
        "warnings": [],
    }


@direct_jobs_router.post(
    "/{job_id}/cancel",
    response_model=JobOut,
    summary="Cancel an active reconstruction job",
)
async def cancel_reconstruction_job(
    job_id: UUID = Path(...),
    job: ReconstructionJob = Depends(require_operator_job_access),
    session: AsyncSession = Depends(get_session),
) -> ReconstructionJob:
    """Cancel an active or queued reconstruction job."""
    terminal_states = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    if job.status in terminal_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in terminal state '{job.status.value}'.",
        )

    job.status = JobStatus.CANCELLED
    job.current_stage = "CANCELLED"
    await session.commit()
    await session.refresh(job)
    return job

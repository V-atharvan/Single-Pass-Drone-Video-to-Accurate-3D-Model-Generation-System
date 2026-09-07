"""
Flights Registration and Metadata Linkage Endpoints – TASK-015
Registers uploaded video and GPS assets as cohesive flight entities ready for validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.rbac import (
    require_flight_access,
    require_operator_project_access,
    require_project_access,
    require_viewer_flight_access,
    require_viewer_project_access,
)
from src.db.base import get_session
from src.db.models import Flight, FlightStatus, FlightTelemetry, Project, ReconstructionJob, UserRole


# Two sub-routers:
# 1. project_flights_router for /projects/{project_id}/flights
# 2. direct_flights_router for /flights/{id}
project_flights_router = APIRouter(prefix="/projects/{project_id}/flights", tags=["Flights"])
direct_flights_router = APIRouter(prefix="/flights", tags=["Flights"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class FlightCreateIn(BaseModel):
    original_filename: str = Field(..., min_length=1, max_length=512, description="Original filename of drone video")
    s3_video_key: str = Field(..., min_length=1, max_length=1024, description="S3 object key for video file")
    s3_telemetry_key: Optional[str] = Field(None, max_length=1024, description="Optional S3 key for telemetry file")
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    fps: Optional[float] = Field(None, gt=0.0)
    resolution_width: Optional[int] = Field(None, gt=0)
    resolution_height: Optional[int] = Field(None, gt=0)
    total_frames: Optional[int] = Field(None, ge=0)


class FlightOut(BaseModel):
    id: UUID
    project_id: UUID
    original_filename: str
    s3_video_key: Optional[str]
    s3_telemetry_key: Optional[str]
    duration_seconds: Optional[float]
    fps: Optional[float]
    resolution_width: Optional[int]
    resolution_height: Optional[int]
    total_frames: Optional[int]
    status: FlightStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class FlightDetailOut(FlightOut):
    telemetry_record_count: int = 0
    reconstruction_job_count: int = 0


# ---------------------------------------------------------------------------
# Routes: Project-scoped Flights
# ---------------------------------------------------------------------------

@project_flights_router.post(
    "",
    response_model=FlightOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register uploaded flight assets",
)
async def create_flight(
    body: FlightCreateIn,
    project_id: UUID = Path(...),
    project: Project = Depends(require_operator_project_access),
    session: AsyncSession = Depends(get_session),
) -> Flight:
    """
    Register a newly uploaded drone video and optional companion GPS file as a cohesive flight asset.
    Initial status is set to PENDING (awaiting pre-flight quality validation).
    """
    expected_prefix = f"projects/{project_id}/"
    if not body.s3_video_key.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"S3 video key must belong to project path '{expected_prefix}'",
        )

    if body.s3_telemetry_key and not body.s3_telemetry_key.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"S3 telemetry key must belong to project path '{expected_prefix}'",
        )

    flight = Flight(
        project_id=project_id,
        original_filename=body.original_filename,
        s3_video_key=body.s3_video_key,
        s3_telemetry_key=body.s3_telemetry_key,
        duration_seconds=body.duration_seconds,
        fps=body.fps,
        resolution_width=body.resolution_width,
        resolution_height=body.resolution_height,
        total_frames=body.total_frames,
        status=FlightStatus.PENDING,
    )
    session.add(flight)
    await session.commit()
    await session.refresh(flight)
    return flight


@project_flights_router.get(
    "",
    response_model=list[FlightOut],
    summary="List all flights in project",
)
async def list_project_flights(
    project_id: UUID = Path(...),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    project: Project = Depends(require_viewer_project_access),
    session: AsyncSession = Depends(get_session),
) -> list[Flight]:
    """List all registered flights belonging to a specific project."""
    stmt = (
        select(Flight)
        .where(Flight.project_id == project_id)
        .order_by(Flight.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Routes: Flight-scoped details
# ---------------------------------------------------------------------------

@direct_flights_router.get(
    "/{flight_id}",
    response_model=FlightDetailOut,
    summary="Get flight details and telemetry counts",
)
async def get_flight(
    flight_id: UUID = Path(...),
    flight: Flight = Depends(require_viewer_flight_access),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Fetch complete metadata for a flight, including linked telemetry and job counts."""
    telemetry_count_result = await session.execute(
        select(func.count()).where(FlightTelemetry.flight_id == flight_id)
    )
    telemetry_count = telemetry_count_result.scalar_one()

    job_count_result = await session.execute(
        select(func.count()).where(ReconstructionJob.flight_id == flight_id)
    )
    job_count = job_count_result.scalar_one()

    return {
        **{col.name: getattr(flight, col.name) for col in flight.__table__.columns},
        "telemetry_record_count": telemetry_count,
        "reconstruction_job_count": job_count,
    }

"""
Model Assets Registration & Final Pipeline Completion — TASK-056.

Finalizes the 3D reconstruction pipeline:
  1. Creates or updates the `models` record in PostgreSQL with geographic bounding box,
     PostGIS geometry polygon (EPSG:4326), coverage, RMSE, and overall quality score.
  2. Inserts asset records in `model_assets` for all generated files (GLB, OBJ, LAS, PLY,
     OGC 3D Tiles, DEM GeoTIFF, DSM GeoTIFF).
  3. Transitions `reconstruction_jobs.status` to `COMPLETED` with `completed_at = NOW()`.
  4. Broadcasts completion events via Redis state tracker and WebSocket channels.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
_APPS_API_DIR = _ROOT_DIR / "apps" / "api"
if str(_APPS_API_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_API_DIR))

from apps.api.src.db.models import (
    AssetType,
    JobStatus,
    Model,
    ModelAsset,
    ModelStatus,
    ReconstructionJob,
)
from packages.schemas.python.single_pass_schemas import AccuracyReport
from packages.shared.python.redis_client import RedisStateTracker
from workers.geospatial.georeferencer import GeographicBoundingBox

logger = logging.getLogger("orchestrator.job_finalizer")


@dataclass
class AssetRegistrationInput:
    """Specification of an asset file to register."""

    asset_type: AssetType
    s3_key: str
    file_size_bytes: int
    lod_level: Optional[int] = None


@dataclass
class FinalizationResult:
    """Summary of the pipeline finalization and asset registration."""

    model_id: uuid.UUID
    job_id: uuid.UUID
    project_id: uuid.UUID
    status: str
    asset_count: int
    assets: List[Dict[str, Any]]
    completed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "job_id": str(self.job_id),
            "project_id": str(self.project_id),
            "status": self.status,
            "asset_count": self.asset_count,
            "assets": self.assets,
            "completed_at": self.completed_at.isoformat(),
        }


class JobFinalizer:
    """
    Orchestrates final database persistence, PostGIS polygon generation,
    and completion signaling.
    """

    def __init__(
        self,
        redis_tracker: Optional[RedisStateTracker] = None,
    ) -> None:
        self.redis_tracker = redis_tracker

    def format_postgis_wkt_polygon(self, bbox: GeographicBoundingBox) -> str:
        """
        Creates a closed WKT polygon for PostGIS geometry with SRID 4326.
        """
        return (
            f"SRID=4326;POLYGON(("
            f"{bbox.min_lon} {bbox.min_lat}, "
            f"{bbox.max_lon} {bbox.min_lat}, "
            f"{bbox.max_lon} {bbox.max_lat}, "
            f"{bbox.min_lon} {bbox.max_lat}, "
            f"{bbox.min_lon} {bbox.min_lat}))"
        )

    async def finalize_job(
        self,
        job_id: uuid.UUID,
        project_id: uuid.UUID,
        bbox: GeographicBoundingBox,
        accuracy_report: AccuracyReport,
        assets: Sequence[AssetRegistrationInput],
        model_name: str = "Georeferenced 3D Model",
        session: Optional[AsyncSession] = None,
        model_id: Optional[uuid.UUID] = None,
    ) -> FinalizationResult:
        """
        Registers all assets in DB, transitions job status to COMPLETED,
        and broadcasts progress 100%.
        """
        now = datetime.now(timezone.utc)
        target_model_id = model_id or uuid.uuid4()
        s3_prefix = f"projects/{project_id}/models/{target_model_id}/"
        polygon_wkt = self.format_postgis_wkt_polygon(bbox)

        registered_assets: List[Dict[str, Any]] = []

        if session is not None:
            # 1. Look for existing Model or create new one
            model_query = await session.execute(
                select(Model).where(Model.job_id == job_id)
            )
            model_record = model_query.scalar_one_or_none()

            if model_record is None:
                model_record = Model(
                    id=target_model_id,
                    job_id=job_id,
                    project_id=project_id,
                    name=model_name,
                    crs="EPSG:4326",
                    bbox=polygon_wkt,
                    center_lat=bbox.center_lat,
                    center_lon=bbox.center_lon,
                    min_altitude=bbox.min_alt_m,
                    max_altitude=bbox.max_alt_m,
                    coverage_percent=accuracy_report.coverage_percent,
                    confidence_score=accuracy_report.overall_quality_score,
                    position_rmse=accuracy_report.horizontal_rmse_meters,
                    vertical_rmse=accuracy_report.vertical_rmse_meters,
                    s3_prefix=s3_prefix,
                    status=ModelStatus.READY,
                )
                session.add(model_record)
            else:
                target_model_id = model_record.id
                model_record.name = model_name
                model_record.bbox = polygon_wkt
                model_record.center_lat = bbox.center_lat
                model_record.center_lon = bbox.center_lon
                model_record.min_altitude = bbox.min_alt_m
                model_record.max_altitude = bbox.max_alt_m
                model_record.coverage_percent = accuracy_report.coverage_percent
                model_record.confidence_score = accuracy_report.overall_quality_score
                model_record.position_rmse = accuracy_report.horizontal_rmse_meters
                model_record.vertical_rmse = accuracy_report.vertical_rmse_meters
                model_record.s3_prefix = s3_prefix
                model_record.status = ModelStatus.READY

            # 2. Insert Asset Records
            for asset_in in assets:
                asset_id = uuid.uuid4()
                asset_record = ModelAsset(
                    id=asset_id,
                    model_id=target_model_id,
                    asset_type=asset_in.asset_type,
                    s3_key=asset_in.s3_key,
                    file_size_bytes=asset_in.file_size_bytes,
                    lod_level=asset_in.lod_level,
                )
                session.add(asset_record)
                registered_assets.append({
                    "id": str(asset_id),
                    "asset_type": asset_in.asset_type.value,
                    "s3_key": asset_in.s3_key,
                    "file_size_bytes": asset_in.file_size_bytes,
                })

            # 3. Transition Job to COMPLETED
            job_query = await session.execute(
                select(ReconstructionJob).where(ReconstructionJob.id == job_id)
            )
            job_record = job_query.scalar_one_or_none()
            if job_record is not None:
                job_record.status = JobStatus.COMPLETED
                job_record.current_stage = "COMPLETED"
                job_record.progress = 100
                job_record.completed_at = now

            await session.commit()
        else:
            # Standalone/test mode without live DB session
            for asset_in in assets:
                registered_assets.append({
                    "id": str(uuid.uuid4()),
                    "asset_type": asset_in.asset_type.value,
                    "s3_key": asset_in.s3_key,
                    "file_size_bytes": asset_in.file_size_bytes,
                })

        # 4. Update Redis State and Broadcast WebSocket Event
        if self.redis_tracker is not None:
            try:
                await self.redis_tracker.set_job_progress(
                    job_id=job_id,
                    stage="COMPLETED",
                    progress=100,
                    frames_processed=0,
                    total_frames=0,
                )
                event_payload = {
                    "event": "JOB_COMPLETED",
                    "job_id": str(job_id),
                    "model_id": str(target_model_id),
                    "stage": "completed",
                    "progress": 1.0,
                    "completed_at": now.isoformat(),
                    "quality_score": accuracy_report.overall_quality_score,
                }
                await self.redis_tracker.publish_event(
                    f"job:{job_id}:events",
                    event_payload,
                )
            except Exception as exc:
                logger.warning("Failed to publish Redis completion event: %s", exc)

        logger.info(
            "Reconstruction Job %s finalized successfully: Model %s ready with %d registered assets.",
            job_id, target_model_id, len(registered_assets)
        )

        return FinalizationResult(
            model_id=target_model_id,
            job_id=job_id,
            project_id=project_id,
            status="COMPLETED",
            asset_count=len(registered_assets),
            assets=registered_assets,
            completed_at=now,
        )

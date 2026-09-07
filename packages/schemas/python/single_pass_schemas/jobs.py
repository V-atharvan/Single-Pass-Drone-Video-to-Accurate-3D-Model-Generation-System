"""Job state machine and execution schemas for Single-Pass 3D Reconstruction Platform."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class JobState(str, Enum):
    """The 15 canonical reconstruction job lifecycle states defined in PRD Section 30."""

    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    EXTRACTING_FRAMES = "EXTRACTING_FRAMES"
    ESTIMATING_POSE = "ESTIMATING_POSE"
    ESTIMATING_DEPTH = "ESTIMATING_DEPTH"
    SEGMENTING = "SEGMENTING"
    FUSING = "FUSING"
    RECONSTRUCTING = "RECONSTRUCTING"
    TEXTURING = "TEXTURING"
    GEOREFERENCING = "GEOREFERENCING"
    QUALITY_CHECK = "QUALITY_CHECK"
    GENERATING_TILES = "GENERATING_TILES"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class QualityPreset(str, Enum):
    """Reconstruction resolution and processing density preset."""

    LOW = "LOW"
    BALANCED = "BALANCED"
    HIGH = "HIGH"


class GPUStatistics(BaseModel):
    """Real-time GPU worker performance snapshot."""

    gpu_utilization_pct: float = Field(0.0, ge=0.0, le=100.0)
    vram_used_mb: float = Field(0.0, ge=0.0)
    vram_total_mb: float = Field(0.0, ge=0.0)
    temperature_celsius: Optional[float] = None


class JobProgressUpdate(BaseModel):
    """Real-time progress telemetry streamed over WebSocket to frontend clients."""

    job_id: UUID
    state: JobState
    stage: str = Field(..., description="Human-readable active stage description")
    progress_percent: float = Field(..., ge=0.0, le=100.0, description="Overall job progress percentage (0-100)")
    frames_processed: int = Field(0, ge=0)
    frames_total: int = Field(0, ge=0)
    gpu_stats: Optional[GPUStatistics] = None
    estimated_seconds_remaining: Optional[float] = Field(None, ge=0.0)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReconstructionJobCreate(BaseModel):
    """Payload for submitting a new reconstruction job."""

    flight_id: UUID
    quality_preset: QualityPreset = Field(default=QualityPreset.BALANCED)
    target_crs: str = Field("EPSG:4326", description="Target Coordinate Reference System (e.g. EPSG:4326 or UTM)")
    enable_dynamic_removal: bool = Field(True, description="Detect and mask moving vehicles/pedestrians")
    enable_semantics: bool = Field(True, description="Classify scene elements (buildings, terrain, roads)")


class ReconstructionJobResponse(BaseModel):
    """API response model for a reconstruction job."""

    id: UUID
    flight_id: UUID
    project_id: UUID
    org_id: UUID
    status: JobState
    quality_preset: QualityPreset
    current_stage: str
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    frames_processed: int = Field(0, ge=0)
    frames_total: int = Field(0, ge=0)
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

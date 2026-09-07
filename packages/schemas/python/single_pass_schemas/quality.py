"""Input quality assessment and flight validation schemas."""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class QualityClassification(str, Enum):
    """Categorical classification of dataset suitability."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class VisualQualityMetrics(BaseModel):
    """Computer-vision visual quality indicators."""

    blur_score: float = Field(..., ge=0.0, le=100.0, description="Laplacian variance sharpness metric (0-100)")
    exposure_score: float = Field(..., ge=0.0, le=100.0, description="Luminance balance and exposure metric (0-100)")
    texture_score: float = Field(..., ge=0.0, le=100.0, description="Spatial gradient detail density (0-100)")
    compression_score: float = Field(100.0, ge=0.0, le=100.0, description="Compression artifact absence score (0-100)")
    shadow_score: float = Field(100.0, ge=0.0, le=100.0, description="Shadow absence score (0-100)")
    illumination_score: float = Field(100.0, ge=0.0, le=100.0, description="Illumination stability score (0-100)")
    shadow_coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Percentage of scene under deep shadow")


class TrajectoryQualityMetrics(BaseModel):
    """Flight path continuity and baseline overlap indicators."""

    gps_continuity_score: float = Field(..., ge=0.0, le=100.0, description="GPS gap and dropout absence score (0-100)")
    speed_variance_score: float = Field(..., ge=0.0, le=100.0, description="Velocity vector smoothness score (0-100)")
    baseline_overlap_score: float = Field(..., ge=0.0, le=100.0, description="Inter-frame baseline overlap score (0-100)")


class InputQualityScore(BaseModel):
    """Comprehensive pre-flight data validation report matching PRD FR-005 and Design Doc Section 12."""

    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted aggregate quality score (0-100)")
    classification: QualityClassification
    video_quality_percent: float = Field(..., ge=0.0, le=100.0, description="Video quality score (0-100%)")
    gps_quality_percent: float = Field(..., ge=0.0, le=100.0, description="GPS quality score (0-100%)")
    motion_blur_percent: float = Field(..., ge=0.0, le=100.0, description="Sharpness score (0-100%)")
    scene_texture_percent: float = Field(..., ge=0.0, le=100.0, description="Texture density score (0-100%)")
    camera_metadata_complete: bool = Field(True, description="Whether camera parameters were detected")
    expected_quality: QualityClassification
    visual_metrics: VisualQualityMetrics
    trajectory_metrics: TrajectoryQualityMetrics
    warnings: List[str] = Field(default_factory=list, description="Actionable pre-flight warnings")
    is_reconstructible: bool = Field(..., description="Whether dataset meets minimum threshold for processing")


class VideoQualityReport(BaseModel):
    """
    Detailed video stream integrity and compression quality assessment report.
    Matches TASK-019 specification.
    """

    width: int = Field(..., description="Video frame width in pixels")
    height: int = Field(..., description="Video frame height in pixels")
    fps: float = Field(..., description="Frames per second")
    duration_seconds: float = Field(..., description="Duration in seconds")
    total_frames: int = Field(..., description="Total decoded frame count")
    codec: str = Field(..., description="Video codec name (e.g. h264, hevc)")
    bitrate_mbps: float = Field(0.0, description="Estimated average bitrate in Mbps")
    avg_gop_size: float = Field(0.0, description="Average keyframe GOP interval")
    blur_score: float = Field(..., ge=0.0, le=100.0, description="Sharpness metric across sampled frames (0-100)")
    compression_artifact_score: float = Field(
        ..., ge=0.0, le=100.0, description="Compression quality score (0=severe artifacts, 100=clean)"
    )
    noise_score: float = Field(..., ge=0.0, le=100.0, description="Noise metric (0=high noise, 100=pristine)")
    exposure_score: float = Field(..., ge=0.0, le=100.0, description="Luminance balance and exposure score (0-100)")
    frame_usability_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of frames usable for 3D reconstruction"
    )
    gps_availability: bool = Field(False, description="Whether GPS/telemetry data is available alongside video")
    metadata_completeness_score: float = Field(
        100.0, ge=0.0, le=100.0, description="Completeness of stream metadata"
    )
    warnings: List[str] = Field(default_factory=list, description="Quality warnings and flags")

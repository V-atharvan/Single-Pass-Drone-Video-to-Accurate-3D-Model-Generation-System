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

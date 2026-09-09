"""3D model metadata, accuracy reports, measurements, and export contracts."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


class ObservationState(str, Enum):
    """Surface visibility classification defined in PRD Section 19."""

    OBSERVED = "OBSERVED"
    PARTIALLY_OBSERVED = "PARTIALLY_OBSERVED"
    AI_INFERRED = "AI_INFERRED"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    """Categorical confidence rating."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PositioningMode(str, Enum):
    """Authoritative positioning sensor configuration from PRD Section 39."""

    RTK_PPK = "RTK_PPK"
    RTK = "RTK"
    GPS_IMU = "GPS_IMU"
    GPS_ONLY = "GPS_ONLY"
    GPS_DEGRADED = "GPS_DEGRADED"
    VISUAL_ONLY = "VISUAL_ONLY"


class ScaleSource(str, Enum):
    """Source of absolute metric scale estimation."""

    RTK = "RTK"
    GPS_BASELINE = "GPS_BASELINE"
    BAROMETRIC = "BAROMETRIC"
    VISUAL_ONLY = "VISUAL_ONLY"


class AccuracyReport(BaseModel):
    """Quantitative reconstruction accuracy report matching PRD Section 39."""

    overall_quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall model score (0-100)")
    horizontal_rmse_meters: float = Field(..., ge=0.0, description="Horizontal RMSE in meters (PRD target <= 0.5m)")
    vertical_rmse_meters: float = Field(..., ge=0.0, description="Vertical RMSE in meters (PRD target <= 0.75m)")
    coverage_percent: float = Field(..., ge=0.0, le=100.0, description="Observable surface coverage % (PRD target >= 90%)")
    dynamic_contamination_percent: float = Field(..., ge=0.0, le=100.0, description="Moving-object residual % (PRD target < 5%)")
    observed_surface_percent: float = Field(..., ge=0.0, le=100.0)
    partially_observed_percent: float = Field(..., ge=0.0, le=100.0)
    ai_inferred_percent: float = Field(..., ge=0.0, le=100.0)
    pose_confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    depth_confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    geolocation_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    warnings: List[str] = Field(default_factory=list)
    reference_data_used: List[str] = Field(default_factory=list)

    # Enhanced TASK-053 fields
    positioning_mode: PositioningMode = PositioningMode.GPS_IMU
    ground_control_used: bool = False
    estimated_horizontal_uncertainty_m: Optional[float] = None
    estimated_vertical_uncertainty_m: Optional[float] = None
    scale_source: ScaleSource = ScaleSource.GPS_BASELINE
    measurement_error_percent: Optional[float] = None

    @model_validator(mode="after")
    def validate_surface_sum(self) -> "AccuracyReport":
        total = self.observed_surface_percent + self.partially_observed_percent + self.ai_inferred_percent
        if total > 100.1:  # allow 0.1 for float rounding
            raise ValueError(
                f"Surface observation percentages sum ({total:.1f}%) exceeds 100%"
            )
        return self


class MeasurementType(str, Enum):
    """Supported interactive measurement tools matching Design Doc Section 17."""

    DISTANCE = "DISTANCE"
    HEIGHT = "HEIGHT"
    AREA = "AREA"
    VOLUME = "VOLUME"


class MeasurementResult(BaseModel):
    """User-created geospatial measurement."""

    id: UUID
    model_id: UUID
    user_id: UUID
    type: MeasurementType
    value: float = Field(..., description="Measured quantity value")
    unit: str = Field(..., description="Measurement unit (e.g. 'm', 'm²', 'm³')")
    estimated_error_margin: float = Field(..., ge=0.0, description="Uncertainty margin (±value)")
    points_geojson: Dict[str, Any] = Field(..., description="GeoJSON geometry of points/polygon")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExportFormat(str, Enum):
    """Supported export file formats matching PRD Section 26 and Design Doc Section 21."""

    GLB = "GLB"
    OBJ = "OBJ"
    PLY = "PLY"
    LAS = "LAS"
    LAZ = "LAZ"
    TILES_3D = "TILES_3D"
    DEM_TIF = "DEM_TIF"
    DSM_TIF = "DSM_TIF"


class ModelAsset(BaseModel):
    """Registered 3D file asset associated with a model."""

    id: UUID
    model_id: UUID
    asset_type: ExportFormat
    s3_key: str
    file_size_bytes: int = Field(..., ge=0)
    lod_level: Optional[int] = Field(None, ge=0, le=2)


class BoundingBox(BaseModel):
    """Geographic bounding box in decimal degrees and elevation."""

    min_latitude: float = Field(..., ge=-90.0, le=90.0)
    max_latitude: float = Field(..., ge=-90.0, le=90.0)
    min_longitude: float = Field(..., ge=-180.0, le=180.0)
    max_longitude: float = Field(..., ge=-180.0, le=180.0)
    min_altitude: float
    max_altitude: float


class ModelMetadata(BaseModel):
    """Authoritative metadata record for a reconstructed 3D scene."""

    id: UUID
    project_id: UUID
    job_id: UUID
    name: str
    crs: str = Field("EPSG:4326")
    bbox: BoundingBox
    center_latitude: float = Field(..., ge=-90.0, le=90.0)
    center_longitude: float = Field(..., ge=-180.0, le=180.0)
    accuracy_report: AccuracyReport
    assets: List[ModelAsset] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

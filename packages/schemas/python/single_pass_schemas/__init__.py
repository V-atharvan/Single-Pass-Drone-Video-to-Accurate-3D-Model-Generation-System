"""Single-Pass 3D Reconstruction Platform - Authoritative Data Contracts."""

from .telemetry import (
    GPSRecord,
    Quaternion,
    IMURecord,
    CameraIntrinsics,
    CameraPose,
    KeyframeMetadata,
)
from .jobs import (
    JobState,
    QualityPreset,
    GPUStatistics,
    JobProgressUpdate,
    ReconstructionJobCreate,
    ReconstructionJobResponse,
)
from .quality import (
    QualityClassification,
    VisualQualityMetrics,
    TrajectoryQualityMetrics,
    InputQualityScore,
)
from .models import (
    ObservationState,
    ConfidenceLevel,
    AccuracyReport,
    MeasurementType,
    MeasurementResult,
    ExportFormat,
    ModelAsset,
    BoundingBox,
    ModelMetadata,
)

__all__ = [
    # Telemetry
    "GPSRecord",
    "Quaternion",
    "IMURecord",
    "CameraIntrinsics",
    "CameraPose",
    "KeyframeMetadata",
    # Jobs
    "JobState",
    "QualityPreset",
    "GPUStatistics",
    "JobProgressUpdate",
    "ReconstructionJobCreate",
    "ReconstructionJobResponse",
    # Quality
    "QualityClassification",
    "VisualQualityMetrics",
    "TrajectoryQualityMetrics",
    "InputQualityScore",
    # Models
    "ObservationState",
    "ConfidenceLevel",
    "AccuracyReport",
    "MeasurementType",
    "MeasurementResult",
    "ExportFormat",
    "ModelAsset",
    "BoundingBox",
    "ModelMetadata",
]

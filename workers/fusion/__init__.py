"""
3D Point Cloud Fusion, Surface Completion & Outlier Filtering Package — Phase 7 (TASK-040 to TASK-045).

Modules:
  - unprojector: 2D depth back-projection with camera pinhole intrinsics and 6-DoF extrinsics (TASK-040).
  - point_fusion: Spatial voxel hashing and confidence-weighted multi-frame fusion (TASK-041).
  - outlier_filter: Statistical Outlier Removal (SOR), Radius Outlier Removal (ROR), and dynamic filtering (TASK-042).
  - normal_estimator: Preset voxel downsampling, PCA surface normals, and camera orientation (TASK-043).
  - occlusion_analyzer: 6-state ObservationState categorization and global coverage percentage validation (TASK-044).
  - las_exporter: ASPRS LAS 1.2/1.4 binary exporter, standard PLY writer, and S3 manifest coordination (TASK-045).
"""
from workers.fusion.las_exporter import (
    ExportedPointCloudFiles,
    PointCloudSerializer,
    encode_las_binary,
    encode_ply_binary,
)
from workers.fusion.normal_estimator import (
    NormalEstimator,
    OrientedPointCloud,
)
from workers.fusion.occlusion_analyzer import (
    OBSERVATION_STATE_COLORS,
    ClassifiedObservationCloud,
    ObservationState,
    ObservationStatistics,
    OcclusionAnalyzer,
)
from workers.fusion.outlier_filter import (
    FilterStatistics,
    OutlierFilter,
    OutlierFilterResult,
)
from workers.fusion.point_fusion import (
    FusedPointCloud,
    PointFusionEngine,
)
from workers.fusion.unprojector import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthUnprojector,
    FramePointCloud,
)

__all__ = [
    # TASK-040
    "CameraIntrinsics",
    "CameraExtrinsics",
    "FramePointCloud",
    "DepthUnprojector",
    # TASK-041
    "FusedPointCloud",
    "PointFusionEngine",
    # TASK-042
    "FilterStatistics",
    "OutlierFilterResult",
    "OutlierFilter",
    # TASK-043
    "OrientedPointCloud",
    "NormalEstimator",
    # TASK-044
    "ObservationState",
    "OBSERVATION_STATE_COLORS",
    "ObservationStatistics",
    "ClassifiedObservationCloud",
    "OcclusionAnalyzer",
    # TASK-045
    "ExportedPointCloudFiles",
    "PointCloudSerializer",
    "encode_las_binary",
    "encode_ply_binary",
]

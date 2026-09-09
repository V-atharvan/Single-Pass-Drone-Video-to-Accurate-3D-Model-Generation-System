"""
workers/geospatial — Georeferencing, Accuracy Assessment, DEM/DSM Rasterization, and 3D Tiling.
"""
from .georeferencer import (
    Georeferencer,
    GeographicBoundingBox,
    Sim3Transform,
    wgs84_to_ecef,
    ecef_to_wgs84,
    wgs84_to_enu,
    enu_to_wgs84,
    estimate_sim3_umeyama,
)
from .accuracy_evaluator import (
    AccuracyEvaluator,
    AccuracyEvaluationResult,
    RelativeAccuracyMetrics,
    AbsoluteGeospatialMetrics,
    MeasurementUncertaintyMetrics,
)
from .dem_rasterizer import (
    DemRasterizer,
    RasterMetadata,
    encode_geotiff_float32,
)
from .tileset_generator import (
    TilesetGenerator,
    TilesetMetadata,
    TileNode,
    compute_geographic_region_radians,
)

__all__ = [
    # Georeferencer
    "Georeferencer",
    "GeographicBoundingBox",
    "Sim3Transform",
    "wgs84_to_ecef",
    "ecef_to_wgs84",
    "wgs84_to_enu",
    "enu_to_wgs84",
    "estimate_sim3_umeyama",
    # Accuracy
    "AccuracyEvaluator",
    "AccuracyEvaluationResult",
    "RelativeAccuracyMetrics",
    "AbsoluteGeospatialMetrics",
    "MeasurementUncertaintyMetrics",
    # DEM / DSM
    "DemRasterizer",
    "RasterMetadata",
    "encode_geotiff_float32",
    # Tiling
    "TilesetGenerator",
    "TilesetMetadata",
    "TileNode",
    "compute_geographic_region_radians",
]

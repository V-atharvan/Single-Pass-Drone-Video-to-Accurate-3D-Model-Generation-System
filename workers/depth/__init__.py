"""
Depth Estimation & Confidence Worker Package — Phase 5 (TASK-030 to TASK-034).

Modules:
  - runtime: PyTorch GPU runtime, CUDA device detection, model loaders, autocast context.
  - depth_estimator: Monocular metric depth estimation pipeline with edge-preserving bilateral upsampling.
  - confidence_estimator: Uncertainty and confidence field prediction (gradient, plausibility, reprojection).
  - scale_aligner: Metric scale calibration against GPS baseline, barometric altitude, and sparse 3D geometry.
  - serializer: Serialization to 16-bit PNG, NPZ, confidence PNG, Turbo previews, S3 interim upload, Redis progress.
"""
from workers.depth.confidence_estimator import (
    ConfidenceEstimationResult,
    ConfidenceEstimator,
    ConfidenceMap,
)
from workers.depth.depth_estimator import (
    DepthEstimationResult,
    DepthEstimator,
    DepthMap,
)
from workers.depth.runtime import (
    DeviceInfo,
    DepthModelLoader,
    LoadedModel,
    ModelWeightManager,
    autocast_context,
    clear_gpu_cache,
    detect_device,
)
from workers.depth.scale_aligner import (
    FrameScaleFactor,
    ScaleAlignmentResult,
    ScaleAligner,
    ScaleCalibrationMode,
)
from workers.depth.serializer import (
    DEFAULT_DEPTH_SCALE,
    DepthSerializationResult,
    DepthSerializer,
    SerializedFrame,
    compute_sha256,
    decode_confidence_png,
    decode_depth_npz,
    decode_depth_png,
    encode_confidence_png,
    encode_depth_npz,
    encode_depth_png,
    encode_depth_preview,
)

__all__ = [
    # runtime
    "detect_device",
    "DeviceInfo",
    "clear_gpu_cache",
    "autocast_context",
    "ModelWeightManager",
    "DepthModelLoader",
    "LoadedModel",
    # depth_estimator
    "DepthEstimator",
    "DepthMap",
    "DepthEstimationResult",
    # confidence_estimator
    "ConfidenceEstimator",
    "ConfidenceMap",
    "ConfidenceEstimationResult",
    # scale_aligner
    "ScaleAligner",
    "ScaleAlignmentResult",
    "ScaleCalibrationMode",
    "FrameScaleFactor",
    # serializer
    "DepthSerializer",
    "SerializedFrame",
    "DepthSerializationResult",
    "DEFAULT_DEPTH_SCALE",
    "encode_depth_png",
    "decode_depth_png",
    "encode_depth_npz",
    "decode_depth_npz",
    "encode_confidence_png",
    "decode_confidence_png",
    "encode_depth_preview",
    "compute_sha256",
]

"""
Semantic Segmentation & Dynamic Object Understanding Package — Phase 6 (TASK-035 to TASK-039).

Modules:
  - semantic_classifier: Semantic segmentation into geospatial categories, animal & dynamic object fallback (TASK-035).
  - dynamic_tracker: Multi-frame optical flow motion divergence tracking and static/dynamic classification (TASK-036).
  - mask_generator: Dilated binary exclusion masks for dynamic entities (TASK-037).
  - mask_combiner: Unified 3-channel semantic tensors and multi-resolution pyramids (TASK-038).
  - serializer: Scratch NVMe caching, S3 interim uploads, Redis progress, and contamination metrics (TASK-039).
"""
from workers.segmentation.dynamic_tracker import (
    CameraEgoMotionEstimator,
    DynamicObjectTracker,
    FrameTrackingResult,
    MotionDivergenceAnalyzer,
    MotionState,
    MultiFrameTrackingResult,
    TrackedObject,
)
from workers.segmentation.mask_combiner import (
    MaskPyramid,
    MultiChannelSemanticMask,
    SemanticMaskCombiner,
)
from workers.segmentation.mask_generator import (
    DynamicMask,
    DynamicMaskGenerator,
)
from workers.segmentation.semantic_classifier import (
    CLASS_COLOR_PALETTE,
    DYNAMIC_CLASSES,
    STATIC_CLASSES,
    AuxiliaryDetection,
    LoadedSegmentationModel,
    SemanticClass,
    SemanticClassifier,
    SemanticMap,
    SemanticSegmentationResult,
    SegmentationModelLoader,
    apply_auxiliary_detections,
)
from workers.segmentation.serializer import (
    DynamicContaminationMetrics,
    SegmentationSerializationResult,
    SemanticMaskSerializer,
    SerializedSemanticFrame,
    compute_sha256,
    encode_dynamic_png,
    encode_pyramid_npz,
    encode_semantic_png,
    encode_semantic_preview_png,
)

__all__ = [
    # Classifier (TASK-035)
    "CLASS_COLOR_PALETTE",
    "DYNAMIC_CLASSES",
    "STATIC_CLASSES",
    "AuxiliaryDetection",
    "LoadedSegmentationModel",
    "SemanticClass",
    "SemanticClassifier",
    "SemanticMap",
    "SemanticSegmentationResult",
    "SegmentationModelLoader",
    "apply_auxiliary_detections",
    # Tracker (TASK-036)
    "CameraEgoMotionEstimator",
    "DynamicObjectTracker",
    "FrameTrackingResult",
    "MotionDivergenceAnalyzer",
    "MotionState",
    "MultiFrameTrackingResult",
    "TrackedObject",
    # Mask Generator (TASK-037)
    "DynamicMask",
    "DynamicMaskGenerator",
    # Mask Combiner (TASK-038)
    "MaskPyramid",
    "MultiChannelSemanticMask",
    "SemanticMaskCombiner",
    # Serializer (TASK-039)
    "DynamicContaminationMetrics",
    "SegmentationSerializationResult",
    "SemanticMaskSerializer",
    "SerializedSemanticFrame",
    "compute_sha256",
    "encode_dynamic_png",
    "encode_pyramid_npz",
    "encode_semantic_png",
    "encode_semantic_preview_png",
]

"""
Unit tests for Semantic Segmentation Pipeline — TASK-035 [ENHANCED].

Covers:
  - SemanticClass enum integrity and 11 PRD target classes.
  - Color palette mappings for all classes.
  - Frame preprocessing and shape contracts.
  - Nearest-neighbor class mask upsampling (preventing intermediate invalid class values).
  - Auxiliary detector logic:
      - Confident animal classification -> SemanticClass.ANIMAL.
      - Ambiguous animal classification (< threshold) -> fallback to SemanticClass.DYNAMIC_OBJECT.
      - Generic moving pixel region -> fallback to SemanticClass.DYNAMIC_OBJECT.
  - Batched array-based and disk-based classification.
  - Error handling: missing files, mismatched indices.
  - Class proportion computation and colorized mask generation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from workers.segmentation.semantic_classifier import (
    CLASS_COLOR_PALETTE,
    DYNAMIC_CLASSES,
    STATIC_CLASSES,
    AuxiliaryDetection,
    SemanticClass,
    SemanticClassifier,
    SemanticMap,
    SemanticSegmentationResult,
    SegmentationModelLoader,
    _compute_class_proportions,
    _preprocess_frame,
    _upsample_mask_and_confidence,
    apply_auxiliary_detections,
)


@pytest.fixture
def classifier():
    loader = SegmentationModelLoader()
    stub_model = loader.load(simulation_mode=True)
    return SemanticClassifier(loaded_model=stub_model, batch_size=2)


@pytest.fixture
def synthetic_aerial_frames():
    """Generates 3 synthetic aerial BGR frames (480x640)."""
    frames = []
    for i in range(3):
        # Base terrain background (soil brown / olive)
        img = np.full((480, 640, 3), (40, 100, 140), dtype=np.uint8)
        # Forest / vegetation patch (green in BGR)
        cv2.rectangle(img, (50, 50), (250, 250), (35, 140, 45), -1)
        # Road corridor (dark gray in BGR)
        cv2.rectangle(img, (280, 0), (360, 480), (70, 70, 70), -1)
        # Water body (blue in BGR)
        cv2.circle(img, (500, 350), 70, (180, 120, 30), -1)
        # Building roof (high contrast rectangle)
        cv2.rectangle(img, (400, 50), (550, 180), (200, 200, 200), -1)
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Enum and Class Schema Tests
# ---------------------------------------------------------------------------

def test_semantic_class_enum_values():
    """Validates that all 11 required classes exist with distinct IDs."""
    expected_classes = {
        "UNKNOWN": 0,
        "BUILDING": 1,
        "ROAD": 2,
        "TERRAIN": 3,
        "VEGETATION": 4,
        "VEHICLE": 5,
        "PERSON": 6,
        "UTILITY_INFRASTRUCTURE": 7,
        "WATER": 8,
        "ANIMAL": 9,
        "DYNAMIC_OBJECT": 10,
    }
    for name, val in expected_classes.items():
        assert hasattr(SemanticClass, name)
        assert getattr(SemanticClass, name).value == val

    assert len(SemanticClass) == 11


def test_palette_and_dynamics_sets():
    """Ensures color palette covers all classes and static/dynamic sets partition them."""
    for s_class in SemanticClass:
        assert s_class in CLASS_COLOR_PALETTE
        color = CLASS_COLOR_PALETTE[s_class]
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    # Dynamic and static categories must not overlap
    assert len(DYNAMIC_CLASSES.intersection(STATIC_CLASSES)) == 0
    assert len(DYNAMIC_CLASSES.union(STATIC_CLASSES)) == len(SemanticClass)
    assert SemanticClass.ANIMAL in DYNAMIC_CLASSES
    assert SemanticClass.DYNAMIC_OBJECT in DYNAMIC_CLASSES


# ---------------------------------------------------------------------------
# Preprocessing and Postprocessing Tests
# ---------------------------------------------------------------------------

def test_preprocess_frame_shape_and_dtype():
    """Preprocessed frame should be channel-first float32 (3, H, W)."""
    bgr = np.zeros((200, 300, 3), dtype=np.uint8)
    tensor = _preprocess_frame(bgr, target_size=256)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == np.float32


def test_upsample_mask_preserves_discrete_classes():
    """Nearest-neighbor upsampling must never introduce fractional or non-existent class IDs."""
    mask_low = np.array([[1, 4], [8, 9]], dtype=np.uint8)
    conf_low = np.array([[0.8, 0.9], [0.85, 0.95]], dtype=np.float32)
    guide = np.zeros((100, 100, 3), dtype=np.uint8)

    mask_up, conf_up = _upsample_mask_and_confidence(mask_low, conf_low, guide, 100, 100)

    assert mask_up.shape == (100, 100)
    assert mask_up.dtype == np.uint8
    unique_classes = set(np.unique(mask_up))
    assert unique_classes.issubset({1, 4, 8, 9})
    assert conf_up.shape == (100, 100)
    assert np.all((conf_up >= 0.0) & (conf_up <= 1.0))


def test_compute_class_proportions():
    """Proportions must sum to 1.0 and cover all 11 classes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = int(SemanticClass.ROAD)      # 50%
    mask[50:, :] = int(SemanticClass.BUILDING)  # 50%

    props = _compute_class_proportions(mask)
    assert len(props) == 11
    assert pytest.approx(props["ROAD"], 0.001) == 0.5
    assert pytest.approx(props["BUILDING"], 0.001) == 0.5
    assert pytest.approx(props["UNKNOWN"], 0.001) == 0.0
    assert pytest.approx(sum(props.values()), 0.001) == 1.0


# ---------------------------------------------------------------------------
# Auxiliary Detection & Enhanced Fallback Logic Tests
# ---------------------------------------------------------------------------

def test_auxiliary_detection_confident_animal():
    """A confidently identified animal species must be labeled as ANIMAL."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    conf = np.full((100, 100), 0.8, dtype=np.float32)

    detection = AuxiliaryDetection(
        bbox_xyxy=(10, 10, 30, 30),
        category="animal",
        species="cattle",
        species_confidence=0.92,
        detection_confidence=0.95,
    )

    out_mask, out_conf, n_animals, n_dyn = apply_auxiliary_detections(
        mask, conf, [detection], animal_threshold=0.70
    )

    assert n_animals == 1
    assert n_dyn == 0
    assert np.all(out_mask[10:30, 10:30] == int(SemanticClass.ANIMAL))
    assert pytest.approx(float(out_conf[15, 15]), 0.01) == 0.92


def test_auxiliary_detection_ambiguous_animal_fallback():
    """
    When animal species classification confidence is below threshold,
    the system must NEVER guess a species — it must fall back to DYNAMIC_OBJECT.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    conf = np.full((100, 100), 0.8, dtype=np.float32)

    # Ambiguous animal detection: confidence 0.55 < threshold 0.70
    detection = AuxiliaryDetection(
        bbox_xyxy=(20, 20, 50, 50),
        category="animal",
        species="unknown_quadruped",
        species_confidence=0.55,
        detection_confidence=0.88,
    )

    out_mask, out_conf, n_animals, n_dyn = apply_auxiliary_detections(
        mask, conf, [detection], animal_threshold=0.70
    )

    assert n_animals == 0
    assert n_dyn == 1
    # Fallback to DYNAMIC_OBJECT
    assert np.all(out_mask[20:50, 20:50] == int(SemanticClass.DYNAMIC_OBJECT))
    assert pytest.approx(float(out_conf[25, 25]), 0.01) == 0.88


def test_auxiliary_detection_generic_motion_fallback():
    """Unclassified moving regions must fall back to DYNAMIC_OBJECT."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    conf = np.full((100, 100), 0.8, dtype=np.float32)

    detection = AuxiliaryDetection(
        bbox_xyxy=(60, 60, 80, 80),
        category="moving_region",
        detection_confidence=0.85,
    )

    out_mask, out_conf, n_animals, n_dyn = apply_auxiliary_detections(
        mask, conf, [detection]
    )

    assert n_animals == 0
    assert n_dyn == 1
    assert np.all(out_mask[60:80, 60:80] == int(SemanticClass.DYNAMIC_OBJECT))


# ---------------------------------------------------------------------------
# Semantic Classifier Pipeline Tests
# ---------------------------------------------------------------------------

def test_classify_frame_simulation(classifier, synthetic_aerial_frames):
    """Running classify_frame produces a complete SemanticMap."""
    frame = synthetic_aerial_frames[0]
    sem_map = classifier.classify_frame(frame, frame_index=42)

    assert isinstance(sem_map, SemanticMap)
    assert sem_map.frame_index == 42
    assert sem_map.class_mask.shape == (480, 640)
    assert sem_map.class_mask.dtype == np.uint8
    assert sem_map.confidence_map.shape == (480, 640)
    assert sem_map.confidence_map.dtype == np.float32
    assert sem_map.original_height == 480
    assert sem_map.original_width == 640
    assert len(sem_map.detected_classes) > 0
    assert sem_map.inference_time_ms > 0.0

    # Colorized mask check
    color_mask = sem_map.colorized_mask
    assert color_mask.shape == (480, 640, 3)
    assert color_mask.dtype == np.uint8


def test_classify_from_arrays_batched(classifier, synthetic_aerial_frames):
    """Batched classification over 3 frames generates a SemanticSegmentationResult."""
    aux_dets = [
        [
            AuxiliaryDetection(
                bbox_xyxy=(50, 50, 100, 100),
                category="animal",
                species="horse",
                species_confidence=0.88,
            )
        ],
        [
            AuxiliaryDetection(
                bbox_xyxy=(200, 200, 260, 260),
                category="moving_region",
            )
        ],
        None,
    ]

    result = classifier.classify_from_arrays(
        bgr_frames=synthetic_aerial_frames,
        frame_indices=[100, 101, 102],
        aux_detections_per_frame=aux_dets,
    )

    assert isinstance(result, SemanticSegmentationResult)
    assert result.total_keyframes == 3
    assert len(result.semantic_maps) == 3
    assert len(result.failed_frame_indices) == 0
    assert result.total_animals_detected == 1
    assert result.total_dynamic_objects_detected == 1
    assert result.mean_inference_time_ms > 0.0
    assert sum(result.overall_class_distribution.values()) == pytest.approx(1.0, rel=1e-2)


def test_classify_from_files_with_missing(classifier, synthetic_aerial_frames):
    """File-based classification handles disk I/O and tracks missing files in failed_frame_indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1 = tmp_path / "frame_001.png"
        p2 = tmp_path / "frame_002.png"
        p_missing = tmp_path / "frame_missing.png"

        cv2.imwrite(str(p1), synthetic_aerial_frames[0])
        cv2.imwrite(str(p2), synthetic_aerial_frames[1])

        paths = [p1, p_missing, p2]
        indices = [1, 2, 3]

        result = classifier.classify_from_files(paths, indices)

        assert result.total_keyframes == 3
        assert len(result.semantic_maps) == 2
        assert result.failed_frame_indices == [2]

"""
Unit tests for Semantic Mask Association & Mask Pyramid Builder — TASK-038.

Covers:
  - Combining SemanticMap and DynamicMask into unified 3-channel tensor.
  - Channel 0 (class ID), Channel 1 (dynamic flag 0/255), Channel 2 (confidence 0..255).
  - static_class_mask property masking dynamic regions.
  - 3-level pyramid creation (1.0x, 0.5x, 0.25x).
  - Nearest-neighbor downsampling of class IDs preventing invalid interpolated classes.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from workers.segmentation.mask_combiner import (
    MaskPyramid,
    MultiChannelSemanticMask,
    SemanticMaskCombiner,
)
from workers.segmentation.mask_generator import DynamicMask
from workers.segmentation.semantic_classifier import (
    SemanticClass,
    SemanticMap,
    _compute_class_proportions,
)


@pytest.fixture
def combiner():
    return SemanticMaskCombiner()


@pytest.fixture
def sample_semantic_and_dynamic():
    h, w = 120, 160
    # Semantic map with terrain and building
    class_mask = np.full((h, w), int(SemanticClass.TERRAIN), dtype=np.uint8)
    class_mask[10:50, 10:50] = int(SemanticClass.BUILDING)
    class_mask[60:100, 60:100] = int(SemanticClass.VEHICLE)

    conf_map = np.full((h, w), 0.85, dtype=np.float32)
    conf_map[10:50, 10:50] = 0.95

    sem_map = SemanticMap(
        frame_index=15,
        class_mask=class_mask,
        confidence_map=conf_map,
        original_height=h,
        original_width=w,
        class_proportions=_compute_class_proportions(class_mask),
        detected_classes=[SemanticClass.TERRAIN, SemanticClass.BUILDING, SemanticClass.VEHICLE],
        animal_count=0,
        dynamic_object_count=0,
        inference_time_ms=5.0,
    )

    # Dynamic mask: vehicle at (60..100, 60..100) is moving
    dyn_arr = np.zeros((h, w), dtype=np.uint8)
    dyn_arr[58:102, 58:102] = 255  # dilated vehicle

    dyn_mask = DynamicMask(
        frame_index=15,
        mask=dyn_arr,
        raw_dynamic_pixels=1600,
        dilated_dynamic_pixels=int(np.count_nonzero(dyn_arr)),
        dynamic_fraction=float(np.count_nonzero(dyn_arr) / (h * w)),
        moving_objects_count=1,
        dilation_margin_px=5,
    )

    return sem_map, dyn_mask


def test_combine_channels(combiner, sample_semantic_and_dynamic):
    """3-channel tensor correctly encodes class, dynamic flag, and quantized confidence."""
    sem_map, dyn_mask = sample_semantic_and_dynamic
    combined = combiner.combine(sem_map, dyn_mask)

    assert isinstance(combined, MultiChannelSemanticMask)
    assert combined.frame_index == 15
    assert combined.tensor.shape == (120, 160, 3)
    assert combined.tensor.dtype == np.uint8

    # Channel 0: class ID
    assert np.all(combined.class_mask == sem_map.class_mask)
    # Channel 1: dynamic flag
    assert np.all(combined.dynamic_mask == dyn_mask.mask)
    # Channel 2: confidence
    assert np.all(combined.confidence_map >= 0.0)
    assert np.all(combined.confidence_map <= 1.0)
    # Building confidence was 0.95 -> in uint8 is ~242
    assert combined.tensor[20, 20, 2] == int(0.95 * 255)

    # Static class mask replaces dynamic pixels with UNKNOWN (0)
    static_mask = combined.static_class_mask
    assert np.all(static_mask[60:100, 60:100] == int(SemanticClass.UNKNOWN))
    # Static building remains BUILDING (1)
    assert np.all(static_mask[20:40, 20:40] == int(SemanticClass.BUILDING))


def test_build_pyramid(combiner, sample_semantic_and_dynamic):
    """Pyramid produces 3 levels with exact scale factors and intact discrete classes."""
    sem_map, dyn_mask = sample_semantic_and_dynamic
    combined = combiner.combine(sem_map, dyn_mask)
    pyramid = combiner.build_pyramid(combined)

    assert isinstance(pyramid, MaskPyramid)
    assert pyramid.frame_index == 15

    # Level 0 (1.0x)
    assert pyramid.level_0.shape == (120, 160, 3)
    # Level 1 (0.5x)
    assert pyramid.level_1.shape == (60, 80, 3)
    # Level 2 (0.25x)
    assert pyramid.level_2.shape == (30, 40, 3)

    # In level 1 and 2, Channel 0 must contain only valid SemanticClass integers
    valid_classes = set(int(c) for c in SemanticClass)
    assert set(np.unique(pyramid.level_1[:, :, 0])).issubset(valid_classes)
    assert set(np.unique(pyramid.level_2[:, :, 0])).issubset(valid_classes)

    # Channel 1 must contain only 0 or 255
    assert set(np.unique(pyramid.level_1[:, :, 1])).issubset({0, 255})
    assert set(np.unique(pyramid.level_2[:, :, 1])).issubset({0, 255})

"""
Unit tests for Dynamic Object Mask Generation & Dilation — TASK-037.

Covers:
  - Dynamic mask generation from moving objects (MotionState.DYNAMIC).
  - Morphological dilation margin creation (5x5 default).
  - Stationary objects (MotionState.STATIC) are excluded from dynamic mask (remain 0).
  - Dynamic fraction calculation.
  - Integration with FrameTrackingResult and SemanticMap.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from workers.segmentation.dynamic_tracker import (
    FrameTrackingResult,
    MotionState,
    TrackedObject,
)
from workers.segmentation.mask_generator import (
    DynamicMask,
    DynamicMaskGenerator,
)
from workers.segmentation.semantic_classifier import (
    SemanticClass,
    SemanticMap,
    _compute_class_proportions,
)


@pytest.fixture
def mask_generator():
    return DynamicMaskGenerator(dilation_kernel_size=5)


def make_test_semantic_map(h: int = 100, w: int = 100) -> SemanticMap:
    mask = np.full((h, w), int(SemanticClass.TERRAIN), dtype=np.uint8)
    conf = np.full((h, w), 0.9, dtype=np.float32)
    return SemanticMap(
        frame_index=0,
        class_mask=mask,
        confidence_map=conf,
        original_height=h,
        original_width=w,
        class_proportions=_compute_class_proportions(mask),
        detected_classes=[SemanticClass.TERRAIN],
        animal_count=0,
        dynamic_object_count=0,
        inference_time_ms=1.0,
    )


def test_dynamic_mask_moving_object_dilated(mask_generator):
    """Moving object gets marked with 255 and expands by dilation margin."""
    moving_obj = TrackedObject(
        track_id=1,
        semantic_class=SemanticClass.VEHICLE,
        bbox_xyxy=(30, 30, 50, 50),
        centroid=(40.0, 40.0),
        motion_vector=(4.0, 0.0),
        ego_motion_vector=(0.0, 0.0),
        divergence_magnitude=4.0,
        motion_state=MotionState.DYNAMIC,
        age_frames=3,
        hits=3,
        confidence=0.95,
    )

    dyn_mask = mask_generator.generate_mask(
        height=100,
        width=100,
        moving_objects=[moving_obj],
        frame_index=5,
    )

    assert isinstance(dyn_mask, DynamicMask)
    assert dyn_mask.frame_index == 5
    assert dyn_mask.mask.shape == (100, 100)
    assert dyn_mask.mask.dtype == np.uint8

    # Raw bbox area is 20x20 = 400
    assert dyn_mask.raw_dynamic_pixels == 400
    # Dilated pixels must be strictly greater than raw pixels due to 5x5 dilation
    assert dyn_mask.dilated_dynamic_pixels > 400
    assert dyn_mask.dynamic_fraction > 0.04
    assert dyn_mask.moving_objects_count == 1
    # Check that mask values are strictly binary: 0 or 255
    assert set(np.unique(dyn_mask.mask)).issubset({0, 255})
    # Centroid of moving object must be 255
    assert dyn_mask.mask[40, 40] == 255


def test_dynamic_mask_stationary_object_ignored(mask_generator):
    """Stationary objects (e.g. parked car or resting cow) produce completely clear mask (all zeros)."""
    static_obj = TrackedObject(
        track_id=2,
        semantic_class=SemanticClass.ANIMAL,
        bbox_xyxy=(20, 20, 40, 40),
        centroid=(30.0, 30.0),
        motion_vector=(0.5, 0.5),
        ego_motion_vector=(0.5, 0.5),
        divergence_magnitude=0.0,
        motion_state=MotionState.STATIC,
        age_frames=4,
        hits=4,
        confidence=0.92,
    )

    dyn_mask = mask_generator.generate_mask(
        height=100,
        width=100,
        moving_objects=[static_obj],
        frame_index=1,
    )

    assert dyn_mask.raw_dynamic_pixels == 0
    assert dyn_mask.dilated_dynamic_pixels == 0
    assert dyn_mask.dynamic_fraction == 0.0
    assert np.all(dyn_mask.mask == 0)


def test_generate_from_tracking_result(mask_generator):
    """Generates dynamic mask correctly using FrameTrackingResult and SemanticMap."""
    sem_map = make_test_semantic_map(h=80, w=80)
    # Set a patch of vehicle pixels in semantic map
    sem_map.class_mask[10:30, 10:30] = int(SemanticClass.VEHICLE)

    moving_car = TrackedObject(
        track_id=10,
        semantic_class=SemanticClass.VEHICLE,
        bbox_xyxy=(10, 10, 30, 30),
        centroid=(20.0, 20.0),
        motion_vector=(5.0, 0.0),
        ego_motion_vector=(0.0, 0.0),
        divergence_magnitude=5.0,
        motion_state=MotionState.DYNAMIC,
        age_frames=2,
        hits=2,
        confidence=0.9,
    )

    tracking_res = FrameTrackingResult(
        frame_index=7,
        tracked_objects=[moving_car],
        moving_objects=[moving_car],
        stationary_objects=[],
        dynamic_pixel_count=400,
        camera_ego_motion=(0.0, 0.0),
        processing_time_ms=2.0,
    )

    mask = mask_generator.generate_from_tracking_result(tracking_res, sem_map)
    assert mask.frame_index == 7
    assert mask.dilated_dynamic_pixels > 0
    assert mask.mask[20, 20] == 255

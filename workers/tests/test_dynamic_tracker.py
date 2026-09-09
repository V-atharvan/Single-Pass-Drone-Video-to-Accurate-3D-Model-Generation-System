"""
Unit tests for Dynamic Object Detection & Multi-Frame Motion Tracking — TASK-036 [ENHANCED].

Covers:
  - MotionState enum values and TrackedObject data contracts.
  - Camera ego-motion background estimation from static semantic masks.
  - Optical flow divergence calculation:
      - Stationary cattle / resting animal -> classified as STATIC (preserved in scene).
      - Moving animal running across field -> classified as DYNAMIC (excluded).
      - Parked car -> classified as STATIC.
      - Moving vehicle -> classified as DYNAMIC.
      - Unclassified moving region (DYNAMIC_OBJECT) -> classified as DYNAMIC.
  - Candidate object extraction from semantic masks.
  - IoU association and multi-frame track identity persistence.
  - Multi-frame sequence tracking execution.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from workers.segmentation.dynamic_tracker import (
    CameraEgoMotionEstimator,
    DynamicObjectTracker,
    FrameTrackingResult,
    MotionDivergenceAnalyzer,
    MotionState,
    MultiFrameTrackingResult,
    TrackedObject,
    _compute_iou,
)
from workers.segmentation.semantic_classifier import (
    SemanticClass,
    SemanticMap,
    _compute_class_proportions,
)


@pytest.fixture
def ego_estimator():
    return CameraEgoMotionEstimator(motion_threshold=1.5)


@pytest.fixture
def divergence_analyzer():
    return MotionDivergenceAnalyzer(motion_threshold=1.5)


@pytest.fixture
def tracker():
    return DynamicObjectTracker(motion_threshold=1.5, iou_threshold=0.3)


def make_test_semantic_map(
    h: int = 200,
    w: int = 300,
    objects: list | None = None,
) -> SemanticMap:
    """Creates a synthetic SemanticMap with terrain background and specified objects."""
    mask = np.full((h, w), int(SemanticClass.TERRAIN), dtype=np.uint8)
    conf = np.full((h, w), 0.90, dtype=np.float32)

    if objects:
        for s_class, (x1, y1, x2, y2) in objects:
            mask[y1:y2, x1:x2] = int(s_class)

    proportions = _compute_class_proportions(mask)
    detected = [SemanticClass(c) for c in np.unique(mask) if c in SemanticClass._value2member_map_]

    return SemanticMap(
        frame_index=0,
        class_mask=mask,
        confidence_map=conf,
        original_height=h,
        original_width=w,
        class_proportions=proportions,
        detected_classes=detected,
        animal_count=0,
        dynamic_object_count=0,
        inference_time_ms=1.0,
    )


# ---------------------------------------------------------------------------
# Ego-Motion and Divergence Tests
# ---------------------------------------------------------------------------

def test_iou_computation():
    """Validates IoU bounding box calculations."""
    boxA = (10, 10, 50, 50)  # area = 1600
    boxB = (10, 10, 50, 50)  # exact match
    boxC = (30, 30, 70, 70)  # overlap = 20x20 = 400
    boxD = (100, 100, 120, 120)  # no overlap

    assert pytest.approx(_compute_iou(boxA, boxB)) == 1.0
    assert pytest.approx(_compute_iou(boxA, boxC), 0.01) == 400 / (1600 + 1600 - 400)
    assert _compute_iou(boxA, boxD) == 0.0


def test_background_ego_motion_estimation(ego_estimator):
    """Estimates background camera motion across static pixels."""
    h, w = 100, 100
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = 3.5   # Camera pan x: +3.5 px
    flow[:, :, 1] = -1.0  # Camera pan y: -1.0 px

    # Static mask over 80% of the image
    static_mask = np.ones((h, w), dtype=bool)
    static_mask[40:60, 40:60] = False  # moving object area

    ego_dx, ego_dy = ego_estimator.estimate_background_ego_motion(flow, static_mask)
    assert pytest.approx(ego_dx, 0.01) == 3.5
    assert pytest.approx(ego_dy, 0.01) == -1.0


def test_motion_divergence_stationary_animal_preservation(divergence_analyzer):
    """
    CRITICAL ENHANCED TEST:
    A stationary animal (e.g. resting cattle) moves only with camera ego-motion.
    Its divergence is near zero, so it must be classified as STATIC and NOT excluded.
    """
    h, w = 100, 100
    flow = np.zeros((h, w, 2), dtype=np.float32)
    # Camera panning: (dx=2.0, dy=0.0) everywhere
    flow[:, :] = (2.0, 0.0)

    ego_dx, ego_dy = 2.0, 0.0
    cattle_bbox = (20, 20, 45, 45)

    motion_vec, divergence, motion_state = divergence_analyzer.analyze_object_motion(
        flow=flow,
        bbox_xyxy=cattle_bbox,
        ego_dx=ego_dx,
        ego_dy=ego_dy,
        semantic_class=SemanticClass.ANIMAL,
    )

    assert pytest.approx(divergence, 0.01) == 0.0
    assert motion_state == MotionState.STATIC


def test_motion_divergence_moving_animal_flagged_dynamic(divergence_analyzer):
    """
    CRITICAL ENHANCED TEST:
    A moving animal running across the field diverges from camera ego-motion.
    It must be classified as DYNAMIC.
    """
    h, w = 100, 100
    flow = np.zeros((h, w, 2), dtype=np.float32)
    # Camera panning: (dx=1.0, dy=0.0)
    flow[:, :] = (1.0, 0.0)

    # Animal running in opposite direction: (dx=-4.0, dy=1.0)
    flow[20:45, 20:45] = (-4.0, 1.0)

    ego_dx, ego_dy = 1.0, 0.0
    animal_bbox = (20, 20, 45, 45)

    motion_vec, divergence, motion_state = divergence_analyzer.analyze_object_motion(
        flow=flow,
        bbox_xyxy=animal_bbox,
        ego_dx=ego_dx,
        ego_dy=ego_dy,
        semantic_class=SemanticClass.ANIMAL,
    )

    # Divergence = sqrt((-4 - 1)^2 + (1 - 0)^2) = sqrt(26) ~= 5.1 px/frame >= 1.5
    assert divergence >= 1.5
    assert motion_state == MotionState.DYNAMIC


def test_motion_divergence_parked_vs_moving_vehicle(divergence_analyzer):
    """Parked vehicle remains STATIC; moving vehicle is flagged DYNAMIC."""
    h, w = 100, 100
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :] = (0.5, 0.5)  # slight drone drift

    # 1. Parked car: moves identically to drone drift
    parked_bbox = (10, 10, 30, 30)
    _, div_parked, state_parked = divergence_analyzer.analyze_object_motion(
        flow, parked_bbox, 0.5, 0.5, SemanticClass.VEHICLE
    )
    assert div_parked < 0.1
    assert state_parked == MotionState.STATIC

    # 2. Moving car: speeds down road at (dx=6.0, dy=0.5)
    moving_bbox = (50, 50, 70, 70)
    flow[50:70, 50:70] = (6.0, 0.5)
    _, div_moving, state_moving = divergence_analyzer.analyze_object_motion(
        flow, moving_bbox, 0.5, 0.5, SemanticClass.VEHICLE
    )
    assert div_moving >= 1.5
    assert state_moving == MotionState.DYNAMIC


def test_motion_divergence_dynamic_object_fallback(divergence_analyzer):
    """Unclassified moving pixels under DYNAMIC_OBJECT fallback are flagged DYNAMIC."""
    h, w = 100, 100
    flow = np.zeros((h, w, 2), dtype=np.float32)
    ego_dx, ego_dy = 0.0, 0.0

    # Moving unclassified patch
    bbox = (30, 30, 60, 60)
    flow[30:60, 30:60] = (3.0, 2.0)

    _, div, state = divergence_analyzer.analyze_object_motion(
        flow, bbox, ego_dx, ego_dy, SemanticClass.DYNAMIC_OBJECT
    )
    assert div >= 1.5
    assert state == MotionState.DYNAMIC


# ---------------------------------------------------------------------------
# Candidate Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_candidates_from_mask(tracker):
    """Extracts bounding boxes for candidate classes and filters small noise."""
    objects = [
        (SemanticClass.VEHICLE, (20, 20, 50, 50)),   # area = 900 >= 25 -> valid
        (SemanticClass.ANIMAL, (70, 70, 95, 95)),    # area = 625 >= 25 -> valid
        (SemanticClass.PERSON, (110, 110, 112, 112)),# area = 4 < 25 -> noise filtered
    ]
    sem_map = make_test_semantic_map(h=150, w=150, objects=objects)
    candidates = tracker.extract_candidates_from_mask(sem_map)

    assert len(candidates) == 2
    classes = [c[0] for c in candidates]
    assert SemanticClass.VEHICLE in classes
    assert SemanticClass.ANIMAL in classes
    assert SemanticClass.PERSON not in classes


# ---------------------------------------------------------------------------
# Multi-Frame Tracking Tests
# ---------------------------------------------------------------------------

def test_multi_frame_tracking_track_continuity(tracker):
    """Verifies that an object maintains the same track_id across consecutive frames."""
    # Frame 1
    f1_bgr = np.full((120, 120, 3), 100, dtype=np.uint8)
    sem1 = make_test_semantic_map(
        h=120, w=120, objects=[(SemanticClass.VEHICLE, (20, 20, 50, 50))]
    )
    res1 = tracker.process_frame(f1_bgr, sem1, frame_index=1)
    assert len(res1.tracked_objects) == 1
    t1_id = res1.tracked_objects[0].track_id

    # Frame 2: Vehicle moved slightly to (25, 20, 55, 50), good IoU
    f2_bgr = np.full((120, 120, 3), 100, dtype=np.uint8)
    sem2 = make_test_semantic_map(
        h=120, w=120, objects=[(SemanticClass.VEHICLE, (25, 20, 55, 50))]
    )
    # Flow override representing independent motion
    flow2 = np.zeros((120, 120, 2), dtype=np.float32)
    flow2[20:55, 20:55] = (5.0, 0.0)  # moving vehicle

    res2 = tracker.process_frame(f2_bgr, sem2, frame_index=2, flow_override=flow2)
    assert len(res2.tracked_objects) == 1
    t2 = res2.tracked_objects[0]

    # Track ID must be preserved
    assert t2.track_id == t1_id
    assert t2.hits == 2
    assert t2.age_frames == 2
    assert t2.motion_state == MotionState.DYNAMIC
    assert len(res2.moving_objects) == 1
    assert len(res2.stationary_objects) == 0


def test_track_sequence_summary(tracker):
    """Runs track_sequence over a 3-frame synthetic dataset and validates MultiFrameTrackingResult."""
    frames = [np.full((100, 100, 3), 80, dtype=np.uint8) for _ in range(3)]
    # Stationary cattle at (10, 10, 35, 35), moving vehicle at (50, 50, 75, 75)
    sem_maps = []
    flows = []

    for i in range(3):
        veh_x = 50 + i * 4
        sem = make_test_semantic_map(
            h=100,
            w=100,
            objects=[
                (SemanticClass.ANIMAL, (10, 10, 35, 35)),
                (SemanticClass.VEHICLE, (veh_x, 50, veh_x + 25, 75)),
            ],
        )
        sem_maps.append(sem)

        flow = np.zeros((100, 100, 2), dtype=np.float32)
        # Background ego-motion is (1.0, 0.0)
        flow[:, :] = (1.0, 0.0)
        # Animal is stationary relative to background -> flow is (1.0, 0.0)
        flow[10:35, 10:35] = (1.0, 0.0)
        # Vehicle is moving fast -> flow is (6.0, 0.0)
        flow[50:75, veh_x:veh_x+25] = (6.0, 0.0)
        flows.append(flow)

    seq_result = tracker.track_sequence(
        bgr_frames=frames,
        semantic_maps=sem_maps,
        frame_indices=[10, 11, 12],
        flows=flows,
    )

    assert isinstance(seq_result, MultiFrameTrackingResult)
    assert seq_result.total_frames == 3
    assert len(seq_result.frame_results) == 3
    # Moving vehicle must be dynamic
    assert seq_result.unique_dynamic_tracks >= 1
    # Stationary cattle must be static
    assert seq_result.unique_static_tracks >= 1
    assert "VEHICLE" in seq_result.class_dynamic_breakdown

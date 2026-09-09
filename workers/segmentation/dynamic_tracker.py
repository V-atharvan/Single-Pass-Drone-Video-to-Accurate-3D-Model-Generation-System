"""
Dynamic Object Detection & Multi-Frame Motion Tracking — TASK-036 [ENHANCED].

Identifies moving vehicles, people, animals, and general dynamic objects across
sequential keyframes to prevent phantom/ghost geometry in 3D reconstruction, while
preserving stationary animals (parked cattle) and parked vehicles as static scene.

Pipeline:
  1. Detect candidate objects from the semantic segmentation mask (VEHICLE, PERSON,
     ANIMAL, DYNAMIC_OBJECT) and bounding box regions.
  2. Estimate camera ego-motion between consecutive keyframes using static semantic
     regions (BUILDING, ROAD, TERRAIN, VEGETATION).
  3. Compute dense or feature optical flow and determine motion vector divergence:
     Delta_v = v_obj - v_ego.
  4. Classify each candidate as DYNAMIC or STATIC based on motion divergence.
  5. Track objects across sequential frames with persistent track IDs and motion
     history smoothing.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.segmentation.semantic_classifier import (
    DYNAMIC_CLASSES,
    STATIC_CLASSES,
    SemanticClass,
    SemanticMap,
)

logger = logging.getLogger("segmentation.dynamic_tracker")

# Default thresholds
_DEFAULT_MOTION_DIVERGENCE_THRESHOLD = 1.5  # pixels / frame
_DEFAULT_IOU_MATCH_THRESHOLD = 0.3
_DEFAULT_MAX_MISSED_FRAMES = 5
_MIN_OBJECT_AREA_PIXELS = 25


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

class MotionState(str, Enum):
    """Motion classification status of an object track."""

    STATIC = "STATIC"            # Stationary relative to ground (preserved in reconstruction)
    DYNAMIC = "DYNAMIC"          # Independent motion diverging from camera ego-motion (excluded)
    UNCERTAIN = "UNCERTAIN"      # Insufficient temporal support to establish motion state


@dataclass
class TrackedObject:
    """
    State of a tracked object at a given keyframe.

    Attributes
    ----------
    track_id:
        Persistent unique tracking identifier across frames.
    semantic_class:
        SemanticClass classification (VEHICLE, PERSON, ANIMAL, DYNAMIC_OBJECT).
    bbox_xyxy:
        Bounding box (x1, y1, x2, y2) in pixel coordinates.
    centroid:
        (cx, cy) sub-pixel center of mass.
    motion_vector:
        (dx, dy) measured pixel motion between frames.
    ego_motion_vector:
        (ego_dx, ego_dy) estimated camera background motion at object centroid.
    divergence_magnitude:
        Euclidean norm of (motion_vector - ego_motion_vector) in pixels/frame.
    motion_state:
        STATIC, DYNAMIC, or UNCERTAIN.
    age_frames:
        Total frames since track initialization.
    hits:
        Total frames this track was successfully detected and matched.
    confidence:
        Detection / classification confidence score [0.0, 1.0].
    """

    track_id: int
    semantic_class: SemanticClass
    bbox_xyxy: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    motion_vector: Tuple[float, float]
    ego_motion_vector: Tuple[float, float]
    divergence_magnitude: float
    motion_state: MotionState
    age_frames: int
    hits: int
    confidence: float


@dataclass
class FrameTrackingResult:
    """Tracking and motion classification output for a single keyframe."""

    frame_index: int
    tracked_objects: List[TrackedObject]
    moving_objects: List[TrackedObject]
    stationary_objects: List[TrackedObject]
    dynamic_pixel_count: int
    camera_ego_motion: Tuple[float, float]   # (mean_dx, mean_dy)
    processing_time_ms: float


@dataclass
class MultiFrameTrackingResult:
    """Aggregated multi-frame tracking sequence results."""

    total_frames: int
    frame_results: List[FrameTrackingResult]
    unique_dynamic_tracks: int
    unique_static_tracks: int
    class_dynamic_breakdown: Dict[str, int]  # count of dynamic objects by class name


# ---------------------------------------------------------------------------
# Ego-Motion & Optical Flow Estimation
# ---------------------------------------------------------------------------

class CameraEgoMotionEstimator:
    """
    Estimates camera ego-motion between consecutive frames using static background pixels.
    """

    def __init__(self, motion_threshold: float = _DEFAULT_MOTION_DIVERGENCE_THRESHOLD):
        self.motion_threshold = motion_threshold

    def compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> np.ndarray:
        """
        Computes dense optical flow field (H, W, 2) using Farneback's algorithm.
        flow[y, x] = (dx, dy).
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow

    def estimate_background_ego_motion(
        self,
        flow: np.ndarray,
        static_mask: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates median background ego-motion (dx, dy) across confirmed static scene pixels.
        """
        if static_mask is None or np.count_nonzero(static_mask) < 100:
            # Fallback to entire image median if static mask is insufficient
            dx = float(np.median(flow[:, :, 0]))
            dy = float(np.median(flow[:, :, 1]))
            return dx, dy

        static_flow_x = flow[:, :, 0][static_mask]
        static_flow_y = flow[:, :, 1][static_mask]

        ego_dx = float(np.median(static_flow_x))
        ego_dy = float(np.median(static_flow_y))
        return ego_dx, ego_dy


# ---------------------------------------------------------------------------
# Motion Divergence Analyzer
# ---------------------------------------------------------------------------

class MotionDivergenceAnalyzer:
    """
    Analyzes whether an object's motion vector diverges from camera ego-motion.
    """

    def __init__(self, motion_threshold: float = _DEFAULT_MOTION_DIVERGENCE_THRESHOLD):
        self.motion_threshold = motion_threshold

    def analyze_object_motion(
        self,
        flow: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        ego_dx: float,
        ego_dy: float,
        semantic_class: SemanticClass,
    ) -> Tuple[Tuple[float, float], float, MotionState]:
        """
        Computes the object's median flow vector, divergence from ego-motion,
        and determines whether the object is STATIC or DYNAMIC.

        Rules:
          - An animal (e.g. cattle, horse) is ONLY classified DYNAMIC if optical flow
            confirms divergence >= motion_threshold. Parked/resting cattle remain STATIC.
          - Vehicles are classified STATIC if parked, DYNAMIC if moving.
          - Fallback DYNAMIC_OBJECT with divergence >= motion_threshold is DYNAMIC.
        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = flow.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        obj_flow_x = flow[y1:y2, x1:x2, 0]
        obj_flow_y = flow[y1:y2, x1:x2, 1]

        if obj_flow_x.size == 0:
            return (0.0, 0.0), 0.0, MotionState.UNCERTAIN

        obj_dx = float(np.median(obj_flow_x))
        obj_dy = float(np.median(obj_flow_y))

        # Residual divergence vector
        diff_x = obj_dx - ego_dx
        diff_y = obj_dy - ego_dy
        divergence = float(math.sqrt(diff_x**2 + diff_y**2))

        if divergence >= self.motion_threshold:
            motion_state = MotionState.DYNAMIC
        else:
            motion_state = MotionState.STATIC

        return (obj_dx, obj_dy), divergence, motion_state


# ---------------------------------------------------------------------------
# Multi-Frame Dynamic Object Tracker
# ---------------------------------------------------------------------------

@dataclass
class _TrackInternal:
    """Internal tracker state maintained across frames."""

    track_id: int
    semantic_class: SemanticClass
    bbox_xyxy: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    motion_vector: Tuple[float, float]
    ego_motion_vector: Tuple[float, float]
    divergence_magnitude: float
    motion_state: MotionState
    age_frames: int = 1
    hits: int = 1
    time_since_update: int = 0
    confidence: float = 0.9
    divergence_history: List[float] = field(default_factory=list)


def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class DynamicObjectTracker:
    """
    Multi-frame dynamic object tracker and motion classifier.

    Extracts candidate dynamic entities from semantic masks, tracks them across frames,
    measures ego-motion compensated optical flow, and flags active moving entities.
    """

    def __init__(
        self,
        motion_threshold: float = _DEFAULT_MOTION_DIVERGENCE_THRESHOLD,
        iou_threshold: float = _DEFAULT_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = _DEFAULT_MAX_MISSED_FRAMES,
    ):
        self.motion_threshold = motion_threshold
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

        self.ego_estimator = CameraEgoMotionEstimator(motion_threshold=motion_threshold)
        self.divergence_analyzer = MotionDivergenceAnalyzer(motion_threshold=motion_threshold)

        self._next_track_id: int = 1
        self._active_tracks: List[_TrackInternal] = []
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_frame_index: Optional[int] = None

    def reset(self) -> None:
        """Resets tracker state."""
        self._next_track_id = 1
        self._active_tracks.clear()
        self._prev_gray = None
        self._prev_frame_index = None

    def extract_candidates_from_mask(
        self,
        semantic_map: SemanticMap,
    ) -> List[Tuple[SemanticClass, Tuple[int, int, int, int], float]]:
        """
        Extracts candidate bounding boxes for dynamic-capable classes from SemanticMap.
        Returns list of (SemanticClass, (x1, y1, x2, y2), confidence).
        """
        mask = semantic_map.class_mask
        conf = semantic_map.confidence_map
        candidates = []

        # Target classes eligible for dynamic tracking
        eligible_classes = [
            SemanticClass.VEHICLE,
            SemanticClass.PERSON,
            SemanticClass.ANIMAL,
            SemanticClass.DYNAMIC_OBJECT,
        ]

        for s_class in eligible_classes:
            binary_mask = (mask == int(s_class)).astype(np.uint8)
            if np.count_nonzero(binary_mask) < _MIN_OBJECT_AREA_PIXELS:
                continue

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < _MIN_OBJECT_AREA_PIXELS:
                    continue

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                bbox = (x, y, x + w, y + h)

                comp_conf = float(np.mean(conf[labels == i])) if conf is not None else 0.90
                candidates.append((s_class, bbox, comp_conf))

        return candidates

    def process_frame(
        self,
        curr_bgr: np.ndarray,
        semantic_map: SemanticMap,
        frame_index: int,
        flow_override: Optional[np.ndarray] = None,
    ) -> FrameTrackingResult:
        """
        Processes a single keyframe against the previous keyframe state:
          1. Computes optical flow against previous keyframe.
          2. Computes background ego-motion using static semantic pixels.
          3. Evaluates motion divergence for each candidate object.
          4. Matches candidates with existing active tracks via IoU bipartite matching.
          5. Returns FrameTrackingResult.
        """
        t_start = time.perf_counter()
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Optical flow calculation
        if flow_override is not None:
            flow = flow_override
        elif self._prev_gray is not None:
            flow = self.ego_estimator.compute_optical_flow(self._prev_gray, curr_gray)
        else:
            # First frame: no motion available yet
            h, w = curr_gray.shape
            flow = np.zeros((h, w, 2), dtype=np.float32)

        # 2. Camera background ego-motion estimation
        static_mask = np.isin(
            semantic_map.class_mask,
            [int(c) for c in STATIC_CLASSES],
        )
        ego_dx, ego_dy = self.ego_estimator.estimate_background_ego_motion(flow, static_mask)

        # 3. Candidate detection extraction
        candidates = self.extract_candidates_from_mask(semantic_map)

        # 4. Associate candidates with existing tracks via IoU
        matched_track_indices: Set[int] = set()
        matched_candidate_indices: Set[int] = set()

        # Build candidate detections data
        detections_data = []
        for s_class, bbox, score in candidates:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            motion_vec, divergence, motion_state = self.divergence_analyzer.analyze_object_motion(
                flow, bbox, ego_dx, ego_dy, s_class
            )
            # If first frame, initial state is UNCERTAIN unless divergence is strictly measured
            if self._prev_gray is None and flow_override is None:
                motion_state = MotionState.UNCERTAIN

            detections_data.append({
                "class": s_class,
                "bbox": bbox,
                "centroid": (cx, cy),
                "motion_vec": motion_vec,
                "divergence": divergence,
                "motion_state": motion_state,
                "score": score,
            })

        # Bipartite matching by IoU
        for t_idx, track in enumerate(self._active_tracks):
            best_iou = 0.0
            best_c_idx = -1
            for c_idx, det in enumerate(detections_data):
                if c_idx in matched_candidate_indices:
                    continue
                iou = _compute_iou(track.bbox_xyxy, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_c_idx = c_idx

            if best_iou >= self.iou_threshold and best_c_idx >= 0:
                det = detections_data[best_c_idx]
                matched_track_indices.add(t_idx)
                matched_candidate_indices.add(best_c_idx)

                # Update existing track
                track.bbox_xyxy = det["bbox"]
                track.centroid = det["centroid"]
                track.motion_vector = det["motion_vec"]
                track.ego_motion_vector = (ego_dx, ego_dy)
                track.divergence_magnitude = det["divergence"]
                track.semantic_class = det["class"]
                track.confidence = det["score"]
                track.hits += 1
                track.age_frames += 1
                track.time_since_update = 0
                track.divergence_history.append(det["divergence"])

                # Smooth motion classification over history: if majority of recent observations
                # indicate dynamic divergence, object is DYNAMIC.
                recent = track.divergence_history[-3:]
                mean_recent_div = float(np.mean(recent))
                if mean_recent_div >= self.motion_threshold:
                    track.motion_state = MotionState.DYNAMIC
                else:
                    track.motion_state = MotionState.STATIC

        # Age unmatched existing tracks and remove stale tracks
        surviving_tracks: List[_TrackInternal] = []
        for t_idx, track in enumerate(self._active_tracks):
            if t_idx not in matched_track_indices:
                track.time_since_update += 1
                track.age_frames += 1

            if track.time_since_update <= self.max_missed_frames:
                surviving_tracks.append(track)
        self._active_tracks = surviving_tracks

        # Create new tracks for unmatched candidates
        for c_idx, det in enumerate(detections_data):
            if c_idx not in matched_candidate_indices:
                new_track = _TrackInternal(
                    track_id=self._next_track_id,
                    semantic_class=det["class"],
                    bbox_xyxy=det["bbox"],
                    centroid=det["centroid"],
                    motion_vector=det["motion_vec"],
                    ego_motion_vector=(ego_dx, ego_dy),
                    divergence_magnitude=det["divergence"],
                    motion_state=det["motion_state"],
                    time_since_update=0,
                    confidence=det["score"],
                    divergence_history=[det["divergence"]],
                )
                self._next_track_id += 1
                self._active_tracks.append(new_track)

        # Prepare exportable TrackedObject instances
        tracked_objects: List[TrackedObject] = []
        moving_objects: List[TrackedObject] = []
        stationary_objects: List[TrackedObject] = []
        dynamic_pixel_count = 0

        for t in self._active_tracks:
            # Only report tracks active in current frame
            if t.time_since_update == 0:
                obj = TrackedObject(
                    track_id=t.track_id,
                    semantic_class=t.semantic_class,
                    bbox_xyxy=t.bbox_xyxy,
                    centroid=t.centroid,
                    motion_vector=t.motion_vector,
                    ego_motion_vector=t.ego_motion_vector,
                    divergence_magnitude=t.divergence_magnitude,
                    motion_state=t.motion_state,
                    age_frames=t.age_frames,
                    hits=t.hits,
                    confidence=t.confidence,
                )
                tracked_objects.append(obj)
                if obj.motion_state == MotionState.DYNAMIC:
                    moving_objects.append(obj)
                    w_box = obj.bbox_xyxy[2] - obj.bbox_xyxy[0]
                    h_box = obj.bbox_xyxy[3] - obj.bbox_xyxy[1]
                    dynamic_pixel_count += w_box * h_box
                elif obj.motion_state == MotionState.STATIC:
                    stationary_objects.append(obj)

        self._prev_gray = curr_gray
        self._prev_frame_index = frame_index
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return FrameTrackingResult(
            frame_index=frame_index,
            tracked_objects=tracked_objects,
            moving_objects=moving_objects,
            stationary_objects=stationary_objects,
            dynamic_pixel_count=dynamic_pixel_count,
            camera_ego_motion=(ego_dx, ego_dy),
            processing_time_ms=t_elapsed_ms,
        )

    def track_sequence(
        self,
        bgr_frames: Sequence[np.ndarray],
        semantic_maps: Sequence[SemanticMap],
        frame_indices: Optional[Sequence[int]] = None,
        flows: Optional[Sequence[np.ndarray]] = None,
    ) -> MultiFrameTrackingResult:
        """Runs multi-frame tracking over an entire keyframe sequence."""
        if frame_indices is None:
            frame_indices = list(range(len(bgr_frames)))

        self.reset()
        frame_results: List[FrameTrackingResult] = []
        all_dynamic_ids: Set[int] = set()
        all_static_ids: Set[int] = set()
        class_dynamic_counts: Dict[str, int] = {c.name: 0 for c in DYNAMIC_CLASSES}

        for i, (frame, sem_map, idx) in enumerate(zip(bgr_frames, semantic_maps, frame_indices)):
            flow_ov = flows[i] if (flows and i < len(flows)) else None
            res = self.process_frame(frame, sem_map, idx, flow_override=flow_ov)
            frame_results.append(res)

            for m_obj in res.moving_objects:
                if m_obj.track_id not in all_dynamic_ids:
                    all_dynamic_ids.add(m_obj.track_id)
                    class_dynamic_counts[m_obj.semantic_class.name] = (
                        class_dynamic_counts.get(m_obj.semantic_class.name, 0) + 1
                    )

            for s_obj in res.stationary_objects:
                all_static_ids.add(s_obj.track_id)

        return MultiFrameTrackingResult(
            total_frames=len(bgr_frames),
            frame_results=frame_results,
            unique_dynamic_tracks=len(all_dynamic_ids),
            unique_static_tracks=len(all_static_ids - all_dynamic_ids),
            class_dynamic_breakdown=class_dynamic_counts,
        )

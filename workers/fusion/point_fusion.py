"""
Multi-Frame Point Cloud Fusion with Confidence Weighting — TASK-041.

Fuses individual frame point clouds into a consolidated global point cloud:
  - Spatial voxel hashing: groups duplicate 3D observations within spatial cells.
  - Confidence-weighted 3D position averaging:
      P_fused = sum(w_i * P_i) / sum(w_i)  where w_i = confidence_i
  - Multi-view visibility tracking: counts distinct keyframes observing each voxel.
  - Attribute fusion: confidence-weighted RGB color, dominant semantic class voting.
  - Multi-view visibility verification: single-view points with confidence < 0.60
    are pruned to eliminate monocular depth noise.
"""
from __future__ import annotations

import collections
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.fusion.unprojector import FramePointCloud

logger = logging.getLogger("fusion.point_fusion")

_DEFAULT_VOXEL_SIZE: float = 0.05  # 5 cm spatial voxel grid
_DEFAULT_SINGLE_VIEW_CONF_THRESHOLD: float = 0.60


@dataclass
class FusedPointCloud:
    """
    Fused, consolidated 3D point cloud.

    Attributes
    ----------
    positions:
        (M, 3) float32 coordinates in world frame (meters).
    colors:
        (M, 3) uint8 RGB colors.
    confidences:
        (M,) float32 fused confidence scores [0.0, 1.0].
    semantic_classes:
        (M,) uint8 consensus semantic class IDs.
    view_counts:
        (M,) int32 number of independent keyframes observing this point.
    voxel_size:
        Spatial cell resolution used during fusion in meters.
    total_input_points:
        Sum of points across all input keyframes before deduplication.
    fused_points_count:
        Number of unique points in the fused cloud.
    """

    positions: np.ndarray          # (M, 3) float32
    colors: np.ndarray             # (M, 3) uint8 RGB
    confidences: np.ndarray        # (M,) float32
    semantic_classes: np.ndarray   # (M,) uint8
    view_counts: np.ndarray        # (M,) int32
    voxel_size: float
    total_input_points: int
    fused_points_count: int

    @property
    def compression_ratio(self) -> float:
        if self.total_input_points == 0:
            return 1.0
        return float(self.fused_points_count / self.total_input_points)


class PointFusionEngine:
    """
    Consolidates multi-view 3D point observations into a unified point cloud.
    """

    def __init__(
        self,
        voxel_size: float = _DEFAULT_VOXEL_SIZE,
        single_view_confidence_threshold: float = _DEFAULT_SINGLE_VIEW_CONF_THRESHOLD,
    ):
        self.voxel_size = max(0.001, float(voxel_size))
        self.single_view_confidence_threshold = single_view_confidence_threshold

    def fuse_frames(
        self,
        frame_clouds: Sequence[FramePointCloud],
    ) -> FusedPointCloud:
        """
        Merges points across keyframes using spatial voxel hashing.
        """
        total_input_points = sum(fc.point_count for fc in frame_clouds)
        if total_input_points == 0:
            return FusedPointCloud(
                positions=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                confidences=np.zeros((0,), dtype=np.float32),
                semantic_classes=np.zeros((0,), dtype=np.uint8),
                view_counts=np.zeros((0,), dtype=np.int32),
                voxel_size=self.voxel_size,
                total_input_points=0,
                fused_points_count=0,
            )

        # Spatial voxel hash map:
        # voxel_coord (ix, iy, iz) -> list of (position, rgb, conf, sem_class, frame_index)
        voxels: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

        inv_v = 1.0 / self.voxel_size

        for fc in frame_clouds:
            if fc.point_count == 0:
                continue

            # Quantize positions into integer voxel grid coordinates
            voxel_coords = np.floor(fc.positions * inv_v).astype(np.int64)

            pos_arr = fc.positions
            col_arr = fc.colors
            conf_arr = fc.confidences
            sem_arr = fc.semantic_classes
            f_idx = fc.frame_index

            for i in range(fc.point_count):
                coord = (int(voxel_coords[i, 0]), int(voxel_coords[i, 1]), int(voxel_coords[i, 2]))
                w = max(0.01, float(conf_arr[i]))

                if coord not in voxels:
                    voxels[coord] = {
                        "weighted_pos": pos_arr[i] * w,
                        "weighted_col": col_arr[i].astype(np.float64) * w,
                        "sum_weights": w,
                        "max_conf": conf_arr[i],
                        "classes": {int(sem_arr[i]): w},
                        "frames": {f_idx},
                    }
                else:
                    v_data = voxels[coord]
                    v_data["weighted_pos"] += pos_arr[i] * w
                    v_data["weighted_col"] += col_arr[i].astype(np.float64) * w
                    v_data["sum_weights"] += w
                    if conf_arr[i] > v_data["max_conf"]:
                        v_data["max_conf"] = conf_arr[i]
                    cls = int(sem_arr[i])
                    v_data["classes"][cls] = v_data["classes"].get(cls, 0.0) + w
                    v_data["frames"].add(f_idx)

        # Aggregate fused points
        fused_positions: List[np.ndarray] = []
        fused_colors: List[np.ndarray] = []
        fused_confidences: List[float] = []
        fused_classes: List[int] = []
        fused_views: List[int] = []

        for coord, v_data in voxels.items():
            num_views = len(v_data["frames"])
            fused_conf = float(v_data["max_conf"])

            # Single-view filtering rule:
            # If observed by only 1 frame and confidence < threshold, discard!
            if num_views == 1 and fused_conf < self.single_view_confidence_threshold:
                continue

            sum_w = v_data["sum_weights"]
            avg_pos = v_data["weighted_pos"] / sum_w
            avg_col = np.clip(np.round(v_data["weighted_col"] / sum_w), 0, 255).astype(np.uint8)

            # Majority weighted semantic class
            best_class = max(v_data["classes"].items(), key=lambda item: item[1])[0]

            fused_positions.append(avg_pos)
            fused_colors.append(avg_col)
            fused_confidences.append(fused_conf)
            fused_classes.append(best_class)
            fused_views.append(num_views)

        if len(fused_positions) == 0:
            return FusedPointCloud(
                positions=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                confidences=np.zeros((0,), dtype=np.float32),
                semantic_classes=np.zeros((0,), dtype=np.uint8),
                view_counts=np.zeros((0,), dtype=np.int32),
                voxel_size=self.voxel_size,
                total_input_points=total_input_points,
                fused_points_count=0,
            )

        return FusedPointCloud(
            positions=np.ascontiguousarray(np.array(fused_positions, dtype=np.float32)),
            colors=np.ascontiguousarray(np.array(fused_colors, dtype=np.uint8)),
            confidences=np.ascontiguousarray(np.array(fused_confidences, dtype=np.float32)),
            semantic_classes=np.ascontiguousarray(np.array(fused_classes, dtype=np.uint8)),
            view_counts=np.ascontiguousarray(np.array(fused_views, dtype=np.int32)),
            voxel_size=self.voxel_size,
            total_input_points=total_input_points,
            fused_points_count=len(fused_positions),
        )

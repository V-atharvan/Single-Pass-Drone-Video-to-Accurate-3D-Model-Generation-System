"""
Dynamic Object & Statistical Outlier Removal — TASK-042.

Cleanses the fused 3D point cloud by:
  1. Stripping any remaining dynamic points or points inside dynamic bounding volumes.
  2. Statistical Outlier Removal (SOR): eliminates sparse noise points whose average distance
     to k-nearest neighbors exceeds (mean + std_ratio * std).
  3. Radius Outlier Removal (ROR): eliminates isolated flying points with fewer than
     min_neighbors within search radius R.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.fusion.point_fusion import FusedPointCloud
from workers.segmentation.semantic_classifier import DYNAMIC_CLASSES

logger = logging.getLogger("fusion.outlier_filter")

_DEFAULT_SOR_NEIGHBORS: int = 20
_DEFAULT_SOR_STD_RATIO: float = 1.5
_DEFAULT_ROR_RADIUS: float = 0.20
_DEFAULT_ROR_MIN_NEIGHBORS: int = 5


@dataclass
class FilterStatistics:
    """Quantitative summary of point cloud outlier filtering."""

    initial_points: int
    dynamic_points_removed: int
    sor_outliers_removed: int
    ror_outliers_removed: int
    final_points: int

    @property
    def total_outliers_removed(self) -> int:
        return self.initial_points - self.final_points

    @property
    def retention_rate(self) -> float:
        if self.initial_points == 0:
            return 1.0
        return float(self.final_points / self.initial_points)


@dataclass
class OutlierFilterResult:
    """Result containing cleansed point cloud and filtering metrics."""

    point_cloud: FusedPointCloud
    stats: FilterStatistics


class OutlierFilter:
    """
    Cleanses 3D point clouds using KD-Tree spatial neighborhood analysis.
    """

    def __init__(
        self,
        nb_neighbors: int = _DEFAULT_SOR_NEIGHBORS,
        std_ratio: float = _DEFAULT_SOR_STD_RATIO,
        radius: float = _DEFAULT_ROR_RADIUS,
        min_neighbors: int = _DEFAULT_ROR_MIN_NEIGHBORS,
        enable_sor: bool = True,
        enable_ror: bool = True,
    ):
        self.nb_neighbors = max(2, nb_neighbors)
        self.std_ratio = max(0.1, std_ratio)
        self.radius = max(0.01, radius)
        self.min_neighbors = max(1, min_neighbors)
        self.enable_sor = enable_sor
        self.enable_ror = enable_ror

    def filter_cloud(
        self,
        cloud: FusedPointCloud,
        dynamic_boxes_xyz: Optional[Sequence[Tuple[float, float, float, float, float, float]]] = None,
        filter_dynamic_classes: bool = False,
    ) -> OutlierFilterResult:
        """
        Cleanses point cloud by filtering dynamic points, SOR outliers, and ROR isolated points.
        """
        n_pts = cloud.fused_points_count
        if n_pts == 0:
            stats = FilterStatistics(0, 0, 0, 0, 0)
            return OutlierFilterResult(point_cloud=cloud, stats=stats)

        keep_mask = np.ones(n_pts, dtype=bool)
        pos = cloud.positions
        dyn_removed = 0

        # 1. Dynamic Bounding Box Filtering (x_min, y_min, z_min, x_max, y_max, z_max)
        if dynamic_boxes_xyz:
            for b in dynamic_boxes_xyz:
                in_box = (
                    (pos[:, 0] >= b[0]) & (pos[:, 0] <= b[3]) &
                    (pos[:, 1] >= b[1]) & (pos[:, 1] <= b[4]) &
                    (pos[:, 2] >= b[2]) & (pos[:, 2] <= b[5])
                )
                keep_mask[in_box] = False

        if filter_dynamic_classes:
            dynamic_vals = [int(c) for c in DYNAMIC_CLASSES]
            is_dyn_class = np.isin(cloud.semantic_classes, dynamic_vals)
            keep_mask[is_dyn_class] = False

        dyn_removed = int(n_pts - np.count_nonzero(keep_mask))

        # 2. Statistical Outlier Removal (SOR)
        sor_removed = 0
        active_indices = np.where(keep_mask)[0]
        if self.enable_sor and len(active_indices) > self.nb_neighbors:
            active_pos = pos[active_indices]
            tree = cKDTree(active_pos)
            # Query k nearest neighbors (including the point itself at index 0)
            k = min(self.nb_neighbors + 1, len(active_pos))
            distances, _ = tree.query(active_pos, k=k)

            # Mean distance to k nearest neighbors (excluding self at distance 0)
            mean_distances = np.mean(distances[:, 1:], axis=1)

            mu = float(np.mean(mean_distances))
            sigma = float(np.std(mean_distances))
            threshold = mu + self.std_ratio * sigma

            sor_inliers = mean_distances <= threshold
            sor_outliers = ~sor_inliers
            sor_removed = int(np.count_nonzero(sor_outliers))

            # Mark removed points in global keep_mask
            outlier_indices = active_indices[sor_outliers]
            keep_mask[outlier_indices] = False

        # 3. Radius Outlier Removal (ROR)
        ror_removed = 0
        active_indices = np.where(keep_mask)[0]
        if self.enable_ror and len(active_indices) > self.min_neighbors:
            active_pos = pos[active_indices]
            tree = cKDTree(active_pos)
            # Count neighbors within radius R (including self)
            neighbor_counts = tree.query_ball_point(active_pos, r=self.radius, return_sorted=False)
            ror_inliers = np.array([len(nbrs) >= (self.min_neighbors + 1) for nbrs in neighbor_counts], dtype=bool)
            ror_outliers = ~ror_inliers
            ror_removed = int(np.count_nonzero(ror_outliers))

            outlier_indices = active_indices[ror_outliers]
            keep_mask[outlier_indices] = False

        # Build cleansed cloud
        final_indices = np.where(keep_mask)[0]
        cleansed_cloud = FusedPointCloud(
            positions=pos[final_indices].copy(),
            colors=cloud.colors[final_indices].copy(),
            confidences=cloud.confidences[final_indices].copy(),
            semantic_classes=cloud.semantic_classes[final_indices].copy(),
            view_counts=cloud.view_counts[final_indices].copy(),
            voxel_size=cloud.voxel_size,
            total_input_points=cloud.total_input_points,
            fused_points_count=len(final_indices),
        )

        stats = FilterStatistics(
            initial_points=n_pts,
            dynamic_points_removed=dyn_removed,
            sor_outliers_removed=sor_removed,
            ror_outliers_removed=ror_removed,
            final_points=len(final_indices),
        )

        return OutlierFilterResult(point_cloud=cleansed_cloud, stats=stats)

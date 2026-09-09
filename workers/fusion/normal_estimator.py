"""
Voxel Downsampling and Normal Vector Estimation — TASK-043.

Resamples the point cloud to uniform spatial density and estimates accurate,
consistently oriented 3D surface normals:
  1. Quality Preset Voxel Downsampling (HIGH: 0.05m, BALANCED: 0.10m, LOW: 0.20m).
  2. Principal Component Analysis (PCA) on local k-nearest neighbor covariance (k=30).
  3. Surface normal vector extracted from eigenvector corresponding to minimum eigenvalue.
  4. Consistent camera-facing orientation: (p_cam - p_i) . n_i > 0.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.fusion.point_fusion import FusedPointCloud

logger = logging.getLogger("fusion.normal_estimator")

_QUALITY_PRESET_VOXEL_SIZES: Dict[str, float] = {
    "HIGH": 0.05,
    "BALANCED": 0.10,
    "LOW": 0.20,
}
_DEFAULT_K_NEIGHBORS: int = 30


@dataclass
class OrientedPointCloud:
    """
    Point cloud with surface normal vectors and local surface curvature.

    Attributes
    ----------
    positions:
        (N, 3) float32 coordinates.
    normals:
        (N, 3) float32 unit normal vectors.
    curvatures:
        (N,) float32 surface curvature values [0.0, 1.0].
    colors:
        (N, 3) uint8 RGB colors.
    confidences:
        (N,) float32 confidence scores.
    semantic_classes:
        (N,) uint8 semantic class IDs.
    view_counts:
        (N,) int32 keyframe visibility counts.
    """

    positions: np.ndarray          # (N, 3) float32
    normals: np.ndarray            # (N, 3) float32
    curvatures: np.ndarray         # (N,) float32
    colors: np.ndarray             # (N, 3) uint8
    confidences: np.ndarray        # (N,) float32
    semantic_classes: np.ndarray   # (N,) uint8
    view_counts: np.ndarray        # (N,) int32

    @property
    def point_count(self) -> int:
        return len(self.positions)


class NormalEstimator:
    """
    Estimates consistently oriented surface normals via PCA on local covariance matrices.
    """

    def __init__(self, k_neighbors: int = _DEFAULT_K_NEIGHBORS):
        self.k_neighbors = max(3, k_neighbors)

    def downsample_cloud(
        self,
        cloud: FusedPointCloud,
        voxel_size: float,
    ) -> FusedPointCloud:
        """Resamples point cloud to target voxel grid spacing."""
        if cloud.fused_points_count == 0 or voxel_size <= 0:
            return cloud

        inv_v = 1.0 / voxel_size
        voxel_coords = np.floor(cloud.positions * inv_v).astype(np.int64)

        # Unique voxels
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        unique_indices.sort()

        return FusedPointCloud(
            positions=cloud.positions[unique_indices].copy(),
            colors=cloud.colors[unique_indices].copy(),
            confidences=cloud.confidences[unique_indices].copy(),
            semantic_classes=cloud.semantic_classes[unique_indices].copy(),
            view_counts=cloud.view_counts[unique_indices].copy(),
            voxel_size=voxel_size,
            total_input_points=cloud.total_input_points,
            fused_points_count=len(unique_indices),
        )

    def estimate_normals(
        self,
        cloud: FusedPointCloud,
        camera_positions: Optional[np.ndarray] = None,
        preset: Optional[str] = None,
        voxel_size_override: Optional[float] = None,
    ) -> OrientedPointCloud:
        """
        Estimates oriented normals and surface curvatures.
        """
        target_voxel = voxel_size_override
        if target_voxel is None and preset in _QUALITY_PRESET_VOXEL_SIZES:
            target_voxel = _QUALITY_PRESET_VOXEL_SIZES[preset]

        processed_cloud = cloud
        if target_voxel is not None and target_voxel > cloud.voxel_size:
            processed_cloud = self.downsample_cloud(cloud, target_voxel)

        n_pts = processed_cloud.fused_points_count
        if n_pts == 0:
            return OrientedPointCloud(
                positions=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                curvatures=np.zeros((0,), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                confidences=np.zeros((0,), dtype=np.float32),
                semantic_classes=np.zeros((0,), dtype=np.uint8),
                view_counts=np.zeros((0,), dtype=np.int32),
            )

        pos = processed_cloud.positions
        k = min(self.k_neighbors, n_pts)

        tree = cKDTree(pos)
        _, neighbor_indices = tree.query(pos, k=k)

        normals = np.zeros((n_pts, 3), dtype=np.float32)
        curvatures = np.zeros(n_pts, dtype=np.float32)

        # Vectorized / loop covariance PCA
        for i in range(n_pts):
            nbr_pts = pos[neighbor_indices[i]]  # (k, 3)
            centroid = np.mean(nbr_pts, axis=0)
            centered = nbr_pts - centroid
            # 3x3 Covariance matrix
            cov = (centered.T @ centered) / k

            # Eigendecomposition (eigenvalues in ascending order)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Normal is eigenvector with smallest eigenvalue (eigenvectors[:, 0])
            normal = eigenvectors[:, 0]
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            normals[i] = normal

            # Curvature calculation: lambda_0 / sum(lambda)
            total_eval = np.sum(eigenvalues)
            if total_eval > 1e-6:
                curvatures[i] = float(eigenvalues[0] / total_eval)
            else:
                curvatures[i] = 0.0

        # Consistent orientation toward camera trajectory
        if camera_positions is not None and len(camera_positions) > 0:
            cam_tree = cKDTree(camera_positions)
            _, nearest_cam_indices = cam_tree.query(pos, k=1)
            view_vectors = camera_positions[nearest_cam_indices] - pos
            dots = np.sum(normals * view_vectors, axis=1)
            flip_mask = dots < 0.0
            normals[flip_mask] = -normals[flip_mask]
        else:
            # Default orientation: point towards +Z (upward)
            dots = normals[:, 2]
            flip_mask = dots < 0.0
            normals[flip_mask] = -normals[flip_mask]

        return OrientedPointCloud(
            positions=pos,
            normals=normals,
            curvatures=curvatures,
            colors=processed_cloud.colors,
            confidences=processed_cloud.confidences,
            semantic_classes=processed_cloud.semantic_classes,
            view_counts=processed_cloud.view_counts,
        )

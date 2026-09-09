"""
Depth Unprojection and Point Cloud Back-Projection — TASK-040.

Unprojects calibrated 2D metric depth maps into 3D camera coordinates and transforms
them into global world coordinates using 6-DoF camera extrinsics:

  Camera frame:
    X_c = (u - c_x) * d / f_x
    Y_c = (v - c_y) * d / f_y
    Z_c = d

  World frame:
    P_w = R_w * P_c + t_w

Associates each 3D point with:
  - RGB color from source keyframe.
  - Depth confidence score [0.0, 1.0].
  - Semantic class ID (0..10).
  - Source keyframe pixel coordinates (u, v).

Excludes pixels marked as dynamic in dynamic exclusion masks.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

logger = logging.getLogger("fusion.unprojector")


@dataclass
class CameraIntrinsics:
    """Camera pinhole projection model intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        """Returns 3x3 camera calibration matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class CameraExtrinsics:
    """
    6-DoF camera pose transformation matrix [R | t].
    By convention: P_world = R * P_camera + t.
    """

    R: np.ndarray  # (3, 3) rotation matrix
    t: np.ndarray  # (3,) translation vector

    @property
    def camera_center(self) -> np.ndarray:
        """World coordinates of the camera optical center."""
        return self.t.flatten()


@dataclass
class FramePointCloud:
    """
    Unprojected 3D point cloud for a single keyframe.

    Attributes
    ----------
    frame_index:
        Source keyframe index.
    positions:
        (N, 3) float32 coordinates in global world frame (meters).
    colors:
        (N, 3) uint8 RGB colors.
    confidences:
        (N,) float32 confidence values in [0.0, 1.0].
    semantic_classes:
        (N,) uint8 semantic class indices (0..10).
    pixel_coords:
        (N, 2) int32 (u, v) image coordinates.
    camera_center:
        (3,) float32 camera position in world coordinates.
    """

    frame_index: int
    positions: np.ndarray          # (N, 3) float32
    colors: np.ndarray             # (N, 3) uint8 RGB
    confidences: np.ndarray        # (N,) float32
    semantic_classes: np.ndarray   # (N,) uint8
    pixel_coords: np.ndarray       # (N, 2) int32 (u, v)
    camera_center: np.ndarray      # (3,) float32

    @property
    def point_count(self) -> int:
        return len(self.positions)


class DepthUnprojector:
    """
    Fast, vectorized back-projection engine transforming depth maps into 3D point sets.
    """

    def unproject_frame(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        bgr_image: Optional[np.ndarray] = None,
        confidence_map: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
        dynamic_mask: Optional[np.ndarray] = None,
        min_depth_m: float = 0.1,
        max_depth_m: float = 300.0,
        stride: int = 1,
        frame_index: int = 0,
    ) -> FramePointCloud:
        """
        Back-projects depth pixels into world coordinates.

        Parameters
        ----------
        depth_m:
            (H, W) float32 metric depth array in meters.
        intrinsics:
            Camera pinhole intrinsics.
        extrinsics:
            Camera 6-DoF world-from-camera transformation.
        bgr_image:
            Optional (H, W, 3) uint8 image for color extraction.
        confidence_map:
            Optional (H, W) float32 confidence map in [0.0, 1.0].
        semantic_mask:
            Optional (H, W) uint8 semantic class mask (0..10).
        dynamic_mask:
            Optional (H, W) uint8 dynamic mask (255 = exclude, 0 = keep).
        min_depth_m, max_depth_m:
            Valid metric depth bounds.
        stride:
            Pixel sampling step (stride=1 is full resolution).
        """
        h, w = depth_m.shape[:2]

        # 1. Determine valid mask
        valid = (depth_m >= min_depth_m) & (depth_m <= max_depth_m) & np.isfinite(depth_m)

        # Exclude dynamic object pixels
        if dynamic_mask is not None:
            valid = valid & (dynamic_mask != 255)

        # Apply stride sub-sampling if requested
        if stride > 1:
            stride_mask = np.zeros((h, w), dtype=bool)
            stride_mask[::stride, ::stride] = True
            valid = valid & stride_mask

        v_indices, u_indices = np.where(valid)
        if len(u_indices) == 0:
            # Empty point cloud
            return FramePointCloud(
                frame_index=frame_index,
                positions=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                confidences=np.zeros((0,), dtype=np.float32),
                semantic_classes=np.zeros((0,), dtype=np.uint8),
                pixel_coords=np.zeros((0, 2), dtype=np.int32),
                camera_center=extrinsics.camera_center.astype(np.float32),
            )

        # 2. Extract valid depths and pixel coordinates
        d = depth_m[v_indices, u_indices].astype(np.float32)
        u = u_indices.astype(np.float32)
        v = v_indices.astype(np.float32)

        # 3. Camera coordinates back-projection
        X_c = (u - intrinsics.cx) * d / intrinsics.fx
        Y_c = (v - intrinsics.cy) * d / intrinsics.fy
        Z_c = d

        # (N, 3) points in camera frame
        P_c = np.stack([X_c, Y_c, Z_c], axis=-1)

        # 4. Transform to world coordinates: P_w = P_c @ R.T + t
        R = extrinsics.R.astype(np.float32)
        t = extrinsics.t.flatten().astype(np.float32)
        P_w = (P_c @ R.T) + t

        # 5. Extract colors (RGB)
        if bgr_image is not None:
            bgr_sampled = bgr_image[v_indices, u_indices]
            # Convert BGR to RGB
            rgb_sampled = np.ascontiguousarray(bgr_sampled[:, ::-1], dtype=np.uint8)
        else:
            rgb_sampled = np.full((len(u_indices), 3), 200, dtype=np.uint8)

        # 6. Extract confidence
        if confidence_map is not None:
            conf_sampled = confidence_map[v_indices, u_indices].astype(np.float32)
            conf_sampled = np.clip(conf_sampled, 0.0, 1.0)
        else:
            conf_sampled = np.ones(len(u_indices), dtype=np.float32)

        # 7. Extract semantic classes
        if semantic_mask is not None:
            sem_sampled = semantic_mask[v_indices, u_indices].astype(np.uint8)
        else:
            sem_sampled = np.zeros(len(u_indices), dtype=np.uint8)

        # Pixel coords (u, v)
        pixel_coords = np.stack([u_indices, v_indices], axis=-1).astype(np.int32)

        return FramePointCloud(
            frame_index=frame_index,
            positions=P_w.astype(np.float32),
            colors=rgb_sampled,
            confidences=conf_sampled,
            semantic_classes=sem_sampled,
            pixel_coords=pixel_coords,
            camera_center=t.copy(),
        )

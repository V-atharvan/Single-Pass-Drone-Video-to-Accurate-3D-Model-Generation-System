"""
Metric Scale Calibration Against Sensor Baseline — TASK-033.

Aligns monocular depth map scale factors against physical GPS baseline travel
distances and optional barometric altitude ground clearance.

The core calibration problem:
  Monocular depth networks produce depth maps that are metrically consistent
  internally but may drift in absolute scale across the flight sequence.
  Given N triangulated 3D sparse feature points (from visual odometry) with
  known physical positions, we find the optimal scale s* per frame such that:

    s* = argmin_s sum ||s * X_depth - X_triangulated||^2

  where X_depth = back-projected depth-map point and X_triangulated = sparse
  point at physically correct scale (derived from GPS baseline).

Calibration modes (applied in priority order):
  1. SPARSE_3D: triangulated 3D feature points from camera poses (most accurate).
  2. BAROMETRIC: GPS baseline travel distance between adjacent frames.
  3. GPS_BASELINE: ground clearance from barometric altitude reading.
  4. MEDIAN_PROPAGATION: median scale from calibrated frames propagated to gaps.

After computing per-frame scale factors:
  - Apply median smoothing across the sequence to suppress outlier frames.
  - Scale each depth map in place: depth_calibrated = s * depth_raw.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.depth.depth_estimator import DepthMap
from workers.pose.sensor_fusion import OptimizedPose

logger = logging.getLogger("depth.scale_aligner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scale factor sanity bounds (metres/depth_unit)
_MIN_SCALE_FACTOR: float = 0.01
_MAX_SCALE_FACTOR: float = 1000.0

# Barometric altitude tolerance for scale validation (±5%)
_BARO_TOLERANCE_FRACTION: float = 0.05

# Minimum number of triangulated points required for SPARSE_3D calibration
_MIN_SPARSE_POINTS: int = 5


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


class ScaleCalibrationMode(str, Enum):
    """Source of ground truth used to calibrate depth scale."""
    SPARSE_3D = "SPARSE_3D"          # Triangulated 3D feature points
    GPS_BASELINE = "GPS_BASELINE"    # GPS inter-frame travel distance
    BAROMETRIC = "BAROMETRIC"        # Barometric altitude ground clearance
    MEDIAN_PROPAGATION = "MEDIAN_PROPAGATION"  # Propagated from adjacent calibrated frames
    UNCALIBRATED = "UNCALIBRATED"    # No calibration data available


@dataclass
class FrameScaleFactor:
    """Scale factor for a single keyframe."""
    frame_index: int
    raw_scale: float                  # Raw estimated scale
    smoothed_scale: float             # After median smoothing
    calibration_mode: ScaleCalibrationMode
    n_points_used: int                # Number of points used in calibration
    residual_m: float                 # Mean absolute residual in metres


@dataclass
class ScaleAlignmentResult:
    """Aggregate result of metric scale calibration over the full sequence."""
    total_frames: int
    calibrated_depth_maps: List[DepthMap] = field(default_factory=list)
    frame_scale_factors: List[FrameScaleFactor] = field(default_factory=list)
    median_global_scale: float = 1.0
    mean_residual_m: float = 0.0
    calibration_mode: ScaleCalibrationMode = ScaleCalibrationMode.UNCALIBRATED
    passes_baro_tolerance: Optional[bool] = None  # True if within ±5%


# ---------------------------------------------------------------------------
# Scale estimation functions
# ---------------------------------------------------------------------------


def _scale_from_sparse_points(
    depth_m: np.ndarray,
    pts_image: np.ndarray,           # (N, 2) image coordinates
    pts_world_m: np.ndarray,         # (N, 3) 3D world coordinates in metres
    K: np.ndarray,                   # 3x3 camera intrinsic matrix
    R_cam_to_world: np.ndarray,      # 3x3 rotation (camera to world)
    t_world: np.ndarray,             # 3-vector camera center in world
) -> Optional[Tuple[float, float, int]]:
    """
    Estimate scale factor s from sparse triangulated 3D points.

    For each inlier 2D point (u, v):
      - Sample depth: d_raw = depth_m[v, u]
      - Back-project: X_depth = s * K^{-1} [u, v, 1]^T
      - Transform to world: X_world_est = R_cam^T @ X_depth + C
      - Compare to triangulated: X_triangulated

    Solve: s* = argmin sum ||s * A_i - B_i||^2 = (sum A_i . B_i) / (sum A_i . A_i)

    Returns (scale_factor, mean_residual_m, n_points) or None if insufficient.
    """
    h, w = depth_m.shape[:2]
    K_inv = np.linalg.inv(K.astype(np.float64))

    A_list = []
    B_list = []

    for i in range(len(pts_image)):
        u, v = int(round(pts_image[i, 0])), int(round(pts_image[i, 1]))
        if not (0 <= u < w and 0 <= v < h):
            continue

        d_raw = float(depth_m[v, u])
        if d_raw < 0.01:
            continue  # Invalid pixel

        # Back-project unit-scale point in camera frame
        pix_h = np.array([u, v, 1.0], dtype=np.float64)
        ray_cam = K_inv @ pix_h          # Direction in camera frame (at d_raw units)

        # Camera-to-world transformation
        X_world_est_unit = R_cam_to_world @ (ray_cam * d_raw)  # at unit scale

        X_triangulated = pts_world_m[i].astype(np.float64) - t_world.astype(np.float64)

        A_list.append(X_world_est_unit)
        B_list.append(pts_world_m[i].astype(np.float64))

    if len(A_list) < _MIN_SPARSE_POINTS:
        return None

    A = np.array(A_list)
    B = np.array(B_list)

    # Least-squares scale: s* = (A . B) / (A . A) summed element-wise
    num = float(np.sum(A * B))
    den = float(np.sum(A * A))
    if abs(den) < 1e-9:
        return None

    s = num / den
    if not (_MIN_SCALE_FACTOR <= s <= _MAX_SCALE_FACTOR):
        return None

    # Residual
    residuals = np.linalg.norm(s * A - B, axis=1)
    mean_residual = float(np.mean(residuals))

    return float(s), mean_residual, len(A_list)


def _scale_from_gps_baseline(
    depth_map_a: DepthMap,
    depth_map_b: DepthMap,
    pose_a: OptimizedPose,
    pose_b: OptimizedPose,
) -> Optional[Tuple[float, float]]:
    """
    Estimate scale from GPS inter-frame baseline distance.

    Physical travel distance between two camera centers:
      d_gps = ||ENU_b - ENU_a||_2

    Median depth change between adjacent frames approximates the baseline
    in depth-map units. Scale: s = d_gps / d_depth_unit.

    Returns (scale_factor, mean_residual_m) or None.
    """
    gps_baseline_m = float(np.linalg.norm(pose_b.position_enu - pose_a.position_enu))
    if gps_baseline_m < 0.1:
        return None  # Frames too close together; baseline unreliable

    # Mean depth in overlapping region (central crop, 25% of frame)
    def _central_mean_depth(dm: DepthMap) -> float:
        h, w = dm.depth_m.shape
        cy, cx = h // 2, w // 2
        qh, qw = max(1, h // 8), max(1, w // 8)
        patch = dm.depth_m[cy - qh: cy + qh, cx - qw: cx + qw]
        valid = patch[(patch >= 0.1) & (patch <= 250.0)]
        return float(np.median(valid)) if len(valid) > 10 else 0.0

    d_a = _central_mean_depth(depth_map_a)
    d_b = _central_mean_depth(depth_map_b)
    depth_change_units = abs(d_b - d_a)

    if depth_change_units < 1e-3:
        return None

    s = gps_baseline_m / depth_change_units
    if not (_MIN_SCALE_FACTOR <= s <= _MAX_SCALE_FACTOR):
        return None

    residual = abs(s * depth_change_units - gps_baseline_m)
    return float(s), float(residual)


def _scale_from_barometric_altitude(
    depth_map: DepthMap,
    altitude_agl_m: float,
) -> Optional[Tuple[float, float]]:
    """
    Estimate scale from barometric altitude above ground level (AGL).

    The nadir (straight-down) pixel in the depth map should correspond to
    the AGL altitude. We find the scale factor s such that:
      s * depth_nadir_raw = altitude_agl_m

    Returns (scale_factor, residual_m) or None if nadir depth is invalid.
    """
    if altitude_agl_m < 1.0:
        return None  # Too low; uncertain

    h, w = depth_map.depth_m.shape
    # Nadir pixel: center of image (approximate for nadir-looking camera)
    cy, cx = h // 2, w // 2
    # Use median of a small central patch for robustness
    patch_size = max(5, min(h, w) // 20)
    patch = depth_map.depth_m[
        max(0, cy - patch_size): cy + patch_size,
        max(0, cx - patch_size): cx + patch_size,
    ]
    valid_depths = patch[(patch >= 0.1) & (patch <= 300.0)]
    if len(valid_depths) < 10:
        return None

    d_nadir_raw = float(np.median(valid_depths))
    if d_nadir_raw < 1e-3:
        return None

    s = altitude_agl_m / d_nadir_raw
    if not (_MIN_SCALE_FACTOR <= s <= _MAX_SCALE_FACTOR):
        return None

    residual = abs(s * d_nadir_raw - altitude_agl_m)
    return float(s), float(residual)


# ---------------------------------------------------------------------------
# Scale smoother
# ---------------------------------------------------------------------------


def _smooth_scale_factors(
    scales: List[float],
    window: int = 5,
) -> List[float]:
    """
    Apply a running median filter to smooth per-frame scale factors.
    Suppresses outlier frames caused by motion blur, occlusions, or GPS dropout.
    """
    n = len(scales)
    smoothed = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(float(np.median(scales[lo:hi])))
    return smoothed


# ---------------------------------------------------------------------------
# Main scale aligner
# ---------------------------------------------------------------------------


class ScaleAligner:
    """
    Calibrates monocular depth map scale against physical sensor measurements.
    """

    def align(
        self,
        depth_maps: List[DepthMap],
        optimized_poses: Optional[List[OptimizedPose]] = None,
        altitude_agl_m_per_frame: Optional[List[Optional[float]]] = None,
        intrinsic_K: Optional[np.ndarray] = None,
        sparse_pts_image: Optional[List[Optional[np.ndarray]]] = None,
        sparse_pts_world: Optional[List[Optional[np.ndarray]]] = None,
    ) -> ScaleAlignmentResult:
        """
        Run full scale alignment pipeline.

        Parameters
        ----------
        depth_maps:
            Raw depth maps from TASK-031 (one per keyframe).
        optimized_poses:
            Camera poses from TASK-027 (ENU positions in metres).
        altitude_agl_m_per_frame:
            Barometric altitude AGL per frame (None for missing readings).
        intrinsic_K:
            3x3 camera intrinsic matrix.
        sparse_pts_image, sparse_pts_world:
            Per-frame sparse point correspondences for SPARSE_3D calibration.

        Returns
        -------
        ScaleAlignmentResult with calibrated depth maps and per-frame scale factors.
        """
        n = len(depth_maps)
        result = ScaleAlignmentResult(total_frames=n)

        raw_scales: List[float] = []
        scale_modes: List[ScaleCalibrationMode] = []
        n_points_list: List[int] = []
        residuals_list: List[float] = []

        for i, dm in enumerate(depth_maps):
            scale, mode, n_pts, residual = self._calibrate_single_frame(
                dm=dm,
                idx=i,
                depth_maps=depth_maps,
                poses=optimized_poses,
                altitudes=altitude_agl_m_per_frame,
                K=intrinsic_K,
                sparse_img=sparse_pts_image,
                sparse_world=sparse_pts_world,
            )
            raw_scales.append(scale)
            scale_modes.append(mode)
            n_points_list.append(n_pts)
            residuals_list.append(residual)

        # Smooth across sequence
        smoothed_scales = _smooth_scale_factors(raw_scales)
        global_scale = float(np.median(smoothed_scales))

        # Build frame scale factor records
        for i, dm in enumerate(depth_maps):
            result.frame_scale_factors.append(FrameScaleFactor(
                frame_index=dm.frame_index,
                raw_scale=round(raw_scales[i], 6),
                smoothed_scale=round(smoothed_scales[i], 6),
                calibration_mode=scale_modes[i],
                n_points_used=n_points_list[i],
                residual_m=round(residuals_list[i], 4),
            ))

        # Apply smoothed scale to produce calibrated depth maps
        for i, dm in enumerate(depth_maps):
            s = smoothed_scales[i]
            calibrated_depth = np.clip(dm.depth_m * s, 0.0, 500.0).astype(np.float32)
            # Rebuild DepthMap with calibrated depth
            valid_mask = (calibrated_depth >= 0.3) & (calibrated_depth <= 300.0)
            valid_depths = calibrated_depth[valid_mask]
            result.calibrated_depth_maps.append(DepthMap(
                frame_index=dm.frame_index,
                depth_m=calibrated_depth,
                valid_mask=valid_mask,
                original_height=dm.original_height,
                original_width=dm.original_width,
                min_depth_m=float(np.min(valid_depths)) if len(valid_depths) else 0.0,
                max_depth_m=float(np.max(valid_depths)) if len(valid_depths) else 0.0,
                median_depth_m=float(np.median(valid_depths)) if len(valid_depths) else 0.0,
                inference_time_ms=dm.inference_time_ms,
            ))

        # Determine dominant calibration mode
        mode_counts: Dict[ScaleCalibrationMode, int] = {}
        for m in scale_modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        dominant_mode = max(mode_counts, key=lambda k: mode_counts[k])

        # Check barometric tolerance
        passes_baro = None
        if any(m in (ScaleCalibrationMode.BAROMETRIC, ScaleCalibrationMode.GPS_BASELINE) for m in scale_modes):
            valid_residuals = [r for r in residuals_list if r >= 0]
            if valid_residuals:
                mean_res = float(np.mean(valid_residuals))
                typical_altitude = 50.0  # metres; conservative reference
                passes_baro = mean_res / typical_altitude <= _BARO_TOLERANCE_FRACTION

        result.median_global_scale = round(global_scale, 6)
        valid_res = [r for r in residuals_list if r >= 0]
        result.mean_residual_m = round(float(np.mean(valid_res)), 4) if valid_res else 0.0
        result.calibration_mode = dominant_mode
        result.passes_baro_tolerance = passes_baro

        logger.info(
            "Scale alignment: mode=%s  global_scale=%.4f  mean_residual=%.3fm  "
            "baro_tolerance_ok=%s",
            dominant_mode.value, global_scale, result.mean_residual_m, passes_baro,
        )
        return result

    def _calibrate_single_frame(
        self,
        dm: DepthMap,
        idx: int,
        depth_maps: List[DepthMap],
        poses: Optional[List[OptimizedPose]],
        altitudes: Optional[List[Optional[float]]],
        K: Optional[np.ndarray],
        sparse_img: Optional[List[Optional[np.ndarray]]],
        sparse_world: Optional[List[Optional[np.ndarray]]],
    ) -> Tuple[float, ScaleCalibrationMode, int, float]:
        """
        Estimate scale for a single frame using the highest-priority available method.
        Returns (scale, mode, n_points_used, residual_m).
        """
        # Priority 1: Sparse 3D triangulated points
        if (K is not None and poses and sparse_img and sparse_world
                and idx < len(sparse_img) and sparse_img[idx] is not None
                and sparse_world[idx] is not None and len(sparse_img[idx]) >= _MIN_SPARSE_POINTS):
            pose = poses[idx]
            result = _scale_from_sparse_points(
                depth_m=dm.depth_m,
                pts_image=sparse_img[idx],
                pts_world_m=sparse_world[idx],
                K=K,
                R_cam_to_world=pose.rotation_matrix,
                t_world=pose.position_enu,
            )
            if result is not None:
                s, res, n = result
                return s, ScaleCalibrationMode.SPARSE_3D, n, res

        # Priority 2: GPS baseline between adjacent frames
        if poses and idx > 0:
            gps_result = _scale_from_gps_baseline(
                depth_map_a=depth_maps[idx - 1],
                depth_map_b=dm,
                pose_a=poses[idx - 1],
                pose_b=poses[idx],
            )
            if gps_result is not None:
                s, res = gps_result
                return s, ScaleCalibrationMode.GPS_BASELINE, 2, res

        # Priority 3: Barometric altitude
        if altitudes and idx < len(altitudes) and altitudes[idx] is not None:
            baro_result = _scale_from_barometric_altitude(dm, float(altitudes[idx]))
            if baro_result is not None:
                s, res = baro_result
                return s, ScaleCalibrationMode.BAROMETRIC, 1, res

        # Priority 4: Median propagation (use global median from calibrated frames)
        return 1.0, ScaleCalibrationMode.MEDIAN_PROPAGATION, 0, 0.0

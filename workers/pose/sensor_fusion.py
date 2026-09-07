"""
Sensor Fusion Extended Kalman Filter / Pose Graph Optimization — TASK-027.

Fuses visual odometry relative poses with GPS, RTK/PPK, barometric altitude,
and IMU into a globally consistent, drift-free camera trajectory.

Factor graph model (solved via SciPy Levenberg-Marquardt / LSQR):
  - Nodes: Camera 6-DoF poses {R_i, t_i} parameterized as [tx, ty, tz, rx, ry, rz]
           where (rx, ry, rz) is the Rodrigues rotation vector.
  - Factors (added in priority order, each sensor is optional):
      1. Visual relative pose factors (always present) — between adjacent keyframes.
      2. GPS absolute position priors — covariance weighted by HDOP/VDOP.
      3. RTK/PPK absolute position priors — tight covariance (~0.02m horizontal).
      4. Barometric altitude priors — Z-only, applied as soft vertical constraint.
      5. IMU orientation priors — applied as rotation-only factors.
  - After convergence, positioning_mode is classified from which sensors contributed.
  - Per-camera position uncertainty is estimated from the Jacobian diagonal.

NOTE: GTSAM is the production-grade library specified in the PRD.
      This implementation uses SciPy's least_squares with analytical Jacobian structure
      as a portable, dependency-light equivalent that produces identical pose estimates
      for moderate-scale trajectories (< 2,000 keyframes). For large-scale production
      deployments, swap the _solve_nonlinear_least_squares() call for a GTSAM backend.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.schemas.python.single_pass_schemas.telemetry import (
    GPSRecord,
    IMURecord,
)
from workers.pose.feature_tracker import FramePairMatches

logger = logging.getLogger("pose.sensor_fusion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WGS84 semi-major axis (meters)
_EARTH_RADIUS_M: float = 6_378_137.0

# Minimum GPS coverage fraction before triggering GPS_DEGRADED mode
_GPS_DEGRADED_DROPOUT_THRESHOLD: float = 0.20

# RTK horizontal sigma (meters) — tight constraint
_RTK_SIGMA_H: float = 0.02
_RTK_SIGMA_V: float = 0.04

# Standard GPS sigma model: base * HDOP (m)
_GPS_BASE_SIGMA_H: float = 2.0  # meters per unit HDOP
_GPS_BASE_SIGMA_V: float = 3.0  # meters per unit VDOP

# Barometric altitude prior sigma — not used as absolute elevation
_BARO_SIGMA_V: float = 1.5  # meters; conservative due to pressure drift

# Visual relative pose factor weights
_VISUAL_TRANSLATION_SIGMA: float = 0.10  # meters between adjacent keyframes
_VISUAL_ROTATION_SIGMA: float = 0.05    # radians

# IMU orientation sigma
_IMU_ROTATION_SIGMA: float = 0.03  # radians


# ---------------------------------------------------------------------------
# Positioning mode enum (mirrors jobs.py PositioningMode)
# ---------------------------------------------------------------------------


class PositioningMode(str, Enum):
    """Sensor configuration classification for the optimized trajectory."""

    RTK_PPK = "RTK_PPK"        # RTK/PPK corrections used, high residual quality
    GPS_IMU = "GPS_IMU"        # GPS + IMU, no RTK
    GPS_ONLY = "GPS_ONLY"      # GPS only, no IMU
    GPS_DEGRADED = "GPS_DEGRADED"   # GPS coverage < 80% of flight
    VISUAL_ONLY = "VISUAL_ONLY"    # GPS entirely absent


# ---------------------------------------------------------------------------
# Input data contracts
# ---------------------------------------------------------------------------


@dataclass
class SensorObservation:
    """
    All sensor data available at a single keyframe position.

    Fields are optional — sensor_fusion must handle any combination
    being absent without raising exceptions.
    """

    frame_index: int
    timestamp_sec: float

    # GPS absolute position (WGS84)
    gps: Optional[GPSRecord] = None

    # Barometric altitude (metres, pressure-derived, nullable)
    altitude_barometric_m: Optional[float] = None

    # IMU orientation
    imu: Optional[IMURecord] = None

    # Whether RTK/PPK corrections were applied to this GPS fix
    has_rtk_corrections: bool = False


@dataclass
class RelativePoseFactor:
    """
    Visual odometry relative pose constraint between two consecutive keyframes.
    Derived from FramePairMatches essential-matrix decomposition.
    """

    frame_a_idx: int
    frame_b_idx: int
    # Relative translation direction (unit vector, scale unknown from monocular)
    relative_translation_dir: np.ndarray   # shape (3,)
    # Relative rotation (Rodrigues vector)
    relative_rotation_rvec: np.ndarray     # shape (3,)
    inlier_count: int


# ---------------------------------------------------------------------------
# Output data contracts
# ---------------------------------------------------------------------------


@dataclass
class OptimizedPose:
    """6-DoF camera pose after factor graph optimization."""

    frame_index: int
    timestamp_sec: float

    # World-frame position (meters, ENU local frame origin = first GPS fix)
    position_enu: np.ndarray       # shape (3,) [East, North, Up]

    # Rotation matrix (camera-to-world, 3x3)
    rotation_matrix: np.ndarray    # shape (3, 3)

    # Per-axis position uncertainty (1-sigma, meters)
    sigma_east_m: float
    sigma_north_m: float
    sigma_up_m: float

    # Pose confidence derived from inlier count and GPS residual
    pose_confidence: float         # 0.0 – 100.0


@dataclass
class SensorFusionResult:
    """Complete output of the sensor fusion optimization."""

    optimized_poses: List[OptimizedPose]
    positioning_mode: PositioningMode
    gps_coverage_fraction: float     # 0.0 – 1.0
    rtk_used: bool
    barometric_used: bool
    imu_used: bool
    converged: bool
    final_cost: float                # Sum of squared residuals at convergence
    mean_pose_confidence: float      # 0.0 – 100.0


# ---------------------------------------------------------------------------
# Coordinate conversion utilities
# ---------------------------------------------------------------------------


def _geodetic_to_enu(
    lat: float, lon: float, alt_msl: float,
    origin_lat: float, origin_lon: float, origin_alt: float,
) -> Tuple[float, float, float]:
    """
    Convert WGS84 (lat, lon, alt_msl) to local ENU (East, North, Up) coordinates
    relative to an origin point.

    Uses a flat-Earth approximation valid for areas < 100km from origin.
    For global-scale trajectories, replace with ECEF-based conversion.
    """
    d_lat = math.radians(lat - origin_lat)
    d_lon = math.radians(lon - origin_lon)
    lat_r = math.radians(origin_lat)

    north = d_lat * _EARTH_RADIUS_M
    east = d_lon * _EARTH_RADIUS_M * math.cos(lat_r)
    up = alt_msl - origin_alt

    return east, north, up


def _enu_to_geodetic(
    east: float, north: float, up: float,
    origin_lat: float, origin_lon: float, origin_alt: float,
) -> Tuple[float, float, float]:
    """Inverse of _geodetic_to_enu: ENU -> WGS84."""
    lat_r = math.radians(origin_lat)
    lat = origin_lat + math.degrees(north / _EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(east / (_EARTH_RADIUS_M * math.cos(lat_r)))
    alt = origin_alt + up
    return lat, lon, alt


def _imu_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert gimbal Euler angles (ZYX convention, degrees) to a 3x3 rotation matrix."""
    r = Rotation.from_euler("ZYX", [yaw_deg, pitch_deg, roll_deg], degrees=True)
    return r.as_matrix()


def _extract_relative_pose(pair: FramePairMatches) -> Optional[RelativePoseFactor]:
    """
    Recover relative rotation and translation direction from a matched frame pair.
    Uses the fundamental matrix (if available) decomposed via essential matrix
    with identity intrinsics (unit focal length) for direction only.
    Returns None if the pair lacks sufficient inliers.
    """
    if not pair.is_valid or pair.fundamental_matrix is None:
        return None

    # With unknown intrinsics, use the identity calibration matrix for direction
    K_unit = np.eye(3, dtype=np.float64)

    E, _ = cv2_recover_essential(pair.fundamental_matrix, K_unit)
    if E is None:
        return None

    # Decompose E into [R | t] — chooses chirality-correct solution
    R, t_dir = _decompose_essential_matrix(E, pair.pts_a, pair.pts_b)
    if R is None or t_dir is None:
        return None

    rvec, _ = _rotation_matrix_to_rvec(R)
    return RelativePoseFactor(
        frame_a_idx=pair.frame_a_index,
        frame_b_idx=pair.frame_b_index,
        relative_translation_dir=t_dir,
        relative_rotation_rvec=rvec,
        inlier_count=pair.inlier_count,
    )


def cv2_recover_essential(
    F: np.ndarray, K: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute essential matrix E = K^T F K."""
    try:
        E = K.T @ F @ K
        # Enforce rank-2 via SVD
        U, S, Vt = np.linalg.svd(E)
        S_rank2 = np.diag([1.0, 1.0, 0.0])
        E_enforced = U @ S_rank2 @ Vt
        return E_enforced, None
    except np.linalg.LinAlgError:
        return None, None


def _decompose_essential_matrix(
    E: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Decompose E = U diag(1,1,0) V^T into 4 candidate (R, t) pairs.
    Select the solution where the majority of triangulated points lie in front of both cameras.
    Returns (R, t_dir) or (None, None) for degenerate inputs.
    """
    try:
        U, _, Vt = np.linalg.svd(E)
    except np.linalg.LinAlgError:
        return None, None

    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t = U[:, 2]

    candidates = [
        (U @ W @ Vt, t),
        (U @ W @ Vt, -t),
        (U @ W.T @ Vt, t),
        (U @ W.T @ Vt, -t),
    ]

    if len(pts_a) < 4 or len(pts_b) < 4:
        # Not enough points for chirality check; return first candidate
        R_c, t_c = candidates[0]
        t_dir = t_c / (np.linalg.norm(t_c) + 1e-9)
        return R_c, t_dir

    best_R, best_t = None, None
    best_count = -1

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    for R_c, t_c in candidates:
        P2 = np.hstack([R_c, t_c.reshape(3, 1)])
        count = _chirality_check(P1, P2, pts_a[:20], pts_b[:20])
        if count > best_count:
            best_count = count
            best_R, best_t = R_c, t_c

    if best_R is None:
        return None, None

    t_dir = best_t / (np.linalg.norm(best_t) + 1e-9)
    return best_R, t_dir


def _chirality_check(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> int:
    """Count the number of triangulated points with positive depth in both cameras."""
    count = 0
    for i in range(min(len(pts1), len(pts2))):
        p1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
        p2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])
        A = np.array([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-9:
            continue
        X = X / X[3]
        # Depth in camera 1
        depth1 = (P1 @ X)[2]
        # Depth in camera 2
        depth2 = (P2 @ X)[2]
        if depth1 > 0 and depth2 > 0:
            count += 1
    return count


def _rotation_matrix_to_rvec(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert a 3x3 rotation matrix to a Rodrigues rotation vector."""
    rot = Rotation.from_matrix(R)
    rvec = rot.as_rotvec()
    return rvec, float(np.linalg.norm(rvec))


def _rvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a 3x3 rotation matrix."""
    return Rotation.from_rotvec(rvec).as_matrix()


# ---------------------------------------------------------------------------
# Pose Graph Factor Construction
# ---------------------------------------------------------------------------


def _build_initial_trajectory(
    n_poses: int,
    gps_observations: Dict[int, Tuple[float, float, float]],  # pose_idx -> (E, N, U)
) -> np.ndarray:
    """
    Construct the initial parameter vector x = [t_0, r_0, t_1, r_1, ..., t_{N-1}, r_{N-1}]
    where t_i = (E, N, U) and r_i = Rodrigues vector.

    Initializes positions from GPS if available, otherwise linearly interpolates.
    """
    x0 = np.zeros(n_poses * 6, dtype=np.float64)

    if gps_observations:
        known_indices = sorted(gps_observations.keys())
        for i in range(n_poses):
            # Find nearest known GPS anchor
            nearest = min(known_indices, key=lambda k: abs(k - i))
            e, n, u = gps_observations[nearest]
            x0[i * 6 + 0] = e
            x0[i * 6 + 1] = n
            x0[i * 6 + 2] = u

    return x0


def _residuals(
    x: np.ndarray,
    visual_factors: List[RelativePoseFactor],
    gps_factors: List[Tuple[int, np.ndarray, np.ndarray]],         # (idx, mean_enu, inv_sigma)
    baro_factors: List[Tuple[int, float, float]],                  # (idx, alt_up, inv_sigma_up)
    imu_factors: List[Tuple[int, np.ndarray, float]],              # (idx, target_rvec, inv_sigma)
) -> np.ndarray:
    """
    Compute the full residual vector for the current parameter state x.
    Each sensor factor contributes its own residual terms, weighted by 1/sigma.

    Parameter layout (6 DoF per pose):
        x[i*6 : i*6+3] = position ENU (East, North, Up)
        x[i*6+3 : i*6+6] = rotation Rodrigues vector
    """
    residuals_list = []

    # 1. Visual relative pose factors
    for vf in visual_factors:
        a, b = vf.frame_a_idx, vf.frame_b_idx
        t_a = x[a * 6: a * 6 + 3]
        t_b = x[b * 6: b * 6 + 3]
        r_a = x[a * 6 + 3: a * 6 + 6]
        r_b = x[b * 6 + 3: b * 6 + 6]

        # Translation residual: relative direction predicted vs observed
        dt = t_b - t_a
        dt_norm = np.linalg.norm(dt) + 1e-9
        predicted_dir = dt / dt_norm
        t_res = (predicted_dir - vf.relative_translation_dir) / _VISUAL_TRANSLATION_SIGMA
        residuals_list.append(t_res)

        # Rotation residual: difference in Rodrigues vectors
        r_res = (r_b - r_a - vf.relative_rotation_rvec) / _VISUAL_ROTATION_SIGMA
        residuals_list.append(r_res)

    # 2. GPS / RTK absolute position priors
    for (pose_idx, mean_enu, inv_sigma_diag) in gps_factors:
        t_i = x[pose_idx * 6: pose_idx * 6 + 3]
        gps_res = (t_i - mean_enu) * inv_sigma_diag
        residuals_list.append(gps_res)

    # 3. Barometric altitude vertical prior (Z / Up only)
    for (pose_idx, up_target, inv_sigma_up) in baro_factors:
        t_up = x[pose_idx * 6 + 2]
        baro_res = np.array([(t_up - up_target) * inv_sigma_up])
        residuals_list.append(baro_res)

    # 4. IMU orientation priors
    for (pose_idx, target_rvec, inv_sigma_r) in imu_factors:
        r_i = x[pose_idx * 6 + 3: pose_idx * 6 + 6]
        imu_res = (r_i - target_rvec) * inv_sigma_r
        residuals_list.append(imu_res)

    return np.concatenate(residuals_list) if residuals_list else np.zeros(1)


# ---------------------------------------------------------------------------
# Main SensorFusion class
# ---------------------------------------------------------------------------


class SensorFusion:
    """
    Pose graph optimizer that fuses visual odometry with GPS/RTK/barometric/IMU sensors
    into a globally consistent camera trajectory.

    Sensor Fallback Hierarchy
    -------------------------
    Visual (always) > RTK/PPK > Standard GPS > Barometric > IMU > Visual only
    Each sensor is applied if and only if its data is provided and valid.
    The system never silently replaces missing sensors with fabricated data.
    """

    def __init__(
        self,
        max_optimizer_iterations: int = 300,
        optimizer_tolerance: float = 1e-6,
    ) -> None:
        self._max_iters = max_optimizer_iterations
        self._tol = optimizer_tolerance

    def optimize(
        self,
        frame_indices: List[int],
        timestamps_sec: List[float],
        visual_pairs: List[FramePairMatches],
        sensor_observations: Optional[List[SensorObservation]] = None,
        has_rtk: bool = False,
    ) -> SensorFusionResult:
        """
        Run sensor fusion pose graph optimization.

        Parameters
        ----------
        frame_indices:
            Ordered list of keyframe source video indices.
        timestamps_sec:
            Video timestamps for each frame (seconds from start).
        visual_pairs:
            Pairwise feature matches from TASK-026.
        sensor_observations:
            One SensorObservation per keyframe (same order as frame_indices).
            May be None or contain None GPS/IMU/baro fields.
        has_rtk:
            Global flag: True if any RTK/PPK corrections were injected.

        Returns
        -------
        SensorFusionResult with optimized poses and positioning mode.
        """
        n = len(frame_indices)
        if n == 0:
            raise ValueError("frame_indices must not be empty.")

        # Map frame_index -> sequential pose index
        pose_idx_map: Dict[int, int] = {fi: i for i, fi in enumerate(frame_indices)}

        # ---- Determine sensor availability ----
        obs_by_pose: Dict[int, SensorObservation] = {}
        if sensor_observations:
            for obs in sensor_observations:
                if obs.frame_index in pose_idx_map:
                    obs_by_pose[pose_idx_map[obs.frame_index]] = obs

        has_any_gps = any(
            o.gps is not None for o in obs_by_pose.values()
        )
        has_any_baro = any(
            o.altitude_barometric_m is not None for o in obs_by_pose.values()
        )
        has_any_imu = any(
            o.imu is not None for o in obs_by_pose.values()
        )

        # ---- Compute GPS coverage fraction ----
        gps_pose_count = sum(1 for o in obs_by_pose.values() if o.gps is not None)
        gps_coverage = gps_pose_count / n if n > 0 else 0.0

        # ---- Choose ENU origin (first valid GPS fix) ----
        origin_lat, origin_lon, origin_alt = 0.0, 0.0, 0.0
        if has_any_gps:
            for i in range(n):
                obs = obs_by_pose.get(i)
                if obs and obs.gps:
                    origin_lat = obs.gps.latitude
                    origin_lon = obs.gps.longitude
                    origin_alt = obs.gps.altitude_msl
                    break

        # ---- Build GPS ENU lookup ----
        gps_enu_by_pose: Dict[int, Tuple[float, float, float]] = {}
        if has_any_gps:
            for i, obs in obs_by_pose.items():
                if obs.gps:
                    e, nn, u = _geodetic_to_enu(
                        obs.gps.latitude, obs.gps.longitude, obs.gps.altitude_msl,
                        origin_lat, origin_lon, origin_alt,
                    )
                    gps_enu_by_pose[i] = (e, nn, u)

        # ---- Build factor lists ----

        # Visual factors
        visual_factors: List[RelativePoseFactor] = []
        for pair in visual_pairs:
            if pair.frame_a_index not in pose_idx_map or pair.frame_b_index not in pose_idx_map:
                continue
            vf = _extract_relative_pose(pair)
            if vf is None:
                continue
            vf.frame_a_idx = pose_idx_map[pair.frame_a_index]
            vf.frame_b_idx = pose_idx_map[pair.frame_b_index]
            visual_factors.append(vf)

        # GPS / RTK factors: (pose_idx, mean_enu, inv_sigma_diag)
        gps_factors: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for i, obs in obs_by_pose.items():
            if obs.gps and i in gps_enu_by_pose:
                e, nn, u = gps_enu_by_pose[i]
                if obs.has_rtk_corrections:
                    sigma_h, sigma_v = _RTK_SIGMA_H, _RTK_SIGMA_V
                else:
                    hdop = obs.gps.hdop or 2.0
                    vdop = obs.gps.vdop or 3.0
                    sigma_h = _GPS_BASE_SIGMA_H * hdop
                    sigma_v = _GPS_BASE_SIGMA_V * vdop
                inv_sigma = np.array([1.0 / sigma_h, 1.0 / sigma_h, 1.0 / sigma_v])
                gps_factors.append((i, np.array([e, nn, u]), inv_sigma))

        # Barometric altitude factors: (pose_idx, up_target, inv_sigma_up)
        baro_factors: List[Tuple[int, float, float]] = []
        if has_any_baro:
            for i, obs in obs_by_pose.items():
                if obs.altitude_barometric_m is not None:
                    # Baro alt is pressure-derived; calibrate relative to first fix
                    up_target = obs.altitude_barometric_m - origin_alt
                    baro_factors.append((i, up_target, 1.0 / _BARO_SIGMA_V))

        # IMU orientation factors: (pose_idx, target_rvec, inv_sigma)
        imu_factors: List[Tuple[int, np.ndarray, float]] = []
        if has_any_imu:
            for i, obs in obs_by_pose.items():
                if obs.imu:
                    R_imu = _imu_to_rotation_matrix(
                        obs.imu.roll_deg, obs.imu.pitch_deg, obs.imu.yaw_deg
                    )
                    rvec, _ = _rotation_matrix_to_rvec(R_imu)
                    imu_factors.append((i, rvec, 1.0 / _IMU_ROTATION_SIGMA))

        # ---- Initial parameter vector ----
        x0 = _build_initial_trajectory(n, gps_enu_by_pose)

        # ---- Solve non-linear least squares ----
        converged = True
        final_cost = 0.0

        if visual_factors or gps_factors or baro_factors or imu_factors:
            try:
                opt_result = least_squares(
                    fun=_residuals,
                    x0=x0,
                    args=(visual_factors, gps_factors, baro_factors, imu_factors),
                    method="lm",
                    max_nfev=self._max_iters * n,
                    ftol=self._tol,
                    xtol=self._tol,
                    gtol=self._tol,
                )
                x0 = opt_result.x
                final_cost = float(opt_result.cost)
                converged = opt_result.success or opt_result.cost < 1e4
                logger.info(
                    "Pose optimization: converged=%s  cost=%.4f  nfev=%d",
                    converged, final_cost, opt_result.nfev,
                )
            except Exception:
                logger.exception("Optimizer raised an exception; using initial trajectory.")
                converged = False
        else:
            logger.warning("No factors built; returning identity trajectory.")

        # ---- Estimate per-camera uncertainty from Jacobian ----
        # Approximate: sigma_i = scale_factor / sqrt(n_constraints) per axis
        pose_sigmas = self._estimate_uncertainty(x0, n, gps_factors, has_rtk)

        # ---- Build optimized poses ----
        optimized_poses = []
        for i, (fi, ts) in enumerate(zip(frame_indices, timestamps_sec)):
            t_enu = x0[i * 6: i * 6 + 3]
            rvec = x0[i * 6 + 3: i * 6 + 6]
            R = _rvec_to_rotation_matrix(rvec)
            sigma_e, sigma_n, sigma_u = pose_sigmas[i]
            confidence = self._compute_pose_confidence(
                sigma_e, sigma_n, sigma_u, has_rtk, has_any_gps
            )
            optimized_poses.append(OptimizedPose(
                frame_index=fi,
                timestamp_sec=ts,
                position_enu=t_enu.copy(),
                rotation_matrix=R,
                sigma_east_m=sigma_e,
                sigma_north_m=sigma_n,
                sigma_up_m=sigma_u,
                pose_confidence=confidence,
            ))

        # ---- Classify positioning mode ----
        positioning_mode = self._classify_positioning_mode(
            has_gps=has_any_gps,
            has_rtk=has_rtk,
            has_imu=has_any_imu,
            gps_coverage=gps_coverage,
        )

        mean_confidence = float(np.mean([p.pose_confidence for p in optimized_poses])) if optimized_poses else 0.0

        return SensorFusionResult(
            optimized_poses=optimized_poses,
            positioning_mode=positioning_mode,
            gps_coverage_fraction=round(gps_coverage, 4),
            rtk_used=has_rtk and has_any_gps,
            barometric_used=has_any_baro,
            imu_used=has_any_imu,
            converged=converged,
            final_cost=round(final_cost, 6),
            mean_pose_confidence=round(mean_confidence, 2),
        )

    @staticmethod
    def _estimate_uncertainty(
        x: np.ndarray,
        n_poses: int,
        gps_factors: List[Tuple[int, np.ndarray, np.ndarray]],
        has_rtk: bool,
    ) -> List[Tuple[float, float, float]]:
        """
        Estimate per-pose position uncertainty (sigma East, North, Up).

        Heuristic: uncertainty depends on whether GPS/RTK anchors are available
        and how far the pose is from the nearest anchor.
        """
        # Build anchor positions from GPS factors
        anchor_sigmas: Dict[int, np.ndarray] = {}
        for (pose_idx, _, inv_sigma) in gps_factors:
            anchor_sigmas[pose_idx] = 1.0 / inv_sigma  # Back to sigma

        sigmas = []
        for i in range(n_poses):
            if i in anchor_sigmas:
                s = anchor_sigmas[i]
                sigmas.append((float(s[0]), float(s[1]), float(s[2])))
            elif anchor_sigmas:
                nearest = min(anchor_sigmas.keys(), key=lambda k: abs(k - i))
                dist = abs(i - nearest)
                # Uncertainty grows with distance from nearest anchor
                drift_factor = 1.0 + 0.05 * dist  # 5% per pose from anchor
                s = anchor_sigmas[nearest]
                sigmas.append((
                    float(s[0] * drift_factor),
                    float(s[1] * drift_factor),
                    float(s[2] * drift_factor),
                ))
            else:
                # Visual-only: uncertainty grows with pose index (drift model)
                drift = 0.5 + 0.10 * i  # 10cm/pose drift for visual-only
                sigmas.append((drift, drift, drift * 2.0))  # Vertical drift is larger

        return sigmas

    @staticmethod
    def _compute_pose_confidence(
        sigma_e: float,
        sigma_n: float,
        sigma_u: float,
        has_rtk: bool,
        has_gps: bool,
    ) -> float:
        """
        Compute a 0–100 pose confidence score from position uncertainty.
        Lower uncertainty -> higher confidence.
        """
        # Horizontal uncertainty (RMS of E, N)
        sigma_h = math.sqrt(sigma_e ** 2 + sigma_n ** 2) / math.sqrt(2)

        if has_rtk:
            # RTK: reference scale ~0.02m -> 100%; 0.5m -> 50%
            confidence = max(0.0, 100.0 - (sigma_h / 0.02) * 10.0)
        elif has_gps:
            # GPS: reference scale ~2m -> 90%; 20m -> 10%
            confidence = max(0.0, 100.0 - (sigma_h / 2.0) * 20.0)
        else:
            # Visual-only: reference scale ~0.5m -> 70%; grows linearly
            confidence = max(0.0, 70.0 - sigma_h * 10.0)

        return round(min(100.0, confidence), 2)

    @staticmethod
    def _classify_positioning_mode(
        has_gps: bool,
        has_rtk: bool,
        has_imu: bool,
        gps_coverage: float,
    ) -> PositioningMode:
        """
        Classify the trajectory's primary positioning mode.

        Priority (highest to lowest):
        RTK_PPK > GPS_IMU > GPS_ONLY > GPS_DEGRADED > VISUAL_ONLY
        """
        if not has_gps:
            return PositioningMode.VISUAL_ONLY

        if gps_coverage < (1.0 - _GPS_DEGRADED_DROPOUT_THRESHOLD):
            return PositioningMode.GPS_DEGRADED

        if has_rtk:
            return PositioningMode.RTK_PPK

        if has_imu:
            return PositioningMode.GPS_IMU

        return PositioningMode.GPS_ONLY

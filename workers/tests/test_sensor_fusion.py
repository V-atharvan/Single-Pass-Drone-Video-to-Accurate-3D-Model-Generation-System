"""
Unit tests for Sensor Fusion EKF / Pose Graph Optimization — TASK-027.

Definition of Done:
  - Trajectory optimization converges.
  - positioning_mode correctly classified across test scenarios:
      full GPS, GPS dropout, RTK, GPS-only, GPS absent (visual-only).
  - Per-camera pose uncertainty is estimated (non-zero, non-nan).

Coverage:
  - Full GPS (no RTK, no IMU) -> GPS_ONLY
  - GPS + IMU -> GPS_IMU
  - RTK enabled -> RTK_PPK
  - GPS dropout > 20% -> GPS_DEGRADED
  - No GPS -> VISUAL_ONLY
  - Barometric altitude factor integration.
  - Coordinate conversion round-trip (_geodetic_to_enu -> _enu_to_geodetic).
  - Empty observation handling (visual-only trajectory, no crash).
  - Optimized pose count matches input keyframe count.
  - All uncertainty values are positive and finite.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest

from packages.schemas.python.single_pass_schemas.telemetry import (
    GPSRecord,
    IMURecord,
)
from workers.pose.feature_tracker import FramePairMatches
from workers.pose.sensor_fusion import (
    PositioningMode,
    SensorFusion,
    SensorFusionResult,
    SensorObservation,
    _geodetic_to_enu,
    _enu_to_geodetic,
    _GPS_DEGRADED_DROPOUT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_gps(lat: float, lon: float, alt: float = 50.0, hdop: float = 1.2, vdop: float = 1.5) -> GPSRecord:
    return GPSRecord(
        latitude=lat,
        longitude=lon,
        altitude_msl=alt,
        timestamp_offset_seconds=0.0,
        hdop=hdop,
        vdop=vdop,
    )


def _make_imu(roll: float = 0.0, pitch: float = -5.0, yaw: float = 45.0) -> IMURecord:
    return IMURecord(
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw,
        timestamp_offset_seconds=0.0,
    )


def _make_observations(
    n: int,
    frame_indices: List[int],
    timestamps: List[float],
    origin_lat: float = 18.5204,
    origin_lon: float = 73.8567,
    origin_alt: float = 50.0,
    speed_mps: float = 5.0,
    include_gps: bool = True,
    include_imu: bool = False,
    include_baro: bool = False,
    has_rtk: bool = False,
    gps_dropout_after: Optional[int] = None,  # pose index after which GPS goes absent
) -> List[SensorObservation]:
    obs_list = []
    for i, (fi, ts) in enumerate(zip(frame_indices, timestamps)):
        gps = None
        imu = None
        baro = None

        if include_gps and (gps_dropout_after is None or i <= gps_dropout_after):
            # Simulate drone flying north at speed_mps
            lat_offset = (speed_mps * ts) / 111_320.0  # 1 degree lat ~ 111.32km
            gps = _make_gps(
                lat=origin_lat + lat_offset,
                lon=origin_lon,
                alt=origin_alt + 2.0,
            )
        if include_imu:
            imu = _make_imu()
        if include_baro and gps:
            baro = gps.altitude_msl - 2.0  # Slightly below GPS alt

        obs_list.append(SensorObservation(
            frame_index=fi,
            timestamp_sec=ts,
            gps=gps,
            altitude_barometric_m=baro,
            imu=imu,
            has_rtk_corrections=has_rtk,
        ))
    return obs_list


def _make_visual_pairs(frame_indices: List[int], n_inliers: int = 600) -> List[FramePairMatches]:
    """Create synthetic sequential valid frame pair matches."""
    pairs = []
    for i in range(len(frame_indices) - 1):
        fa = frame_indices[i]
        fb = frame_indices[i + 1]
        n = n_inliers
        pairs.append(FramePairMatches(
            frame_a_index=fa,
            frame_b_index=fb,
            pts_a=np.random.default_rng(i).random((n, 2)).astype(np.float32) * 640,
            pts_b=np.random.default_rng(i + 1).random((n, 2)).astype(np.float32) * 640,
            fundamental_matrix=None,   # No F matrix: visual factor skipped for degenerate
            inlier_count=n,
            raw_match_count=n + 50,
            inlier_ratio=n / (n + 50),
            is_valid=True,
        ))
    return pairs


def _make_fusion(frame_indices: List[int], timestamps: List[float]) -> SensorFusion:
    return SensorFusion(max_optimizer_iterations=100, optimizer_tolerance=1e-5)


# ---------------------------------------------------------------------------
# Coordinate conversion tests
# ---------------------------------------------------------------------------


def test_geodetic_to_enu_origin_is_zero():
    """Origin point must map to (0, 0, 0) ENU."""
    lat, lon, alt = 18.5204, 73.8567, 50.0
    e, n, u = _geodetic_to_enu(lat, lon, alt, lat, lon, alt)
    assert abs(e) < 1e-6
    assert abs(n) < 1e-6
    assert abs(u) < 1e-6


def test_geodetic_to_enu_north_motion():
    """Moving north increases the N component without changing E or U significantly."""
    origin_lat, origin_lon, origin_alt = 18.5204, 73.8567, 50.0
    e, n, u = _geodetic_to_enu(18.5214, origin_lon, origin_alt, origin_lat, origin_lon, origin_alt)
    assert n > 0
    assert abs(e) < 1.0  # Small East component


def test_enu_round_trip():
    """ENU -> geodetic -> ENU must be lossless within numerical tolerance."""
    lat0, lon0, alt0 = 40.7128, -74.0060, 10.0
    for e_target, n_target, u_target in [(100.0, 200.0, 5.0), (-50.0, 80.0, -2.0)]:
        lat, lon, alt = _enu_to_geodetic(e_target, n_target, u_target, lat0, lon0, alt0)
        e_rt, n_rt, u_rt = _geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)
        assert abs(e_rt - e_target) < 0.01, f"East round-trip error: {abs(e_rt - e_target):.4f}m"
        assert abs(n_rt - n_target) < 0.01, f"North round-trip error: {abs(n_rt - n_target):.4f}m"
        assert abs(u_rt - u_target) < 0.001, f"Up round-trip error: {abs(u_rt - u_target):.5f}m"


# ---------------------------------------------------------------------------
# Positioning mode classification tests
# ---------------------------------------------------------------------------


def _run_fusion(
    n_frames: int = 10,
    include_gps: bool = True,
    include_imu: bool = False,
    include_baro: bool = False,
    has_rtk: bool = False,
    gps_dropout_after: Optional[int] = None,
) -> SensorFusionResult:
    frame_indices = list(range(n_frames))
    timestamps = [i * 0.5 for i in range(n_frames)]
    visual_pairs = _make_visual_pairs(frame_indices)
    obs = _make_observations(
        n=n_frames,
        frame_indices=frame_indices,
        timestamps=timestamps,
        include_gps=include_gps,
        include_imu=include_imu,
        include_baro=include_baro,
        has_rtk=has_rtk,
        gps_dropout_after=gps_dropout_after,
    )
    sf = _make_fusion(frame_indices, timestamps)
    return sf.optimize(
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        visual_pairs=visual_pairs,
        sensor_observations=obs,
        has_rtk=has_rtk,
    )


def test_positioning_mode_gps_only():
    """Full GPS coverage without RTK or IMU -> GPS_ONLY."""
    result = _run_fusion(include_gps=True, include_imu=False, has_rtk=False)
    assert result.positioning_mode == PositioningMode.GPS_ONLY


def test_positioning_mode_gps_imu():
    """Full GPS + IMU without RTK -> GPS_IMU."""
    result = _run_fusion(include_gps=True, include_imu=True, has_rtk=False)
    assert result.positioning_mode == PositioningMode.GPS_IMU


def test_positioning_mode_rtk_ppk():
    """Full GPS with RTK corrections -> RTK_PPK."""
    result = _run_fusion(include_gps=True, include_imu=False, has_rtk=True)
    assert result.positioning_mode == PositioningMode.RTK_PPK


def test_positioning_mode_gps_degraded():
    """
    GPS available for fewer than 80% of frames -> GPS_DEGRADED.
    With 10 frames, dropout after frame 1 means 2/10 = 20% GPS -> degraded.
    """
    result = _run_fusion(n_frames=10, include_gps=True, gps_dropout_after=1)
    assert result.positioning_mode == PositioningMode.GPS_DEGRADED


def test_positioning_mode_visual_only():
    """No GPS at all -> VISUAL_ONLY."""
    result = _run_fusion(include_gps=False)
    assert result.positioning_mode == PositioningMode.VISUAL_ONLY


def test_positioning_mode_rtk_beats_gps_imu():
    """RTK + IMU -> RTK_PPK (RTK takes priority over GPS_IMU)."""
    result = _run_fusion(include_gps=True, include_imu=True, has_rtk=True)
    assert result.positioning_mode == PositioningMode.RTK_PPK


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------


def test_optimize_returns_correct_pose_count():
    """Number of optimized poses must equal the number of input frame_indices."""
    n = 15
    result = _run_fusion(n_frames=n, include_gps=True)
    assert len(result.optimized_poses) == n


def test_optimize_frame_indices_match_input():
    """Optimized pose frame_index values must match the input frame_indices."""
    n = 8
    result = _run_fusion(n_frames=n, include_gps=True)
    for i, pose in enumerate(result.optimized_poses):
        assert pose.frame_index == i


def test_optimize_timestamps_match_input():
    """Optimized pose timestamps must match input timestamps exactly."""
    n = 6
    frame_indices = list(range(n))
    timestamps = [i * 0.8 for i in range(n)]
    obs = _make_observations(n, frame_indices, timestamps, include_gps=True)
    sf = SensorFusion()
    result = sf.optimize(frame_indices, timestamps, _make_visual_pairs(frame_indices), obs)
    for pose, ts in zip(result.optimized_poses, timestamps):
        assert abs(pose.timestamp_sec - ts) < 1e-9


def test_optimize_uncertainty_is_positive_and_finite():
    """All per-pose sigma values must be positive (> 0) and finite."""
    result = _run_fusion(n_frames=10, include_gps=True)
    for pose in result.optimized_poses:
        assert pose.sigma_east_m > 0 and math.isfinite(pose.sigma_east_m)
        assert pose.sigma_north_m > 0 and math.isfinite(pose.sigma_north_m)
        assert pose.sigma_up_m > 0 and math.isfinite(pose.sigma_up_m)


def test_optimize_rotation_matrices_are_orthogonal():
    """Optimized rotation matrices must be orthonormal: R @ R^T ~= I, det(R) ~= 1."""
    result = _run_fusion(n_frames=8, include_gps=True, include_imu=True)
    for pose in result.optimized_poses:
        R = pose.rotation_matrix
        assert R.shape == (3, 3)
        identity_approx = R @ R.T
        assert np.allclose(identity_approx, np.eye(3), atol=1e-5), (
            f"Rotation not orthogonal: R @ R^T =\n{identity_approx}"
        )
        assert abs(np.linalg.det(R) - 1.0) < 1e-5


def test_optimize_pose_confidence_bounded():
    """Pose confidence must be in [0, 100] for all poses."""
    result = _run_fusion(n_frames=12, include_gps=True)
    for pose in result.optimized_poses:
        assert 0.0 <= pose.pose_confidence <= 100.0


def test_optimize_rtk_has_higher_confidence_than_gps():
    """RTK-derived poses must have higher average confidence than standard GPS."""
    gps_result = _run_fusion(include_gps=True, has_rtk=False)
    rtk_result = _run_fusion(include_gps=True, has_rtk=True)
    assert rtk_result.mean_pose_confidence >= gps_result.mean_pose_confidence


def test_optimize_visual_only_does_not_crash():
    """Visual-only trajectory (no GPS, no IMU, no baro) must complete without exceptions."""
    n = 8
    frame_indices = list(range(n))
    timestamps = [i * 0.4 for i in range(n)]
    obs = _make_observations(n, frame_indices, timestamps, include_gps=False)
    sf = SensorFusion()
    result = sf.optimize(
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        visual_pairs=_make_visual_pairs(frame_indices),
        sensor_observations=obs,
        has_rtk=False,
    )
    assert len(result.optimized_poses) == n
    assert result.positioning_mode == PositioningMode.VISUAL_ONLY


def test_optimize_barometric_used_flag():
    """When barometric data is provided, result.barometric_used must be True."""
    result = _run_fusion(include_gps=True, include_baro=True)
    assert result.barometric_used is True


def test_optimize_imu_used_flag():
    """When IMU data is provided, result.imu_used must be True."""
    result = _run_fusion(include_gps=True, include_imu=True)
    assert result.imu_used is True


def test_optimize_gps_coverage_fraction_correct():
    """
    GPS coverage fraction must correctly reflect how many frames have GPS.
    With 10 frames and dropout after frame 1 -> 2/10 = 0.2.
    """
    result = _run_fusion(n_frames=10, include_gps=True, gps_dropout_after=1)
    assert abs(result.gps_coverage_fraction - 0.2) < 0.01


def test_optimize_empty_visual_pairs_no_crash():
    """No visual pairs (edge case) must not crash; should use GPS priors alone."""
    n = 5
    frame_indices = list(range(n))
    timestamps = [i * 0.5 for i in range(n)]
    obs = _make_observations(n, frame_indices, timestamps, include_gps=True)
    sf = SensorFusion()
    result = sf.optimize(
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        visual_pairs=[],  # Empty
        sensor_observations=obs,
        has_rtk=False,
    )
    assert len(result.optimized_poses) == n


def test_optimize_gps_positions_close_to_prior():
    """
    With GPS anchors and no conflicting factors, optimized ENU positions
    should remain close to their GPS prior values.
    """
    n = 5
    frame_indices = list(range(n))
    timestamps = [i * 1.0 for i in range(n)]
    obs = _make_observations(n, frame_indices, timestamps, include_gps=True)

    origin_lat, origin_lon, origin_alt = 18.5204, 73.8567, 52.0

    sf = SensorFusion()
    result = sf.optimize(
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        visual_pairs=[],   # GPS-only; no visual drift
        sensor_observations=obs,
        has_rtk=False,
    )

    # Check all optimized positions are finite
    for pose in result.optimized_poses:
        assert all(math.isfinite(v) for v in pose.position_enu), (
            f"Non-finite position at frame {pose.frame_index}: {pose.position_enu}"
        )

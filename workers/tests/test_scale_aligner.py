"""
Unit tests for Metric Scale Calibration Against Sensor Baseline — TASK-033.

Covers:
  - Triangulated sparse 3D point cloud scale estimation (least-squares solve).
  - GPS baseline distance scale estimation.
  - Barometric altitude AGL ground clearance scale estimation.
  - Running median filter smoothing across frame sequences.
  - Full ScaleAligner pipeline with 4-level fallback hierarchy:
      SPARSE_3D -> GPS_BASELINE -> BAROMETRIC -> MEDIAN_PROPAGATION.
  - Barometric tolerance check (±5%).
"""
from __future__ import annotations

import numpy as np
import pytest

from workers.depth.depth_estimator import DepthMap
from workers.depth.scale_aligner import (
    FrameScaleFactor,
    ScaleAlignmentResult,
    ScaleAligner,
    ScaleCalibrationMode,
    _scale_from_barometric_altitude,
    _scale_from_gps_baseline,
    _scale_from_sparse_points,
    _smooth_scale_factors,
)
from workers.pose.sensor_fusion import OptimizedPose


@pytest.fixture
def sample_depth_maps():
    """Generates 3 synthetic DepthMaps (360x640) at depth ~10 units."""
    maps = []
    for i in range(3):
        depth = np.full((360, 640), 10.0 + i * 0.5, dtype=np.float32)
        mask = np.ones((360, 640), dtype=bool)
        maps.append(
            DepthMap(
                frame_index=i,
                depth_m=depth,
                valid_mask=mask,
                original_height=360,
                original_width=640,
                min_depth_m=10.0,
                max_depth_m=11.0,
                median_depth_m=10.0 + i * 0.5,
                inference_time_ms=15.0,
            )
        )
    return maps


@pytest.fixture
def sample_poses():
    """Generates 3 camera poses with a 5-metre forward travel distance per frame."""
    poses = []
    for i in range(3):
        poses.append(
            OptimizedPose(
                frame_index=i,
                timestamp_sec=float(i),
                position_enu=np.array([float(i * 5.0), 0.0, 50.0], dtype=np.float64),
                rotation_matrix=np.eye(3, dtype=np.float64),
                sigma_east_m=0.1,
                sigma_north_m=0.1,
                sigma_up_m=0.2,
                pose_confidence=95.0,
            )
        )
    return poses


# ---------------------------------------------------------------------------
# Individual Estimation Tests
# ---------------------------------------------------------------------------


def test_scale_from_sparse_points_recovers_ground_truth_scale():
    """Least-squares scale estimation correctly recovers ground truth scale."""
    gt_scale = 2.5
    h, w = 360, 640
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # 10 random 2D image points and their raw depths
    pts_img = np.array([
        [100, 100], [200, 150], [300, 200], [400, 250], [500, 120],
        [150, 300], [250, 80], [350, 160], [450, 220], [550, 190],
    ], dtype=np.float64)
    raw_depths = np.array([12.0, 14.0, 10.0, 16.0, 15.0, 11.0, 13.0, 17.0, 14.0, 12.0])

    depth_map = np.full((h, w), 10.0, dtype=np.float32)
    for (u, v), d in zip(pts_img, raw_depths):
        depth_map[int(v), int(u)] = float(d)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)

    # Create ground-truth 3D world points scaled by gt_scale
    pts_world = []
    for (u, v), d in zip(pts_img, raw_depths):
        ray = K_inv @ np.array([u, v, 1.0], dtype=np.float64) * d
        p_world = gt_scale * ray
        pts_world.append(p_world)
    pts_world = np.array(pts_world)

    result = _scale_from_sparse_points(
        depth_m=depth_map,
        pts_image=pts_img,
        pts_world_m=pts_world,
        K=K,
        R_cam_to_world=R,
        t_world=t,
    )

    assert result is not None
    estimated_scale, residual, n_pts = result
    assert n_pts == 10
    assert pytest.approx(estimated_scale, rel=1e-3) == gt_scale
    assert residual < 1e-3


def test_scale_from_barometric_altitude():
    """Recovers scale from nadir altitude ground clearance."""
    depth_map = DepthMap(
        frame_index=0,
        depth_m=np.full((360, 640), 20.0, dtype=np.float32),
        valid_mask=np.ones((360, 640), dtype=bool),
        original_height=360,
        original_width=640,
        min_depth_m=20.0,
        max_depth_m=20.0,
        median_depth_m=20.0,
        inference_time_ms=10.0,
    )
    # Altitude 50m, raw nadir depth 20m -> scale should be 50 / 20 = 2.5
    res = _scale_from_barometric_altitude(depth_map, altitude_agl_m=50.0)
    assert res is not None
    s, residual = res
    assert pytest.approx(s, rel=1e-3) == 2.5
    assert residual < 1e-3


def test_smooth_scale_factors_removes_outlier():
    """Median filter suppresses isolated scale spike."""
    raw = [2.0, 2.0, 100.0, 2.0, 2.0]  # Outlier spike at index 2
    smoothed = _smooth_scale_factors(raw, window=3)
    assert smoothed[2] == 2.0  # Spike eliminated by median filter


# ---------------------------------------------------------------------------
# ScaleAligner Pipeline Tests
# ---------------------------------------------------------------------------


def test_scale_aligner_barometric_mode(sample_depth_maps):
    """Calibrates scale using barometric altitude."""
    aligner = ScaleAligner()
    result = aligner.align(
        depth_maps=sample_depth_maps,
        optimized_poses=None,
        altitude_agl_m_per_frame=[50.0, 50.0, 50.0],
    )

    assert isinstance(result, ScaleAlignmentResult)
    assert result.total_frames == 3
    assert len(result.calibrated_depth_maps) == 3
    assert len(result.frame_scale_factors) == 3
    assert result.median_global_scale > 0.0
    assert result.passes_baro_tolerance is True
    assert result.calibration_mode == ScaleCalibrationMode.BAROMETRIC

    # Calibrated depth should be approximately 50m (raw 10m * scale 5.0)
    for cdm in result.calibrated_depth_maps:
        assert pytest.approx(cdm.median_depth_m, rel=0.1) == 50.0


def test_scale_aligner_gps_baseline_mode(sample_depth_maps, sample_poses):
    """Calibrates scale using GPS inter-frame baseline."""
    aligner = ScaleAligner()
    result = aligner.align(
        depth_maps=sample_depth_maps,
        optimized_poses=sample_poses,
        altitude_agl_m_per_frame=None,
    )

    assert isinstance(result, ScaleAlignmentResult)
    assert result.calibration_mode == ScaleCalibrationMode.GPS_BASELINE
    assert result.median_global_scale > 0.0
    assert len(result.calibrated_depth_maps) == 3


def test_scale_aligner_median_propagation_when_no_sensors(sample_depth_maps):
    """When no GPS/altitudes provided, uses default median propagation."""
    aligner = ScaleAligner()
    result = aligner.align(
        depth_maps=sample_depth_maps,
        optimized_poses=None,
        altitude_agl_m_per_frame=None,
    )

    assert result.calibration_mode == ScaleCalibrationMode.MEDIAN_PROPAGATION
    assert result.median_global_scale == 1.0
    assert len(result.calibrated_depth_maps) == 3

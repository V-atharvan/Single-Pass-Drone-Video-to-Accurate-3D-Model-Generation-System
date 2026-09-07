"""
Unit tests for Camera Extrinsics/Intrinsics Solver and Trajectory Exporter — TASK-028.

Definition of Done:
  - cameras.json saved in S3 (or mocked).
  - Every keyframe has a valid 4x4 world-to-camera transformation matrix.
  - Per-camera position uncertainty fields are present and positive.
  - Mean reprojection error is logged as a measured metric (not a guarantee).
  - Accuracy note is present in the payload.

Coverage:
  - Intrinsic matrix construction from provided CameraIntrinsics.
  - Intrinsic matrix estimation from image dimensions (fallback).
  - World-to-camera 4x4 matrix validity (orthogonal R block, correct shape).
  - Camera center world coordinates are valid 3-vectors.
  - Pose confidence is bounded [0, 100].
  - Sigma fields are positive and finite.
  - cameras.json contains all expected fields.
  - S3 upload called with correct key pattern.
  - S3 failure is non-fatal.
  - Reprojection RMSE is a float in the payload (not a guaranteed bound).
  - Accuracy note is present.
"""
from __future__ import annotations

import json
import math
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from packages.schemas.python.single_pass_schemas.telemetry import CameraIntrinsics
from workers.pose.feature_tracker import FramePairMatches
from workers.pose.pose_output import (
    PoseOutputExporter,
    _build_intrinsic_matrix,
    _compute_pose_confidence_from_inliers,
    _REPROJECTION_BENCHMARK_TARGET_PX,
)
from workers.pose.sensor_fusion import (
    OptimizedPose,
    PositioningMode,
    SensorFusionResult,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_fusion_result(n: int = 10, sigma: float = 2.0) -> SensorFusionResult:
    """Build a minimal SensorFusionResult with identity rotations."""
    poses = []
    for i in range(n):
        poses.append(OptimizedPose(
            frame_index=i,
            timestamp_sec=float(i) * 0.5,
            position_enu=np.array([float(i * 2), 0.0, 50.0]),
            rotation_matrix=np.eye(3),
            sigma_east_m=sigma,
            sigma_north_m=sigma,
            sigma_up_m=sigma * 1.5,
            pose_confidence=85.0,
        ))
    return SensorFusionResult(
        optimized_poses=poses,
        positioning_mode=PositioningMode.GPS_ONLY,
        gps_coverage_fraction=1.0,
        rtk_used=False,
        barometric_used=False,
        imu_used=False,
        converged=True,
        final_cost=0.05,
        mean_pose_confidence=85.0,
    )


def _make_visual_pairs(frame_indices: List[int], n_inliers: int = 700) -> List[FramePairMatches]:
    pairs = []
    for i in range(len(frame_indices) - 1):
        pairs.append(FramePairMatches(
            frame_a_index=frame_indices[i],
            frame_b_index=frame_indices[i + 1],
            pts_a=np.zeros((n_inliers, 2), dtype=np.float32),
            pts_b=np.zeros((n_inliers, 2), dtype=np.float32),
            fundamental_matrix=None,
            inlier_count=n_inliers,
            raw_match_count=n_inliers + 100,
            inlier_ratio=n_inliers / (n_inliers + 100),
            is_valid=True,
        ))
    return pairs


def _make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=2000.0, fy=2000.0, cx=1920.0, cy=1080.0, width=3840, height=2160)


# ---------------------------------------------------------------------------
# Intrinsic matrix tests
# ---------------------------------------------------------------------------


def test_build_intrinsic_matrix_from_calibration():
    """Provided CameraIntrinsics must produce exact K matrix values."""
    intr = _make_intrinsics()
    K = _build_intrinsic_matrix(intr, image_width=3840, image_height=2160)
    assert K.shape == (3, 3)
    assert abs(K[0, 0] - 2000.0) < 1e-6   # fx
    assert abs(K[1, 1] - 2000.0) < 1e-6   # fy
    assert abs(K[0, 2] - 1920.0) < 1e-6   # cx
    assert abs(K[1, 2] - 1080.0) < 1e-6   # cy
    assert abs(K[2, 2] - 1.0) < 1e-6


def test_build_intrinsic_matrix_fallback_from_image_size():
    """When intrinsics are None, K must be estimated from image dimensions."""
    K = _build_intrinsic_matrix(None, image_width=1920, image_height=1080)
    assert K.shape == (3, 3)
    # fx estimated from 70 deg FoV
    expected_fx = (1920 / 2.0) / math.tan(math.radians(70.0 / 2.0))
    assert abs(K[0, 0] - expected_fx) < 1.0
    assert abs(K[0, 2] - 960.0) < 1.0   # cx = width/2
    assert abs(K[1, 2] - 540.0) < 1.0   # cy = height/2


# ---------------------------------------------------------------------------
# Pose confidence tests
# ---------------------------------------------------------------------------


def test_pose_confidence_maximum():
    """Maximum inliers + zero GPS residual -> confidence near 100."""
    confidence = _compute_pose_confidence_from_inliers(
        inlier_count=1000,  # = MAX_INLIER_BASELINE
        gps_visual_residual_m=0.0,
    )
    assert confidence == 100.0


def test_pose_confidence_minimum():
    """Zero inliers + large GPS residual -> confidence near 0."""
    confidence = _compute_pose_confidence_from_inliers(
        inlier_count=0,
        gps_visual_residual_m=50.0,
    )
    assert confidence == 0.0


def test_pose_confidence_bounded():
    """Confidence must always stay in [0, 100]."""
    for inliers in [0, 100, 500, 1000, 2000]:
        for residual in [0.0, 5.0, 20.0, 100.0]:
            c = _compute_pose_confidence_from_inliers(inliers, residual)
            assert 0.0 <= c <= 100.0, f"Out of range: inliers={inliers}, residual={residual}, c={c}"


# ---------------------------------------------------------------------------
# PoseOutputExporter output structure tests
# ---------------------------------------------------------------------------


def test_export_produces_correct_camera_count():
    """cameras.json must contain one record per optimized pose."""
    n = 12
    fusion = _make_fusion_result(n=n)
    pairs = _make_visual_pairs(list(range(n)))
    exporter = PoseOutputExporter()
    result = exporter.export("job-001", fusion, pairs)
    assert result.total_cameras == n
    assert len(result.cameras_payload.cameras) == n


def test_export_world_to_camera_is_4x4():
    """World-to-camera matrix in each camera record must be 4x4."""
    fusion = _make_fusion_result(n=5)
    exporter = PoseOutputExporter()
    result = exporter.export("job-002", fusion, [])
    for cam in result.cameras_payload.cameras:
        Wc = cam.world_to_camera
        assert len(Wc) == 4, "world_to_camera must have 4 rows"
        for row in Wc:
            assert len(row) == 4, "Each row must have 4 elements"


def test_export_camera_center_is_3_vector():
    """camera_center_world must be a 3-element list."""
    fusion = _make_fusion_result(n=4)
    exporter = PoseOutputExporter()
    result = exporter.export("job-003", fusion, [])
    for cam in result.cameras_payload.cameras:
        assert len(cam.camera_center_world) == 3


def test_export_world_to_camera_rotation_block_is_orthogonal():
    """
    The top-left 3x3 rotation block of world_to_camera must be orthonormal:
    R @ R^T ~ I, det(R) ~ 1.
    """
    fusion = _make_fusion_result(n=6)
    exporter = PoseOutputExporter()
    result = exporter.export("job-004", fusion, [])
    for cam in result.cameras_payload.cameras:
        Wc = np.array(cam.world_to_camera)
        R = Wc[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), (
            f"Rotation block not orthogonal at frame {cam.frame_index}"
        )
        assert abs(np.linalg.det(R) - 1.0) < 1e-5


def test_export_intrinsic_matrix_shape():
    """K matrix in every camera record must be 3x3."""
    fusion = _make_fusion_result(n=3)
    exporter = PoseOutputExporter()
    result = exporter.export("job-005", fusion, [], intrinsics=_make_intrinsics())
    for cam in result.cameras_payload.cameras:
        K = cam.K
        assert len(K) == 3
        for row in K:
            assert len(row) == 3


def test_export_intrinsic_values_match_calibration():
    """Provided CameraIntrinsics must be reflected exactly in the K matrix field."""
    intr = _make_intrinsics()
    fusion = _make_fusion_result(n=2)
    exporter = PoseOutputExporter()
    result = exporter.export("job-006", fusion, [], intrinsics=intr)
    for cam in result.cameras_payload.cameras:
        assert abs(cam.K[0][0] - 2000.0) < 1e-4   # fx
        assert abs(cam.K[1][1] - 2000.0) < 1e-4   # fy


def test_export_sigma_fields_positive_and_finite():
    """All per-camera sigma fields must be positive and finite."""
    fusion = _make_fusion_result(n=8, sigma=3.0)
    exporter = PoseOutputExporter()
    result = exporter.export("job-007", fusion, [])
    for cam in result.cameras_payload.cameras:
        assert cam.sigma_east_m > 0 and math.isfinite(cam.sigma_east_m)
        assert cam.sigma_north_m > 0 and math.isfinite(cam.sigma_north_m)
        assert cam.sigma_up_m > 0 and math.isfinite(cam.sigma_up_m)


def test_export_pose_confidence_bounded():
    """Pose confidence in every camera record must be in [0, 100]."""
    fusion = _make_fusion_result(n=6)
    exporter = PoseOutputExporter()
    result = exporter.export("job-008", fusion, _make_visual_pairs(list(range(6))))
    for cam in result.cameras_payload.cameras:
        assert 0.0 <= cam.pose_confidence <= 100.0


def test_export_accuracy_note_present():
    """The accuracy_note field must be present and non-empty in the payload."""
    fusion = _make_fusion_result(n=3)
    exporter = PoseOutputExporter()
    result = exporter.export("job-009", fusion, [])
    assert result.cameras_payload.accuracy_note
    assert "measured" in result.cameras_payload.accuracy_note.lower()


def test_export_reprojection_rmse_is_float():
    """mean_reprojection_rmse_px must be a float (not a guarantee string)."""
    fusion = _make_fusion_result(n=4)
    exporter = PoseOutputExporter()
    result = exporter.export("job-010", fusion, [])
    assert isinstance(result.mean_reprojection_rmse_px, float)


def test_export_cameras_json_s3_key_pattern():
    """S3 key must follow the canonical pattern: jobs/{job_id}/interim/cameras/cameras.json"""
    fusion = _make_fusion_result(n=2)
    exporter = PoseOutputExporter()
    result = exporter.export("test-job-xyz", fusion, [])
    assert result.cameras_s3_key == "jobs/test-job-xyz/interim/cameras/cameras.json"


def test_export_s3_upload_called_with_correct_args():
    """When a storage client is provided, upload_bytes must be called with the cameras.json key."""
    mock_storage = MagicMock()
    mock_storage.INTERIM_BUCKET = "single-pass-3d-interim"
    mock_storage.upload_bytes = MagicMock()

    fusion = _make_fusion_result(n=3)
    exporter = PoseOutputExporter(storage_client=mock_storage)
    result = exporter.export("job-s3-test", fusion, [])

    assert mock_storage.upload_bytes.call_count == 1
    call_kwargs = mock_storage.upload_bytes.call_args.kwargs
    assert "cameras.json" in call_kwargs["key"]
    assert call_kwargs["content_type"] == "application/json"
    assert result.upload_ok is True


def test_export_s3_failure_is_non_fatal():
    """S3 upload failure must set upload_ok=False but not raise an exception."""
    mock_storage = MagicMock()
    mock_storage.INTERIM_BUCKET = "single-pass-3d-interim"
    mock_storage.upload_bytes = MagicMock(side_effect=ConnectionError("S3 down"))

    fusion = _make_fusion_result(n=3)
    exporter = PoseOutputExporter(storage_client=mock_storage)
    result = exporter.export("job-s3-fail", fusion, [])

    assert result.upload_ok is False
    assert result.total_cameras == 3   # Output still complete despite upload failure


def test_export_frame_indices_in_cameras_json_match_input():
    """Frame index in each CameraRecord must match the corresponding optimized pose."""
    n = 7
    fusion = _make_fusion_result(n=n)
    exporter = PoseOutputExporter()
    result = exporter.export("job-idx-check", fusion, [])
    for i, cam in enumerate(result.cameras_payload.cameras):
        assert cam.frame_index == i


def test_export_benchmark_target_in_payload():
    """The benchmark target value must be present in the payload."""
    fusion = _make_fusion_result(n=2)
    exporter = PoseOutputExporter()
    result = exporter.export("job-target", fusion, [])
    assert result.cameras_payload.reprojection_benchmark_target_px == _REPROJECTION_BENCHMARK_TARGET_PX

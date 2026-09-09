"""
Unit tests for 3D Point Cloud Fusion, Surface Completion & Outlier Filtering — Phase 7 (TASK-040 to TASK-045).

Covers:
  - TASK-040: Depth unprojection with pinhole intrinsics and 6-DoF extrinsics into world coordinates, dynamic exclusion.
  - TASK-041: Multi-frame spatial voxel hashing, confidence-weighted 3D position averaging, single-view discard.
  - TASK-042: Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR).
  - TASK-043: Voxel downsampling, PCA surface normal estimation, and camera-facing orientation.
  - TASK-044: Canonical 6-state ObservationState categorization and 100% sum verification.
  - TASK-045: Binary PLY and ASPRS LAS 1.2 serialization with classification codes and manifests.
"""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from workers.fusion.las_exporter import (
    ExportedPointCloudFiles,
    PointCloudSerializer,
    encode_las_binary,
    encode_ply_binary,
)
from workers.fusion.normal_estimator import (
    NormalEstimator,
    OrientedPointCloud,
)
from workers.fusion.occlusion_analyzer import (
    ClassifiedObservationCloud,
    ObservationState,
    ObservationStatistics,
    OcclusionAnalyzer,
)
from workers.fusion.outlier_filter import (
    FilterStatistics,
    OutlierFilter,
    OutlierFilterResult,
)
from workers.fusion.point_fusion import (
    FusedPointCloud,
    PointFusionEngine,
)
from workers.fusion.unprojector import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthUnprojector,
    FramePointCloud,
)
from workers.segmentation.semantic_classifier import SemanticClass


@pytest.fixture
def pinhole_intrinsics():
    # 640x480 camera with fx=fy=500, cx=320, cy=240
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


@pytest.fixture
def identity_extrinsics():
    return CameraExtrinsics(R=np.eye(3, dtype=np.float32), t=np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# TASK-040: Depth Unprojection Tests
# ---------------------------------------------------------------------------

def test_unprojector_geometry(pinhole_intrinsics, identity_extrinsics):
    """Verifies camera frame back-projection math and world transformation."""
    unprojector = DepthUnprojector()

    h, w = 480, 640
    depth_m = np.zeros((h, w), dtype=np.float32)
    # Center pixel (cx=320, cy=240) at depth 10.0m
    depth_m[240, 320] = 10.0
    # Top-left pixel (u=120, v=140) at depth 10.0m
    # X_c = (120 - 320) * 10 / 500 = -4.0m
    # Y_c = (140 - 240) * 10 / 500 = -2.0m
    depth_m[140, 120] = 10.0

    cloud = unprojector.unproject_frame(
        depth_m=depth_m,
        intrinsics=pinhole_intrinsics,
        extrinsics=identity_extrinsics,
        frame_index=1,
    )

    assert isinstance(cloud, FramePointCloud)
    assert cloud.point_count == 2
    assert cloud.frame_index == 1

    # Center pixel should have X=0, Y=0, Z=10
    pos_center = cloud.positions[cloud.pixel_coords[:, 0] == 320][0]
    assert pytest.approx(pos_center[0], 0.01) == 0.0
    assert pytest.approx(pos_center[1], 0.01) == 0.0
    assert pytest.approx(pos_center[2], 0.01) == 10.0

    # Top-left pixel: X=-4.0, Y=-2.0, Z=10.0
    pos_tl = cloud.positions[cloud.pixel_coords[:, 0] == 120][0]
    assert pytest.approx(pos_tl[0], 0.01) == -4.0
    assert pytest.approx(pos_tl[1], 0.01) == -2.0
    assert pytest.approx(pos_tl[2], 0.01) == 10.0


def test_unprojector_dynamic_exclusion(pinhole_intrinsics, identity_extrinsics):
    """Dynamic mask pixels (255) must be strictly excluded from 3D back-projection."""
    unprojector = DepthUnprojector()

    h, w = 100, 100
    depth_m = np.full((h, w), 5.0, dtype=np.float32)

    # Dynamic mask: moving vehicle at (20..40, 20..40)
    dyn_mask = np.zeros((h, w), dtype=np.uint8)
    dyn_mask[20:40, 20:40] = 255

    cloud = unprojector.unproject_frame(
        depth_m=depth_m,
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
        extrinsics=identity_extrinsics,
        dynamic_mask=dyn_mask,
    )

    # Total pixels = 10000. Excluded pixels = 20x20 = 400.
    assert cloud.point_count == 9600
    # Confirm no pixel in the excluded region exists
    u_coords = cloud.pixel_coords[:, 0]
    v_coords = cloud.pixel_coords[:, 1]
    in_dyn_region = (u_coords >= 20) & (u_coords < 40) & (v_coords >= 20) & (v_coords < 40)
    assert not np.any(in_dyn_region)


# ---------------------------------------------------------------------------
# TASK-041: Multi-Frame Point Cloud Fusion Tests
# ---------------------------------------------------------------------------

def test_point_fusion_weighted_average():
    """Fuses multiple overlapping observations into confidence-weighted 3D positions."""
    engine = PointFusionEngine(voxel_size=0.10, single_view_confidence_threshold=0.60)

    # Frame 1: point at (1.0, 1.0, 5.0) with confidence 0.9
    fc1 = FramePointCloud(
        frame_index=1,
        positions=np.array([[1.0, 1.0, 5.0]], dtype=np.float32),
        colors=np.array([[200, 0, 0]], dtype=np.uint8),
        confidences=np.array([0.9], dtype=np.float32),
        semantic_classes=np.array([int(SemanticClass.BUILDING)], dtype=np.uint8),
        pixel_coords=np.array([[50, 50]], dtype=np.int32),
        camera_center=np.zeros(3, dtype=np.float32),
    )

    # Frame 2: point in the same voxel at (1.04, 1.02, 5.02) with confidence 0.3
    fc2 = FramePointCloud(
        frame_index=2,
        positions=np.array([[1.04, 1.02, 5.02]], dtype=np.float32),
        colors=np.array([[0, 200, 0]], dtype=np.uint8),
        confidences=np.array([0.3], dtype=np.float32),
        semantic_classes=np.array([int(SemanticClass.BUILDING)], dtype=np.uint8),
        pixel_coords=np.array([[52, 51]], dtype=np.int32),
        camera_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    fused = engine.fuse_frames([fc1, fc2])

    assert isinstance(fused, FusedPointCloud)
    assert fused.fused_points_count == 1
    assert fused.view_counts[0] == 2

    # Weighted position: (1.0 * 0.9 + 1.04 * 0.3) / 1.2 = (0.9 + 0.312) / 1.2 = 1.01
    expected_x = (1.0 * 0.9 + 1.04 * 0.3) / 1.2
    assert pytest.approx(float(fused.positions[0, 0]), 0.001) == expected_x
    assert fused.semantic_classes[0] == int(SemanticClass.BUILDING)


def test_point_fusion_single_view_filter():
    """Discards single-view points with low confidence (< 0.60)."""
    engine = PointFusionEngine(voxel_size=0.10, single_view_confidence_threshold=0.60)

    # Single-view point with confidence 0.40 (< 0.60 threshold)
    fc_low = FramePointCloud(
        frame_index=1,
        positions=np.array([[2.0, 2.0, 8.0]], dtype=np.float32),
        colors=np.array([[100, 100, 100]], dtype=np.uint8),
        confidences=np.array([0.40], dtype=np.float32),
        semantic_classes=np.array([int(SemanticClass.TERRAIN)], dtype=np.uint8),
        pixel_coords=np.array([[10, 10]], dtype=np.int32),
        camera_center=np.zeros(3, dtype=np.float32),
    )

    # Single-view point with confidence 0.85 (>= 0.60 threshold)
    fc_high = FramePointCloud(
        frame_index=1,
        positions=np.array([[4.0, 4.0, 8.0]], dtype=np.float32),
        colors=np.array([[100, 100, 100]], dtype=np.uint8),
        confidences=np.array([0.85], dtype=np.float32),
        semantic_classes=np.array([int(SemanticClass.TERRAIN)], dtype=np.uint8),
        pixel_coords=np.array([[20, 20]], dtype=np.int32),
        camera_center=np.zeros(3, dtype=np.float32),
    )

    fused = engine.fuse_frames([fc_low, fc_high])
    # Low confidence single-view point must be discarded!
    assert fused.fused_points_count == 1
    assert pytest.approx(float(fused.positions[0, 0]), 0.01) == 4.0


# ---------------------------------------------------------------------------
# TASK-042: Outlier Removal Tests
# ---------------------------------------------------------------------------

def test_statistical_and_radius_outlier_removal():
    """Filters out noise points via SOR and isolated points via ROR."""
    # Create a dense planar surface of 100 points
    xs, ys = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    plane_pts = np.column_stack([xs.flatten(), ys.flatten(), np.zeros(100)])

    # Add 2 extreme outlier points far from the plane
    outliers = np.array([[10.0, 10.0, 20.0], [-15.0, -10.0, 30.0]])
    all_pts = np.vstack([plane_pts, outliers]).astype(np.float32)

    cloud = FusedPointCloud(
        positions=all_pts,
        colors=np.full((len(all_pts), 3), 150, dtype=np.uint8),
        confidences=np.ones(len(all_pts), dtype=np.float32),
        semantic_classes=np.ones(len(all_pts), dtype=np.uint8),
        view_counts=np.full(len(all_pts), 2, dtype=np.int32),
        voxel_size=0.05,
        total_input_points=len(all_pts),
        fused_points_count=len(all_pts),
    )

    filter_engine = OutlierFilter(nb_neighbors=10, std_ratio=1.5, radius=0.5, min_neighbors=3)
    res = filter_engine.filter_cloud(cloud)

    assert isinstance(res, OutlierFilterResult)
    assert res.stats.final_points == 100
    assert res.stats.total_outliers_removed == 2
    # Ensure neither outlier point survived
    assert np.all(res.point_cloud.positions[:, 0] <= 2.0)


# ---------------------------------------------------------------------------
# TASK-043: Normal Vector Estimation Tests
# ---------------------------------------------------------------------------

def test_normal_estimator_horizontal_plane():
    """Surface normals of a horizontal XY plane must point toward the camera in +Z."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    pts = np.column_stack([xs.flatten(), ys.flatten(), np.full(225, 5.0)]).astype(np.float32)

    cloud = FusedPointCloud(
        positions=pts,
        colors=np.full((len(pts), 3), 120, dtype=np.uint8),
        confidences=np.ones(len(pts), dtype=np.float32),
        semantic_classes=np.full(len(pts), int(SemanticClass.TERRAIN), dtype=np.uint8),
        view_counts=np.full(len(pts), 3, dtype=np.int32),
        voxel_size=0.05,
        total_input_points=len(pts),
        fused_points_count=len(pts),
    )

    # Camera looking down from (0, 0, 0) towards Z=5
    camera_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    estimator = NormalEstimator(k_neighbors=15)
    oriented = estimator.estimate_normals(cloud, camera_positions=camera_pos)

    assert isinstance(oriented, OrientedPointCloud)
    assert oriented.point_count == 225
    # Unit normal vectors
    norms = np.linalg.norm(oriented.normals, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-4)

    # All normals on this horizontal plane should point toward camera (Z negative, since camera is at Z=0 and points at Z=5)
    # view_vector is (0 - x, 0 - y, 0 - 5) = -Z direction, so normal.Z should be negative
    assert np.all(oriented.normals[:, 2] < -0.9)


# ---------------------------------------------------------------------------
# TASK-044: Occlusion Analysis & 6-State Categorization Tests
# ---------------------------------------------------------------------------

def test_occlusion_analyzer_6_states():
    """Validates categorization into the full 6-state ObservationState and 100% sum rule."""
    n = 100
    pts = np.zeros((n, 3), dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0

    # 40 points: multi-view (views=3, conf=0.9) -> OBSERVED
    # 20 points: single-view (views=1, conf=0.9) -> PARTIAL
    # 20 points: low confidence (views=2, conf=0.15) -> LOW_CONFIDENCE
    # 20 points: inferred indices -> INFERRED
    conf = np.full(n, 0.9, dtype=np.float32)
    views = np.full(n, 3, dtype=np.int32)

    views[40:60] = 1       # PARTIAL
    conf[60:80] = 0.15     # LOW_CONFIDENCE

    oriented = OrientedPointCloud(
        positions=pts,
        normals=normals,
        curvatures=np.zeros(n, dtype=np.float32),
        colors=np.zeros((n, 3), dtype=np.uint8),
        confidences=conf,
        semantic_classes=np.zeros(n, dtype=np.uint8),
        view_counts=views,
    )

    analyzer = OcclusionAnalyzer(low_confidence_threshold=0.30, min_multi_view_count=2)
    res = analyzer.analyze_cloud(
        cloud=oriented,
        dynamic_excluded_count=50,  # 50 dynamic excluded points
        unknown_region_points=50,   # 50 unobserved unknown points
        inferred_indices=list(range(80, 100)), # 20 inferred points
    )

    assert isinstance(res, ClassifiedObservationCloud)
    assert res.point_count == 100

    # Verify presence of all 6 observation states in distribution
    stats = res.stats
    assert stats.verify_sum() is True
    assert stats.total_points_evaluated == 200  # 100 in cloud + 50 dyn + 50 unk

    assert pytest.approx(stats.observed_pct) == (40 / 200) * 100.0
    assert pytest.approx(stats.partial_pct) == (20 / 200) * 100.0
    assert pytest.approx(stats.low_confidence_pct) == (20 / 200) * 100.0
    assert pytest.approx(stats.inferred_pct) == (20 / 200) * 100.0
    assert pytest.approx(stats.dynamic_excluded_pct) == (50 / 200) * 100.0
    assert pytest.approx(stats.unknown_pct) == (50 / 200) * 100.0


# ---------------------------------------------------------------------------
# TASK-045: LAS/LAZ and PLY Serialization Tests
# ---------------------------------------------------------------------------

def test_ply_and_las_export():
    """Validates binary PLY and ASPRS LAS 1.2 generation with classifications and manifest."""
    n = 10
    pts = np.linspace(0, 10, n * 3).reshape(n, 3).astype(np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0

    oriented = OrientedPointCloud(
        positions=pts,
        normals=normals,
        curvatures=np.zeros(n, dtype=np.float32),
        colors=np.full((n, 3), 180, dtype=np.uint8),
        confidences=np.full(n, 0.88, dtype=np.float32),
        semantic_classes=np.full(n, int(SemanticClass.BUILDING), dtype=np.uint8),
        view_counts=np.full(n, 3, dtype=np.int32),
    )

    analyzer = OcclusionAnalyzer()
    cloud = analyzer.analyze_cloud(oriented)

    with tempfile.TemporaryDirectory() as tmpdir:
        serializer = PointCloudSerializer(output_dir=tmpdir)
        exported = serializer.export(cloud, base_filename="survey_corridor", job_id="job_p7_test")

        assert isinstance(exported, ExportedPointCloudFiles)
        assert exported.total_points == 10

        # Check PLY file
        ply_bytes = Path(exported.ply_path).read_bytes()
        assert ply_bytes.startswith(b"ply\nformat binary_little_endian 1.0\n")
        assert b"property uchar observation_state\n" in ply_bytes
        assert len(exported.ply_sha256) == 64

        # Check LAS 1.2 file
        las_bytes = Path(exported.las_path).read_bytes()
        assert las_bytes.startswith(b"LASF")
        # Version 1.2 check at offsets 24 and 25
        assert las_bytes[24] == 1
        assert las_bytes[25] == 2
        # Point Format 3 check at offset 104
        assert las_bytes[104] == 3
        # Point record count check (10 points)
        rec_count = struct.unpack_from("<I", las_bytes, 107)[0]
        assert rec_count == 10

        # Check manifest JSON
        manifest_path = Path(exported.manifest_path)
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["total_points"] == 10
        assert manifest["files"]["las"]["asprs_version"] == "1.2"
        assert manifest["observation_statistics"]["observed_pct"] == 100.0

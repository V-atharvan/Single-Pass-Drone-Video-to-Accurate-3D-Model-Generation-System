"""
Unit tests for Trajectory Visualization GeoJSON / CZML Generator — TASK-029.

Definition of Done:
  - Valid CZML file is produced that parses cleanly against the CZML schema.
  - Valid GeoJSON FeatureCollection is produced with correct structure.
  - trajectory.czml and trajectory.geojson are uploaded to S3 with correct key patterns.

Coverage:
  - CZML document header (id="document", version="1.0", clock interval).
  - CZML flight path polyline packet (cartographicDegrees, Electric Cyan color).
  - CZML per-keyframe orientation packets (quaternion, properties).
  - GeoJSON FeatureCollection with LineString and Point features.
  - GeoJSON bounding box correctness.
  - Electric Cyan color (#00F0FF = RGBA [0, 240, 255, 255]) in polyline.
  - S3 key patterns for CZML and GeoJSON.
  - S3 upload called with correct content types.
  - S3 upload failure is non-fatal.
  - Empty pose list produces minimal valid CZML and GeoJSON.
  - Quaternion output is 4-element list (Cesium [x, y, z, w]).
  - ISO8601 timestamp generation.
  - Coordinate conversion (ENU -> geodetic) produces valid WGS84 ranges.
"""
from __future__ import annotations

import json
import math
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from workers.pose.czml_generator import (
    CZMLBuilder,
    GeoJSONBuilder,
    TrajectoryGenerator,
    _iso8601_from_offset,
    _rotation_to_quaternion,
    _TRAJECTORY_COLOR_RGBA,
)
from workers.pose.sensor_fusion import (
    OptimizedPose,
    PositioningMode,
    SensorFusionResult,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_ORIGIN_LAT = 18.5204
_ORIGIN_LON = 73.8567
_ORIGIN_ALT = 50.0


def _make_poses(n: int = 8, speed_mps: float = 5.0) -> List[OptimizedPose]:
    """Generate synthetic poses flying north at speed_mps."""
    poses = []
    for i in range(n):
        ts = float(i) * 0.5
        # ENU: East stays 0, North increases with speed
        e, nn, u = 0.0, speed_mps * ts, 20.0
        poses.append(OptimizedPose(
            frame_index=i,
            timestamp_sec=ts,
            position_enu=np.array([e, nn, u]),
            rotation_matrix=np.eye(3),
            sigma_east_m=2.0,
            sigma_north_m=2.0,
            sigma_up_m=3.0,
            pose_confidence=85.0,
        ))
    return poses


def _make_fusion_result(n: int = 8) -> SensorFusionResult:
    return SensorFusionResult(
        optimized_poses=_make_poses(n),
        positioning_mode=PositioningMode.GPS_ONLY,
        gps_coverage_fraction=1.0,
        rtk_used=False,
        barometric_used=False,
        imu_used=False,
        converged=True,
        final_cost=0.1,
        mean_pose_confidence=85.0,
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_iso8601_zero_offset():
    """Offset 0.0 must produce 2026-01-01T00:00:00.000Z."""
    ts = _iso8601_from_offset(0.0)
    assert ts.startswith("2026-01-01T00:00:00")


def test_iso8601_one_minute():
    """Offset 60.0 should produce T00:01:00.000Z."""
    ts = _iso8601_from_offset(60.0)
    assert "00:01:00" in ts


def test_iso8601_large_offset():
    """Offset of 3661.5 sec = 1h 1m 1.5s."""
    ts = _iso8601_from_offset(3661.5)
    assert "01:01:01" in ts


def test_rotation_to_quaternion_identity():
    """Identity rotation must produce quaternion [0, 0, 0, 1] (x, y, z, w)."""
    q = _rotation_to_quaternion(np.eye(3))
    assert len(q) == 4
    # w component (index 3) should be 1.0 for identity
    assert abs(q[3] - 1.0) < 1e-6
    assert abs(q[0]) < 1e-6  # x
    assert abs(q[1]) < 1e-6  # y
    assert abs(q[2]) < 1e-6  # z


def test_rotation_to_quaternion_unit_norm():
    """Quaternion produced from any valid rotation must have unit norm."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("ZYX", [45.0, -10.0, 5.0], degrees=True).as_matrix()
    q = _rotation_to_quaternion(R)
    norm = math.sqrt(sum(v ** 2 for v in q))
    assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# CZML builder tests
# ---------------------------------------------------------------------------


def test_czml_document_header_packet():
    """First CZML packet must be the document header with id='document'."""
    builder = CZMLBuilder()
    poses = _make_poses(5)
    packets = builder.build("job-czml-01", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    assert len(packets) >= 1
    doc = packets[0]
    assert doc["id"] == "document"
    assert doc["version"] == "1.0"
    assert "clock" in doc


def test_czml_clock_interval_format():
    """CZML clock interval must be in 'start/stop' ISO8601 format."""
    builder = CZMLBuilder()
    poses = _make_poses(3)
    packets = builder.build("job-czml-02", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    clock = packets[0]["clock"]
    assert "/" in clock["interval"]
    parts = clock["interval"].split("/")
    assert len(parts) == 2
    assert parts[0].endswith("Z")
    assert parts[1].endswith("Z")


def test_czml_flight_path_polyline_packet():
    """Second CZML packet must be the flight path polyline."""
    builder = CZMLBuilder()
    poses = _make_poses(5)
    packets = builder.build("job-czml-03", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    path_packet = packets[1]
    assert "polyline" in path_packet
    poly = path_packet["polyline"]
    assert "positions" in poly
    assert "cartographicDegrees" in poly["positions"]
    # 5 poses -> 5 * 3 = 15 values
    assert len(poly["positions"]["cartographicDegrees"]) == 15


def test_czml_flight_path_electric_cyan_color():
    """Polyline material must use Electric Cyan RGBA [0, 240, 255, 255]."""
    builder = CZMLBuilder()
    poses = _make_poses(4)
    packets = builder.build("job-czml-04", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    poly = packets[1]["polyline"]
    rgba = poly["material"]["solidColor"]["color"]["rgba"]
    assert rgba == _TRAJECTORY_COLOR_RGBA
    assert rgba[0] == 0     # R
    assert rgba[1] == 240   # G
    assert rgba[2] == 255   # B
    assert rgba[3] == 255   # A (fully opaque)


def test_czml_per_keyframe_packets():
    """Must produce one orientation packet per keyframe (after doc + path = N+2 total)."""
    n = 6
    builder = CZMLBuilder()
    poses = _make_poses(n)
    packets = builder.build("job-czml-05", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    # 1 doc + 1 path + n keyframes
    assert len(packets) == n + 2


def test_czml_keyframe_packet_has_orientation():
    """Each keyframe packet must contain orientation with unitQuaternion [x, y, z, w]."""
    builder = CZMLBuilder()
    poses = _make_poses(3)
    packets = builder.build("job-czml-06", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    for packet in packets[2:]:  # Skip doc + path
        assert "orientation" in packet
        q = packet["orientation"]["unitQuaternion"]
        assert len(q) == 4
        norm = math.sqrt(sum(v ** 2 for v in q))
        assert abs(norm - 1.0) < 1e-5


def test_czml_keyframe_packet_has_position():
    """Each keyframe packet must contain a cartographicDegrees position [lon, lat, alt]."""
    builder = CZMLBuilder()
    poses = _make_poses(4)
    packets = builder.build("job-czml-07", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    for packet in packets[2:]:
        assert "position" in packet
        pos = packet["position"]["cartographicDegrees"]
        assert len(pos) == 3
        lon, lat, alt = pos
        assert -180.0 <= lon <= 180.0
        assert -90.0 <= lat <= 90.0
        assert alt >= 0.0


def test_czml_keyframe_has_pose_properties():
    """Each keyframe packet must carry frame_index and pose_confidence in properties."""
    builder = CZMLBuilder()
    poses = _make_poses(3)
    packets = builder.build("job-czml-08", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    for i, packet in enumerate(packets[2:]):
        props = packet.get("properties", {})
        assert "frame_index" in props
        assert "pose_confidence" in props
        assert props["frame_index"] == i


def test_czml_empty_poses_produces_minimal_document():
    """An empty pose list must produce a minimal CZML document without crashing."""
    builder = CZMLBuilder()
    packets = builder.build("job-czml-empty", [], _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    assert len(packets) >= 1
    assert packets[0]["id"] == "document"


def test_czml_is_json_serializable():
    """CZML output must be fully JSON-serializable without any non-serializable types."""
    builder = CZMLBuilder()
    poses = _make_poses(5)
    packets = builder.build("job-czml-json", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    # Should not raise
    serialized = json.dumps(packets)
    assert len(serialized) > 0


# ---------------------------------------------------------------------------
# GeoJSON builder tests
# ---------------------------------------------------------------------------


def test_geojson_feature_collection_type():
    """GeoJSON output must be a FeatureCollection."""
    builder = GeoJSONBuilder()
    poses = _make_poses(5)
    doc = builder.build("job-geo-01", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    assert doc["type"] == "FeatureCollection"


def test_geojson_contains_linestring_feature():
    """GeoJSON must contain a LineString feature for the flight path."""
    builder = GeoJSONBuilder()
    poses = _make_poses(5)
    doc = builder.build("job-geo-02", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    features = doc["features"]
    line_features = [f for f in features if f["geometry"]["type"] == "LineString"]
    assert len(line_features) == 1
    coords = line_features[0]["geometry"]["coordinates"]
    assert len(coords) == 5  # 5 poses


def test_geojson_linestring_has_trajectory_color():
    """LineString properties must declare the trajectory color as Electric Cyan."""
    builder = GeoJSONBuilder()
    poses = _make_poses(3)
    doc = builder.build("job-geo-03", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    line = [f for f in doc["features"] if f["geometry"]["type"] == "LineString"][0]
    assert line["properties"]["color"] == "#00F0FF"


def test_geojson_contains_point_features():
    """GeoJSON must contain one Point feature per keyframe."""
    n = 7
    builder = GeoJSONBuilder()
    poses = _make_poses(n)
    doc = builder.build("job-geo-04", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    point_features = [f for f in doc["features"] if f["geometry"]["type"] == "Point"]
    assert len(point_features) == n


def test_geojson_point_coordinates_are_valid_wgs84():
    """Point feature coordinates must be [lon, lat, alt] in valid WGS84 ranges."""
    builder = GeoJSONBuilder()
    poses = _make_poses(5)
    doc = builder.build("job-geo-05", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    for f in doc["features"]:
        if f["geometry"]["type"] == "Point":
            lon, lat, alt = f["geometry"]["coordinates"]
            assert -180.0 <= lon <= 180.0
            assert -90.0 <= lat <= 90.0


def test_geojson_bounding_box_present():
    """GeoJSON must include a bbox array [min_lon, min_lat, max_lon, max_lat]."""
    builder = GeoJSONBuilder()
    poses = _make_poses(5)
    doc = builder.build("job-geo-06", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    assert "bbox" in doc
    assert len(doc["bbox"]) == 4
    min_lon, min_lat, max_lon, max_lat = doc["bbox"]
    assert min_lon <= max_lon
    assert min_lat <= max_lat


def test_geojson_empty_poses_produces_empty_collection():
    """Empty pose list must produce an empty FeatureCollection without crashing."""
    builder = GeoJSONBuilder()
    doc = builder.build("job-geo-empty", [], _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "VISUAL_ONLY")
    assert doc["type"] == "FeatureCollection"
    assert doc["features"] == []


def test_geojson_is_json_serializable():
    """GeoJSON output must be fully JSON-serializable."""
    builder = GeoJSONBuilder()
    poses = _make_poses(4)
    doc = builder.build("job-geo-json", poses, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT, "GPS_ONLY")
    serialized = json.dumps(doc)
    assert len(serialized) > 0


# ---------------------------------------------------------------------------
# TrajectoryGenerator orchestration tests
# ---------------------------------------------------------------------------


def test_trajectory_generator_s3_key_patterns():
    """CZML and GeoJSON S3 keys must follow canonical path structure."""
    fusion = _make_fusion_result()
    gen = TrajectoryGenerator()
    result = gen.generate("my-job-123", fusion, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    assert result.czml_s3_key == "jobs/my-job-123/interim/trajectory/trajectory.czml"
    assert result.geojson_s3_key == "jobs/my-job-123/interim/trajectory/trajectory.geojson"


def test_trajectory_generator_waypoint_count():
    """total_waypoints in result must equal number of optimized poses."""
    n = 9
    fusion = _make_fusion_result(n=n)
    gen = TrajectoryGenerator()
    result = gen.generate("job-wc", fusion, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    assert result.total_waypoints == n


def test_trajectory_generator_s3_uploads_both_files():
    """Both CZML and GeoJSON must be uploaded to S3 with correct content types."""
    mock_storage = MagicMock()
    mock_storage.INTERIM_BUCKET = "single-pass-3d-interim"
    mock_storage.upload_bytes = MagicMock()

    fusion = _make_fusion_result(n=4)
    gen = TrajectoryGenerator(storage_client=mock_storage)
    result = gen.generate("job-s3-upload", fusion, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    assert mock_storage.upload_bytes.call_count == 2
    call_args_list = mock_storage.upload_bytes.call_args_list

    content_types = {call.kwargs["content_type"] for call in call_args_list}
    assert "application/json" in content_types
    assert "application/geo+json" in content_types

    assert result.czml_upload_ok is True
    assert result.geojson_upload_ok is True


def test_trajectory_generator_s3_failure_is_non_fatal():
    """S3 upload failure must set upload_ok=False without raising exceptions."""
    mock_storage = MagicMock()
    mock_storage.INTERIM_BUCKET = "single-pass-3d-interim"
    mock_storage.upload_bytes = MagicMock(side_effect=ConnectionError("S3 down"))

    fusion = _make_fusion_result(n=3)
    gen = TrajectoryGenerator(storage_client=mock_storage)
    result = gen.generate("job-s3-fail", fusion, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)

    assert result.czml_upload_ok is False
    assert result.geojson_upload_ok is False
    assert result.total_waypoints == 3  # Result still populated


def test_trajectory_generator_bounding_box_is_4_elements():
    """Bounding box in result must be a 4-element list."""
    fusion = _make_fusion_result(n=5)
    gen = TrajectoryGenerator()
    result = gen.generate("job-bbox", fusion, _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT)
    assert len(result.bounding_box) == 4
    min_lon, min_lat, max_lon, max_lat = result.bounding_box
    assert min_lon <= max_lon
    assert min_lat <= max_lat

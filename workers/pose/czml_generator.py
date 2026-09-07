"""
Trajectory Visualization GeoJSON / CZML Generator — TASK-029.

Converts the optimized 3D camera trajectory into two formats for frontend rendering:

  1. CZML (Cesium Markup Language) — time-indexed flight path with:
     - Camera positions as a polyline drawn in Electric Cyan (#00F0FF).
     - Camera frustum orientations at every keyframe position.
     - Timestamped waypoints consumable by the Cesium timeline.

  2. GeoJSON — spatial feature collection with:
     - LineString of camera positions (WGS84).
     - Point features for each keyframe with pose metadata.

Both files are uploaded to S3 interim storage and their S3 keys returned.

CZML schema references:
  https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide

Color palette: Electric Cyan #00F0FF per the design specification.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.shared.python.storage import S3StorageClient
from workers.pose.sensor_fusion import OptimizedPose, SensorFusionResult, _enu_to_geodetic

logger = logging.getLogger("pose.czml_generator")

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

# Electric Cyan per design specification (RGBA 0-255)
_TRAJECTORY_COLOR_RGBA = [0, 240, 255, 255]       # #00F0FF, full opacity
_FRUSTUM_COLOR_RGBA = [0, 240, 255, 180]           # Slightly transparent for frustums
_WAYPOINT_COLOR_RGBA = [255, 220, 0, 255]          # Amber waypoint markers

# Frustum half-angles (radians) for camera cone visualisation
_FRUSTUM_HALF_ANGLE_RAD = math.radians(35.0)      # ~70 deg horizontal FoV / 2

# Path width in Cesium (pixels)
_PATH_WIDTH_PX = 3.0

# Default ENU origin for fallback when no GPS (New York City)
_DEFAULT_ORIGIN = (40.7128, -74.0060, 0.0)


# ---------------------------------------------------------------------------
# Output data contracts
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryExportResult:
    """Paths and metadata for the exported trajectory files."""

    job_id: str
    czml_s3_key: str
    geojson_s3_key: str
    czml_upload_ok: bool
    geojson_upload_ok: bool
    total_waypoints: int
    bounding_box: List[float]  # [min_lon, min_lat, max_lon, max_lat]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _rotation_to_quaternion(R: np.ndarray) -> List[float]:
    """
    Convert a 3x3 rotation matrix to a Cesium-compatible [X, Y, Z, W] quaternion.
    Cesium uses (x, y, z, w) ordering.
    """
    rot = Rotation.from_matrix(R)
    # scipy returns [x, y, z, w] by default
    q = rot.as_quat()
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _iso8601_from_offset(offset_sec: float, epoch: str = "2026-01-01T00:00:00Z") -> str:
    """
    Convert a video timestamp offset (seconds) to an ISO8601 datetime string
    relative to a fixed epoch. Used for CZML timeline compatibility.
    """
    h = int(offset_sec // 3600)
    m = int((offset_sec % 3600) // 60)
    s = offset_sec % 60
    return f"2026-01-01T{h:02d}:{m:02d}:{s:06.3f}Z"


# ---------------------------------------------------------------------------
# CZML Builder
# ---------------------------------------------------------------------------


class CZMLBuilder:
    """
    Constructs a valid CZML document for the optimized camera trajectory.

    CZML structure:
        Packet 0: document header (name, clock, version)
        Packet 1: flight path polyline
        Packets 2..N+1: per-keyframe camera orientation frustums
    """

    CZML_VERSION = "1.0"
    DOCUMENT_NAME = "Single-Pass 3D Reconstruction — Camera Trajectory"

    def build(
        self,
        job_id: str,
        poses: List[OptimizedPose],
        origin_lat: float,
        origin_lon: float,
        origin_alt: float,
    ) -> List[Dict[str, Any]]:
        """
        Build the full CZML packet list.

        Parameters
        ----------
        job_id:
            Reconstruction job UUID string (used for packet IDs).
        poses:
            Ordered list of optimized camera poses from TASK-027.
        origin_lat, origin_lon, origin_alt:
            WGS84 ENU origin for converting ENU positions to geodetic.

        Returns
        -------
        CZML packet list (list of dicts) ready for JSON serialisation.
        """
        if not poses:
            return self._empty_document()

        # Convert ENU positions to WGS84
        geodetic_positions: List[tuple[float, float, float]] = []
        for pose in poses:
            e, n, u = pose.position_enu
            lat, lon, alt = _enu_to_geodetic(e, n, u, origin_lat, origin_lon, origin_alt)
            geodetic_positions.append((lat, lon, alt))

        # Build timestamps
        timestamps = [_iso8601_from_offset(p.timestamp_sec) for p in poses]
        start_time = timestamps[0]
        stop_time = timestamps[-1]

        packets: List[Dict[str, Any]] = []

        # Packet 0: Document header
        packets.append({
            "id": "document",
            "name": self.DOCUMENT_NAME,
            "version": self.CZML_VERSION,
            "clock": {
                "interval": f"{start_time}/{stop_time}",
                "currentTime": start_time,
                "multiplier": 1,
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER",
            },
        })

        # Packet 1: Flight path polyline
        # Positions in Cartographic Degrees: [lon, lat, alt, lon, lat, alt, ...]
        cart_flat: List[float] = []
        for lat, lon, alt in geodetic_positions:
            cart_flat.extend([lon, lat, alt])

        packets.append({
            "id": f"{job_id}_path",
            "name": "Camera Flight Path",
            "availability": f"{start_time}/{stop_time}",
            "polyline": {
                "positions": {
                    "cartographicDegrees": cart_flat,
                },
                "material": {
                    "solidColor": {
                        "color": {"rgba": _TRAJECTORY_COLOR_RGBA},
                    },
                },
                "width": _PATH_WIDTH_PX,
                "clampToGround": False,
                "classificationType": "BOTH",
            },
        })

        # Packets 2..N+1: Per-keyframe camera frustum orientation waypoints
        for i, (pose, (lat, lon, alt), ts) in enumerate(zip(poses, geodetic_positions, timestamps)):
            q = _rotation_to_quaternion(pose.rotation_matrix)
            packets.append({
                "id": f"{job_id}_cam_{pose.frame_index:05d}",
                "name": f"Keyframe {pose.frame_index}",
                "availability": f"{ts}/{stop_time}",
                "position": {
                    "cartographicDegrees": [lon, lat, alt],
                },
                "orientation": {
                    "unitQuaternion": q,  # [x, y, z, w]
                },
                "properties": {
                    "frame_index": pose.frame_index,
                    "timestamp_sec": pose.timestamp_sec,
                    "pose_confidence": pose.pose_confidence,
                    "sigma_east_m": pose.sigma_east_m,
                    "sigma_north_m": pose.sigma_north_m,
                    "sigma_up_m": pose.sigma_up_m,
                },
            })

        return packets

    @staticmethod
    def _empty_document() -> List[Dict[str, Any]]:
        return [{"id": "document", "name": "Empty Trajectory", "version": "1.0"}]


# ---------------------------------------------------------------------------
# GeoJSON Builder
# ---------------------------------------------------------------------------


class GeoJSONBuilder:
    """
    Constructs a GeoJSON FeatureCollection from the optimized trajectory.

    Features:
      - Feature 0: LineString of camera positions (WGS84, 3D).
      - Features 1..N: Point features per keyframe with pose metadata.
    """

    def build(
        self,
        job_id: str,
        poses: List[OptimizedPose],
        origin_lat: float,
        origin_lon: float,
        origin_alt: float,
        positioning_mode: str,
    ) -> Dict[str, Any]:
        """Build the GeoJSON FeatureCollection."""
        if not poses:
            return {"type": "FeatureCollection", "features": []}

        geodetic_positions: List[tuple[float, float, float]] = []
        for pose in poses:
            e, n, u = pose.position_enu
            lat, lon, alt = _enu_to_geodetic(e, n, u, origin_lat, origin_lon, origin_alt)
            geodetic_positions.append((lat, lon, alt))

        features: List[Dict[str, Any]] = []

        # LineString — flight path
        line_coords = [[lon, lat, alt] for lat, lon, alt in geodetic_positions]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line_coords,
            },
            "properties": {
                "job_id": job_id,
                "positioning_mode": positioning_mode,
                "total_waypoints": len(poses),
                "color": "#00F0FF",
            },
        })

        # Point features per keyframe
        for pose, (lat, lon, alt) in zip(poses, geodetic_positions):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat, alt],
                },
                "properties": {
                    "frame_index": pose.frame_index,
                    "timestamp_sec": pose.timestamp_sec,
                    "pose_confidence": pose.pose_confidence,
                    "sigma_east_m": pose.sigma_east_m,
                    "sigma_north_m": pose.sigma_north_m,
                    "sigma_up_m": pose.sigma_up_m,
                },
            })

        # Compute bounding box [min_lon, min_lat, max_lon, max_lat]
        lons = [lon for _, lon, _ in geodetic_positions]
        lats = [lat for lat, _, _ in geodetic_positions]
        bbox = [min(lons), min(lats), max(lons), max(lats)]

        return {
            "type": "FeatureCollection",
            "bbox": bbox,
            "features": features,
        }


# ---------------------------------------------------------------------------
# Main CZML generator orchestrator
# ---------------------------------------------------------------------------


class TrajectoryGenerator:
    """
    Orchestrates CZML and GeoJSON generation and S3 upload for a trajectory.
    """

    def __init__(self, storage_client: Optional[S3StorageClient] = None) -> None:
        self._storage = storage_client
        self._czml_builder = CZMLBuilder()
        self._geojson_builder = GeoJSONBuilder()

    def generate(
        self,
        job_id: Union[str, UUID],
        fusion_result: SensorFusionResult,
        origin_lat: float = _DEFAULT_ORIGIN[0],
        origin_lon: float = _DEFAULT_ORIGIN[1],
        origin_alt: float = _DEFAULT_ORIGIN[2],
    ) -> TrajectoryExportResult:
        """
        Generate trajectory.czml and trajectory.geojson, upload to S3.

        Parameters
        ----------
        job_id:
            Reconstruction job UUID.
        fusion_result:
            SensorFusionResult from TASK-027.
        origin_lat, origin_lon, origin_alt:
            WGS84 ENU origin (first GPS fix or default).

        Returns
        -------
        TrajectoryExportResult with S3 keys and upload status.
        """
        job_id_str = str(job_id)
        poses = fusion_result.optimized_poses

        # Build CZML
        czml_packets = self._czml_builder.build(
            job_id=job_id_str,
            poses=poses,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_alt=origin_alt,
        )

        # Build GeoJSON
        geojson_doc = self._geojson_builder.build(
            job_id=job_id_str,
            poses=poses,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_alt=origin_alt,
            positioning_mode=fusion_result.positioning_mode.value,
        )

        # Compute bounding box from GeoJSON (may be empty for no-GPS visual-only)
        bbox = geojson_doc.get("bbox", [0.0, 0.0, 0.0, 0.0])

        # S3 keys
        czml_key = f"jobs/{job_id_str}/interim/trajectory/trajectory.czml"
        geojson_key = f"jobs/{job_id_str}/interim/trajectory/trajectory.geojson"

        # Upload
        czml_ok = self._upload_json(job_id_str, czml_key, czml_packets, "application/json")
        geojson_ok = self._upload_json(job_id_str, geojson_key, geojson_doc, "application/geo+json")

        logger.info(
            "[%s] Trajectory export: waypoints=%d  CZML=%s  GeoJSON=%s  bbox=%s",
            job_id_str, len(poses),
            "OK" if czml_ok else "FAILED",
            "OK" if geojson_ok else "FAILED",
            bbox,
        )

        return TrajectoryExportResult(
            job_id=job_id_str,
            czml_s3_key=czml_key,
            geojson_s3_key=geojson_key,
            czml_upload_ok=czml_ok,
            geojson_upload_ok=geojson_ok,
            total_waypoints=len(poses),
            bounding_box=bbox,
        )

    def _upload_json(
        self,
        job_id: str,
        s3_key: str,
        payload: Any,
        content_type: str,
    ) -> bool:
        """Serialize payload to JSON and upload to S3. Returns True on success."""
        if self._storage is None:
            logger.debug("[%s] No storage client; skipping upload for %s", job_id, s3_key)
            return True

        try:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self._storage.upload_bytes(
                bucket=S3StorageClient.INTERIM_BUCKET,
                key=s3_key,
                data=data,
                content_type=content_type,
            )
            logger.info("[%s] Uploaded %s (%d bytes)", job_id, s3_key, len(data))
            return True
        except Exception:
            logger.exception("[%s] Failed to upload %s", job_id, s3_key)
            return False

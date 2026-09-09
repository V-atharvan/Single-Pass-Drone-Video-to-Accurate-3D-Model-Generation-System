"""
LAS/LAZ and PLY Point Cloud Serializer — TASK-045.

Exports georeferenced, classified 3D point clouds into standard GIS formats:
  1. Standard ASPRS LAS 1.2 / 1.4 Binary Point Cloud (`pointcloud.las`).
     - Standard ASPRS classifications:
         Class 2: Ground / Terrain
         Class 5: High Vegetation
         Class 6: Building
         Class 9: Water
         Class 11: Road Surface
         Class 17: Utility Infrastructure
     - Confidence encoded in 16-bit Intensity [0..65535].
     - ObservationState (0..5) stored in User Data.
     - 16-bit RGB color channels.
  2. Standard Binary PLY (`pointcloud.ply`) with positions, normals, colors,
     confidence, semantic class ID, and observation state.
  3. Manifest and S3 upload coordination (`pointcloud_manifest.json`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.fusion.occlusion_analyzer import (
    ClassifiedObservationCloud,
    ObservationState,
)
from workers.segmentation.semantic_classifier import SemanticClass

logger = logging.getLogger("fusion.las_exporter")

# ASPRS Standard LAS Classification mapping
_ASPRS_CLASS_MAP: Dict[SemanticClass, int] = {
    SemanticClass.UNKNOWN: 1,                # Unclassified
    SemanticClass.BUILDING: 6,               # Building
    SemanticClass.ROAD: 11,                  # Road surface
    SemanticClass.TERRAIN: 2,                # Ground
    SemanticClass.VEGETATION: 5,             # High Vegetation
    SemanticClass.VEHICLE: 13,               # Vehicle
    SemanticClass.PERSON: 14,                # Pedestrian
    SemanticClass.UTILITY_INFRASTRUCTURE: 17,# Infrastructure / Bridge
    SemanticClass.WATER: 9,                  # Water
    SemanticClass.ANIMAL: 1,                 # Unclassified
    SemanticClass.DYNAMIC_OBJECT: 1,         # Unclassified
}


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_ply_binary(cloud: ClassifiedObservationCloud) -> bytes:
    """
    Encodes point cloud to standard binary little-endian PLY format.
    """
    oc = cloud.oriented_cloud
    n = oc.point_count

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Single-Pass 3D Reconstruction Platform\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float confidence\n"
        "property uchar semantic_class\n"
        "property uchar observation_state\n"
        "end_header\n"
    ).encode("ascii")

    if n == 0:
        return header

    # Define structured array matching PLY properties
    ply_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("confidence", "<f4"),
        ("semantic_class", "u1"),
        ("observation_state", "u1"),
    ])

    data = np.empty(n, dtype=ply_dtype)
    data["x"] = oc.positions[:, 0]
    data["y"] = oc.positions[:, 1]
    data["z"] = oc.positions[:, 2]
    data["nx"] = oc.normals[:, 0]
    data["ny"] = oc.normals[:, 1]
    data["nz"] = oc.normals[:, 2]
    data["red"] = oc.colors[:, 0]
    data["green"] = oc.colors[:, 1]
    data["blue"] = oc.colors[:, 2]
    data["confidence"] = oc.confidences
    data["semantic_class"] = oc.semantic_classes
    data["observation_state"] = cloud.observation_states

    return header + data.tobytes()


def encode_las_binary(cloud: ClassifiedObservationCloud) -> bytes:
    """
    Encodes point cloud to standard ASPRS LAS 1.2 Binary format (Point Format 3).
    Point Record Format 3 (34 bytes):
      - X, Y, Z: int32 (scaled)
      - Intensity: uint16 (confidence * 65535)
      - Flags (return number, number of returns, scan direction, edge of line): uint8
      - Classification: uint8 (ASPRS code)
      - Scan Angle Rank: int8
      - User Data: uint8 (ObservationState 0..5)
      - Point Source ID: uint16
      - GPS Time: float64 (0.0)
      - Red, Green, Blue: uint16, uint16, uint16 (scaled 8-bit color << 8)
    """
    oc = cloud.oriented_cloud
    n = oc.point_count

    pos = oc.positions
    if n == 0:
        min_x, max_x = 0.0, 0.0
        min_y, max_y = 0.0, 0.0
        min_z, max_z = 0.0, 0.0
    else:
        min_x, min_y, min_z = float(np.min(pos[:, 0])), float(np.min(pos[:, 1])), float(np.min(pos[:, 2]))
        max_x, max_y, max_z = float(np.max(pos[:, 0])), float(np.max(pos[:, 1])), float(np.max(pos[:, 2]))

    scale_x, scale_y, scale_z = 0.001, 0.001, 0.001  # 1 mm precision
    offset_x, offset_y, offset_z = min_x, min_y, min_z

    header_size = 227
    offset_to_point_data = 227
    point_data_record_length = 34  # Format 3

    # Construct LAS 1.2 Header
    header = bytearray(header_size)
    header[0:4] = b"LASF"                             # File Signature
    struct.pack_into("<H", header, 4, 0)              # File Source ID
    struct.pack_into("<H", header, 6, 0)              # Global Encoding
    # Project GUID (16 bytes, offset 8..24)
    header[24] = 1                                    # Version Major
    header[25] = 2                                    # Version Minor (LAS 1.2)
    header[26:58] = b"SinglePass3D".ljust(32, b"\x00")# System Identifier
    header[58:90] = b"LASExporter".ljust(32, b"\x00") # Generating Software
    struct.pack_into("<H", header, 90, 1)             # File Creation Day
    struct.pack_into("<H", header, 92, 2026)          # File Creation Year
    struct.pack_into("<H", header, 94, header_size)   # Header Size
    struct.pack_into("<I", header, 96, offset_to_point_data) # Offset to data
    struct.pack_into("<I", header, 100, 0)            # Number of Variable Length Records
    header[104] = 3                                   # Point Data Format ID (Format 3)
    struct.pack_into("<H", header, 105, point_data_record_length) # Record Length
    struct.pack_into("<I", header, 107, n)            # Number of point records
    # Number of points by return (5 x 4 bytes = 20 bytes, offset 111..131)
    struct.pack_into("<I", header, 111, n)            # Return 1
    # Scale factors
    struct.pack_into("<d", header, 131, scale_x)
    struct.pack_into("<d", header, 139, scale_y)
    struct.pack_into("<d", header, 147, scale_z)
    # Offsets
    struct.pack_into("<d", header, 155, offset_x)
    struct.pack_into("<d", header, 163, offset_y)
    struct.pack_into("<d", header, 171, offset_z)
    # Min / Max bounds
    struct.pack_into("<d", header, 179, max_x)
    struct.pack_into("<d", header, 187, min_x)
    struct.pack_into("<d", header, 195, max_y)
    struct.pack_into("<d", header, 203, min_y)
    struct.pack_into("<d", header, 211, max_z)
    struct.pack_into("<d", header, 219, min_z)

    if n == 0:
        return bytes(header)

    # Convert coordinates to scaled int32
    x_scaled = np.round((pos[:, 0] - offset_x) / scale_x).astype(np.int32)
    y_scaled = np.round((pos[:, 1] - offset_y) / scale_y).astype(np.int32)
    z_scaled = np.round((pos[:, 2] - offset_z) / scale_z).astype(np.int32)

    # Intensity from confidence
    intensity = np.clip(np.round(oc.confidences * 65535.0), 0, 65535).astype(np.uint16)

    # Classification mapping to ASPRS
    asprs_classes = np.ones(n, dtype=np.uint8)
    for sem_val in np.unique(oc.semantic_classes):
        s_class = SemanticClass(sem_val) if sem_val in SemanticClass._value2member_map_ else SemanticClass.UNKNOWN
        asprs_code = _ASPRS_CLASS_MAP.get(s_class, 1)
        asprs_classes[oc.semantic_classes == sem_val] = asprs_code

    # User data: observation state
    user_data = cloud.observation_states.astype(np.uint8)

    # Colors: convert 8-bit (0..255) to 16-bit (0..65535) via bit-shift
    r_16 = (oc.colors[:, 0].astype(np.uint16) << 8)
    g_16 = (oc.colors[:, 1].astype(np.uint16) << 8)
    b_16 = (oc.colors[:, 2].astype(np.uint16) << 8)

    # Binary structured array for Point Format 3
    las_dtype = np.dtype([
        ("x", "<i4"),
        ("y", "<i4"),
        ("z", "<i4"),
        ("intensity", "<u2"),
        ("return_flags", "u1"),
        ("classification", "u1"),
        ("scan_angle", "i1"),
        ("user_data", "u1"),
        ("point_source_id", "<u2"),
        ("gps_time", "<f8"),
        ("red", "<u2"),
        ("green", "<u2"),
        ("blue", "<u2"),
    ])

    records = np.empty(n, dtype=las_dtype)
    records["x"] = x_scaled
    records["y"] = y_scaled
    records["z"] = z_scaled
    records["intensity"] = intensity
    records["return_flags"] = 1  # Return 1 of 1
    records["classification"] = asprs_classes
    records["scan_angle"] = 0
    records["user_data"] = user_data
    records["point_source_id"] = 1
    records["gps_time"] = 0.0
    records["red"] = r_16
    records["green"] = g_16
    records["blue"] = b_16

    return bytes(header) + records.tobytes()


@dataclass
class ExportedPointCloudFiles:
    """Paths and metadata for exported point cloud files."""

    ply_path: str
    las_path: str
    manifest_path: str
    total_points: int
    ply_size_bytes: int
    las_size_bytes: int
    ply_sha256: str
    las_sha256: str
    observation_stats: Dict[str, Any]
    s3_uploaded: bool = False


class PointCloudSerializer:
    """
    Serializes classified point clouds to disk and coordinates S3 uploads.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        s3_client: Optional[Any] = None,
        s3_bucket: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket

    def export(
        self,
        cloud: ClassifiedObservationCloud,
        base_filename: str = "pointcloud",
        job_id: Optional[str] = None,
    ) -> ExportedPointCloudFiles:
        """
        Exports PLY and LAS formats and writes a complete JSON manifest.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Encode and write PLY
        ply_bytes = encode_ply_binary(cloud)
        ply_sha = compute_sha256(ply_bytes)
        ply_file = self.output_dir / f"{base_filename}.ply"
        ply_file.write_bytes(ply_bytes)

        # 2. Encode and write LAS
        las_bytes = encode_las_binary(cloud)
        las_sha = compute_sha256(las_bytes)
        las_file = self.output_dir / f"{base_filename}.las"
        las_file.write_bytes(las_bytes)

        # 3. Create manifest
        stats_dict = cloud.stats.to_dict()
        manifest_data = {
            "job_id": job_id or "local",
            "base_filename": base_filename,
            "total_points": cloud.point_count,
            "observation_statistics": stats_dict,
            "files": {
                "ply": {
                    "filename": f"{base_filename}.ply",
                    "file_size_bytes": len(ply_bytes),
                    "sha256": ply_sha,
                },
                "las": {
                    "filename": f"{base_filename}.las",
                    "file_size_bytes": len(las_bytes),
                    "sha256": las_sha,
                    "asprs_version": "1.2",
                    "point_format": 3,
                },
            },
        }
        manifest_path = self.output_dir / f"{base_filename}_manifest.json"
        manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)

        s3_uploaded = False
        if self.s3_client and self.s3_bucket and job_id:
            try:
                pfx = f"jobs/{job_id}/pointcloud/"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{pfx}{base_filename}.ply",
                    Body=ply_bytes,
                    ContentType="application/octet-stream",
                )
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{pfx}{base_filename}.las",
                    Body=las_bytes,
                    ContentType="application/octet-stream",
                )
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{pfx}{base_filename}_manifest.json",
                    Body=manifest_bytes,
                    ContentType="application/json",
                )
                s3_uploaded = True
            except Exception as exc:
                logger.error(f"Failed to upload point clouds to S3: {exc}")

        return ExportedPointCloudFiles(
            ply_path=str(ply_file),
            las_path=str(las_file),
            manifest_path=str(manifest_path),
            total_points=cloud.point_count,
            ply_size_bytes=len(ply_bytes),
            las_size_bytes=len(las_bytes),
            ply_sha256=ply_sha,
            las_sha256=las_sha,
            observation_stats=stats_dict,
            s3_uploaded=s3_uploaded,
        )

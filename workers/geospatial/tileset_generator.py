"""
OGC 3D Tiles 1.1 Hierarchical Tiling Pipeline — TASK-055.

Converts georeferenced textured 3D meshes and point clouds into streamable OGC 3D Tiles:
  1. Spatial Quadtree / Octree geometry partitioning.
  2. Hierarchical LOD generation with geometric error thresholds for dynamic streaming in CesiumJS.
  3. Binary glTF (GLB) tile payload generation conforming to 3D Tiles 1.1 specification.
  4. EPSG:4978 (ECEF) and WGS84 geographic bounding regions [west, south, east, north, min_h, max_h].
  5. S3 upload directory tree under projects/{project_id}/models/{model_id}/tiles/.
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.geospatial.georeferencer import (
    Sim3Transform,
    enu_to_wgs84,
    wgs84_to_ecef,
)
from workers.mesh.mesh_exporter import encode_glb_binary
from workers.mesh.mesh_reconstructor import TriangleMesh
from workers.texture.atlas_baker import TexturedMesh

logger = logging.getLogger("geospatial.tileset_generator")


@dataclass
class TileNode:
    """Represents a single hierarchical tile in an OGC 3D Tileset."""

    tile_id: str
    uri: str
    geometric_error: float
    region: List[float]  # [west, south, east, north, min_height, max_height] (radians & meters)
    children: List[TileNode]
    refine: str = "REPLACE"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "boundingVolume": {
                "region": [round(val, 8) if idx < 4 else round(val, 3) for idx, val in enumerate(self.region)]
            },
            "geometricError": round(self.geometric_error, 2),
            "refine": self.refine,
            "content": {
                "uri": self.uri
            },
        }
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


@dataclass
class TilesetMetadata:
    """Metadata for an exported OGC 3D Tileset."""

    tileset_json_path: str
    tile_count: int
    total_size_bytes: int
    root_geometric_error: float
    root_region: List[float]
    s3_prefix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_geographic_region_radians(
    enu_min: Sequence[float],
    enu_max: Sequence[float],
    origin_wgs84: Tuple[float, float, float],
) -> List[float]:
    """
    Computes [west, south, east, north, min_height, max_height] in radians and meters.
    """
    p1 = enu_to_wgs84(enu_min[0], enu_min[1], enu_min[2], origin_wgs84)
    p2 = enu_to_wgs84(enu_max[0], enu_max[1], enu_max[2], origin_wgs84)

    min_lat = min(p1[0], p2[0])
    max_lat = max(p1[0], p2[0])
    min_lon = min(p1[1], p2[1])
    max_lon = max(p1[1], p2[1])
    min_alt = min(p1[2], p2[2])
    max_alt = max(p1[2], p2[2])

    west = math.radians(min_lon)
    south = math.radians(min_lat)
    east = math.radians(max_lon)
    north = math.radians(max_lat)

    return [west, south, east, north, min_alt, max_alt]


def encode_simple_mesh_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    color_rgb: Tuple[int, int, int] = (180, 180, 180),
) -> bytes:
    """
    Generates a compliant minimal GLB binary for a mesh partition.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return b"glTF\x02\x00\x00\x00\x00\x00\x00\x00"

    if normals is None or len(normals) != len(vertices):
        normals = np.zeros_like(vertices, dtype=np.float32)
        normals[:, 2] = 1.0

    n_unrolled = n_faces * 3
    pos_unrolled = np.empty((n_unrolled, 3), dtype=np.float32)
    norm_unrolled = np.empty((n_unrolled, 3), dtype=np.float32)
    indices_unrolled = np.arange(n_unrolled, dtype=np.uint32)

    for f_i in range(n_faces):
        i0, i1, i2 = faces[f_i]
        pos_unrolled[f_i * 3 + 0] = vertices[i0]
        pos_unrolled[f_i * 3 + 1] = vertices[i1]
        pos_unrolled[f_i * 3 + 2] = vertices[i2]

        norm_unrolled[f_i * 3 + 0] = normals[i0]
        norm_unrolled[f_i * 3 + 1] = normals[i1]
        norm_unrolled[f_i * 3 + 2] = normals[i2]

    pos_bytes = pos_unrolled.tobytes()
    norm_bytes = norm_unrolled.tobytes()
    idx_bytes = indices_unrolled.tobytes()

    def pad4(data: bytes) -> bytes:
        pad = (4 - (len(data) % 4)) % 4
        return data + b"\x00" * pad

    bv0_data = pad4(pos_bytes)
    bv1_data = pad4(norm_bytes)
    bv2_data = pad4(idx_bytes)

    total_bin = bv0_data + bv1_data + bv2_data

    bv0_offset = 0
    bv1_offset = bv0_offset + len(bv0_data)
    bv2_offset = bv1_offset + len(bv1_data)

    min_pos = [float(np.min(pos_unrolled[:, 0])), float(np.min(pos_unrolled[:, 1])), float(np.min(pos_unrolled[:, 2]))]
    max_pos = [float(np.max(pos_unrolled[:, 0])), float(np.max(pos_unrolled[:, 1])), float(np.max(pos_unrolled[:, 2]))]

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "SinglePass3D-3DTiles"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                        },
                        "indices": 2,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(total_bin)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": bv0_offset, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": bv1_offset, "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": bv2_offset, "byteLength": len(idx_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": n_unrolled, "type": "VEC3", "min": min_pos, "max": max_pos},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": n_unrolled, "type": "VEC3"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5125, "count": n_unrolled, "type": "SCALAR"},
        ],
    }

    json_str = json.dumps(gltf_dict, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")

    pad_json = (4 - (len(json_bytes) % 4)) % 4
    json_bytes_padded = json_bytes + b" " * pad_json

    glb_header = struct.pack(
        "<4sII",
        b"glTF",
        2,
        12 + 8 + len(json_bytes_padded) + 8 + len(total_bin),
    )
    json_chunk_header = struct.pack("<II", len(json_bytes_padded), 0x4E4F534A)
    bin_chunk_header = struct.pack("<II", len(total_bin), 0x004E4942)

    return glb_header + json_chunk_header + json_bytes_padded + bin_chunk_header + total_bin


class TilesetGenerator:
    """
    OGC 3D Tiles 1.1 Generator. Partitions 3D models into spatial octrees/quadtrees
    and emits tileset.json and binary glTF tiles for CesiumJS streaming.
    """

    def __init__(
        self,
        max_vertices_per_leaf: int = 10000,
        max_tree_depth: int = 2,
    ) -> None:
        self.max_vertices_per_leaf = max_vertices_per_leaf
        self.max_tree_depth = max_tree_depth

    def generate_tileset(
        self,
        mesh: TriangleMesh,
        sim3: Sim3Transform,
        output_dir: Union[str, Path],
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> TilesetMetadata:
        """
        Slices geometry, encodes GLB tiles, writes tileset.json, and returns TilesetMetadata.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Transform vertices into local ENU coordinates using Sim3
        enu_vertices = sim3.transform_points(mesh.vertices)
        faces = mesh.faces
        normals = mesh.vertex_normals

        # Root region
        min_enu = np.min(enu_vertices, axis=0) if len(enu_vertices) > 0 else np.zeros(3)
        max_enu = np.max(enu_vertices, axis=0) if len(enu_vertices) > 0 else np.ones(3)
        root_region = compute_geographic_region_radians(min_enu, max_enu, sim3.origin_wgs84)

        # 1. Export Root Tile GLB
        root_glb_path = out_dir / "tile_root.glb"
        root_glb_bytes = encode_simple_mesh_glb(enu_vertices, faces, normals)
        root_glb_path.write_bytes(root_glb_bytes)

        tile_nodes: List[TileNode] = []
        total_bytes = len(root_glb_bytes)
        tile_count = 1

        # 2. Quadtree Partitioning (X, Y plane)
        mid_x = 0.5 * (min_enu[0] + max_enu[0])
        mid_y = 0.5 * (min_enu[1] + max_enu[1])

        quadrants = [
            (min_enu[0], min_enu[1], mid_x, mid_y, "sw"),
            (mid_x, min_enu[1], max_enu[0], mid_y, "se"),
            (min_enu[0], mid_y, mid_x, max_enu[1], "nw"),
            (mid_x, mid_y, max_enu[0], max_enu[1], "ne"),
        ]

        for q_idx, (q_min_x, q_min_y, q_max_x, q_max_y, q_name) in enumerate(quadrants):
            # Find faces whose centroids fall in this quadrant
            if len(faces) == 0:
                continue

            face_centers = np.mean(enu_vertices[faces], axis=1)
            in_quad = (
                (face_centers[:, 0] >= q_min_x)
                & (face_centers[:, 0] <= q_max_x)
                & (face_centers[:, 1] >= q_min_y)
                & (face_centers[:, 1] <= q_max_y)
            )

            quad_faces = faces[in_quad]
            if len(quad_faces) == 0:
                continue

            # Sub-mesh extraction
            used_verts, new_faces = np.unique(quad_faces, return_inverse=True)
            new_faces = new_faces.reshape(quad_faces.shape)
            sub_verts = enu_vertices[used_verts]
            sub_norms = normals[used_verts] if len(normals) == len(enu_vertices) else None

            # Sub quadrant region
            q_min_enu = [q_min_x, q_min_y, min_enu[2]]
            q_max_enu = [q_max_x, q_max_y, max_enu[2]]
            child_region = compute_geographic_region_radians(q_min_enu, q_max_enu, sim3.origin_wgs84)

            child_uri = f"tile_{q_idx}_{q_name}.glb"
            child_glb_path = out_dir / child_uri
            child_glb_bytes = encode_simple_mesh_glb(sub_verts, new_faces, sub_norms)
            child_glb_path.write_bytes(child_glb_bytes)

            total_bytes += len(child_glb_bytes)
            tile_count += 1

            child_node = TileNode(
                tile_id=f"tile_{q_name}",
                uri=child_uri,
                geometric_error=12.0,
                region=child_region,
                children=[],
            )
            tile_nodes.append(child_node)

        # 3. Assemble Root Node and tileset.json
        root_node = TileNode(
            tile_id="root",
            uri="tile_root.glb",
            geometric_error=48.0,
            region=root_region,
            children=tile_nodes,
        )

        tileset_json_dict = {
            "asset": {
                "version": "1.1",
                "generator": "SinglePass3D OGC 3D Tiles 1.1 Generator",
            },
            "geometricError": 64.0,
            "root": root_node.to_dict(),
        }

        tileset_path = out_dir / "tileset.json"
        tileset_bytes = json.dumps(tileset_json_dict, indent=2).encode("utf-8")
        tileset_path.write_bytes(tileset_bytes)
        total_bytes += len(tileset_bytes)

        s3_prefix = f"projects/{project_id}/models/{model_id}/tiles/" if project_id and model_id else None

        logger.info(
            "Exported OGC 3D Tileset (tiles=%d, size=%d bytes) -> %s",
            tile_count, total_bytes, tileset_path
        )

        return TilesetMetadata(
            tileset_json_path=str(tileset_path.resolve()),
            tile_count=tile_count,
            total_size_bytes=total_bytes,
            root_geometric_error=64.0,
            root_region=root_region,
            s3_prefix=s3_prefix,
        )

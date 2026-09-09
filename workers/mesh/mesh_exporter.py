"""
GLB, glTF, and OBJ Mesh Exporter — TASK-051.

Exports textured 3D models into industry-standard formats:
  1. Standard Binary glTF 2.0 (`model.glb`):
     - Self-contained single-file binary with embedded geometry, vertex normals,
       UV texture coordinates, PBR metallic-roughness material, and embedded diffuse texture.
  2. Wavefront OBJ + Material Template Library (`model.obj`, `model.mtl`, `diffuse_00.png`):
     - Universal CAD/BIM/GIS interoperability.
  3. Export manifest and S3 asset synchronization under `projects/{project_id}/models/{model_id}/`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.texture.atlas_baker import TexturedMesh

logger = logging.getLogger("mesh.mesh_exporter")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class ExportedMeshFiles:
    """Paths and metadata for exported mesh assets."""

    glb_path: str
    obj_path: str
    mtl_path: str
    texture_path: str
    manifest_path: str
    total_vertices: int
    total_faces: int
    glb_size_bytes: int
    obj_size_bytes: int
    glb_sha256: str
    s3_uploaded: bool = False


def encode_obj_mtl(
    textured_mesh: TexturedMesh,
    base_name: str = "model",
) -> Tuple[str, str, bytes]:
    """
    Generates Wavefront OBJ text, MTL material text, and diffuse PNG bytes.
    """
    mesh = textured_mesh.mesh
    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals
    face_uvs = textured_mesh.face_uvs

    # 1. MTL Content
    mtl_filename = f"{base_name}.mtl"
    tex_filename = f"{base_name}_diffuse.png"
    mtl_lines = [
        f"# Single-Pass 3D Reconstruction Platform MTL",
        f"newmtl DefaultMaterial",
        f"Ka 1.000 1.000 1.000",
        f"Kd 1.000 1.000 1.000",
        f"Ks 0.000 0.000 0.000",
        f"d 1.0",
        f"illum 2",
        f"map_Kd {tex_filename}\n",
    ]
    mtl_text = "\n".join(mtl_lines)

    # 2. OBJ Content
    obj_lines = [
        f"# Single-Pass 3D Reconstruction Platform OBJ",
        f"mtllib {mtl_filename}",
        f"usemtl DefaultMaterial",
    ]

    # Geometric vertices: v x y z
    for v in verts:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # Vertex normals: vn nx ny nz
    for vn in normals:
        obj_lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}")

    # Texture coordinates: vt u v
    # Collect unique (u, v) or per-corner UVs
    n_faces = len(faces)
    vt_index = 1
    face_vt_indices = []
    for f_i in range(n_faces):
        corner_vts = []
        for c in range(3):
            u, v = face_uvs[f_i, c]
            obj_lines.append(f"vt {u:.6f} {v:.6f}")
            corner_vts.append(vt_index)
            vt_index += 1
        face_vt_indices.append(corner_vts)

    # Faces: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 (1-indexed)
    for f_i in range(n_faces):
        i0, i1, i2 = faces[f_i] + 1
        vt0, vt1, vt2 = face_vt_indices[f_i]
        obj_lines.append(f"f {i0}/{vt0}/{i0} {i1}/{vt1}/{i1} {i2}/{vt2}/{i2}")

    obj_text = "\n".join(obj_lines) + "\n"

    # 3. Diffuse PNG bytes
    success, buf = cv2.imencode(".png", cv2.cvtColor(textured_mesh.diffuse_atlas, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode texture atlas to PNG")
    tex_bytes = bytes(buf)

    return obj_text, mtl_text, tex_bytes


def encode_glb_binary(
    textured_mesh: TexturedMesh,
) -> bytes:
    """
    Encodes TexturedMesh into self-contained binary glTF 2.0 (GLB) with embedded texture.
    """
    mesh = textured_mesh.mesh
    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals
    face_uvs = textured_mesh.face_uvs

    n_faces = len(faces)
    if n_faces == 0:
        return b"glTF\x02\x00\x00\x00\x00\x00\x00\x00"

    # Unroll per-face-vertex data to ensure continuous (position, normal, UV) indexing
    n_unrolled = n_faces * 3
    pos_unrolled = np.empty((n_unrolled, 3), dtype=np.float32)
    norm_unrolled = np.empty((n_unrolled, 3), dtype=np.float32)
    uv_unrolled = np.empty((n_unrolled, 2), dtype=np.float32)
    indices_unrolled = np.arange(n_unrolled, dtype=np.uint32)

    for f_i in range(n_faces):
        i0, i1, i2 = faces[f_i]
        pos_unrolled[f_i * 3 + 0] = verts[i0]
        pos_unrolled[f_i * 3 + 1] = verts[i1]
        pos_unrolled[f_i * 3 + 2] = verts[i2]

        norm_unrolled[f_i * 3 + 0] = normals[i0]
        norm_unrolled[f_i * 3 + 1] = normals[i1]
        norm_unrolled[f_i * 3 + 2] = normals[i2]

        uv_unrolled[f_i * 3 + 0] = face_uvs[f_i, 0]
        uv_unrolled[f_i * 3 + 1] = face_uvs[f_i, 1]
        uv_unrolled[f_i * 3 + 2] = face_uvs[f_i, 2]

    # Invert V axis for glTF standard (origin at top-left)
    uv_unrolled[:, 1] = 1.0 - uv_unrolled[:, 1]

    # Encode diffuse texture to PNG
    success, img_buf = cv2.imencode(".png", cv2.cvtColor(textured_mesh.diffuse_atlas, cv2.COLOR_RGB2BGR))
    img_bytes = bytes(img_buf)

    # Convert binary arrays
    pos_bytes = pos_unrolled.tobytes()
    norm_bytes = norm_unrolled.tobytes()
    uv_bytes = uv_unrolled.tobytes()
    idx_bytes = indices_unrolled.tobytes()

    # Buffer Views Alignment (4-byte alignment for glTF buffers)
    def pad4(data: bytes) -> bytes:
        pad = (4 - (len(data) % 4)) % 4
        return data + b"\x00" * pad

    bv0_data = pad4(pos_bytes)
    bv1_data = pad4(norm_bytes)
    bv2_data = pad4(uv_bytes)
    bv3_data = pad4(idx_bytes)
    bv4_data = pad4(img_bytes)

    total_bin = bv0_data + bv1_data + bv2_data + bv3_data + bv4_data

    bv0_offset = 0
    bv1_offset = bv0_offset + len(bv0_data)
    bv2_offset = bv1_offset + len(bv1_data)
    bv3_offset = bv2_offset + len(bv2_data)
    bv4_offset = bv3_offset + len(bv3_data)

    min_pos = [float(np.min(pos_unrolled[:, 0])), float(np.min(pos_unrolled[:, 1])), float(np.min(pos_unrolled[:, 2]))]
    max_pos = [float(np.max(pos_unrolled[:, 0])), float(np.max(pos_unrolled[:, 1])), float(np.max(pos_unrolled[:, 2]))]

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "SinglePass3D-Platform"},
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
                            "TEXCOORD_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "SurfaceMaterial",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
        "accessors": [
            # 0: POSITION
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": n_unrolled,
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos,
            },
            # 1: NORMAL
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": n_unrolled,
                "type": "VEC3",
            },
            # 2: TEXCOORD_0
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": n_unrolled,
                "type": "VEC2",
            },
            # 3: INDICES
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": n_unrolled,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": bv0_offset, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": bv1_offset, "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": bv2_offset, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": bv3_offset, "byteLength": len(idx_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": bv4_offset, "byteLength": len(img_bytes)},
        ],
        "buffers": [{"byteLength": len(total_bin)}],
    }

    json_str = json.dumps(gltf_dict, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON chunk to 4 bytes with spaces (0x20)
    json_padded = json_bytes + b" " * ((4 - (len(json_bytes) % 4)) % 4)

    # GLB Header: magic (4), version (4), length (4) = 12 bytes
    total_length = 12 + (8 + len(json_padded)) + (8 + len(total_bin))

    header = struct.pack("<4sII", b"glTF", 2, total_length)
    chunk0_header = struct.pack("<II", len(json_padded), 0x4E4F534A)  # 0x4E4F534A = JSON
    chunk1_header = struct.pack("<II", len(total_bin), 0x004E4942)    # 0x004E4942 = BIN\0

    return header + chunk0_header + json_padded + chunk1_header + total_bin


class MeshExporter:
    """
    Coordinates GLB, OBJ, MTL, and texture atlas file exports and S3 uploads.
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
        textured_mesh: TexturedMesh,
        base_name: str = "model",
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> ExportedMeshFiles:
        """
        Exports GLB, OBJ, MTL, texture, and manifest files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export GLB
        glb_bytes = encode_glb_binary(textured_mesh)
        glb_sha = compute_sha256(glb_bytes)
        glb_path = self.output_dir / f"{base_name}.glb"
        glb_path.write_bytes(glb_bytes)

        # 2. Export OBJ + MTL + Diffuse PNG
        obj_text, mtl_text, tex_bytes = encode_obj_mtl(textured_mesh, base_name=base_name)
        obj_path = self.output_dir / f"{base_name}.obj"
        obj_path.write_text(obj_text, encoding="utf-8")
        mtl_path = self.output_dir / f"{base_name}.mtl"
        mtl_path.write_text(mtl_text, encoding="utf-8")
        tex_path = self.output_dir / f"{base_name}_diffuse.png"
        tex_path.write_bytes(tex_bytes)

        # 3. Export Manifest
        manifest_data = {
            "project_id": project_id or "local",
            "model_id": model_id or "local",
            "base_name": base_name,
            "vertex_count": textured_mesh.mesh.vertex_count,
            "face_count": textured_mesh.mesh.face_count,
            "files": {
                "glb": {
                    "filename": f"{base_name}.glb",
                    "size_bytes": len(glb_bytes),
                    "sha256": glb_sha,
                    "format": "glTF 2.0 Binary",
                },
                "obj": {
                    "filename": f"{base_name}.obj",
                    "size_bytes": len(obj_text.encode('utf-8')),
                },
                "mtl": {
                    "filename": f"{base_name}.mtl",
                },
                "texture": {
                    "filename": f"{base_name}_diffuse.png",
                    "size_bytes": len(tex_bytes),
                },
            },
        }
        manifest_path = self.output_dir / f"{base_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        s3_uploaded = False
        if self.s3_client and self.s3_bucket and project_id and model_id:
            try:
                pfx = f"projects/{project_id}/models/{model_id}/"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{pfx}{base_name}.glb",
                    Body=glb_bytes,
                    ContentType="model/gltf-binary",
                )
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{pfx}{base_name}.obj",
                    Body=obj_text.encode('utf-8'),
                    ContentType="text/plain",
                )
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{pfx}{base_name}_diffuse.png",
                    Body=tex_bytes,
                    ContentType="image/png",
                )
                s3_uploaded = True
            except Exception as exc:
                logger.error(f"Failed to upload mesh files to S3: {exc}")

        return ExportedMeshFiles(
            glb_path=str(glb_path),
            obj_path=str(obj_path),
            mtl_path=str(mtl_path),
            texture_path=str(tex_path),
            manifest_path=str(manifest_path),
            total_vertices=textured_mesh.mesh.vertex_count,
            total_faces=textured_mesh.mesh.face_count,
            glb_size_bytes=len(glb_bytes),
            obj_size_bytes=len(obj_text.encode('utf-8')),
            glb_sha256=glb_sha,
            s3_uploaded=s3_uploaded,
        )

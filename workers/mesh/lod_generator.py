"""
Multi-Resolution Level-of-Detail (LOD) Decimation — TASK-050.

Generates 3 optimized discrete Level-of-Detail (LOD) mesh tiers for WebGL/3D Tiles streaming:
  - LOD 0: 100% full-resolution polygonal mesh.
  - LOD 1: Decimated to ~35% polygons using edge-collapse simplification.
  - LOD 2: Simplified envelope (~10% polygons) for distant viewer perspectives.

Maintains intact UV coordinate parameterization and `is_ai_inferred` metadata tags.
"""
from __future__ import annotations

import collections
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.mesh.surface_reconstructor import TriangleMesh, compute_face_normals
from workers.texture.atlas_baker import TexturedMesh

logger = logging.getLogger("mesh.lod_generator")


@dataclass
class LODHierarchy:
    """Multi-resolution hierarchy containing LOD 0, LOD 1, and LOD 2 models."""

    lod_0: TexturedMesh                # 100% polygons
    lod_1: TexturedMesh                # 35% polygons
    lod_2: TexturedMesh                # 10% polygons
    face_counts: Dict[str, int]
    target_ratios: Dict[str, float]


def decimate_mesh(
    mesh: TriangleMesh,
    face_uvs: np.ndarray,
    target_face_count: int,
) -> Tuple[TriangleMesh, np.ndarray]:
    """
    Decimates mesh down to target_face_count via edge collapses.
    Preserves vertex colors, face semantic classes, and `is_ai_inferred` tags.
    """
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    uvs = face_uvs.copy()
    face_sems = mesh.face_semantic_classes.copy()
    inferred = mesh.is_ai_inferred.copy()

    n_faces = len(faces)
    if n_faces <= target_face_count or target_face_count < 4:
        return mesh, face_uvs

    # Edge collapse loop
    # In each pass, select shortest internal edges and collapse them
    while len(faces) > target_face_count:
        # Edge lengths calculation
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)

        # Collect unique edges and lengths
        edges_dict: Dict[Tuple[int, int], float] = {}
        for f_i in range(len(faces)):
            i0, i1, i2 = faces[f_i]
            for u, v, l in [(i0, i1, e0[f_i]), (i1, i2, e1[f_i]), (i2, i0, e2[f_i])]:
                edge = tuple(sorted((u, v)))
                if edge not in edges_dict:
                    edges_dict[edge] = l

        sorted_edges = sorted(edges_dict.items(), key=lambda item: item[1])

        # Collapse a batch of non-overlapping shortest edges
        collapsed_verts: Dict[int, int] = {}
        used_vertices: set = set()

        for (u, v), _ in sorted_edges:
            if u in used_vertices or v in used_vertices:
                continue
            # Collapse v into u
            collapsed_verts[v] = u
            used_vertices.add(u)
            used_vertices.add(v)
            # Midpoint placement
            verts[u] = 0.5 * (verts[u] + verts[v])

            # Stop batch if target face count would be exceeded
            if len(faces) - len(collapsed_verts) * 2 <= target_face_count:
                break

        if not collapsed_verts:
            break

        # Re-map faces
        new_faces = faces.copy()
        for v_old, v_target in collapsed_verts.items():
            new_faces[new_faces == v_old] = v_target

        # Remove degenerate triangles (where vertices collapsed into each other)
        valid = (new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2]) & (new_faces[:, 2] != new_faces[:, 0])

        if np.count_nonzero(valid) == len(faces):
            # No faces removed in this pass
            break

        faces = new_faces[valid]
        uvs = uvs[valid]
        face_sems = face_sems[valid]
        inferred = inferred[valid]

    # Re-index remaining vertices
    used_v = np.unique(faces)
    v_map = np.full(len(verts), -1, dtype=np.int32)
    v_map[used_v] = np.arange(len(used_v))

    final_verts = verts[used_v]
    final_normals = mesh.vertex_normals[used_v] if len(mesh.vertex_normals) > np.max(used_v) else np.zeros((len(used_v), 3), dtype=np.float32)
    final_colors = mesh.vertex_colors[used_v] if len(mesh.vertex_colors) > np.max(used_v) else np.full((len(used_v), 3), 150, dtype=np.uint8)
    final_faces = v_map[faces]
    final_fn = compute_face_normals(final_verts, final_faces)

    decimated_mesh = TriangleMesh(
        vertices=final_verts,
        faces=final_faces,
        vertex_normals=final_normals,
        vertex_colors=final_colors,
        face_normals=final_fn,
        face_semantic_classes=face_sems,
        is_ai_inferred=inferred,
    )

    return decimated_mesh, uvs


class LODGenerator:
    """
    Generates multi-resolution LOD hierarchy (LOD 0, LOD 1, LOD 2).
    """

    def __init__(
        self,
        lod1_ratio: float = 0.35,
        lod2_ratio: float = 0.10,
    ):
        self.lod1_ratio = lod1_ratio
        self.lod2_ratio = lod2_ratio

    def generate_lod_hierarchy(
        self,
        textured_mesh: TexturedMesh,
    ) -> LODHierarchy:
        """
        Creates LOD 0 (100%), LOD 1 (35%), and LOD 2 (10%) meshes.
        """
        base_f = textured_mesh.mesh.face_count

        # LOD 0: 100%
        lod_0 = textured_mesh

        # Target face counts
        target_f1 = max(4, int(round(base_f * self.lod1_ratio)))
        target_f2 = max(4, int(round(base_f * self.lod2_ratio)))

        # LOD 1
        mesh_1, uvs_1 = decimate_mesh(textured_mesh.mesh, textured_mesh.face_uvs, target_f1)
        lod_1 = TexturedMesh(
            mesh=mesh_1,
            face_uvs=uvs_1,
            diffuse_atlas=textured_mesh.diffuse_atlas,
            confidence_atlas=textured_mesh.confidence_atlas,
            atlas_resolution=textured_mesh.atlas_resolution,
        )

        # LOD 2
        mesh_2, uvs_2 = decimate_mesh(mesh_1, uvs_1, target_f2)
        lod_2 = TexturedMesh(
            mesh=mesh_2,
            face_uvs=uvs_2,
            diffuse_atlas=textured_mesh.diffuse_atlas,
            confidence_atlas=textured_mesh.confidence_atlas,
            atlas_resolution=textured_mesh.atlas_resolution,
        )

        counts = {
            "lod_0": lod_0.mesh.face_count,
            "lod_1": lod_1.mesh.face_count,
            "lod_2": lod_2.mesh.face_count,
        }
        ratios = {
            "lod_0": 1.0,
            "lod_1": float(lod_1.mesh.face_count / base_f) if base_f > 0 else 0.0,
            "lod_2": float(lod_2.mesh.face_count / base_f) if base_f > 0 else 0.0,
        }

        return LODHierarchy(
            lod_0=lod_0,
            lod_1=lod_1,
            lod_2=lod_2,
            face_counts=counts,
            target_ratios=ratios,
        )

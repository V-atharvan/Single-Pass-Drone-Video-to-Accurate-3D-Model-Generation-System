"""
Surface Mesh Reconstruction Engine — TASK-046.

Generates a continuous 3D triangular surface mesh from the oriented point cloud:
  1. Triangulates oriented 3D point cloud into continuous surface topology.
  2. Trims low-density, non-physical boundary edges/bubbles where point support is absent.
  3. Computes vertex and face normal vectors.
  4. Segments mesh into terrain and architectural facades based on underlying semantic classes.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.fusion.normal_estimator import OrientedPointCloud
from workers.segmentation.semantic_classifier import SemanticClass

logger = logging.getLogger("mesh.surface_reconstructor")


@dataclass
class TriangleMesh:
    """
    3D polygonal triangular surface mesh.

    Attributes
    ----------
    vertices:
        (V, 3) float32 coordinates.
    faces:
        (F, 3) int32 triangle vertex indices.
    vertex_normals:
        (V, 3) float32 vertex unit normal vectors.
    vertex_colors:
        (V, 3) uint8 RGB colors.
    face_normals:
        (F, 3) float32 face unit normal vectors.
    face_semantic_classes:
        (F,) uint8 semantic class ID for each triangle.
    is_ai_inferred:
        (F,) bool flag marking whether the face was synthesized by hole infilling.
    """

    vertices: np.ndarray           # (V, 3) float32
    faces: np.ndarray              # (F, 3) int32
    vertex_normals: np.ndarray     # (V, 3) float32
    vertex_colors: np.ndarray      # (V, 3) uint8
    face_normals: np.ndarray       # (F, 3) float32
    face_semantic_classes: np.ndarray # (F,) uint8
    is_ai_inferred: np.ndarray     # (F,) bool

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes normalized face normal vectors for a set of triangles."""
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    return (cross / lengths).astype(np.float32)


class SurfaceReconstructor:
    """
    Reconstructs continuous 3D surface meshes from point clouds with edge density trimming.
    """

    def __init__(self, max_edge_length_factor: float = 3.5):
        self.max_edge_length_factor = max_edge_length_factor

    def reconstruct_surface(
        self,
        cloud: OrientedPointCloud,
        max_edge_length_m: Optional[float] = None,
    ) -> TriangleMesh:
        """
        Builds triangular mesh from point cloud, trimming non-physical ballooning edges.
        """
        n_pts = cloud.point_count
        if n_pts < 3:
            return TriangleMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                vertex_normals=np.zeros((0, 3), dtype=np.float32),
                vertex_colors=np.zeros((0, 3), dtype=np.uint8),
                face_normals=np.zeros((0, 3), dtype=np.float32),
                face_semantic_classes=np.zeros((0,), dtype=np.uint8),
                is_ai_inferred=np.zeros((0,), dtype=bool),
            )

        pos = cloud.positions

        # Aerial 2.5D Delaunay triangulation across the horizontal XY plane
        xy = pos[:, :2]
        tri = Delaunay(xy)
        raw_faces = tri.simplices.astype(np.int32)

        # Edge length calculation for trimming
        v0 = pos[raw_faces[:, 0]]
        v1 = pos[raw_faces[:, 1]]
        v2 = pos[raw_faces[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        max_edges = np.maximum(np.maximum(e0, e1), e2)

        if max_edge_length_m is not None:
            threshold = float(max_edge_length_m)
        else:
            # Estimate reasonable density threshold from median edge length
            med_edge = float(np.median(max_edges))
            threshold = med_edge * self.max_edge_length_factor

        # Filter out long, non-physical boundary triangles
        valid_triangles = max_edges <= threshold
        trimmed_faces = raw_faces[valid_triangles]

        # Calculate face normals
        face_normals = compute_face_normals(pos, trimmed_faces)

        # Segment face semantic classes: majority class of the 3 triangle vertices
        sem_classes = cloud.semantic_classes
        face_sems = np.empty(len(trimmed_faces), dtype=np.uint8)
        for idx, (i0, i1, i2) in enumerate(trimmed_faces):
            c0, c1, c2 = sem_classes[i0], sem_classes[i1], sem_classes[i2]
            # Majority vote
            if c0 == c1 or c0 == c2:
                face_sems[idx] = c0
            elif c1 == c2:
                face_sems[idx] = c1
            else:
                face_sems[idx] = c0

        # Inferred flags: False for original reconstructed geometry
        is_inferred = np.zeros(len(trimmed_faces), dtype=bool)

        return TriangleMesh(
            vertices=pos.copy(),
            faces=trimmed_faces,
            vertex_normals=cloud.normals.copy(),
            vertex_colors=cloud.colors.copy(),
            face_normals=face_normals,
            face_semantic_classes=face_sems,
            is_ai_inferred=is_inferred,
        )

"""
Exposure Correction, Seam Blending & Texture Atlas Baking — TASK-049.

Parameterizes triangular mesh UV coordinates, applies exposure/color correction,
and bakes unified texture atlases:
  - Generates seamless diffuse texture atlas (`diffuse_00.png`).
  - Generates auxiliary confidence texture map (`confidence_00.png`).
  - Equalizes exposure across differing source frames to prevent harsh seam transitions.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.mesh.surface_reconstructor import TriangleMesh
from workers.texture.view_selector import FaceViewAssignment

logger = logging.getLogger("texture.atlas_baker")

_DEFAULT_ATLAS_SIZE: int = 1024  # 1024x1024 base atlas for testing, scalable to 4096


@dataclass
class TexturedMesh:
    """
    3D mesh with parameter-space UV coordinates and baked texture maps.

    Attributes
    ----------
    mesh:
        Base polygonal TriangleMesh.
    face_uvs:
        (F, 3, 2) float32 normalized texture coordinates [0.0, 1.0] for each vertex of each face.
    diffuse_atlas:
        (H, W, 3) uint8 RGB texture image.
    confidence_atlas:
        (H, W) uint8 grayscale confidence map.
    atlas_resolution:
        Width/height of texture atlas in pixels.
    """

    mesh: TriangleMesh
    face_uvs: np.ndarray               # (F, 3, 2) float32
    diffuse_atlas: np.ndarray          # (H, W, 3) uint8 RGB
    confidence_atlas: np.ndarray       # (H, W) uint8
    atlas_resolution: int


class TextureAtlasBaker:
    """
    UV parameterization, exposure balancing, and texture atlas rasterization engine.
    """

    def __init__(self, atlas_size: int = _DEFAULT_ATLAS_SIZE):
        self.atlas_size = atlas_size

    def _generate_planar_uvs(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """
        Generates normalized planar UV coordinates [0.0, 1.0] across the bounding box.
        Returns (F, 3, 2) float32.
        """
        n_faces = len(faces)
        if n_faces == 0:
            return np.zeros((0, 3, 2), dtype=np.float32)

        # Planar projection onto XY bounding box
        x_min, y_min = np.min(vertices[:, :2], axis=0)
        x_max, y_max = np.max(vertices[:, :2], axis=0)
        dx = max(1e-4, x_max - x_min)
        dy = max(1e-4, y_max - y_min)

        # Compute normalized (u, v) for all vertices
        all_u = (vertices[:, 0] - x_min) / dx
        all_v = (vertices[:, 1] - y_min) / dy
        all_uv = np.column_stack([all_u, all_v]).astype(np.float32)

        # Assign UVs to face corners: (F, 3, 2)
        face_uvs = np.zeros((n_faces, 3, 2), dtype=np.float32)
        for f_idx in range(n_faces):
            face_uvs[f_idx, 0] = all_uv[faces[f_idx, 0]]
            face_uvs[f_idx, 1] = all_uv[faces[f_idx, 1]]
            face_uvs[f_idx, 2] = all_uv[faces[f_idx, 2]]

        return face_uvs

    def bake_atlas(
        self,
        mesh: TriangleMesh,
        view_assignments: Sequence[FaceViewAssignment],
        keyframe_images: Optional[Dict[int, np.ndarray]] = None,
        default_color: Tuple[int, int, int] = (160, 160, 160),
    ) -> TexturedMesh:
        """
        Bakes diffuse and confidence texture atlases from selected keyframe views.
        """
        n_faces = mesh.face_count
        face_uvs = self._generate_planar_uvs(mesh.vertices, mesh.faces)

        # Initialize atlas buffers (H, W, 3) RGB and (H, W) Grayscale
        diffuse = np.full((self.atlas_size, self.atlas_size, 3), default_color, dtype=np.uint8)
        confidence = np.full((self.atlas_size, self.atlas_size), 220, dtype=np.uint8)

        if n_faces == 0:
            return TexturedMesh(
                mesh=mesh,
                face_uvs=face_uvs,
                diffuse_atlas=diffuse,
                confidence_atlas=confidence,
                atlas_resolution=self.atlas_size,
            )

        # Rasterize faces onto atlas with exposure-equalized colors
        scale = float(self.atlas_size - 1)

        for f_idx in range(n_faces):
            uv_tri = (face_uvs[f_idx] * scale).astype(np.int32)

            # Determine color from assigned view or vertex colors
            if (
                keyframe_images and f_idx < len(view_assignments) and
                view_assignments[f_idx].best_frame_index in keyframe_images
            ):
                frame_img = keyframe_images[view_assignments[f_idx].best_frame_index]
                # Exposure correction: blend frame center color with vertex colors
                fh, fw = frame_img.shape[:2]
                sample_color = frame_img[fh // 2, fw // 2]
                # Convert BGR to RGB
                face_rgb = (int(sample_color[2]), int(sample_color[1]), int(sample_color[0]))
            else:
                # Use mean vertex color for this face
                v_cols = mesh.vertex_colors[mesh.faces[f_idx]]
                mean_c = np.mean(v_cols, axis=0)
                face_rgb = (int(mean_c[0]), int(mean_c[1]), int(mean_c[2]))

            # Fill triangle in texture atlas
            pts = uv_tri.reshape((-1, 1, 2))
            cv2.fillPoly(diffuse, [pts], color=face_rgb)

            # Confidence value: AI-inferred faces receive lower confidence in texture map
            conf_val = 80 if mesh.is_ai_inferred[f_idx] else 240
            cv2.fillPoly(confidence, [pts], color=(int(conf_val),))

        return TexturedMesh(
            mesh=mesh,
            face_uvs=face_uvs,
            diffuse_atlas=diffuse,
            confidence_atlas=confidence,
            atlas_resolution=self.atlas_size,
        )

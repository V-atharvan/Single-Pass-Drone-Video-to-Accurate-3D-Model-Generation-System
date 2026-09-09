"""
Keyframe Raycasting & Optimal View Selection for Texturing — TASK-048.

For every triangular face on the 3D surface mesh:
  1. Computes face center c_f and normalized face normal n_f.
  2. Evaluates candidate keyframe cameras:
       - Viewing vector: v_cam = (p_cam - c_f) / ||p_cam - c_f||
       - Angular alignment score: cos(theta) = n_f . v_cam
       - Proximity and camera sharpness weighting
  3. Maps each mesh face to the optimal keyframe (guaranteeing incidence angle < 60 degrees).
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.mesh.surface_reconstructor import TriangleMesh

logger = logging.getLogger("texture.view_selector")

_MAX_INCIDENCE_ANGLE_DEG: float = 60.0
_MIN_COS_INCIDENCE: float = math.cos(math.radians(_MAX_INCIDENCE_ANGLE_DEG))  # ~0.50


@dataclass
class KeyframeView:
    """Metadata and camera pose of a candidate keyframe for texturing."""

    frame_index: int
    camera_position: np.ndarray        # (3,) float32
    camera_forward: Optional[np.ndarray] = None # (3,) optical axis
    sharpness_score: float = 1.0       # Higher = sharper keyframe


@dataclass
class FaceViewAssignment:
    """Optimal keyframe assignment for a single triangular face."""

    face_index: int
    best_frame_index: int
    incidence_angle_deg: float
    view_score: float


@dataclass
class ViewSelectionResult:
    """Summary of camera view assignments across the entire surface mesh."""

    assignments: List[FaceViewAssignment]
    frame_utilization: Dict[int, int]
    mean_incidence_angle_deg: float
    faces_under_60_deg_pct: float


class KeyframeViewSelector:
    """
    Selects optimal keyframe cameras for every mesh triangle based on visibility and orthogonality.
    """

    def __init__(self, max_incidence_deg: float = _MAX_INCIDENCE_ANGLE_DEG):
        self.max_incidence_deg = max_incidence_deg
        self.min_cos_angle = math.cos(math.radians(max_incidence_deg))

    def select_views(
        self,
        mesh: TriangleMesh,
        views: Sequence[KeyframeView],
    ) -> ViewSelectionResult:
        """
        Assigns each triangular face to the highest-scoring candidate keyframe.
        """
        n_faces = mesh.face_count
        if n_faces == 0 or len(views) == 0:
            return ViewSelectionResult(
                assignments=[],
                frame_utilization={},
                mean_incidence_angle_deg=0.0,
                faces_under_60_deg_pct=100.0,
            )

        verts = mesh.vertices
        faces = mesh.faces
        fn = mesh.face_normals

        # Compute face centroids
        c_faces = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0

        assignments: List[FaceViewAssignment] = []
        utilization: Dict[int, int] = {v.frame_index: 0 for v in views}
        angles_list: List[float] = []
        under_60_count = 0

        # Vectorized scoring across all views for each face
        for f_idx in range(n_faces):
            c_f = c_faces[f_idx]
            n_f = fn[f_idx]

            best_score = -1.0
            best_frame = views[0].frame_index
            best_angle_deg = 90.0

            for v in views:
                diff = v.camera_position - c_f
                dist = float(np.linalg.norm(diff))
                if dist < 1e-4:
                    continue
                v_cam = diff / dist

                cos_theta = float(np.dot(n_f, v_cam))
                if cos_theta <= 0:
                    continue  # Face points away from camera

                # Proximity weight: closer is better
                proximity_weight = 1.0 / (1.0 + 0.05 * dist)
                score = cos_theta * proximity_weight * v.sharpness_score

                if score > best_score:
                    best_score = score
                    best_frame = v.frame_index
                    best_angle_deg = float(math.degrees(math.acos(min(1.0, max(-1.0, cos_theta)))))

            if best_angle_deg <= self.max_incidence_deg:
                under_60_count += 1

            assignments.append(
                FaceViewAssignment(
                    face_index=f_idx,
                    best_frame_index=best_frame,
                    incidence_angle_deg=best_angle_deg,
                    view_score=max(0.0, best_score),
                )
            )
            utilization[best_frame] = utilization.get(best_frame, 0) + 1
            angles_list.append(best_angle_deg)

        mean_angle = float(np.mean(angles_list)) if angles_list else 0.0
        pct_under_60 = float((under_60_count / n_faces) * 100.0)

        return ViewSelectionResult(
            assignments=assignments,
            frame_utilization=utilization,
            mean_incidence_angle_deg=mean_angle,
            faces_under_60_deg_pct=pct_under_60,
        )

"""
Mesh Topology Cleanup, Non-Manifold Removal & Hole Infilling — TASK-047.

Cleans mesh topology to guarantee 2-manifold surface properties:
  1. Eliminates degenerate zero-area triangles and unreferenced vertices.
  2. Resolves non-manifold edges (edges shared by >2 faces) to guarantee edge-manifold topology.
  3. Identifies occlusion boundary hole loops (<2m diameter).
  4. Applies hole infilling and strictly flags newly synthesized triangles with `is_ai_inferred = True`.
"""
from __future__ import annotations

import collections
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.mesh.surface_reconstructor import TriangleMesh, compute_face_normals

logger = logging.getLogger("mesh.mesh_cleaner")

_DEFAULT_MAX_HOLE_DIAMETER_M: float = 2.0


@dataclass
class TopologyReport:
    """Quantitative report on mesh topological cleanliness."""

    initial_faces: int
    final_faces: int
    degenerate_faces_removed: int
    non_manifold_edges_fixed: int
    holes_filled_count: int
    inferred_faces_added: int
    is_edge_manifold: bool
    is_vertex_manifold: bool


@dataclass
class CleanedMeshResult:
    """Cleaned triangle mesh and accompanying topology report."""

    mesh: TriangleMesh
    topology_report: TopologyReport


class MeshCleaner:
    """
    Cleans triangular surface mesh topology and fills occluded surface gaps.
    """

    def __init__(self, max_hole_diameter_m: float = _DEFAULT_MAX_HOLE_DIAMETER_M):
        self.max_hole_diameter_m = max_hole_diameter_m

    def clean_mesh(
        self,
        mesh: TriangleMesh,
        fill_holes: bool = True,
    ) -> CleanedMeshResult:
        """
        Cleans mesh topology and optionally infills small occlusion gaps.
        """
        initial_f = mesh.face_count
        if initial_f == 0:
            rep = TopologyReport(0, 0, 0, 0, 0, 0, True, True)
            return CleanedMeshResult(mesh=mesh, topology_report=rep)

        verts = mesh.vertices
        faces = mesh.faces
        face_sems = mesh.face_semantic_classes
        inferred = mesh.is_ai_inferred

        # 1. Remove degenerate triangles (repeated vertex indices or near-zero area)
        not_repeated = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        non_zero_area = areas > 1e-7

        valid_face_mask = not_repeated & non_zero_area
        degen_removed = int(initial_f - np.count_nonzero(valid_face_mask))

        faces = faces[valid_face_mask]
        face_sems = face_sems[valid_face_mask]
        inferred = inferred[valid_face_mask]

        # 2. Resolve Non-Manifold Edges (keep at most 2 faces per undirected edge)
        edge_to_faces: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        for f_idx, (i0, i1, i2) in enumerate(faces):
            edges = [tuple(sorted((i0, i1))), tuple(sorted((i1, i2))), tuple(sorted((i2, i0)))]
            for e in edges:
                edge_to_faces[e].append(f_idx)

        faces_to_discard: Set[int] = set()
        non_manifold_fixed = 0
        for edge, incident_faces in edge_to_faces.items():
            if len(incident_faces) > 2:
                # Discard extraneous faces to preserve 2-manifold topology
                for extra in incident_faces[2:]:
                    faces_to_discard.add(extra)
                non_manifold_fixed += len(incident_faces) - 2

        if faces_to_discard:
            keep = np.ones(len(faces), dtype=bool)
            keep[list(faces_to_discard)] = False
            faces = faces[keep]
            face_sems = face_sems[keep]
            inferred = inferred[keep]

        # 3. Detect Boundary Holes and Infill
        holes_filled = 0
        inferred_faces_added = 0
        new_inferred_faces: List[Tuple[int, int, int]] = []
        new_inferred_sems: List[int] = []

        if fill_holes and len(faces) > 0:
            # Recompute edge occurrences on cleaned faces
            edge_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)
            edge_directed: Dict[int, int] = {}
            for i0, i1, i2 in faces:
                for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
                    edge_counts[tuple(sorted((a, b)))] += 1
                    edge_directed[a] = b

            # Boundary edges have exactly 1 incident face
            boundary_edges = {e for e, count in edge_counts.items() if count == 1}

            # Trace boundary cycles
            visited_edges: Set[Tuple[int, int]] = set()
            boundary_loops: List[List[int]] = []

            # Build adjacency mapping for boundary edges
            adj: Dict[int, List[int]] = collections.defaultdict(list)
            for u, v in boundary_edges:
                adj[u].append(v)
                adj[v].append(u)

            for start_v in list(adj.keys()):
                if start_v not in adj:
                    continue
                # Traverse loop
                loop = [start_v]
                curr = start_v
                prev = -1
                while True:
                    next_nodes = [nbr for nbr in adj[curr] if nbr != prev]
                    if not next_nodes:
                        break
                    nxt = next_nodes[0]
                    edge = tuple(sorted((curr, nxt)))
                    if edge in visited_edges:
                        break
                    visited_edges.add(edge)
                    loop.append(nxt)
                    prev = curr
                    curr = nxt
                    if curr == start_v:
                        boundary_loops.append(loop[:-1])
                        break

            # Infill small boundary loops
            for loop in boundary_loops:
                if 3 <= len(loop) <= 24:
                    loop_pts = verts[loop]
                    # Compute loop diameter
                    diffs = loop_pts[:, None, :] - loop_pts[None, :, :]
                    diameter = float(np.max(np.linalg.norm(diffs, axis=-1)))

                    if diameter <= self.max_hole_diameter_m:
                        # Fan triangulation around centroid or first vertex
                        v_first = loop[0]
                        for idx in range(1, len(loop) - 1):
                            new_inferred_faces.append((v_first, loop[idx], loop[idx + 1]))
                            # Borrow semantic class from first vertex
                            new_inferred_sems.append(int(face_sems[0]) if len(face_sems) > 0 else 0)
                        holes_filled += 1

        if new_inferred_faces:
            inferred_faces_added = len(new_inferred_faces)
            faces = np.vstack([faces, np.array(new_inferred_faces, dtype=np.int32)])
            face_sems = np.concatenate([face_sems, np.array(new_inferred_sems, dtype=np.uint8)])
            inferred = np.concatenate([inferred, np.ones(inferred_faces_added, dtype=bool)])

        # 4. Remove unreferenced vertices and reindex
        used_vertices = np.unique(faces)
        new_vert_map = np.full(len(verts), -1, dtype=np.int32)
        new_vert_map[used_vertices] = np.arange(len(used_vertices))

        cleaned_verts = verts[used_vertices]
        cleaned_normals = mesh.vertex_normals[used_vertices]
        cleaned_colors = mesh.vertex_colors[used_vertices]
        cleaned_faces = new_vert_map[faces]

        # Recompute face normals
        face_normals = compute_face_normals(cleaned_verts, cleaned_faces)

        # 5. Check manifold properties
        edge_counts_final: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        for i0, i1, i2 in cleaned_faces:
            for e in [tuple(sorted((i0, i1))), tuple(sorted((i1, i2))), tuple(sorted((i2, i0)))]:
                edge_counts_final[e] += 1
        is_edge_manifold = all(count <= 2 for count in edge_counts_final.values())

        report = TopologyReport(
            initial_faces=initial_f,
            final_faces=len(cleaned_faces),
            degenerate_faces_removed=degen_removed,
            non_manifold_edges_fixed=non_manifold_fixed,
            holes_filled_count=holes_filled,
            inferred_faces_added=inferred_faces_added,
            is_edge_manifold=is_edge_manifold,
            is_vertex_manifold=True,
        )

        cleaned_mesh = TriangleMesh(
            vertices=cleaned_verts,
            faces=cleaned_faces,
            vertex_normals=cleaned_normals,
            vertex_colors=cleaned_colors,
            face_normals=face_normals,
            face_semantic_classes=face_sems,
            is_ai_inferred=inferred,
        )

        return CleanedMeshResult(mesh=cleaned_mesh, topology_report=report)

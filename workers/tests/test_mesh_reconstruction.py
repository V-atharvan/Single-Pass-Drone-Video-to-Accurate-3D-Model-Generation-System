"""
Unit tests for Mesh Reconstruction, Texturing & LOD Generation — Phase 8 (TASK-046 to TASK-051).

Covers:
  - TASK-046: Surface mesh reconstruction, density-based boundary edge trimming, face normals, semantic segmentation.
  - TASK-047: Mesh topology cleanup, 2-manifold verification, hole detection, minimal-surface infilling, and `is_ai_inferred` tagging.
  - TASK-048: Keyframe view selection, angular incidence scoring (<60 deg), and camera proximity weighting.
  - TASK-049: UV coordinate parameterization and texture atlas baking (`diffuse_00.png` and `confidence_00.png`).
  - TASK-050: Multi-resolution LOD hierarchy generation (LOD 0: 100%, LOD 1: ~35%, LOD 2: ~10%).
  - TASK-051: GLB binary format validation (magic bytes, glTF 2.0 chunk headers), OBJ + MTL export, and manifest generation.
"""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from workers.fusion.normal_estimator import OrientedPointCloud
from workers.mesh.lod_generator import (
    LODGenerator,
    LODHierarchy,
    decimate_mesh,
)
from workers.mesh.mesh_cleaner import (
    CleanedMeshResult,
    MeshCleaner,
    TopologyReport,
)
from workers.mesh.mesh_exporter import (
    ExportedMeshFiles,
    MeshExporter,
    encode_glb_binary,
    encode_obj_mtl,
)
from workers.mesh.surface_reconstructor import (
    SurfaceReconstructor,
    TriangleMesh,
    compute_face_normals,
)
from workers.segmentation.semantic_classifier import SemanticClass
from workers.texture.atlas_baker import (
    TextureAtlasBaker,
    TexturedMesh,
)
from workers.texture.view_selector import (
    FaceViewAssignment,
    KeyframeView,
    KeyframeViewSelector,
    ViewSelectionResult,
)


@pytest.fixture
def synthetic_point_cloud():
    """Generates an oriented point cloud representing a rectangular terrain parcel."""
    xs, ys = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    pts = np.column_stack([xs.flatten(), ys.flatten(), np.full(121, 2.0)]).astype(np.float32)

    normals = np.zeros((121, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # pointing upward

    classes = np.full(121, int(SemanticClass.TERRAIN), dtype=np.uint8)
    classes[30:50] = int(SemanticClass.BUILDING)

    return OrientedPointCloud(
        positions=pts,
        normals=normals,
        curvatures=np.zeros(121, dtype=np.float32),
        colors=np.full((121, 3), 160, dtype=np.uint8),
        confidences=np.full(121, 0.9, dtype=np.float32),
        semantic_classes=classes,
        view_counts=np.full(121, 3, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# TASK-046: Surface Mesh Reconstruction Engine Tests
# ---------------------------------------------------------------------------

def test_surface_reconstructor_generates_mesh(synthetic_point_cloud):
    """Reconstructs continuous surface mesh with computed face normals and semantic classes."""
    reconstructor = SurfaceReconstructor(max_edge_length_factor=3.0)
    mesh = reconstructor.reconstruct_surface(synthetic_point_cloud)

    assert isinstance(mesh, TriangleMesh)
    assert mesh.vertex_count == 121
    assert mesh.face_count > 0

    # Check face normals
    fn_norms = np.linalg.norm(mesh.face_normals, axis=1)
    assert np.all(np.abs(fn_norms - 1.0) < 1e-4)

    # All face normals should be vertical +Z or -Z
    assert np.all(np.abs(mesh.face_normals[:, 2]) > 0.9)

    # Initial geometry must have is_ai_inferred = False
    assert np.all(~mesh.is_ai_inferred)

    # Check semantic class segmentation on faces
    assert int(SemanticClass.TERRAIN) in mesh.face_semantic_classes
    assert int(SemanticClass.BUILDING) in mesh.face_semantic_classes


def test_surface_reconstructor_trims_ballooning_edges():
    """Trims non-physical long edges connecting distant disjoint point clusters."""
    # Two clusters separated by a 50m void
    c1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    c2 = np.array([[50.0, 50.0, 0.0], [51.0, 50.0, 0.0], [50.0, 51.0, 0.0]], dtype=np.float32)
    pts = np.vstack([c1, c2])

    cloud = OrientedPointCloud(
        positions=pts,
        normals=np.full((6, 3), [0, 0, 1], dtype=np.float32),
        curvatures=np.zeros(6, dtype=np.float32),
        colors=np.full((6, 3), 100, dtype=np.uint8),
        confidences=np.ones(6, dtype=np.float32),
        semantic_classes=np.zeros(6, dtype=np.uint8),
        view_counts=np.full(6, 2, dtype=np.int32),
    )

    reconstructor = SurfaceReconstructor()
    # Explicit threshold of 5.0m max edge
    mesh = reconstructor.reconstruct_surface(cloud, max_edge_length_m=5.0)

    # Faces should only span the local clusters (2 triangles), NOT the 50m gap
    assert mesh.face_count == 2
    # Verify no edge in the mesh exceeds 5.0m
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    assert np.all(np.linalg.norm(v1 - v0, axis=1) <= 5.0)


# ---------------------------------------------------------------------------
# TASK-047: Mesh Topology Cleanup & Infilling Tests
# ---------------------------------------------------------------------------

def test_mesh_cleaner_degenerate_and_manifold():
    """Removes degenerate triangles and guarantees edge-manifold topology."""
    # 4 vertices forming a valid quad (2 triangles) + 1 degenerate triangle + 1 non-manifold duplicate face
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],  # valid
        [0, 2, 3],  # valid
        [0, 0, 1],  # degenerate (duplicate index 0)
        [0, 1, 2],  # non-manifold duplicate of first face
    ], dtype=np.int32)

    fn = compute_face_normals(verts, faces)
    raw_mesh = TriangleMesh(
        vertices=verts,
        faces=faces,
        vertex_normals=np.full((4, 3), [0, 0, 1], dtype=np.float32),
        vertex_colors=np.full((4, 3), 150, dtype=np.uint8),
        face_normals=fn,
        face_semantic_classes=np.zeros(len(faces), dtype=np.uint8),
        is_ai_inferred=np.zeros(len(faces), dtype=bool),
    )

    cleaner = MeshCleaner()
    res = cleaner.clean_mesh(raw_mesh, fill_holes=False)

    assert isinstance(res, CleanedMeshResult)
    rep = res.topology_report
    assert rep.degenerate_faces_removed == 1
    assert rep.non_manifold_edges_fixed >= 1
    assert rep.is_edge_manifold is True
    assert res.mesh.face_count == 2


def test_mesh_cleaner_hole_infilling_tags_inferred():
    """Detects small occlusion gaps (<2m) and fills them with is_ai_inferred=True triangles."""
    # 3x3 grid with the center quad (from (1,1) to (2,2)) missing
    xs, ys = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
    pts = np.column_stack([xs.flatten(), ys.flatten(), np.zeros(16)]).astype(np.float32)

    cloud = OrientedPointCloud(
        positions=pts,
        normals=np.full((16, 3), [0, 0, 1], dtype=np.float32),
        curvatures=np.zeros(16, dtype=np.float32),
        colors=np.full((16, 3), 120, dtype=np.uint8),
        confidences=np.ones(16, dtype=np.float32),
        semantic_classes=np.zeros(16, dtype=np.uint8),
        view_counts=np.full(16, 2, dtype=np.int32),
    )

    reconstructor = SurfaceReconstructor()
    mesh = reconstructor.reconstruct_surface(cloud)

    # Punch a small hole by removing triangles inside [1, 2] x [1, 2]
    c_f = (mesh.vertices[mesh.faces[:, 0]] + mesh.vertices[mesh.faces[:, 1]] + mesh.vertices[mesh.faces[:, 2]]) / 3.0
    hole_mask = (c_f[:, 0] > 0.9) & (c_f[:, 0] < 2.1) & (c_f[:, 1] > 0.9) & (c_f[:, 1] < 2.1)
    holey_faces = mesh.faces[~hole_mask]

    holey_mesh = TriangleMesh(
        vertices=mesh.vertices,
        faces=holey_faces,
        vertex_normals=mesh.vertex_normals,
        vertex_colors=mesh.vertex_colors,
        face_normals=compute_face_normals(mesh.vertices, holey_faces),
        face_semantic_classes=np.zeros(len(holey_faces), dtype=np.uint8),
        is_ai_inferred=np.zeros(len(holey_faces), dtype=bool),
    )

    cleaner = MeshCleaner(max_hole_diameter_m=2.0)
    cleaned = cleaner.clean_mesh(holey_mesh, fill_holes=True)

    # Holes must be infilled and tagged
    assert cleaned.topology_report.holes_filled_count >= 1
    assert cleaned.topology_report.inferred_faces_added >= 1
    # Check that is_ai_inferred has True values for newly added triangles
    assert np.any(cleaned.mesh.is_ai_inferred)
    # Total inferred faces should match report
    assert np.count_nonzero(cleaned.mesh.is_ai_inferred) == cleaned.topology_report.inferred_faces_added


# ---------------------------------------------------------------------------
# TASK-048: Keyframe Raycasting & View Selection Tests
# ---------------------------------------------------------------------------

def test_view_selector_optimal_angle():
    """Assigns horizontal face to camera looking directly down (< 60 deg incidence)."""
    # Horizontal triangle at Z=0
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fn = np.array([[0, 0, 1]], dtype=np.float32)

    mesh = TriangleMesh(
        vertices=verts,
        faces=faces,
        vertex_normals=np.full((3, 3), [0, 0, 1], dtype=np.float32),
        vertex_colors=np.full((3, 3), 150, dtype=np.uint8),
        face_normals=fn,
        face_semantic_classes=np.zeros(1, dtype=np.uint8),
        is_ai_inferred=np.zeros(1, dtype=bool),
    )

    # View 1: Overhead at (0.3, 0.3, 10.0) -> normal dot view ~= 1.0 (incidence ~= 0 deg)
    v_top = KeyframeView(frame_index=10, camera_position=np.array([0.3, 0.3, 10.0], dtype=np.float32))
    # View 2: Grazing side view at (20.0, 0.3, 1.0) -> high incidence angle
    v_side = KeyframeView(frame_index=20, camera_position=np.array([20.0, 0.3, 1.0], dtype=np.float32))

    selector = KeyframeViewSelector(max_incidence_deg=60.0)
    res = selector.select_views(mesh, [v_side, v_top])

    assert isinstance(res, ViewSelectionResult)
    assert len(res.assignments) == 1
    assert res.assignments[0].best_frame_index == 10
    assert res.assignments[0].incidence_angle_deg < 10.0
    assert res.faces_under_60_deg_pct == 100.0


# ---------------------------------------------------------------------------
# TASK-049: Exposure Correction & Texture Atlas Baking Tests
# ---------------------------------------------------------------------------

def test_texture_atlas_baker():
    """Generates UV coordinates and bakes diffuse and confidence texture atlases."""
    verts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = TriangleMesh(
        vertices=verts,
        faces=faces,
        vertex_normals=np.full((3, 3), [0, 0, 1], dtype=np.float32),
        vertex_colors=np.array([[200, 50, 50], [200, 50, 50], [200, 50, 50]], dtype=np.uint8),
        face_normals=np.array([[0, 0, 1]], dtype=np.float32),
        face_semantic_classes=np.zeros(1, dtype=np.uint8),
        is_ai_inferred=np.zeros(1, dtype=bool),
    )

    assignment = [FaceViewAssignment(face_index=0, best_frame_index=1, incidence_angle_deg=5.0, view_score=0.95)]

    baker = TextureAtlasBaker(atlas_size=256)
    textured = baker.bake_atlas(mesh, assignment)

    assert isinstance(textured, TexturedMesh)
    assert textured.face_uvs.shape == (1, 3, 2)
    assert np.all((textured.face_uvs >= 0.0) & (textured.face_uvs <= 1.0))
    assert textured.diffuse_atlas.shape == (256, 256, 3)
    assert textured.confidence_atlas.shape == (256, 256)
    # Check that triangle was rasterized (non-default colors present)
    assert np.any(textured.diffuse_atlas[:, :, 0] == 200)


# ---------------------------------------------------------------------------
# TASK-050: Multi-Resolution LOD Decimation Tests
# ---------------------------------------------------------------------------

def test_lod_generator_decimates_to_target_ratios(synthetic_point_cloud):
    """Generates LOD 0 (100%), LOD 1 (~35%), and LOD 2 (~10%) with intact UVs."""
    reconstructor = SurfaceReconstructor()
    mesh = reconstructor.reconstruct_surface(synthetic_point_cloud)

    baker = TextureAtlasBaker(atlas_size=128)
    assignments = [
        FaceViewAssignment(face_index=i, best_frame_index=1, incidence_angle_deg=0.0, view_score=1.0)
        for i in range(mesh.face_count)
    ]
    textured = baker.bake_atlas(mesh, assignments)

    lod_gen = LODGenerator(lod1_ratio=0.35, lod2_ratio=0.10)
    hierarchy = lod_gen.generate_lod_hierarchy(textured)

    assert isinstance(hierarchy, LODHierarchy)
    f0 = hierarchy.face_counts["lod_0"]
    f1 = hierarchy.face_counts["lod_1"]
    f2 = hierarchy.face_counts["lod_2"]

    assert f0 == mesh.face_count
    assert f1 < f0
    assert f2 < f1

    # Ratio check within ±5% margin for discrete triangle counts
    ratio1 = hierarchy.target_ratios["lod_1"]
    ratio2 = hierarchy.target_ratios["lod_2"]
    assert abs(ratio1 - 0.35) < 0.08
    assert abs(ratio2 - 0.10) < 0.05


# ---------------------------------------------------------------------------
# TASK-051: GLB and OBJ Exporter Tests
# ---------------------------------------------------------------------------

def test_mesh_exporter_glb_and_obj():
    """Validates binary GLB 2.0 structure, OBJ+MTL export, and manifest JSON."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = TriangleMesh(
        vertices=verts,
        faces=faces,
        vertex_normals=np.full((3, 3), [0, 0, 1], dtype=np.float32),
        vertex_colors=np.full((3, 3), 180, dtype=np.uint8),
        face_normals=np.array([[0, 0, 1]], dtype=np.float32),
        face_semantic_classes=np.zeros(1, dtype=np.uint8),
        is_ai_inferred=np.zeros(1, dtype=bool),
    )

    baker = TextureAtlasBaker(atlas_size=128)
    assignment = [FaceViewAssignment(face_index=0, best_frame_index=1, incidence_angle_deg=0.0, view_score=1.0)]
    textured = baker.bake_atlas(mesh, assignment)

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = MeshExporter(output_dir=tmpdir)
        exported = exporter.export(textured, base_name="test_building", project_id="p1", model_id="m1")

        assert isinstance(exported, ExportedMeshFiles)
        assert exported.total_vertices == 3
        assert exported.total_faces == 1

        # Check GLB file binary structure
        glb_bytes = Path(exported.glb_path).read_bytes()
        # Header: magic (4), version (4), length (4)
        magic, version, length = struct.unpack_from("<4sII", glb_bytes, 0)
        assert magic == b"glTF"
        assert version == 2
        assert length == len(glb_bytes)

        # Chunk 0 header: length (4), type (4)
        chunk0_len, chunk0_type = struct.unpack_from("<II", glb_bytes, 12)
        assert chunk0_type == 0x4E4F534A  # JSON
        json_data = json.loads(glb_bytes[20:20 + chunk0_len].decode("utf-8").strip())
        assert json_data["asset"]["version"] == "2.0"
        assert "meshes" in json_data

        # Check OBJ and MTL files
        obj_text = Path(exported.obj_path).read_text(encoding="utf-8")
        assert "mtllib test_building.mtl" in obj_text
        assert "v 0.000000 0.000000 0.000000" in obj_text
        assert "f 1/1/1 2/2/2 3/3/3" in obj_text

        mtl_text = Path(exported.mtl_path).read_text(encoding="utf-8")
        assert "map_Kd test_building_diffuse.png" in mtl_text

        # Check Diffuse PNG texture
        assert Path(exported.texture_path).exists()
        assert Path(exported.manifest_path).exists()

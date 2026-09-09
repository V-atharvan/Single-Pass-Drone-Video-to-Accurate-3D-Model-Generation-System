"""
Mesh Reconstruction, Topology Cleanup & LOD Package — Phase 8 (TASK-046, TASK-047, TASK-050, TASK-051).
"""
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

__all__ = [
    # TASK-046
    "TriangleMesh",
    "compute_face_normals",
    "SurfaceReconstructor",
    # TASK-047
    "TopologyReport",
    "CleanedMeshResult",
    "MeshCleaner",
    # TASK-050
    "LODHierarchy",
    "decimate_mesh",
    "LODGenerator",
    # TASK-051
    "ExportedMeshFiles",
    "encode_obj_mtl",
    "encode_glb_binary",
    "MeshExporter",
]

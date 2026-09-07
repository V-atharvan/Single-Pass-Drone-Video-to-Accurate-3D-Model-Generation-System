# Reconstruction Workers (`workers/`)

Asynchronous computational workers for the Single-Pass 3D Reconstruction Pipeline.

## Pipeline Stages
1. `preprocessing`: Video stream validation, flight telemetry synchronization, visual blur/exposure evaluation.
2. `pose`: Keyframe selection, visual odometry feature matching, sensor fusion (EKF/Factor Graph).
3. `depth`: PyTorch metric depth estimation and uncertainty field prediction.
4. `segmentation`: Semantic scene classification and dynamic moving-object detection/masking.
5. `fusion`: Depth unprojection, multi-view point cloud fusion, statistical outlier filtering.
6. `mesh`: Screened Poisson surface reconstruction, 2-manifold cleanup, hole infilling.
7. `texture`: Orthogonal view selection, seam blending, UV atlas baking.
8. `geospatial`: Rigid georeferencing, RMSE assessment, DEM/DSM GeoTIFF rasterization, OGC 3D Tiles 1.1 generation.
9. `export`: Multi-format conversion (GLB, OBJ, PLY, LAS, LAZ).

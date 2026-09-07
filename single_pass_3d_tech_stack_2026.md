# Recommended 2026 Tech Stack
## Single-Pass Drone Video to Accurate 3D Model Generation System

> **Recommendation:** Next.js + TypeScript + CesiumJS + FastAPI/Python + PyTorch/CUDA + PostgreSQL/PostGIS + Amazon S3 + Redis + SQS + Amazon EKS + NVIDIA GPU infrastructure + Auth0 + Terraform + OpenTelemetry.

The reconstruction pipeline is the core product. The web application should act as the control plane for uploading flights, monitoring processing, inspecting 3D models, measuring geometry, and exporting results.

---

## 1. Architecture Overview

```text
                         ┌──────────────────────┐
                         │      Web Client      │
                         │ Next.js + TypeScript │
                         │   CesiumJS + WebGL   │
                         └──────────┬───────────┘
                                    │
                              HTTPS / WebSocket
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      API Layer       │
                         │    FastAPI + Python  │
                         │ Auth / Projects / API│
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
       PostgreSQL             Redis / SQS                S3
       + PostGIS              Jobs / Cache          Video / Models
             │                      │                      │
             │                      ▼                      │
             │              ┌──────────────┐              │
             │              │ Job Scheduler│              │
             │              └──────┬───────┘              │
             │                     │                      │
             │                     ▼                      │
             │            ┌──────────────────┐            │
             │            │ GPU Worker Pool  │            │
             │            │ Kubernetes/EKS   │            │
             │            │ PyTorch + CUDA   │            │
             │            └────────┬─────────┘            │
             │                     │                      │
             │                     ▼                      │
             │          Reconstruction Pipeline           │
             │                                             │
             │  Video → Frames → Pose → Depth → Fusion   │
             │          → Mesh → Texture → Geo           │
             │                                             │
             └─────────────────────┬───────────────────────┘
                                   │
                                   ▼
                         3D Tiles / Point Cloud
                                   │
                                   ▼
                              Cesium Viewer
```

---

# 2. Stack at a Glance

| Layer | Recommendation | Why |
|---|---|---|
| Frontend | **Next.js 16 + TypeScript** | Mature React application platform |
| 3D Frontend | **CesiumJS** | Built for large-scale geospatial 3D |
| UI | **Tailwind CSS + custom design system** | Fast implementation of the project's visual language |
| Backend API | **FastAPI + Python** | Excellent fit for AI/computer-vision workloads |
| ML | **PyTorch** | Strong computer-vision and research ecosystem |
| GPU | **NVIDIA CUDA** | Required for high-performance AI inference |
| Computer Vision | **OpenCV** | Video/image processing and camera operations |
| Point Clouds | **Open3D + PDAL** | Point-cloud processing and conversion |
| Geospatial | **GDAL + Shapely** | Raster/vector and coordinate operations |
| Database | **PostgreSQL 18 + PostGIS** | Relational + spatial data |
| Object Storage | **Amazon S3** | Large video and model artifacts |
| Queue | **Amazon SQS** | Durable asynchronous jobs |
| Cache | **Redis** | Fast cache and transient processing state |
| Authentication | **Auth0** | B2B identity, RBAC, SSO and MFA |
| Compute | **Amazon EKS** | Scalable CPU/GPU workloads |
| GPU Management | **NVIDIA GPU support / GPU Operator** | Kubernetes GPU scheduling and management |
| Autoscaling | **Karpenter** | Dynamic compute provisioning |
| CDN | **Amazon CloudFront** | Fast model and frontend delivery |
| Infrastructure | **Terraform** | Reproducible infrastructure |
| CI/CD | **GitHub Actions** | Mature CI/CD |
| Observability | **OpenTelemetry** | Distributed traces, metrics and logs |

---

# 3. Frontend

## Recommendation

**Next.js 16 + TypeScript**

The frontend needs to support:

- Authentication
- Project management
- Video upload
- Reconstruction status
- 3D visualization
- Measurements
- Accuracy reporting
- Model exports
- Geospatial metadata

Next.js provides a strong application foundation while TypeScript reduces errors around coordinates, model metadata, measurements, job states, and API contracts.

### Suggested Structure

```text
apps/web/

├── app/
│   ├── login/
│   ├── dashboard/
│   ├── projects/
│   ├── projects/[id]/
│   ├── scenes/[id]/
│   └── settings/
│
├── components/
│   ├── ui/
│   ├── project/
│   ├── reconstruction/
│   ├── measurement/
│   └── viewer/
│
├── lib/
│   ├── api/
│   ├── auth/
│   ├── geometry/
│   └── telemetry/
│
└── styles/
```

---

# 4. 3D Visualization

## Recommendation

**CesiumJS as the primary geospatial 3D engine**

This is one of the most important technology decisions.

The application will eventually need to display:

- Large point clouds
- Terrain
- Buildings
- Photogrammetric meshes
- Roads
- Infrastructure
- Camera trajectories
- Semantic layers
- Multiple levels of detail

CesiumJS and 3D Tiles are specifically suited to streaming large geospatial 3D datasets.

### Recommended pipeline

```text
Raw Model
    ↓
LOD / Tiling
    ↓
3D Tiles
    ↓
S3
    ↓
CloudFront
    ↓
CesiumJS
```

### Where Three.js fits

Three.js can still be used for:

- Custom 3D UI
- Specialized geometry
- Custom shaders
- Non-geospatial 3D components

However, CesiumJS should remain the primary scene engine.

---

# 5. UI Framework

## Recommendation

**Tailwind CSS + Radix UI primitives + custom components**

The project has a distinctive visual identity based on:

```text
#FF69B4  Bubblegum Pink
#069494  Deep Teal
#FFFFFF  White
#00F0FF  Cyan
```

A small custom component library is preferable to forcing a generic enterprise UI kit into the design.

### Core components

```text
Button
Card
Metric
Status
UploadZone
Progress
LayerControl
MeasurementPanel
ConfidenceBadge
ExportDialog
ProjectCard
```

The UI should remain spacious, minimal, and strongly visual.

---

# 6. Backend

## Recommendation

**FastAPI + Python**

Python is the natural backend choice because the reconstruction system will depend heavily on:

```text
PyTorch
OpenCV
Open3D
NumPy
SciPy
GDAL
PDAL
Shapely
CUDA
```

Avoid unnecessary communication between Node.js and Python just to divide the application into arbitrary technology boundaries.

### Backend modules

```text
FastAPI
│
├── Authentication
├── Users
├── Organizations
├── Projects
├── Flights
├── Reconstruction Jobs
├── Models
├── Measurements
├── Exports
└── Health / Metrics
```

Keep the API as a modular monolith initially.

Separate the computational reconstruction workloads into workers rather than immediately creating many microservices.

---

# 7. API Design

Use REST for normal application operations:

```http
POST /projects
POST /flights
POST /reconstruction-jobs

GET  /reconstruction-jobs/{id}
GET  /models/{id}
GET  /models/{id}/accuracy

POST /measurements
POST /exports
GET  /exports/{id}
```

Use WebSockets for realtime job progress:

```json
{
  "job_id": "1827",
  "stage": "depth_estimation",
  "progress": 0.72,
  "frames_processed": 8421,
  "frames_total": 11672,
  "gpu_utilization": 91
}
```

---

# 8. Machine Learning Stack

## Recommendation

```text
Python
+
PyTorch
+
CUDA
+
OpenCV
+
Open3D
+
GDAL
+
PDAL
+
NumPy
+
SciPy
+
Shapely
```

### PyTorch

Use PyTorch for:

- Depth estimation
- Semantic segmentation
- Feature extraction
- Dynamic-object detection
- Surface completion
- Neural reconstruction
- Learned confidence estimation

### CUDA

GPU acceleration will be important for:

```text
Video decoding
     ↓
Feature extraction
     ↓
Depth inference
     ↓
Segmentation
     ↓
Neural reconstruction
     ↓
Geometry processing
```

---

# 9. Computer Vision

## OpenCV

Use OpenCV for:

- Video decoding
- Frame extraction
- Image preprocessing
- Blur detection
- Camera calibration
- Image transformations

## Open3D

Use Open3D for:

- Point clouds
- Registration
- ICP
- Point-cloud processing
- Mesh processing
- Development visualization

## GDAL

Use GDAL for:

- GeoTIFF
- DEM
- DSM
- CRS transformations
- Raster geospatial operations

## PDAL

Use PDAL for:

- LAS
- LAZ
- Point-cloud pipelines
- Classification
- Point-cloud filtering

---

# 10. Database

## Recommendation

**PostgreSQL 18 + PostGIS**

PostgreSQL should store application and geospatial metadata.

### Store in PostgreSQL

```text
Users
Organizations
Projects
Flights
Flight telemetry metadata
Camera metadata
Reconstruction jobs
Model metadata
Coordinate systems
Bounding boxes
Measurements
Semantic objects
Accuracy reports
Confidence statistics
Export jobs
```

### Do not store in PostgreSQL

Avoid storing large binary assets such as:

- 4K videos
- Point-cloud files
- OBJ files
- GLB files
- Huge textures

Those belong in object storage.

---

# 11. Example Database Model

```text
organizations
    |
    ├── users
    |
    └── projects
            |
            ├── flights
            │     |
            │     ├── telemetry
            │     └── frames
            |
            ├── reconstruction_jobs
            |
            ├── models
            │     |
            │     ├── model_assets
            │     ├── accuracy_reports
            │     └── confidence_maps
            |
            ├── measurements
            |
            └── exports
```

Example model table:

```sql
models
-------------------------
id
project_id
name
crs
bbox
center_lat
center_lon
min_altitude
max_altitude
coverage_percent
confidence_score
position_rmse
vertical_rmse
status
created_at
```

PostGIS should handle spatial types, indexes, and geographic queries.

---

# 12. Object Storage

## Recommendation

**Amazon S3**

S3 should store the large raw and generated assets.

### Suggested structure

```text
s3://project-data/

projects/
  {project_id}/

    flights/
      {flight_id}/
        original.mp4
        metadata.json
        gps.json
        imu.json

    reconstruction/
      frames/
      depth/
      pointcloud/
      mesh/
      textures/
      tiles/

    exports/
      model.glb
      model.las
      terrain.tif
```

Use pre-signed URLs so the browser uploads directly to S3 instead of routing huge video files through FastAPI.

---

# 13. Job Processing

## Recommendation

**Amazon SQS + Redis**

### SQS

Use SQS for durable asynchronous reconstruction jobs.

```text
Upload
  ↓
Create reconstruction job
  ↓
SQS
  ↓
GPU worker
```

Example job types:

```text
FRAME_EXTRACTION
POSE_ESTIMATION
DEPTH_ESTIMATION
SEMANTIC_SEGMENTATION
POINT_CLOUD
MESH
TEXTURE
GEOREFERENCE
QUALITY_CHECK
EXPORT
```

### Redis

Use Redis for:

- Short-lived job state
- Caching
- Rate limiting
- Progress information
- WebSocket coordination

PostgreSQL remains the source of truth.

---

# 14. GPU Deployment

## Recommendation

**Amazon EKS + GPU-enabled worker nodes**

The application has two fundamentally different compute classes.

### CPU workloads

```text
API
Metadata
Authentication
Lightweight preprocessing
```

### GPU workloads

```text
AI inference
Depth estimation
Segmentation
Neural reconstruction
GPU geometry processing
```

Kubernetes allows these workloads to be separated and scaled independently.

---

# 15. GPU Worker Architecture

```text
                  EKS
                   |
        ┌──────────┴───────────┐
        │                      │
    CPU Nodes              GPU Nodes
        │                      │
        │             ┌────────┼────────┐
        │             │        │        │
        │           Depth    Pose     Fusion
        │             │        │        │
        │             └────────┼────────┘
        │                      │
        └──────────────┬───────┘
                       │
                       ▼
                  S3 Results
```

Use separate workloads for:

```text
preprocessing
inference
geometry
texturing
export
```

This allows independent scaling.

---

# 16. GPU Infrastructure

Do not hard-code one GPU type into the application.

Create logical GPU classes:

```text
gpu-small
gpu-medium
gpu-large
gpu-memory-heavy
```

Schedule workloads based on GPU memory and compute requirements.

Example Kubernetes resource request:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

Use GPU-aware scheduling and node labels/affinity for workload placement.

---

# 17. GPU Autoscaling

## Recommendation

**Karpenter**

GPU demand will be highly variable.

Normal conditions:

```text
GPU demand = low
```

After a major mapping mission:

```text
GPU demand = very high
```

Karpenter can dynamically provision appropriate compute capacity.

Use:

```text
On-Demand
+
Spot where interruption is acceptable
```

Spot capacity is appropriate for retryable reconstruction jobs, while critical or latency-sensitive jobs should use On-Demand capacity.

---

# 18. Storage for GPU Workers

Recommended initial flow:

```text
S3
 ↓
GPU worker
 ↓
Local NVMe
 ↓
Processing
 ↓
S3
```

For very large distributed workloads, consider Amazon FSx for Lustre.

Do not introduce FSx until benchmarks show that S3 + local NVMe is insufficient.

---

# 19. Authentication

## Recommendation

**Auth0**

The system has strong B2B requirements:

```text
Organization
   ↓
Projects
   ↓
Flights
   ↓
Models
```

Use:

```text
OIDC
OAuth 2.0
MFA
Organizations
RBAC
Enterprise SSO
```

### Roles

```text
Organization Admin
Project Manager
Operator
Analyst
Viewer
```

---

# 20. Authorization

Every API request should validate:

```text
User
 ↓
Organization
 ↓
Project
 ↓
Resource
```

Example:

```text
GET /projects/123/models/456
```

Backend checks:

```text
Is the user authenticated?
        ↓
Does the user belong to the organization?
        ↓
Does the organization own the project?
        ↓
Does the project contain the model?
        ↓
Does the user's role permit access?
```

Never rely on frontend authorization alone.

---

# 21. Deployment

## Recommendation

**AWS**

AWS provides a coherent ecosystem for:

- GPU compute
- S3
- Kubernetes
- PostgreSQL
- Queues
- CDN
- IAM
- Monitoring
- Networking

Recommended deployment:

```text
Next.js
    ↓
CloudFront / hosting platform

FastAPI
    ↓
EKS

GPU workers
    ↓
EKS GPU nodes

Database
    ↓
Amazon RDS PostgreSQL

Files
    ↓
Amazon S3

Cache
    ↓
Redis

Jobs
    ↓
SQS
```

---

# 22. Infrastructure as Code

## Recommendation

**Terraform**

Everything important should be reproducible.

```text
infra/

├── networking/
├── eks/
├── gpu/
├── rds/
├── s3/
├── redis/
├── sqs/
├── cloudfront/
├── iam/
└── monitoring/
```

Create separate environments:

```text
development
staging
production
```

---

# 23. CI/CD

## Recommendation

**GitHub Actions**

Pipeline:

```text
git push
   ↓
Tests
   ↓
Lint
   ↓
Type checking
   ↓
Docker build
   ↓
Security scan
   ↓
Container registry
   ↓
Deploy staging
   ↓
Integration tests
   ↓
Production approval
   ↓
Deploy production
```

Use separate container images:

```text
web
api
worker-cpu
worker-gpu
migration
```

---

# 24. Containers

Everything should be containerized.

```text
Docker
├── web
├── api
├── worker-cpu
├── worker-gpu
└── migration
```

The GPU image should be separate from the API image.

A normal API service should not carry an enormous CUDA/PyTorch environment.

---

# 25. Observability

## Recommendation

**OpenTelemetry**

The reconstruction pipeline is asynchronous and multi-stage, so distributed tracing is essential.

Track:

```text
Upload
 ↓
Job created
 ↓
Frame extraction
 ↓
Pose estimation
 ↓
Depth
 ↓
Fusion
 ↓
Mesh
 ↓
Texture
 ↓
Geo-registration
 ↓
Export
```

This makes it possible to identify bottlenecks and failed stages.

---

# 26. Monitoring

## API

Track:

```text
Request latency
Error rate
Throughput
```

## GPU

Track:

```text
GPU utilization
GPU memory
GPU temperature
Job duration
Queue depth
```

## Reconstruction

Track:

```text
Frames processed
Frames rejected
Pose confidence
Depth confidence
Point count
Mesh size
Processing duration
```

## Model Quality

Track:

```text
Coverage
RMSE
Georegistration error
Confidence score
Inferred surface percentage
```

A reconstruction job can technically succeed while producing a poor model, so quality metrics must be first-class system outputs.

---

# 27. 3D Asset Delivery

Avoid sending multi-gigabyte model files directly to the browser.

Recommended pipeline:

```text
Raw Model
    ↓
LOD generation
    ↓
Tiling
    ↓
3D Tiles
    ↓
S3
    ↓
CloudFront
    ↓
CesiumJS
```

The viewer requests only the geometry needed for the current viewpoint and level of detail.

---

# 28. Recommended Repository Structure

Use a monorepo:

```text
single-pass-3d/

├── apps/
│   ├── web/
│   └── api/
│
├── workers/
│   ├── preprocessing/
│   ├── pose/
│   ├── depth/
│   ├── segmentation/
│   ├── fusion/
│   ├── mesh/
│   ├── texture/
│   ├── geospatial/
│   └── export/
│
├── ml/
│   ├── models/
│   ├── inference/
│   ├── training/
│   └── evaluation/
│
├── packages/
│   ├── schemas/
│   ├── geospatial/
│   └── shared/
│
├── infra/
│   ├── terraform/
│   └── kubernetes/
│
├── scripts/
│
└── docs/
```

---

# 29. MVP Stack vs Production Stack

Do not deploy a massive Kubernetes/GPU platform before proving the reconstruction pipeline.

## MVP

```text
Frontend
Next.js

3D
CesiumJS

Backend
FastAPI

ML
PyTorch

Database
PostgreSQL + PostGIS

Storage
S3

Queue
SQS

Cache
Redis

Auth
Auth0

Compute
Small GPU infrastructure

Deployment
Docker

IaC
Terraform
```

## Production

```text
Next.js
      +
CesiumJS
      +
FastAPI
      +
PostgreSQL/PostGIS
      +
S3
      +
SQS
      +
Redis
      +
EKS
      +
Karpenter
      +
NVIDIA GPU infrastructure
      +
CloudFront
      +
OpenTelemetry
```

---

# 30. What Not to Use

## MongoDB

Not recommended as the primary database.

The system has strongly related entities and substantial spatial requirements. PostgreSQL + PostGIS is a better fit.

## Firebase

Not appropriate as the core backend because the difficult workloads involve Python, GPUs, spatial processing, and asynchronous pipelines.

## Supabase as the Entire Backend

Useful for a quick prototype, but it should not become the architectural center of a GPU-heavy reconstruction platform.

## Vercel for the Entire Product

Vercel can be excellent for the web frontend, but GPU reconstruction should run on dedicated GPU infrastructure.

## Three.js as the Only 3D Engine

Three.js is useful, but CesiumJS is better aligned with large-scale geospatial visualization and 3D Tiles.

## Kafka Initially

Do not introduce Kafka unless actual scale and event-stream requirements justify it.

Start with SQS + Redis.

---

# 31. Most Important Architectural Decision

Separate the **control plane** from the **reconstruction plane**.

## Control Plane

```text
Next.js
FastAPI
PostgreSQL
Auth0
Redis
```

Responsible for:

- Users
- Projects
- Permissions
- Jobs
- Metadata
- UI
- Measurements
- Model status

## Reconstruction Plane

```text
S3
SQS
EKS
GPU
PyTorch
CUDA
Open3D
GDAL
PDAL
```

Responsible for:

- Video processing
- Pose estimation
- Depth
- Segmentation
- 3D fusion
- Mesh generation
- Texturing
- Georeferencing
- Quality estimation

This separation allows the reconstruction algorithms to evolve without forcing a rewrite of the SaaS application.

---

# 32. Final Recommended Stack

| Category | Technology | Rating |
|---|---|---:|
| Frontend | **Next.js 16** | 5/5 |
| Language | **TypeScript** | 5/5 |
| 3D Engine | **CesiumJS** | 5/5 |
| UI | **Tailwind + custom components** | 5/5 |
| Backend | **FastAPI** | 5/5 |
| ML | **PyTorch** | 5/5 |
| GPU | **CUDA / NVIDIA** | 5/5 |
| CV | **OpenCV** | 5/5 |
| Point Clouds | **Open3D + PDAL** | 5/5 |
| Geospatial | **GDAL + Shapely** | 5/5 |
| Database | **PostgreSQL 18 + PostGIS** | 5/5 |
| Object Storage | **Amazon S3** | 5/5 |
| Queue | **Amazon SQS** | 5/5 |
| Cache | **Redis** | 5/5 |
| Authentication | **Auth0** | 5/5 |
| Containers | **Docker** | 5/5 |
| Orchestration | **Amazon EKS** | 5/5 |
| GPU orchestration | **NVIDIA GPU support / GPU Operator** | 5/5 |
| Autoscaling | **Karpenter** | 5/5 |
| CDN | **CloudFront** | 5/5 |
| Infrastructure | **Terraform** | 5/5 |
| CI/CD | **GitHub Actions** | 5/5 |
| Observability | **OpenTelemetry** | 5/5 |

---

# 33. Final Architecture

```text
                    ┌─────────────────────┐
                    │     Next.js 16      │
                    │    TypeScript       │
                    └──────────┬──────────┘
                               │
                         CesiumJS / 3D Tiles
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Python        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       PostgreSQL 18         Redis             S3
          + PostGIS         Cache          Video/Models
              │                │                │
              └────────────────┼────────────────┘
                               │
                              SQS
                               │
                               ▼
                     ┌───────────────────┐
                     │     Amazon EKS    │
                     │                   │
                     │ CPU Workers       │
                     │ GPU Workers       │
                     └─────────┬─────────┘
                               │
                         NVIDIA CUDA
                               │
                         PyTorch / CV
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
              Pose           Depth          Scene
           Estimation      Estimation     Segmentation
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                         3D Fusion
                               │
                               ▼
                         Mesh / Point Cloud
                               │
                               ▼
                       Geo Registration
                               │
                               ▼
                         3D Tiles / GLB
                               │
                               ▼
                         Cesium Viewer
```

## Final Recommendation

The strongest 2026 architecture for this product is:

**Next.js 16 + TypeScript + CesiumJS** for the user experience, **FastAPI + Python** for the API, **PyTorch/CUDA** for reconstruction, **PostgreSQL 18 + PostGIS** for metadata and spatial data, **S3** for raw/generated assets, **SQS + Redis** for asynchronous processing, **Auth0** for B2B identity, and **Amazon EKS + GPU nodes + Karpenter** for scalable reconstruction.

The critical architectural rule is:

> **The API submits reconstruction jobs. GPU workers perform the reconstruction. S3 stores the heavy assets. PostgreSQL stores metadata. Cesium streams the resulting spatial model.**

That gives the system a clean path from research prototype to a production geospatial platform without forcing a rewrite when reconstruction workloads become significantly larger.

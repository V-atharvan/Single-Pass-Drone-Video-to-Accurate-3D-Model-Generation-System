# Single-Pass 3D Reconstruction Platform
## Master Project Todo List & Sequential Implementation Roadmap

**Project Name:** Single-Pass 3D Reconstruction Platform  
**Target Milestone:** MVP → Production Platform (2026)  
**Execution Model:** Strictly Sequential, Atomic Tasks, Zero Overlapping Dependencies  
**Tech Stack:** Next.js 16, TypeScript, CesiumJS, Tailwind CSS, FastAPI, Python, PyTorch, CUDA, OpenCV, Open3D, GDAL, PDAL, PostgreSQL 18 + PostGIS, Redis, Amazon SQS, Amazon S3, Amazon EKS, Karpenter, Auth0, Terraform.

---

### Dependency & Execution Rules
1. **Strict Atomicity:** Every task represents a single, independent, testable unit of work.
2. **Zero Overlapping Dependencies:** A task may only begin when its predecessor has completed and its verification criteria have passed.
3. **Traceability:** Every task explicitly specifies its Prerequisites (Inputs), Exact Implementation Steps (Files/Tools), and Definition of Done (Verification).

---

## Phase 1: Environment, Monorepo Architecture & Shared Data Contracts

### TASK-001: Initialize Monorepo Architecture and Workspace Tooling
- **Prerequisites:** Git, Node.js 22 LTS, Python 3.12+, Docker Desktop / Podman installed locally.
- **Description:** Set up the foundational monorepo directory layout supporting both the web frontend (`apps/web`), backend API (`apps/api`), asynchronous reconstruction workers (`workers/`), shared packages (`packages/`), and infrastructure (`infra/`).
- **Implementation Steps:**
  1. Initialize root directory with `pnpm` workspace configuration (`pnpm-workspace.yaml`).
  2. Configure root `package.json` with Turborepo (`turbo.json`) for orchestrating build, lint, and type-checking pipelines.
  3. Create root `.gitignore`, `.editorconfig`, and Prettier/ESLint configs.
  4. Scaffold directories: `apps/web`, `apps/api`, `packages/schemas`, `packages/geospatial`, `packages/shared`, `workers/`, `infra/terraform`, `scripts/`.
- **Definition of Done:** Running `pnpm install` and `pnpm lint` executes cleanly across all monorepo directories with zero errors.

---

### TASK-002: Establish Core Pydantic Schemas in Shared Schemas Package
- **Prerequisites:** TASK-001 completed.
- **Description:** Define centralized, authoritative data contracts in Python using Pydantic V2 for all domain entities, telemetry records, reconstruction job states, quality metrics, and 3D metadata.
- **Implementation Steps:**
  1. In `packages/schemas/python/`, configure `pyproject.toml` with `pydantic>=2.7.0`.
  2. Implement `telemetry.py`: Define `GPSRecord` (lat, lon, alt_msl, alt_rel, speed, heading, timestamp, accuracy), `IMURecord` (roll, pitch, yaw, q0..q3, timestamp), `CameraIntrinsics` (fx, fy, cx, cy, k1, k2, p1, p2, width, height).
  3. Implement `jobs.py`: Define `JobState` enum (15 states from PRD: `QUEUED`, `VALIDATING`, `EXTRACTING_FRAMES`, `ESTIMATING_POSE`, `ESTIMATING_DEPTH`, `SEGMENTING`, `FUSING`, `RECONSTRUCTING`, `TEXTURING`, `GEOREFERENCING`, `QUALITY_CHECK`, `GENERATING_TILES`, `COMPLETED`, `FAILED`, `CANCELLED`), `QualityPreset` enum (`LOW`, `BALANCED`, `HIGH`), and `JobProgressUpdate`.
  4. Implement `quality.py`: Define `InputQualityScore` (overall, video_score, gps_score, blur_score, texture_score, overlap_score, warnings).
  5. Implement `models.py`: Define `ModelMetadata`, `AccuracyReport` (rmse_horizontal, rmse_vertical, coverage_pct, observed_pct, inferred_pct, confidence_score), and `MeasurementResult`.
- **Definition of Done:** Pytest suite in `packages/schemas/python/tests/` passes validation tests for valid and malformed domain payloads.

---

### TASK-003: Implement TypeScript Type Generation from Pydantic Contracts
- **Prerequisites:** TASK-002 completed.
- **Description:** Establish an automated type synchronization pipeline that converts the shared Pydantic models into TypeScript definitions for `apps/web`.
- **Implementation Steps:**
  1. In `packages/schemas/`, create a build script (`scripts/export-types.py`) using `pydantic-to-typescript` or JSON Schema export + `json-schema-to-typescript`.
  2. Configure output directory `packages/schemas/ts/src/` to package generated `.d.ts` and `.ts` files.
  3. Add build target `pnpm build:types` in root `turbo.json`.
- **Definition of Done:** Running `pnpm build:types` outputs valid TypeScript interfaces in `packages/schemas/ts/` matching every Pydantic model with zero missing fields.

---

### TASK-004: Containerized Local Infrastructure Setup
- **Prerequisites:** TASK-003 completed.
- **Description:** Build a reproducible local development environment using Docker Compose for PostgreSQL with PostGIS, Redis, and MinIO (emulating Amazon S3).
- **Implementation Steps:**
  1. Create `docker-compose.dev.yml` at root.
  2. Service 1: `postgres` with image `postgis/postgis:16-3.4` (configured with persistent volume, credentials, and initial DB `single_pass_3d_dev`).
  3. Service 2: `redis` with image `redis:7.2-alpine` for caching and state tracking.
  4. Service 3: `minio` with image `minio/minio` + `minio/mc` setup container to provision local buckets: `single-pass-3d-raw`, `single-pass-3d-interim`, and `single-pass-3d-models`.
  5. Add healthcheck directives for all services.
- **Definition of Done:** Running `docker compose -f docker-compose.dev.yml up -d` brings up all three containers in healthy states; MinIO console accessible at `localhost:9001` with provisioned buckets.

---

### TASK-005: PostgreSQL Database Schema Definition and Migration Scripts
- **Prerequisites:** TASK-004 completed.
- **Description:** Create the relational and geospatial database schema using SQLAlchemy 2.0 and Alembic migrations.
- **Implementation Steps:**
  1. In `apps/api/`, configure `alembic` targeting the PostGIS database.
  2. Define tables:
     - `organizations` (id, name, created_at)
     - `users` (id, org_id, email, full_name, role, auth0_sub, created_at)
     - `projects` (id, org_id, name, description, location_name, crs, bbox geometry(Polygon, 4326), tags, created_at, updated_at)
     - `flights` (id, project_id, original_filename, s3_video_key, s3_telemetry_key, duration_seconds, fps, resolution_width, resolution_height, total_frames, status, created_at)
     - `flight_telemetry` (id, flight_id, timestamp_offset, location geometry(PointZ, 4326), altitude_msl, heading, speed, raw_json)
     - `reconstruction_jobs` (id, flight_id, org_id, status, quality_preset, current_stage, progress, error_message, started_at, completed_at)
     - `models` (id, job_id, project_id, name, crs, bbox geometry(Polygon, 4326), center_lat, center_lon, min_altitude, max_altitude, coverage_percent, confidence_score, position_rmse, vertical_rmse, s3_prefix, status, created_at)
     - `model_assets` (id, model_id, asset_type enum [`GLB`, `OBJ`, `PLY`, `LAS`, `TILES_3D`, `DEM_TIF`], s3_key, file_size_bytes, lod_level)
     - `measurements` (id, model_id, user_id, type enum [`DISTANCE`, `HEIGHT`, `AREA`, `VOLUME`], geometry_json, value, error_margin, created_at)
  3. Execute `alembic upgrade head`.
- **Definition of Done:** Migration applies cleanly against the PostGIS container; spatial indices (`GIST`) are verified on `bbox` and `location` columns.

---

### TASK-006: S3 Object Storage Client & Key Structure Provisioner
- **Prerequisites:** TASK-005 completed.
- **Description:** Implement a standardized storage utility wrapper in Python around `boto3` / `aiobotocore` enforcing bucket structure rules from the tech stack doc.
- **Implementation Steps:**
  1. In `packages/shared/python/storage.py`, create `S3StorageClient`.
  2. Implement key path generators:
     - `get_flight_raw_prefix(project_id, flight_id)` -> `projects/{project_id}/flights/{flight_id}/`
     - `get_reconstruction_interim_prefix(job_id)` -> `jobs/{job_id}/interim/{stage}/`
     - `get_model_output_prefix(project_id, model_id)` -> `projects/{project_id}/models/{model_id}/`
     - `get_export_prefix(project_id, export_id)` -> `projects/{project_id}/exports/{export_id}/`
  3. Implement helper methods for pre-signed upload URL generation (`generate_presigned_upload_url`), download URL generation (`generate_presigned_download_url`), and multipart upload verification.
- **Definition of Done:** Unit test uploads a buffer to MinIO using `S3StorageClient`, generates a pre-signed download URL, and validates content retrieval.

---

### TASK-007: Redis Client and Distributed State Tracker
- **Prerequisites:** TASK-006 completed.
- **Description:** Implement a high-performance Redis client for transient job state caching, worker heartbeats, and pub/sub message propagation.
- **Implementation Steps:**
  1. In `packages/shared/python/redis_client.py`, create `RedisStateTracker` using `redis-py` async.
  2. Implement atomic job stage updates: `set_job_progress(job_id, stage, progress, frames_processed, total_frames, gpu_stats)`.
  3. Implement pub/sub channel publishing on key `job_events:{job_id}` for WebSocket fan-out.
  4. Implement worker liveness heartbeats with TTL expiration (15 seconds).
- **Definition of Done:** Integration test verifies that a published job progress event updates the cached hash in Redis and emits a message to the pub/sub subscriber.

---

### TASK-008: OpenTelemetry Instrumentation Baseline
- **Prerequisites:** TASK-007 completed.
- **Description:** Set up centralized OpenTelemetry tracing and structured logging for both the API and background workers.
- **Implementation Steps:**
  1. In `packages/shared/python/telemetry.py`, configure OpenTelemetry SDK with OTLP exporter support and JSON structured console logging.
  2. Add tracer hooks for tracing async tasks, database transactions, S3 file transfers, and ML inference blocks.
  3. Define standard trace tags: `project_id`, `flight_id`, `job_id`, `worker_stage`.
- **Definition of Done:** A mock script creates a span with attributes and outputs structured JSON logs containing `trace_id` and `span_id`.

---

## Phase 2: Backend Control Plane API & Authentication

### TASK-009: FastAPI Application Bootstrap and Health Probes
- **Prerequisites:** TASK-008 completed.
- **Description:** Initialize the FastAPI REST application with standard middleware, CORS, lifecycle management, and health endpoints.
- **Implementation Steps:**
  1. In `apps/api/src/main.py`, instantiate `FastAPI(title="Single-Pass 3D Reconstruction API", version="1.0.0")`.
  2. Attach CORS middleware allowing Next.js frontend origin.
  3. Implement lifespan context manager connecting/disconnecting SQLAlchemy engine pool and Redis pool.
  4. Implement `/health/liveness` and `/health/readiness` (checking PostgreSQL and Redis connections).
- **Definition of Done:** Running `uvicorn apps.api.src.main:app` starts the server; `GET /health/readiness` returns HTTP 200 with DB and Redis statuses `UP`.

---

### TASK-010: Auth0 JWT Authentication and Token Validation Middleware
- **Prerequisites:** TASK-009 completed.
- **Description:** Implement security middleware to validate Auth0 RS256 Bearer JWT tokens, extract claims, and map to local user identities.
- **Implementation Steps:**
  1. In `apps/api/src/auth/jwt.py`, implement `JWTVerifier` using `PyJWT` and Auth0 JSON Web Key Sets (`JWKS`).
  2. Implement dependency `get_current_user(token: str = Depends(oauth2_scheme))` that caches JWKS public keys, verifies token expiration/audience/issuer, and queries or creates the user record in PostgreSQL.
- **Definition of Done:** Test suite verifies valid JWT allows access; expired, malformed, or missing tokens return HTTP 401 Unauthorized.

---

### TASK-011: Multi-Tenant RBAC Authorization Guard
- **Prerequisites:** TASK-010 completed.
- **Description:** Implement authorization checks enforcing hierarchical resource ownership: `Organization -> Project -> Flight -> Model -> Asset`.
- **Implementation Steps:**
  1. In `apps/api/src/auth/rbac.py`, define permissions for roles: `ORG_ADMIN`, `PROJECT_MANAGER`, `OPERATOR`, `ANALYST`, `VIEWER`.
  2. Create FastAPI dependencies:
     - `require_project_access(project_id: UUID, min_role: Role)`
     - `require_flight_access(flight_id: UUID, min_role: Role)`
     - `require_model_access(model_id: UUID, min_role: Role)`
  3. Ensure that accessing a resource belonging to a different organization returns HTTP 403 Forbidden.
- **Definition of Done:** Unit tests confirm a user in Org A cannot view or edit projects belonging to Org B.

---

### TASK-012: Organizations and Users Management Endpoints
- **Prerequisites:** TASK-011 completed.
- **Description:** Implement REST endpoints for managing organizations, member invitations, and role assignments.
- **Implementation Steps:**
  1. In `apps/api/src/routers/organizations.py`, implement:
     - `GET /organizations/me`: Return caller's organization details.
     - `GET /organizations/me/members`: List users with roles.
     - `PATCH /organizations/me/members/{user_id}`: Update user role (Admin only).
- **Definition of Done:** Verified via integration test: non-admin gets HTTP 403 when updating roles; admin gets HTTP 200 with updated role.

---

### TASK-013: Projects CRUD API Endpoints
- **Prerequisites:** TASK-012 completed.
- **Description:** Implement project management endpoints including spatial bounding box updates and metadata filtering.
- **Implementation Steps:**
  1. In `apps/api/src/routers/projects.py`, implement:
     - `POST /projects`: Create project with name, description, location name, CRS (e.g. `EPSG:4326`), and optional tags.
     - `GET /projects`: List projects for caller's organization with pagination and search query.
     - `GET /projects/{id}`: Return project details, associated flight count, model count, and aggregate quality score.
     - `PATCH /projects/{id}`: Update name, description, or tags.
     - `DELETE /projects/{id}`: Soft-delete project.
- **Definition of Done:** Tests verify project creation, retrieval, updating, and organization isolation.

---

### TASK-014: S3 Pre-Signed Direct Upload URL Service
- **Prerequisites:** TASK-013 completed.
- **Description:** Implement endpoints to facilitate secure, direct-to-S3 multi-gigabyte video and GPS file uploads without routing raw binary through FastAPI.
- **Implementation Steps:**
  1. In `apps/api/src/routers/uploads.py`, implement:
     - `POST /projects/{project_id}/uploads/initiate`: Takes `filename`, `content_type`, `file_size_bytes`, `file_type` (`VIDEO` or `TELEMETRY`).
     - Generates S3 pre-signed upload URL (or multipart upload parts for files > 100MB).
     - Returns `upload_url`, `s3_key`, and `upload_id`.
     - `POST /projects/{project_id}/uploads/complete`: Verifies S3 file presence and headers.
- **Definition of Done:** Integration test requests an upload URL, uploads a test payload directly to MinIO via HTTP PUT, calls `/complete`, and verifies file metadata on MinIO.

---

### TASK-015: Flights Registration and Metadata Linkage Endpoints
- **Prerequisites:** TASK-014 completed.
- **Description:** Implement endpoints to register an uploaded video and GPS file as a cohesive flight asset ready for validation.
- **Implementation Steps:**
  1. In `apps/api/src/routers/flights.py`, implement:
     - `POST /projects/{project_id}/flights`: Links uploaded S3 video key and S3 telemetry key to a new `flights` row.
     - `GET /projects/{project_id}/flights`: List all flights in a project with upload status and duration.
     - `GET /flights/{id}`: Fetch detailed flight metadata.
- **Definition of Done:** Database query confirms flight record is linked to project with initial status `PENDING_VALIDATION`.

---

### TASK-016: Reconstruction Job Submission & State Machine API
- **Prerequisites:** TASK-015 completed.
- **Description:** Implement endpoints to start, inspect, and cancel 3D reconstruction jobs.
- **Implementation Steps:**
  1. In `apps/api/src/routers/reconstruction_jobs.py`, implement:
     - `POST /flights/{flight_id}/reconstruction-jobs`: Accepts `quality_preset` (`LOW`, `BALANCED`, `HIGH`), verifies flight is valid, creates `reconstruction_jobs` entry with state `QUEUED`, publishes job ID to SQS/queue.
     - `GET /reconstruction-jobs/{id}`: Returns state, progress percentage, current stage, frames processed, and warnings.
     - `POST /reconstruction-jobs/{id}/cancel`: Sets state to `CANCELLED` if job is still in progress.
- **Definition of Done:** Submitting a job creates a DB record with state `QUEUED` and enqueues the payload into SQS/mock queue.

---

### TASK-017: Real-Time WebSocket Server for Job Status Streaming
- **Prerequisites:** TASK-016 completed.
- **Description:** Implement WebSocket endpoint streaming realtime progress updates from Redis pub/sub to frontend clients.
- **Implementation Steps:**
  1. In `apps/api/src/routers/websockets.py`, implement `GET /ws/jobs/{job_id}`.
  2. Authenticate WebSocket connection using query token or connection message.
  3. Subscribe to Redis channel `job_events:{job_id}`.
  4. Stream JSON packets containing: `job_id`, `stage`, `progress`, `frames_processed`, `frames_total`, `gpu_utilization`, `warnings`.
- **Definition of Done:** Integration test connects a WebSocket client, simulates publishing a Redis event, and verifies client receives the exact JSON packet within 100ms.

---

## Phase 3: Flight Ingestion & Pre-Flight Quality Assessment Engine

### TASK-018: Reconstruction Worker Dispatcher & SQS Consumer Loop
- **Prerequisites:** TASK-017 completed.
- **Description:** Build the asynchronous Python worker daemon that polls SQS, acquires job locks, and orchestrates pipeline execution.
- **Implementation Steps:**
  1. In `workers/orchestrator/runner.py`, implement `JobWorkerDaemon`.
  2. Poll SQS queue with long-polling (20s).
  3. On message receive: parse `job_id`, fetch job and flight records from PostgreSQL, mark job state as `VALIDATING` in DB and Redis.
  4. Provide graceful SIGTERM handling to release or requeue stalled jobs.
- **Definition of Done:** Worker pulls a test message from the queue, updates PostgreSQL job status to `VALIDATING`, and logs the trace ID.

---

### TASK-019: Video File Validation & Stream Integrity Probing
- **Prerequisites:** TASK-018 completed.
- **Description:** Build the video validation module using FFprobe / OpenCV to inspect video codec, resolution, frame rate, container integrity, and audio/timestamp tracks.
- **Implementation Steps:**
  1. In `workers/preprocessing/video_validator.py`, implement `validate_video_stream(local_video_path: Path) -> VideoMetadata`.
  2. Verify video format (`MP4`, `MOV`), codec (`H.264`, `H.265/HEVC`), minimum resolution (1080p), duration, and valid frame count.
  3. Detect corrupted frames, missing moov atoms, or variable framerate anomalies.
  4. Update `flights` row with detected width, height, fps, total_frames, and duration.
- **Definition of Done:** Tested against valid 4K/1080p MP4s and corrupt/truncated video files; corrupt files trigger actionable error: `"Unsupported codec or truncated video file"`.

---

### TASK-020: Flight Telemetry and GPS Extractor & Time Synchronizer
- **Prerequisites:** TASK-019 completed.
- **Description:** Parse uploaded GPS and flight metadata files (CSV, JSON, NMEA, DJI SRT/subtitle stream) and synchronize coordinates with video frame timestamps.
- **Implementation Steps:**
  1. In `workers/preprocessing/telemetry_parser.py`, implement parser for DJI SRT format, CSV, and standard JSON telemetry.
  2. Extract timestamps, latitude, longitude, altitude (MSL and relative), heading, gimbal pitch/roll/yaw, and horizontal/vertical dilution of precision (HDOP/VDOP).
  3. Perform linear / spline interpolation of GPS fixes to match video keyframe timestamps.
  4. Populate `flight_telemetry` rows in PostgreSQL with PostGIS `PointZ` coordinates.
- **Definition of Done:** An SRT/CSV telemetry sample is parsed and produces synchronized GPS records for every second of video; missing GPS records trigger a `"GPS timestamp mismatch"` error.

---

### TASK-021: Visual Quality Metric Evaluator (Blur, Exposure, Texture)
- **Prerequisites:** TASK-020 completed.
- **Description:** Implement computer-vision algorithms to assess frame quality, motion blur, underexposure, overexposure, and scene texture.
- **Implementation Steps:**
  1. In `workers/preprocessing/quality_evaluator.py`, implement:
     - Blur score: Laplacian variance across frames (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
     - Exposure check: Luminance histogram distribution; flag underexposed (<15% luminance) or overexposed (>85% saturation) frames.
     - Texture score: Spatial gradient density (Sobel operator) to detect low-texture surfaces like blank roofs or water.
- **Definition of Done:** Unit test processes test frames (sharp, blurred, dark, overexposed) and outputs normalized scores [0..100] accurately classifying degraded frames.

---

### TASK-022: Flight Trajectory Continuity and Overlap Quality Estimator
- **Prerequisites:** TASK-021 completed.
- **Description:** Evaluate drone flight trajectory consistency, speed spikes, GPS gaps, and estimated visual overlap between successive views.
- **Implementation Steps:**
  1. In `workers/preprocessing/trajectory_evaluator.py`, calculate drone velocity vectors between consecutive GPS fixes.
  2. Flag GPS outlier jumps (>30 m/s sudden deviation for standard UAVs) and temporal gaps (>2 seconds without position).
  3. Estimate baseline-to-height ratio (B/H) to ensure sufficient parallax for stereoscopic/multi-view reconstruction.
- **Definition of Done:** A simulated trajectory with a 5-second GPS dropout flags a `GPS_GAP_DETECTED` warning and deducts from the trajectory score.

---

### TASK-023: Input Quality Score Synthesizer and Pre-Flight Quality Report
- **Prerequisites:** TASK-022 completed.
- **Description:** Aggregate visual and trajectory metrics into an authoritative 0-100 Input Quality Score, write results to DB, and notify the frontend.
- **Implementation Steps:**
  1. In `workers/preprocessing/quality_synthesizer.py`, compute weighted score:
     $$\text{Score} = 0.35 \cdot \text{Blur} + 0.25 \cdot \text{GPS} + 0.20 \cdot \text{Texture} + 0.10 \cdot \text{Exposure} + 0.10 \cdot \text{Overlap}$$
  2. Classify expected result: `HIGH` (>=80), `MODERATE` (60-79), `LOW` (<60).
  3. Persist `InputQualityScore` into `flights.quality_report_json`.
  4. If score < 40 or critical data missing, mark job `FAILED` with specific actionable remediation instructions; otherwise proceed to keyframe extraction.
- **Definition of Done:** A mock flight run generates the exact output schema from PRD FR-005 and sends a WebSocket message to the client.

---

## Phase 4: Keyframe Selection, Camera Poses & Sensor Fusion

### TASK-024: Keyframe Selection Engine with Preset Control
- **Prerequisites:** TASK-023 completed.
- **Description:** Implement an intelligent keyframe extractor that avoids redundant frames while ensuring optimal visual overlap and sharpness according to quality presets.
- **Implementation Steps:**
  1. In `workers/pose/keyframe_selector.py`, implement keyframe filter.
  2. Configure preset thresholds:
     - `LOW`: ~1 keyframe per 1.5 meters travelled or 1.5 seconds.
     - `BALANCED`: ~1 keyframe per 0.8 meters travelled or 0.8 seconds.
     - `HIGH`: ~1 keyframe per 0.4 meters travelled or 0.4 seconds.
  3. Reject frames identified as blurry by TASK-021 in favor of immediate adjacent sharp frames.
  4. Output keyframe indices, frame timestamps, and source video timestamps.
- **Definition of Done:** Processing a 3-minute 30fps video (5,400 raw frames) in `BALANCED` mode extracts ~200-300 optimal keyframes with zero blur score outliers.

---

### TASK-025: Keyframe Disk and Storage Caching Pipeline
- **Prerequisites:** TASK-024 completed.
- **Description:** Extract selected keyframes at native resolution to local high-speed NVMe scratch storage and upload thumbnails to S3.
- **Implementation Steps:**
  1. In `workers/pose/frame_exporter.py`, use PyAV / OpenCV to extract selected keyframes directly to `/tmp/scratch/{job_id}/frames/frame_%05d.png`.
  2. Generate downscaled 512px WebP thumbnails and upload to `s3://{bucket}/jobs/{job_id}/interim/thumbnails/`.
  3. Update Redis stage: `EXTRACTING_FRAMES`, emitting progress percentage as frames are decoded.
- **Definition of Done:** Keyframe PNGs exist on local disk and thumbnails are viewable from MinIO bucket.

---

### TASK-026: Visual Odometry & Feature Matching Across Keyframes
- **Prerequisites:** TASK-025 completed.
- **Description:** Extract robust keypoint features and compute multi-view correspondences across sequential and overlapping keyframes.
- **Implementation Steps:**
  1. In `workers/pose/feature_tracker.py`, implement feature extraction using OpenCV SIFT / learned keypoints (SuperPoint / DISK).
  2. Match features across adjacent frames and loop-closure candidates using LightGlue / FLANN matcher with ratio test.
  3. Filter outlier matches using RANSAC epipolar constraint (essential matrix estimation $E$).
- **Definition of Done:** Unit test matches 10 sequential drone keyframes, identifying >=500 robust inlier correspondences per frame pair.

---

### TASK-027: Sensor Fusion Extended Kalman Filter / Pose Graph Optimization
- **Prerequisites:** TASK-026 completed.
- **Description:** Fuse visual odometry relative poses with GPS positions, barometric altitude, and IMU orientation into an optimized, drift-free trajectory.
- **Implementation Steps:**
  1. In `workers/pose/sensor_fusion.py`, construct a factor graph using GTSAM / SciPy least squares.
  2. Nodes: Camera 6-DoF poses ($R_i, t_i$).
  3. Factors:
     - Visual relative pose constraints between keyframes.
     - GPS position priors with uncertainty covariance based on HDOP/VDOP.
     - Barometric altitude prior for vertical stability.
     - IMU orientation priors (gimbal pitch/roll/yaw).
  4. Solve non-linear optimization to minimize total reprojection and sensor error.
- **Definition of Done:** Trajectory error optimization converges; camera path stays strictly bound to GPS coordinates while maintaining smooth local inter-frame relative consistency.

---

### TASK-028: Camera Extrinsics/Intrinsics Solver & Trajectory Exporter
- **Prerequisites:** TASK-027 completed.
- **Description:** Format and validate camera calibration matrices (intrinsics $K$) and optimized extrinsics ($[R|t]$) for each keyframe, computing pose confidence metrics.
- **Implementation Steps:**
  1. In `workers/pose/pose_output.py`, calculate reprojection RMSE for each camera pose.
  2. Assign pose confidence score (0-100%) based on inlier count and GPS-visual alignment residual.
  3. Export `cameras.json` to S3 containing camera matrix $K$, world-to-camera matrix, camera-to-world center, and pose confidence per frame.
- **Definition of Done:** `cameras.json` is saved in S3; every keyframe has a valid 4x4 transformation matrix and reprojection error < 1.0 pixel.

---

### TASK-029: Trajectory Visualization GeoJSON / CZML Generator
- **Prerequisites:** TASK-028 completed.
- **Description:** Convert estimated 3D flight trajectory into Cesium-compatible CZML / GeoJSON for interactive rendering in the frontend viewer.
- **Implementation Steps:**
  1. In `workers/pose/czml_generator.py`, generate a CZML document defining the camera flight path, orientation frustums at keyframe intervals, and time-stamped waypoints.
  2. Style path using Electric Cyan `#00F0FF` per design document specifications.
  3. Upload `trajectory.czml` to S3 interim storage.
- **Definition of Done:** Valid CZML file is written; verified to parse cleanly with Cesium CZML parser schema.

---

## Phase 5: Monocular AI Depth Estimation & Confidence Prediction

### TASK-030: PyTorch GPU Worker Runtime & CUDA Model Loader
- **Prerequisites:** TASK-029 completed.
- **Description:** Initialize the GPU worker runtime environment, verifying CUDA availability, TensorRT/torch.compile optimization, and GPU memory management.
- **Implementation Steps:**
  1. In `workers/depth/runtime.py`, verify `torch.cuda.is_available()`, log GPU model name and VRAM capacity.
  2. Implement PyTorch memory cache cleaner and FP16 / BF16 mixed-precision context manager.
  3. Download and cache pre-trained monocular metric depth model weights (e.g., Depth Anything V2 Metric / UniDepth) to local model cache directory.
- **Definition of Done:** Script executes on GPU worker, loads model into VRAM under 6GB memory footprint, and runs dummy inference without CUDA out-of-memory errors.

---

### TASK-031: Monocular Metric Depth Map Inference Pipeline
- **Prerequisites:** TASK-030 completed.
- **Description:** Process keyframes through the depth network to produce dense, high-resolution metric depth maps.
- **Implementation Steps:**
  1. In `workers/depth/depth_estimator.py`, implement batch inference over extracted keyframes.
  2. Normalize and resize keyframes to model input resolution; run batched model forward pass under `torch.inference_mode()`.
  3. Upsample output depth maps back to original keyframe resolution using bilateral / guided filtering to preserve sharp building edges.
- **Definition of Done:** Running depth estimation across 50 keyframes outputs 16-bit / 32-bit floating-point depth arrays ($Z$ in meters) for every keyframe.

---

### TASK-032: Depth Uncertainty and Confidence Field Estimation
- **Prerequisites:** TASK-031 completed.
- **Description:** Estimate pixel-wise depth uncertainty and confidence maps for each keyframe to guide multi-view fusion and flag ungrounded geometry.
- **Implementation Steps:**
  1. In `workers/depth/confidence_estimator.py`, compute depth confidence based on:
     - Network feature entropy / uncertainty head output.
     - Local depth gradient consistency across neighboring pixels (edge sharpness).
     - Photometric reprojection consistency between adjacent frames using warping:
       $$I_{\text{warp}} = \text{warp}(I_{t+1}, D_t, R_{t \to t+1}, t_{t \to t+1})$$
  2. Store confidence as a normalized 8-bit map [0..255] where 255 = highest confidence.
- **Definition of Done:** Confidence maps correctly highlight high confidence on textured building facades/roads and low confidence on reflective water or sky pixels.

---

### TASK-033: Metric Scale Calibration Against Sensor Baseline
- **Prerequisites:** TASK-032 completed.
- **Description:** Align and calibrate estimated depth map scale factors against physical GPS baseline travel distance and barometric altitude ground clearance.
- **Implementation Steps:**
  1. In `workers/depth/scale_aligner.py`, compute scale factor $s$ for each depth map such that 3D triangulated sparse feature points match the physical baseline distance between camera centers:
     $$s^* = \arg\min_s \sum \| s \cdot X_{\text{depth}} - X_{\text{triangulated}} \|^2$$
  2. Apply median scale factor across the keyframe sequence to ensure global metric consistency across all frames.
- **Definition of Done:** Scale-aligned depth maps produce ground-truth-consistent elevation deltas matching barometric flight height within ±5%.

---

### TASK-034: Depth and Confidence Map Serialization to NVMe/S3
- **Prerequisites:** TASK-033 completed.
- **Description:** Compress and serialize metric depth arrays and confidence maps to local NVMe scratch storage and archive them to S3.
- **Implementation Steps:**
  1. In `workers/depth/serializer.py`, save depth maps as compressed 16-bit PNG (with millimetric scale factor) or EXR files: `depth_%05d.png`.
  2. Save confidence maps as 8-bit PNG: `confidence_%05d.png`.
  3. Update Redis stage to `ESTIMATING_DEPTH` and publish progress updates after each batch of 10 frames.
- **Definition of Done:** All depth and confidence maps for the keyframe sequence are written to disk; file sizes and checksums validated.

---

## Phase 6: Semantic Scene Understanding & Dynamic Object Removal

### TASK-035: Semantic Segmentation Pipeline Setup
- **Prerequisites:** TASK-034 completed.
- **Description:** Deploy a lightweight, accurate semantic segmentation model (e.g. SegFormer / Mask2Former) classifying scene regions into target geospatial categories.
- **Implementation Steps:**
  1. In `workers/segmentation/semantic_classifier.py`, load segmentation model trained on drone/aerial imagery.
  2. Target classes: `BUILDING`, `ROAD`, `TERRAIN`, `VEGETATION`, `VEHICLE`, `PERSON`, `UTILITY_INFRASTRUCTURE`, `WATER`, `UNKNOWN`.
  3. Run batched inference across all keyframes; generate 8-bit class index masks.
- **Definition of Done:** Semantic masks correctly categorize buildings, roads, trees, and ground surfaces with mean IoU >= 80% on standard aerial benchmark test images.

---

### TASK-036: Dynamic Object Detection & Multi-Frame Motion Tracking
- **Prerequisites:** TASK-035 completed.
- **Description:** Identify moving vehicles, people, and machinery across sequential frames to prevent phantom/ghost geometry in the 3D reconstruction.
- **Implementation Steps:**
  1. In `workers/segmentation/dynamic_tracker.py`, detect vehicles and pedestrians from the semantic mask and bounding box detector.
  2. Track objects across sequential frames using ByteTrack / BoT-SORT.
  3. Compute optical flow and reprojected background motion: if an object’s motion vector diverges from expected camera epipolar geometry, classify it as `DYNAMIC`.
- **Definition of Done:** A moving car on a roadway is flagged as `DYNAMIC`, while a parked car on a driveway remains classified as static infrastructure.

---

### TASK-037: Dynamic Object Mask Generation & Dilation
- **Prerequisites:** TASK-036 completed.
- **Description:** Generate dilated binary exclusion masks for all dynamic entities to ensure complete exclusion during 3D point cloud back-projection.
- **Implementation Steps:**
  1. In `workers/segmentation/mask_generator.py`, isolate all pixels labeled with active dynamic object tracks.
  2. Apply morphological dilation (kernel size 5x5) to prevent border artifact leaking.
  3. Produce binary dynamic mask `dynamic_%05d.png` (255 = dynamic/exclude, 0 = static).
- **Definition of Done:** Dynamic masks cover 100% of moving vehicle pixels plus safe 5px margin around object contours.

---

### TASK-038: Semantic Mask Association & Mask Pyramid Builder
- **Prerequisites:** TASK-037 completed.
- **Description:** Merge static semantic classification with dynamic exclusion masks into a unified multi-channel semantic tensor per keyframe.
- **Implementation Steps:**
  1. In `workers/segmentation/mask_combiner.py`, combine class ID (Channel 0), dynamic flag (Channel 1), and classification confidence (Channel 2).
  2. Generate multi-resolution pyramids (1x, 0.5x, 0.25x) to support multi-scale LOD 3D fusion.
- **Definition of Done:** Verified multi-channel semantic mask file created for each keyframe with intact label mappings.

---

### TASK-039: Semantic and Dynamic Mask Serialization to Storage
- **Prerequisites:** TASK-038 completed.
- **Description:** Save final semantic masks to scratch NVMe and upload sample previews to S3 interim storage.
- **Implementation Steps:**
  1. In `workers/segmentation/serializer.py`, write mask files to `/tmp/scratch/{job_id}/semantics/`.
  2. Update Redis stage to `SEGMENTING` with progress percentage.
  3. Upload dynamic object statistics (number of moving objects removed) to job metadata.
- **Definition of Done:** Semantic mask files saved; dynamic object contamination metric stored for the final quality report.

---

## Phase 7: 3D Point Cloud Fusion, Surface Completion & Outlier Filtering

### TASK-040: Depth Unprojection and Point Cloud Back-Projection
- **Prerequisites:** TASK-039 completed.
- **Description:** Unproject calibrated 2D metric depth maps into 3D camera coordinates and transform them to global coordinates using 6-DoF poses.
- **Implementation Steps:**
  1. In `workers/fusion/unprojector.py`, use PyTorch / NumPy to back-project depth pixels $(u, v, d)$ to 3D points in camera frame:
     $$X_c = \frac{(u - c_x) \cdot d}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad Z_c = d$$
  2. Transform camera points to world frame using extrinsics matrix $[R_w | t_w]$.
  3. Associate each 3D point with RGB color from the source keyframe, depth confidence score, and semantic class ID.
- **Definition of Done:** Back-projecting 10 keyframes creates 3D point clusters whose overlapping regions physically align in 3D coordinate space.

---

### TASK-041: Multi-Frame Point Cloud Fusion with Confidence Weighting
- **Prerequisites:** TASK-040 completed.
- **Description:** Fuse individual point sets into a consolidated global point cloud, applying confidence weights and multi-view visibility verification.
- **Implementation Steps:**
  1. In `workers/fusion/point_fusion.py`, implement spatial voxel hashing to merge duplicate observations across overlapping frames.
  2. Where a voxel contains observations from multiple frames, compute the weighted average 3D position based on depth confidence:
     $$P_{\text{fused}} = \frac{\sum w_i \cdot P_i}{\sum w_i}$$
  3. Discard points observed by only one frame if their confidence is below 60%.
- **Definition of Done:** Point cloud fusion reduces redundant overlapping points while increasing surface sharpness and geometric signal-to-noise ratio.

---

### TASK-042: Dynamic Object & Statistical Outlier Removal
- **Prerequisites:** TASK-041 completed.
- **Description:** Cleanse the fused point cloud by stripping out points falling inside dynamic masks and removing floating sky/edge artifacts using Open3D.
- **Implementation Steps:**
  1. In `workers/fusion/outlier_filter.py`, filter out any point tagged with a dynamic object mask.
  2. Apply Statistical Outlier Removal (SOR) using Open3D (`nb_neighbors=20, std_ratio=1.5`).
  3. Apply Radius Outlier Removal (ROR) to eliminate sparse flying points far from primary structures.
- **Definition of Done:** Point cloud shows clean structural outlines (crisp building edges, clean ground) with zero floating ghost points from passing vehicles or birds.

---

### TASK-043: Voxel Downsampling and Normal Vector Estimation
- **Prerequisites:** TASK-042 completed.
- **Description:** Resample the cleaned point cloud to a uniform spatial density and compute accurate surface normal vectors.
- **Implementation Steps:**
  1. In `workers/fusion/normal_estimator.py`, perform voxel grid downsampling with voxel size tailored to quality preset (`HIGH` = 0.05m, `BALANCED` = 0.10m, `LOW` = 0.20m).
  2. Estimate normals using covariance analysis over local $k$-nearest neighbors ($k=30$).
  3. Orient normals consistently toward the drone camera positions.
- **Definition of Done:** Fused point cloud has uniform point spacing; 100% of surface points have normalized, consistently oriented normal vectors $(n_x, n_y, n_z)$.

---

### TASK-044: Occlusion Analysis and Surface Observation State Tagging
- **Prerequisites:** TASK-043 completed.
- **Description:** Analyze viewing ray visibility to categorize every region of the 3D scene into observation states per PRD Section 19.
- **Implementation Steps:**
  1. In `workers/fusion/occlusion_analyzer.py`, trace camera viewing rays against the reconstructed point cloud and voxel grid.
  2. Classify points and spatial cells into four states:
     - `OBSERVED`: Visual rays directly intersected from 2+ camera angles.
     - `PARTIALLY_OBSERVED`: Observed from only a single steep angle or near occlusion boundary.
     - `AI_INFERRED`: Occluded shadow/backside surface filled by learned surface prior.
     - `UNKNOWN`: Completely unobserved.
  3. Calculate global percentages: `observed_pct`, `partially_observed_pct`, `inferred_pct`.
- **Definition of Done:** Point cloud attributes contain `observation_state`; statistics verify sum of percentages equals 100%.

---

### TASK-045: LAS/LAZ and PLY Point Cloud Serializer
- **Prerequisites:** TASK-044 completed.
- **Description:** Export georeferenced, classified point clouds into standard GIS LAS/LAZ and PLY formats using PDAL and Open3D.
- **Implementation Steps:**
  1. In `workers/fusion/las_exporter.py`, build a PDAL pipeline saving `pointcloud.laz` and `pointcloud.las`.
  2. Map semantic classes to standard ASPRS LAS classification codes:
     - Class 2: Ground / Terrain
     - Class 5: High Vegetation
     - Class 6: Building
     - Class 9: Water
     - Class 17: Bridge / Infrastructure
  3. Write confidence and observation state as extra LAS dimensions.
  4. Write `pointcloud.ply` with RGB, normals, and confidence.
  5. Upload generated files to S3 model directory.
- **Definition of Done:** `lasinfo` (PDAL) on the exported LAZ file confirms valid ASPRS classifications, bounding box coordinates, and custom confidence dimensions.

---

## Phase 8: Mesh Reconstruction, Texturing & LOD Generation

### TASK-046: Surface Mesh Reconstruction Engine
- **Prerequisites:** TASK-045 completed.
- **Description:** Generate a continuous 3D triangular surface mesh from the oriented point cloud using Screened Poisson Surface Reconstruction.
- **Implementation Steps:**
  1. In `workers/mesh/surface_reconstructor.py`, invoke Open3D Poisson Reconstruction (`depth=10` or `11` depending on preset).
  2. Trim low-density mesh vertices where point cloud support is absent using density thresholding to avoid spurious bubble surfaces.
  3. Segment mesh into terrain surface and architectural building facades using semantic classification from the underlying points.
- **Definition of Done:** Reconstructed polygonal mesh exhibits clean planar roofs, vertical facades, and smooth terrain without non-physical ballooning.

---

### TASK-047: Mesh Topology Cleanup, Non-Manifold Removal & Hole Infilling
- **Prerequisites:** TASK-046 completed.
- **Description:** Clean mesh topology to guarantee 2-manifold surface properties and apply localized hole filling with AI inference tagging.
- **Implementation Steps:**
  1. In `workers/mesh/mesh_cleaner.py`, remove duplicate vertices, zero-area degenerate triangles, and non-manifold edges.
  2. Identify boundary holes caused by minor drone occlusions; apply planar/minimal-surface hole filling for small gaps (<2m diameter).
  3. Flag filled hole faces with metadata attribute `is_ai_inferred = True`.
- **Definition of Done:** Open3D mesh analysis reports `is_edge_manifold() == True` and `is_vertex_manifold() == True`; hole-filled triangles are distinctly tagged.

---

### TASK-048: Keyframe Raycasting & Optimal View Selection for Texturing
- **Prerequisites:** TASK-047 completed.
- **Description:** For every triangular face on the 3D mesh, select the sharpest, least-occluded source keyframe with the most orthogonal viewing angle.
- **Implementation Steps:**
  1. In `workers/texture/view_selector.py`, compute face normal $\vec{n}_f$ and face center $c_f$.
  2. For each candidate keyframe camera:
     - Check viewing ray line-of-sight using Embree / BVH raycasting to ensure no intervening geometry occludes the face.
     - Calculate angular score: $\cos(\theta) = \vec{n}_f \cdot \vec{v}_{\text{cam}}$.
     - Multiply by frame sharpness / blur score.
  3. Assign each mesh face to the highest-scoring keyframe camera.
- **Definition of Done:** Every visible mesh face is mapped to an unoccluded keyframe with an incidence angle < 60 degrees.

---

### TASK-049: Exposure Correction, Seam Blending & Texture Atlas Baking
- **Prerequisites:** TASK-048 completed.
- **Description:** Parameterize UV coordinates, equalize exposure and white balance across differing source frames, and bake seamless texture atlases.
- **Implementation Steps:**
  1. In `workers/texture/atlas_baker.py`, compute UV parameterization using xatlas.
  2. Apply Poisson image blending / multi-band blending across texture boundaries to eliminate visible seam lines caused by drone lighting changes.
  3. Generate 4096x4096 texture atlases (`diffuse_00.png`) and auxiliary confidence texture maps (`confidence_00.png`).
- **Definition of Done:** Textured 3D mesh displays uniform illumination across face seams without visible tiling boundaries or harsh color transitions.

---

### TASK-050: Multi-Resolution Level-of-Detail (LOD) Decimation
- **Prerequisites:** TASK-049 completed.
- **Description:** Generate optimized, multi-resolution mesh levels (LOD 0, LOD 1, LOD 2) for smooth streaming and rendering in WebGL.
- **Implementation Steps:**
  1. In `workers/mesh/lod_generator.py`, generate:
     - LOD 0: Full resolution (100% polygons).
     - LOD 1: Decimated to 35% polygons using quadric error metrics (QEM).
     - LOD 2: Simplified envelope (10% polygons) for distant viewer perspectives.
  2. Recompute UV coordinates and bake simplified normal maps for LOD 1 and LOD 2.
- **Definition of Done:** Three discrete LOD meshes exist on disk; vertex count for LOD 1 and LOD 2 matches target decimation ratios within ±2%.

---

### TASK-051: GLB, glTF, and OBJ Mesh Exporter
- **Prerequisites:** TASK-050 completed.
- **Description:** Export textured 3D models into industry-standard GLB (binary glTF with embedded textures) and OBJ formats.
- **Implementation Steps:**
  1. In `workers/mesh/mesh_exporter.py`, export `model.glb` using Draco geometry compression for fast web loading.
  2. Export `model.obj` + `model.mtl` + textures for CAD/BIM interoperability.
  3. Upload assets to S3 under `projects/{project_id}/models/{model_id}/`.
- **Definition of Done:** Exported GLB model successfully loads in standard 3D viewer (e.g. Three.js / glTF Viewer) with intact textures, materials, and compressed geometry.

---

## Phase 9: Georeferencing, Terrain Rasterization & 3D Tiling

### TASK-052: Georeferencing Rigid Body Transformation Pipeline
- **Prerequisites:** TASK-051 completed.
- **Description:** Transform local 3D reconstruction coordinates into geographic coordinate systems (WGS84 EPSG:4326 and target UTM projection) using PostGIS and GDAL.
- **Implementation Steps:**
  1. In `workers/geospatial/georeferencer.py`, calculate 7-parameter similarity transformation (Sim3: scale $s$, rotation $R$, translation $t$) aligning local reconstruction points with GPS/RTK ground control references.
  2. Transform mesh vertices and point cloud coordinates into target CRS (e.g. UTM Zone coordinate system and WGS84 ellipsoidal height).
  3. Compute geographic bounding box (`min_lat`, `max_lat`, `min_lon`, `max_lon`, `min_alt`, `max_alt`).
- **Definition of Done:** Point cloud and mesh coordinates match absolute GPS ground positions verified against input drone flight telemetry.

---

### TASK-053: Geospatial Accuracy Assessment & RMSE Estimator
- **Prerequisites:** TASK-052 completed.
- **Description:** Statistically evaluate reconstruction accuracy against GPS telemetry / GCPs and generate the quantitative Accuracy Report.
- **Implementation Steps:**
  1. In `workers/geospatial/accuracy_evaluator.py`, compute residuals between estimated camera centers and recorded GPS/RTK positions:
     $$\text{RMSE}_{\text{horiz}} = \sqrt{\frac{1}{N} \sum (\Delta x_i^2 + \Delta y_i^2)}, \quad \text{RMSE}_{\text{vert}} = \sqrt{\frac{1}{N} \sum \Delta z_i^2}$$
  2. Evaluate relative geometric consistency and confidence scores.
  3. Construct `AccuracyReport` data contract matching PRD Section 39: Overall Quality (0-100), Horizontal RMSE, Vertical RMSE, Coverage %, Dynamic Contamination %, and specific warnings.
- **Definition of Done:** Generated accuracy report contains non-zero, mathematically calculated RMSE values; never produces fabricated survey-grade claims when GPS confidence is low.

---

### TASK-054: Digital Elevation Model (DEM) and DSM GeoTIFF Generator
- **Prerequisites:** TASK-053 completed.
- **Description:** Rasterize classified ground and surface point cloud data into georeferenced GeoTIFF elevation models using GDAL.
- **Implementation Steps:**
  1. In `workers/geospatial/dem_rasterizer.py`, filter points labeled `GROUND` (LAS Class 2) to interpolate a Digital Elevation Model (DEM).
  2. Use all surface points to interpolate a Digital Surface Model (DSM).
  3. Write 32-bit floating-point GeoTIFFs with proper CRS metadata, NoData values, and spatial resolution (e.g. 0.1m/pixel).
  4. Upload `dem.tif` and `dsm.tif` to S3.
- **Definition of Done:** `gdalinfo` on the exported GeoTIFF verifies valid spatial projection, bounding bounds, and non-empty elevation raster bands.

---

### TASK-055: OGC 3D Tiles 1.1 Hierarchical Tiling Pipeline
- **Prerequisites:** TASK-054 completed.
- **Description:** Convert georeferenced textured meshes and point clouds into streamable OGC 3D Tiles (`tileset.json`, `.b3dm`, `.pnts` / glTF 3D Tiles 1.1).
- **Implementation Steps:**
  1. In `workers/geospatial/tileset_generator.py`, slice geometry into a spatial octree hierarchy.
  2. Package leaf and intermediate bounding boxes into 3D Tiles 1.1 glTF containers with geometric error thresholds for dynamic LOD streaming.
  3. Write root `tileset.json` with EPSG:4978 (ECEF - Earth-Centered, Earth-Fixed) coordinates required by CesiumJS.
  4. Upload 3D Tiles directory tree to S3 `projects/{project_id}/models/{model_id}/tiles/`.
- **Definition of Done:** `tileset.json` is structurally valid per OGC 3D Tiles specification and references valid child tile nodes in S3.

---

### TASK-056: Model Assets Registration & Final Pipeline Completion
- **Prerequisites:** TASK-055 completed.
- **Description:** Register all generated 3D assets, reports, and spatial bounds in PostgreSQL, mark the reconstruction job as `COMPLETED`, and notify the user.
- **Implementation Steps:**
  1. In `workers/orchestrator/job_finalizer.py`, insert records into `models`, `model_assets`, and `accuracy_reports` tables.
  2. Populate `models.bbox` with PostGIS geometry polygon.
  3. Mark `reconstruction_jobs.status = 'COMPLETED'` with `completed_at = NOW()`.
  4. Emit final WebSocket message to client: `stage = 'completed'`, `progress = 1.0`.
- **Definition of Done:** Database query shows job is `COMPLETED`; all associated model assets (GLB, LAZ, GeoTIFF, 3D Tiles) have valid S3 keys registered.

---

## Phase 10: Frontend Design System & Component Library

### TASK-057: Next.js 16 App Directory Setup & Brand Theme Configuration
- **Prerequisites:** TASK-056 completed (backend pipeline functional).
- **Description:** Initialize the web application frontend using Next.js 16 with TypeScript and configure the exact visual design system and color tokens from the Design Document.
- **Implementation Steps:**
  1. In `apps/web/`, initialize Next.js with App Router and Tailwind CSS.
  2. Configure `tailwind.config.ts` with brand color palette:
     - `brand-pink`: `#FF69B4` (Bubblegum Pink - primary CTAs, active highlights)
     - `brand-teal`: `#069494` (Deep Teal - technical panels, secondary actions)
     - `brand-cyan`: `#00F0FF` (Electric Cyan - active 3D overlays, telemetry)
     - `brand-white`: `#FFFFFF` (Main canvas, clean cards)
     - Neutral palette: Ink `#111111`, Muted `#6B6B6B`, Soft Gray `#F4F4F4`, Border `#E5E5E5`.
  3. Configure Google Font `Inter` in `apps/web/app/layout.tsx` with clean geometric type scale.
- **Definition of Done:** Running `pnpm --filter web dev` launches the frontend; CSS classes `bg-brand-pink`, `text-brand-teal`, `border-brand-cyan` render exact hex codes.

---

### TASK-058: Base Primitive UI Components Library
- **Prerequisites:** TASK-057 completed.
- **Description:** Implement custom UI primitive components matching the design specifications without bloated third-party styling kits.
- **Implementation Steps:**
  1. In `apps/web/components/ui/`:
     - `Button.tsx`: Variants `primary` (pink background, black/white text, rounded), `secondary` (white background, teal border, teal text), `ghost`.
     - `Card.tsx`: White background, subtle `#E5E5E5` border, rounded radius (12px), generous padding.
     - `Badge.tsx`: Compact status and tag pill badges.
     - `Modal.tsx`: Accessible dialog using `@radix-ui/react-dialog`.
     - `Input.tsx` and `Select.tsx`: Minimal, high-contrast form controls.
     - `Tooltip.tsx`: Contextual explanation tooltips.
- **Definition of Done:** Storybook / component preview page displays all button variants, cards, and modal dialogs with correct hover states and focus rings.

---

### TASK-059: Application Shell Header and Navigation Bar
- **Prerequisites:** TASK-058 completed.
- **Description:** Implement the lightweight, non-intrusive application shell header matching Design Doc Section 8.
- **Implementation Steps:**
  1. In `apps/web/components/shell/Header.tsx`, build top navigation:
     - Logo: Minimal geometric brand emblem with pink/teal styling.
     - Nav Links: Dashboard, Projects, Flights, Models, Jobs.
     - Global Search bar.
     - Active Jobs badge with pulsating cyan indicator when jobs are running.
     - User profile dropdown with organization switch and logout.
- **Definition of Done:** Header renders across desktop viewports (max-width 1440px), responds cleanly to navigation route changes, and highlights the active page.

---

### TASK-060: Metric and Status Display Components
- **Prerequisites:** TASK-059 completed.
- **Description:** Implement large-numeral data metric widgets and standardized status badges specified in Design Doc Sections 6 & 24.
- **Implementation Steps:**
  1. In `apps/web/components/ui/Metric.tsx`, build large numeral display: e.g. `93%` (large heading, tight tracking) with small uppercase label `Coverage`.
  2. In `apps/web/components/ui/StatusBadge.tsx`, enforce strict enum states: `READY` (neutral), `PROCESSING` (animated cyan), `WARNING` (amber `#F0A000`), `FAILED` (red `#D64545`), `COMPLETED` (green `#20A36A`).
- **Definition of Done:** Components render sample metrics and statuses matching design specifications with zero ambiguous labels.

---

### TASK-061: Auth0 Client-Side Integration & Route Guards
- **Prerequisites:** TASK-060 completed.
- **Description:** Integrate Auth0 React SDK (`@auth0/auth0-react` or `@auth0/nextjs-auth0`) for client authentication, token refresh, and protected page wrappers.
- **Implementation Steps:**
  1. In `apps/web/lib/auth/`, configure Auth0 provider using environment variables (`AUTH0_DOMAIN`, `AUTH0_CLIENT_ID`, `AUTH0_AUDIENCE`).
  2. Implement `ProtectedRoute` wrapper checking authentication state; redirect unauthenticated users to `/login`.
  3. Attach JWT Bearer token automatically to outgoing REST and WebSocket requests.
- **Definition of Done:** Navigating to `/dashboard` while unauthenticated redirects to Auth0 login; logging in returns user to `/dashboard` with session token.

---

### TASK-062: Typed API Client & WebSocket Hook Infrastructure
- **Prerequisites:** TASK-061 completed.
- **Description:** Implement TanStack React Query hooks and typed fetch wrappers consuming the generated TypeScript contracts from TASK-003.
- **Implementation Steps:**
  1. In `apps/web/lib/api/client.ts`, configure typed HTTP client with error interceptors and automatic token injection.
  2. In `apps/web/lib/api/hooks/`, implement React Query hooks: `useProjects`, `useProject(id)`, `useFlights(projectId)`, `useModel(id)`.
  3. In `apps/web/lib/api/useJobWebSocket.ts`, implement resilient WebSocket hook with exponential backoff auto-reconnect.
- **Definition of Done:** React component successfully queries `/projects` via `useProjects()` hook and updates state automatically without page reload.

---

### TASK-063: Marketing Landing Page & Hero Showcase
- **Prerequisites:** TASK-062 completed.
- **Description:** Build the high-impact, minimalist landing page at `/` matching Design Doc Sections 28-31.
- **Implementation Steps:**
  1. In `apps/web/app/page.tsx`, implement Hero section:
     - Headline: `"ONE FLIGHT. ONE VIDEO. ONE 3D WORLD."`
     - Subhead: `"Turn a single drone pass into a measurable 3D scene."`
     - Primary CTA: `[ Start Reconstruction ]` (brand pink) linking to `/dashboard`.
     - Secondary CTA: `[ Explore Demo ]` linking to interactive demo scene.
  2. Implement 4-step "How It Works" visual flow: 01 Capture $\to$ 02 Process $\to$ 03 Reconstruct $\to$ 04 Analyze.
  3. Implement Use Case visual cards: Disaster Response, Infrastructure, Construction, Mapping.
- **Definition of Done:** Landing page renders cleanly at `/`, passes mobile/desktop responsive breakpoints, and buttons navigate to target views.

---

## Phase 11: Frontend Core Application Views

### TASK-064: Project Dashboard View (`/dashboard`)
- **Prerequisites:** TASK-063 completed.
- **Description:** Build the central operational dashboard answering the three primary operator questions per Design Doc Section 9.
- **Implementation Steps:**
  1. In `apps/web/app/dashboard/page.tsx`, create layout:
     - Top section: Active Projects grid with `+ New Project` button.
     - Project Cards: Display Project Name, Location, Flight Count, Model Count, Quality Score badge, and 3D thumbnail preview.
     - Bottom section: "Active Reconstruction Jobs" monitoring card displaying active flights, animated progress bar, percentage, and current stage.
- **Definition of Done:** Dashboard loads projects from API and renders live progress bars for active jobs updating in realtime.

---

### TASK-065: Project Creation & Project Details Page
- **Prerequisites:** TASK-064 completed.
- **Description:** Build the modal to create new projects and the detail page displaying project flights, models, and metadata.
- **Implementation Steps:**
  1. In `apps/web/components/project/CreateProjectModal.tsx`, implement form: Name, Description, Location, Coordinate Reference System (CRS dropdown: WGS84, UTM zones), and Tags.
  2. In `apps/web/app/projects/[id]/page.tsx`, display project summary, flight list table, and reconstructed models list with links to 3D viewer.
- **Definition of Done:** User creates a new project via the modal; page automatically navigates to `/projects/[id]` with empty state prompting for flight upload.

---

### TASK-066: Resumable Direct-to-S3 Video & GPS Upload Screen
- **Prerequisites:** TASK-065 completed.
- **Description:** Implement drag-and-drop file upload zone for multi-gigabyte drone videos and companion GPS files per Design Doc Section 11.
- **Implementation Steps:**
  1. In `apps/web/components/upload/UploadZone.tsx`, build upload UI supporting MP4/MOV drag-and-drop.
  2. Implement file chunking and direct S3 pre-signed upload using `@uppy/core` or custom multi-part upload worker.
  3. Display upload progress bar, upload speed (MB/s), and estimated remaining upload time.
  4. Provide companion upload slot for GPS files (CSV, JSON, SRT).
  5. Upon upload completion, automatically call `POST /projects/{id}/flights`.
- **Definition of Done:** Dropping a 1GB test video file uploads directly to S3 storage bucket with active progress bar and registers flight in backend.

---

### TASK-067: Pre-Flight Input Quality Assessment Screen
- **Prerequisites:** TASK-066 completed.
- **Description:** Display immediate pre-flight data validation and quality metrics before initiating expensive GPU reconstruction per Design Doc Section 12.
- **Implementation Steps:**
  1. In `apps/web/components/flight/InputQualityScreen.tsx`, render:
     - Video Quality score & status badge (e.g. `91%` - Good).
     - GPS Quality score & status badge (e.g. `74%` - Moderate).
     - Motion Blur score (e.g. `88%` - Low).
     - Scene Texture score (e.g. `94%` - Good).
     - Overall Expected Result badge (`HIGH` / `MODERATE` / `LOW`).
     - Actionable warnings list (e.g., `"GPS uncertainty increased near end of flight"`).
  2. Action button: `[ Proceed to Reconstruction ]` (pink) or `[ Re-upload Data ]`.
- **Definition of Done:** Screen renders the exact layout from Design Doc Section 12; disables reconstruction if critical input failure is detected.

---

### TASK-068: Reconstruction Launch Configuration Modal
- **Prerequisites:** TASK-067 completed.
- **Description:** Implement job parameter modal allowing operators to select processing quality and coordinate reference system.
- **Implementation Steps:**
  1. In `apps/web/components/reconstruction/LaunchJobModal.tsx`, build options:
     - Quality Preset: `Low` (Fastest, preview), `Balanced` (Standard, recommended), `High` (Full resolution, max detail).
     - Feature toggles: Dynamic Object Removal, Semantic Segmentation.
     - Target CRS projection confirmation.
  2. Primary action button: `[ Start Reconstruction ]` with brand pink accent.
  3. Submits `POST /flights/{id}/reconstruction-jobs`.
- **Definition of Done:** Submitting form launches backend job and redirects browser to Reconstruction Progress view.

---

### TASK-069: Reconstruction Real-Time Progress View
- **Prerequisites:** TASK-068 completed.
- **Description:** Build the live processing view tracking reconstruction stages, progress percentage, frame count, and ETA per Design Doc Section 13.
- **Implementation Steps:**
  1. In `apps/web/app/jobs/[id]/page.tsx`, implement:
     - Headline: `"Building your 3D scene"`.
     - Large percentage display: e.g. `78%`.
     - Horizontal progress bar with brand pink/teal fill.
     - Frame counter: `"8,421 / 11,672"`.
     - Stage checklist with visual status dots:
       - `✓ Input validation`
       - `✓ Frame extraction`
       - `✓ Pose estimation`
       - `● Depth estimation` (active animated indicator)
       - `○ Scene fusion`
       - `○ Mesh generation`
       - `○ Texturing`
       - `○ Quality assessment`
  2. Automatically transition to the 3D Viewer when status becomes `COMPLETED`.
- **Definition of Done:** Progress view subscribes to WebSocket, animates stage transitions in realtime, and navigates to the 3D viewer upon completion.

---

## Phase 12: Frontend CesiumJS 3D Viewer & Interactive Tools

### TASK-070: CesiumJS Container Integration with Next.js
- **Prerequisites:** TASK-069 completed.
- **Description:** Mount the CesiumJS 3D geospatial globe cleanly inside the Next.js client component lifecycle without SSR/window conflicts.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/CesiumViewer.tsx`, dynamically import Cesium with `ssr: false`.
  2. Initialize `Cesium.Viewer` with custom minimalist options (disable default Bing imagery, timeline, animation, and info box widgets).
  3. Configure high-precision WGS84 globe terrain with neutral/clean ambient lighting.
  4. Implement smooth camera orbit, pan, zoom, and fly-to controls.
- **Definition of Done:** CesiumJS globe mounts inside React view without browser console errors; frame rate maintains >= 60 FPS in empty scene.

---

### TASK-071: 3D Tiles Streaming and Camera Trajectory Overlay
- **Prerequisites:** TASK-070 completed.
- **Description:** Stream the reconstructed 3D Tileset and render the estimated flight trajectory camera path in Electric Cyan.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/TileLoader.ts`, load `Cesium3DTileset.fromUrl(tilesetUrl)`.
  2. Position and orient the tileset at its exact geographic coordinate center.
  3. In `apps/web/components/viewer/TrajectoryLayer.ts`, load the CZML/GeoJSON flight path and render the camera flight path using a glowing Electric Cyan `#00F0FF` polyline with directional camera frustums at keyframe locations.
- **Definition of Done:** Reconstructed 3D building/terrain tiles stream dynamically as the camera navigates; cyan camera trajectory renders accurately above the model.

---

### TASK-072: Multi-Layer Visibility and Opacity Control Panel
- **Prerequisites:** TASK-071 completed.
- **Description:** Build the viewer sidebar allowing users to toggle layers and adjust opacity per Design Doc Section 16.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/LayerControl.tsx`, implement layer toggles:
     - `☑ Terrain`
     - `☑ Buildings`
     - `☑ Roads`
     - `☑ Vegetation`
     - `☑ Point Cloud`
     - `☑ Mesh`
     - `☑ Camera Path`
     - `☑ Confidence Heatmap`
     - `☑ AI Inferred Areas`
  2. Implement opacity sliders (0-100%) for each layer.
  3. Connect toggles to Cesium feature styling and entity visibility.
- **Definition of Done:** Toggling off "Vegetation" hides tree geometry; toggling "Point Cloud" switches rendering between solid mesh and point cloud points.

---

### TASK-073: Interactive Distance & Height Measurement Tool
- **Prerequisites:** TASK-072 completed.
- **Description:** Implement click-to-measure distance and vertical height tools with uncertainty indicators per Design Doc Section 17.
- **Implementation Steps:**
  1. In `apps/web/components/measurement/MeasurementManager.ts`, attach Cesium `ScreenSpaceEventHandler` for mouse clicks.
  2. Distance Mode: User clicks Point A and Point B; render cyan measurement guide line and floating HUD:
     - Distance: `12.48 m`
     - Estimated error: `±0.18 m`
  3. Height Mode: User clicks ground point and roof point; snap vertical height line and display:
     - Structure Height: `18.2 m`
     - Estimated error: `±0.31 m`
  4. Coordinates HUD: Display live Latitude, Longitude, and Altitude at mouse cursor.
- **Definition of Done:** Clicking two building points displays measured distance and uncertainty; matches known ground truth within ±2%.

---

### TASK-074: Polygon Area and Surface Volume Measurement Tool
- **Prerequisites:** TASK-073 completed.
- **Description:** Implement multi-point polygon selection for area and volumetric calculation (stockpiles, excavation pits).
- **Implementation Steps:**
  1. In `apps/web/components/measurement/PolygonMeasure.ts`, allow users to click multiple vertices to draw a closed polygon.
  2. Compute 2D surface area in square meters ($m^2$) on the georeferenced plane.
  3. In Volume Mode: compute cut/fill volume between triangulated polygon surface and base reference plane.
  4. Render measurement summary in floating teal card.
- **Definition of Done:** Drawing a polygon over a flat surface calculates correct area in square meters; volume tool estimates stockpile volume.

---

### TASK-075: Semantic Object Inspection & Confidence Heatmap Mode
- **Prerequisites:** TASK-074 completed.
- **Description:** Implement click-to-inspect semantic metadata and the dedicated confidence visualization mode per Design Doc Sections 19 & 20.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/ObjectInspector.tsx`, on clicking an object (e.g. building), display contextual panel:
     - Object Type: Commercial Building
     - Height: 18.2 m | Footprint: 1,420 $m^2$
     - Geometry Confidence: 94% | Texture Confidence: 87%
     - Observation State: Mostly Observed (84% observed, 10% partially observed, 6% AI inferred)
  2. In `apps/web/components/viewer/ConfidenceShader.ts`, apply custom shader recoloring geometry:
     - High Confidence: Cyan / Teal (`#00F0FF` / `#069494`)
     - Medium Confidence: Neutral Gray (`#F4F4F4`)
     - Low Confidence: Bubblegum Pink (`#FF69B4`)
- **Definition of Done:** Clicking a building displays semantic attributes and confidence breakdown; toggling confidence mode paints the model with the cyan-to-pink gradient.

---

### TASK-076: Model Quality Panel and Full Accuracy Report View
- **Prerequisites:** TASK-075 completed.
- **Description:** Build the comprehensive Accuracy Report and Model Quality Panel per Design Doc Section 18 and PRD Section 39.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/ModelQualityPanel.tsx`, display:
     - Overall Quality Score: `87 / 100`
     - Coverage: `93%`
     - Horizontal RMSE: `0.32 m`
     - Vertical RMSE: `0.58 m`
     - Observation breakdown: Observed (84%), Partially Observed (10%), AI Inferred (6%)
     - Warnings list: (e.g. `"North-facing facade has limited observations"`, `"Roof texture confidence reduced by shadows"`)
  2. Include `[ Download Full Accuracy PDF/JSON ]` button.
- **Definition of Done:** Panel accurately reflects the backend `AccuracyReport` data contract; renders with clean typography and high-contrast metrics.

---

## Phase 13: Model Export UI & Backend Export Worker

### TASK-077: Backend Export Worker for Multi-Format Conversion
- **Prerequisites:** TASK-076 completed.
- **Description:** Implement asynchronous export conversion jobs supporting GLB, OBJ, PLY, LAS, LAZ, GeoTIFF DEM, and 3D Tiles packages.
- **Implementation Steps:**
  1. In `workers/export/converter.py`, implement `ExportWorker`.
  2. Supported conversions:
     - Point cloud: Filter LAS/LAZ by semantic class or bounding box crop.
     - Mesh: Re-export GLB or OBJ with optional coordinate shift or reduced texture resolution.
     - Terrain: GeoTIFF clip to user polygon.
  3. Package output into a zip archive with `metadata.json` and `accuracy_report.json`.
  4. Upload finished package to `s3://{bucket}/projects/{project_id}/exports/{export_id}/`.
- **Definition of Done:** Integration test triggers export for `GLB` and `LAS`, producing downloadable zip packages on S3.

---

### TASK-078: Frontend Export Dialog & Download Manager
- **Prerequisites:** TASK-077 completed.
- **Description:** Implement the clean export dialog modal per Design Doc Section 21.
- **Implementation Steps:**
  1. In `apps/web/components/export/ExportModal.tsx`, build selection modal:
     - 3D Model: Radio buttons for GLB, OBJ, glTF.
     - Point Cloud: Radio buttons for LAS, LAZ, PLY.
     - Terrain: GeoTIFF DEM/DSM.
     - Streaming: 3D Tiles package.
     - Coordinate system selector (Original UTM or WGS84).
  2. Submits `POST /projects/{id}/exports`.
  3. Displays active export conversion progress with download button when ready.
- **Definition of Done:** Selecting "GLB" and clicking "Export" triggers the backend export job, displays progress, and provides the download link.

---

### TASK-079: Automated Pre-Signed Download URLs & Audit Logging
- **Prerequisites:** TASK-078 completed.
- **Description:** Generate short-lived, secure pre-signed download URLs for export packages and log export events for security compliance.
- **Implementation Steps:**
  1. In `apps/api/src/routers/exports.py`, implement `GET /exports/{id}/download`.
  2. Verify user has project read permissions; generate S3 pre-signed URL with 15-minute expiration.
  3. Record audit event in `audit_logs` table: `user_id`, `export_id`, `format`, `timestamp`, `ip_address`.
- **Definition of Done:** Attempting to download without authentication fails; authenticated download returns expiring pre-signed URL and creates audit log.

---

### TASK-080: Post-Export Cleanup and Temporary Asset Lifecycle Policy
- **Prerequisites:** TASK-079 completed.
- **Description:** Implement automated cleanup for intermediate scratch files and expired export archives to control cloud storage costs.
- **Implementation Steps:**
  1. In `scripts/storage_cleanup.py`, implement lifecycle cleaner.
  2. Purge local NVMe scratch files `/tmp/scratch/{job_id}/` immediately after job completion.
  3. Configure S3 bucket lifecycle rule: delete objects in `jobs/{job_id}/interim/` after 7 days; delete generated export zip files after 48 hours.
- **Definition of Done:** Script verifies local scratch directory is pruned; S3 bucket lifecycle rules verified via AWS CLI / Terraform state.

---

## Phase 14: Cloud Infrastructure, GPU Autoscaling & Production Hardening

### TASK-081: Terraform Infrastructure as Code: VPC, RDS PostGIS, Redis & S3
- **Prerequisites:** TASK-080 completed.
- **Description:** Author reproducible Terraform modules for base AWS cloud infrastructure.
- **Implementation Steps:**
  1. In `infra/terraform/`:
     - `vpc/`: Multi-AZ VPC with public, private, and database subnets.
     - `rds/`: Amazon RDS PostgreSQL 16/18 with PostGIS extension enabled and automated snapshots.
     - `elasticache/`: Redis cluster with in-transit encryption.
     - `s3/`: S3 buckets with server-side encryption (KMS/AES-256), private access blocking, and CORS rules for direct web uploads.
     - `sqs/`: SQS standard queues with dead-letter queue (DLQ) for failed reconstruction jobs.
- **Definition of Done:** Running `terraform validate` and `terraform plan` passes without errors across staging configuration.

---

### TASK-082: Terraform Infrastructure as Code: Amazon EKS Cluster & IAM Roles
- **Prerequisites:** TASK-081 completed.
- **Description:** Provision the Amazon EKS Kubernetes cluster with least-privilege IAM roles for service accounts (IRSA).
- **Implementation Steps:**
  1. In `infra/terraform/eks/`, provision EKS cluster with managed CPU node group for API services.
  2. Configure IAM OIDC provider.
  3. Attach IAM roles for service accounts allowing backend pods to access S3 buckets, SQS queues, and KMS keys without hardcoded secrets.
- **Definition of Done:** `kubectl get nodes` confirms operational cluster; test pod successfully queries S3 via IAM role.

---

### TASK-083: Karpenter Provisioner Configuration for Dynamic GPU Worker Nodes
- **Prerequisites:** TASK-082 completed.
- **Description:** Configure Karpenter on EKS to dynamically provision GPU worker nodes on-demand and scale down to zero when the reconstruction queue is empty.
- **Implementation Steps:**
  1. In `infra/kubernetes/karpenter/gpu-provisioner.yaml`, define `NodePool` targeting EC2 GPU instance families (e.g. `g5.xlarge`, `g5.2xlarge`, `g6`).
  2. Configure Spot instance preference with automatic On-Demand fallback.
  3. Install NVIDIA GPU Operator to automatically manage GPU drivers, container runtime, and Kubernetes device plugins.
  4. Configure scale-to-zero: terminate idle GPU nodes after 5 minutes of empty queue.
- **Definition of Done:** Publishing a job to SQS triggers Karpenter to provision a GPU node in <2 minutes; completing the job terminates the GPU node after idle timeout.

---

### TASK-084: CloudFront CDN Setup for 3D Tiles and Static Web Distribution
- **Prerequisites:** TASK-083 completed.
- **Description:** Deploy Amazon CloudFront CDN distribution in front of S3 3D Tiles and the Next.js web application.
- **Implementation Steps:**
  1. In `infra/terraform/cloudfront/`, configure CloudFront distribution with Origin Access Control (OAC) to S3 model bucket.
  2. Configure caching behavior for 3D Tiles (`tileset.json` with short TTL, `.b3dm` / `.pnts` with long immutable cache headers).
  3. Configure SSL/TLS certificate via AWS Certificate Manager.
- **Definition of Done:** Requesting a 3D tile through CloudFront URL returns `X-Cache: Hit from cloudfront` with gzip/brotli compression.

---

### TASK-085: Distributed Tracing & Observability Dashboards
- **Prerequisites:** TASK-084 completed.
- **Description:** Deploy OpenTelemetry collector, Prometheus, and Grafana dashboards monitoring pipeline metrics, GPU utilization, and API health.
- **Implementation Steps:**
  1. In `infra/monitoring/`, deploy OpenTelemetry collector on EKS.
  2. Create Grafana dashboards tracking:
     - GPU utilization, VRAM usage, and GPU temperature across workers.
     - Reconstruction job stage latency (breakdown of time spent in pose, depth, fusion, texturing).
     - SQS queue depth and worker scaling velocity.
     - Reconstruction success vs failure rates.
- **Definition of Done:** Grafana dashboard displays live GPU utilization metrics and end-to-end trace waterfalls for completed reconstruction jobs.

---

### TASK-086: GitHub Actions CI/CD Pipeline
- **Prerequisites:** TASK-085 completed.
- **Description:** Build automated continuous integration and deployment workflows for frontend, API, and worker container images.
- **Implementation Steps:**
  1. In `.github/workflows/ci.yml`:
     - Step 1: Linting and type-checking (`pnpm lint`, `pnpm build:types`, `mypy`).
     - Step 2: Automated unit and integration tests (`pnpm test`, `pytest`).
     - Step 3: Multi-arch Docker container build for `apps/web`, `apps/api`, and `workers/reconstruction-gpu`.
     - Step 4: Security vulnerability scanning (Trivy).
     - Step 5: Push images to Amazon ECR and deploy to staging EKS via Helm / ArgoCD.
- **Definition of Done:** Merging a pull request to `main` passes all automated tests, builds container images, and deploys to the staging environment without manual intervention.

---

### TASK-087: Controlled Drone Benchmark Dataset Evaluation & Verification Suite
- **Prerequisites:** TASK-086 completed (complete platform deployed).
- **Description:** Run end-to-end evaluation against controlled aerial benchmark datasets to mathematically verify PRD Section 27 & 45 targets.
- **Implementation Steps:**
  1. In `scripts/benchmark_suite.py`, execute end-to-end pipeline across standard aerial datasets with known ground truth (Urban, Rural, Industrial, Mixed Terrain).
  2. Validate quantitative acceptance criteria:
     - Horizontal RMSE $\le 0.5\text{ m}$ target on suitable inputs.
     - Vertical RMSE $\le 0.75\text{ m}$ target on suitable inputs.
     - Observable surface coverage $\ge 90\%$.
     - Dynamic object contamination $< 5\%$.
     - 5-minute 4K flight reconstruction time $\le 15\text{ minutes}$ on GPU workers.
     - Measurement error $\le 2\%$.
     - Interactive viewer performance $\ge 30\text{ FPS}$.
  3. Generate final benchmark verification report document.
- **Definition of Done:** Benchmark run outputs passing verification logs across all test categories, proving the complete system meets all PRD and Design Doc requirements.

---

## Task Verification & Progression Matrix

| Phase | Tasks | Key Deliverable | Primary Tech | Verification Criteria |
|---|---|---|---|---|
| **1. Scaffolding & Schemas** | `TASK-001` - `TASK-008` | Monorepo, shared schemas, local DB/Redis/MinIO | pnpm, Turborepo, Pydantic, Docker | `docker compose up` healthy, schema tests pass |
| **2. Control Plane API** | `TASK-009` - `TASK-017` | REST API, Auth0 RBAC, S3 upload URLs, WebSockets | FastAPI, SQLAlchemy, PostGIS, Auth0 | Auth enforced, signed URLs work, WS streams |
| **3. Flight Ingestion** | `TASK-018` - `TASK-023` | Video/GPS validation, quality scoring (0-100) | OpenCV, FFprobe, Shapely | Quality report generated, bad inputs flagged |
| **4. Pose & Keyframes** | `TASK-024` - `TASK-029` | Blur-free keyframes, 6-DoF camera poses, CZML | OpenCV, SIFT, GTSAM, CZML | Reprojection error < 1px, cyan flight path |
| **5. Monocular Depth** | `TASK-030` - `TASK-034` | Metric depth maps + confidence fields | PyTorch, CUDA, Depth Anything V2 | Millimetric depth saved, scale aligned to GPS |
| **6. Semantics & Dynamic** | `TASK-035` - `TASK-039` | Semantic scene masks, moving object removal | SegFormer, ByteTrack | Moving cars masked, static structures retained |
| **7. 3D Fusion & Point Cloud**| `TASK-040` - `TASK-045` | Fused 3D point cloud, LAS/LAZ/PLY export | Open3D, PDAL | Clean point cloud, ASPRS classes, no ghost points |
| **8. Mesh & Texturing** | `TASK-046` - `TASK-051` | Screened Poisson mesh, seamless UV atlases | Open3D, xatlas, Draco, GLB | Watertight mesh, LOD 0/1/2, seamless textures |
| **9. Georeferencing & Tiles** | `TASK-052` - `TASK-056` | WGS84/UTM coordinates, GeoTIFF, 3D Tiles | GDAL, PostGIS, OGC 3D Tiles | RMSE report generated, 3D Tiles in ECEF |
| **10. UI Design System** | `TASK-057` - `TASK-063` | Bubblegum Pink & Deep Teal component library | Next.js 16, Tailwind, Inter font | Brand colors active, primitives accessible |
| **11. Frontend Core Views** | `TASK-064` - `TASK-069` | Dashboard, Upload, Pre-Flight Quality, Progress | React Query, Radix UI | Upload to S3 works, realtime progress bar active |
| **12. CesiumJS 3D Viewer** | `TASK-070` - `TASK-076` | 3D Tiles streaming, measurements, quality panel | CesiumJS, WebGL | >=30 FPS, distance/height error HUD, layers |
| **13. Exports & Lifecycle** | `TASK-077` - `TASK-080` | GLB/LAS/GeoTIFF exports, secure downloads | S3 presigned URLs, Python zip | Zip generated, audit logged, scratch pruned |
| **14. Cloud & Benchmarks** | `TASK-081` - `TASK-087` | Terraform, EKS, Karpenter GPU, benchmark suite | Terraform, Karpenter, OpenTelemetry | Horiz RMSE $\le 0.5m$, Vert RMSE $\le 0.75m$ |

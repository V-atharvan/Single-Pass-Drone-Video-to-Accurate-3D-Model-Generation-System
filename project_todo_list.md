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

### TASK-001: Initialize Monorepo Architecture and Workspace Tooling [COMPLETED]
- **Prerequisites:** Git, Node.js 22 LTS, Python 3.12+, Docker Desktop / Podman installed locally.
- **Description:** Set up the foundational monorepo directory layout supporting both the web frontend (`apps/web`), backend API (`apps/api`), asynchronous reconstruction workers (`workers/`), shared packages (`packages/`), and infrastructure (`infra/`).
- **Implementation Steps:**
  1. Initialize root directory with `pnpm` workspace configuration (`pnpm-workspace.yaml`).
  2. Configure root `package.json` with Turborepo (`turbo.json`) for orchestrating build, lint, and type-checking pipelines.
  3. Create root `.gitignore`, `.editorconfig`, and Prettier/ESLint configs.
  4. Scaffold directories: `apps/web`, `apps/api`, `packages/schemas`, `packages/geospatial`, `packages/shared`, `workers/`, `infra/terraform`, `scripts/`.
- **Definition of Done:** Running `pnpm install` and `pnpm lint` executes cleanly across all monorepo directories with zero errors.

---

### TASK-002: Establish Core Pydantic Schemas in Shared Schemas Package [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-001 completed.
- **Description:** Define centralized, authoritative data contracts in Python using Pydantic V2 for all domain entities, telemetry records, reconstruction job states, quality metrics, and 3D metadata. Enhanced to include all missing sensor schemas, observation state enum, calibration hierarchy, and positioning mode enum.
- **Implementation Steps:**
  1. In `packages/schemas/python/`, configure `pyproject.toml` with `pydantic>=2.7.0`.
  2. Implement `telemetry.py`: Define `GPSRecord` (lat, lon, alt_msl, alt_rel, speed, heading, timestamp, accuracy, hdop, vdop), `IMURecord` (roll, pitch, yaw, q0..q3, timestamp), `CameraIntrinsics` (fx, fy, cx, cy, k1, k2, p1, p2, width, height), **`BarometerRecord`** (pressure_hpa, altitude_baro_m, temperature_c, timestamp) — optional sensor; system must function without it.
  3. Implement `jobs.py`: Define `JobState` enum (15 states from PRD), `QualityPreset` enum (`LOW`, `BALANCED`, `HIGH`), `JobProgressUpdate`, and **`PositioningMode`** enum (`RTK_PPK`, `RTK`, `GPS_IMU`, `GPS_ONLY`, `GPS_DEGRADED`, `VISUAL_ONLY`) — reported in AccuracyReport and used to set uncertainty expectations.
  4. Implement `quality.py`: Define `InputQualityScore` (overall, video_score, gps_score, blur_score, texture_score, overlap_score, **compression_score**, **shadow_score**, **illumination_score**, warnings, component_warnings).
  5. Implement `models.py`: Define `ModelMetadata`, `ComponentConfidence` (geometry_confidence, depth_confidence, pose_confidence, geolocation_confidence, texture_confidence — all 0-100 float), `AccuracyReport` (rmse_horizontal, rmse_vertical, scale_error_pct, coverage_pct, observed_pct, partial_pct, inferred_pct, dynamic_excluded_pct, unknown_pct, low_confidence_pct, confidence_score, positioning_mode, ground_control_used, estimated_horizontal_uncertainty_m, estimated_vertical_uncertainty_m), and `MeasurementResult` (value, error_margin, observation_state_at_point).
  6. Implement **`observation.py`**: Define `ObservationState` enum with exactly 6 values — `OBSERVED`, `PARTIAL`, `INFERRED`, `UNKNOWN`, `DYNAMIC_EXCLUDED`, `LOW_CONFIDENCE`. INFERRED geometry must never be labeled as measured/observed anywhere in the system.
  7. Implement **`calibration.py`**: Define `CalibrationSource` enum (`USER_PROVIDED`, `KNOWN_PROFILE`, `ESTIMATED`), `CameraCalibration` (intrinsics, distortion_coefficients, calibration_source, calibration_confidence_0_100, sensor_width_mm, sensor_height_mm, camera_model_name). Calibration quality must propagate into reconstruction confidence.
- **Definition of Done:** Pytest suite in `packages/schemas/python/tests/` passes validation tests for valid and malformed domain payloads including all new enums and models.

---

### TASK-003: Implement TypeScript Type Generation from Pydantic Contracts [COMPLETED]
- **Prerequisites:** TASK-002 completed.
- **Description:** Establish an automated type synchronization pipeline that converts the shared Pydantic models into TypeScript definitions for `apps/web`.
- **Implementation Steps:**
  1. In `packages/schemas/`, create a build script (`scripts/export-types.py`) using `pydantic-to-typescript` or JSON Schema export + `json-schema-to-typescript`.
  2. Configure output directory `packages/schemas/ts/src/` to package generated `.d.ts` and `.ts` files.
  3. Add build target `pnpm build:types` in root `turbo.json`.
- **Definition of Done:** Running `pnpm build:types` outputs valid TypeScript interfaces in `packages/schemas/ts/` matching every Pydantic model with zero missing fields.

---

### TASK-004: Containerized Local Infrastructure Setup [COMPLETED]
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

### TASK-005: PostgreSQL Database Schema Definition and Migration Scripts [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-004 completed.
- **Description:** Create the relational and geospatial database schema using SQLAlchemy 2.0 and Alembic migrations. Enhanced to include all missing sensor and confidence columns.
- **Implementation Steps:**
  1. In `apps/api/`, configure `alembic` targeting the PostGIS database.
  2. Define tables:
     - `organizations` (id, name, created_at)
     - `users` (id, org_id, email, full_name, role, auth0_sub, created_at)
     - `projects` (id, org_id, name, description, location_name, crs, bbox geometry(Polygon, 4326), tags, created_at, updated_at)
     - `flights` (id, project_id, original_filename, s3_video_key, s3_telemetry_key, duration_seconds, fps, resolution_width, resolution_height, total_frames, status, **video_quality_report_json**, **has_barometric_altitude**, **has_rtk_corrections**, created_at)
     - `flight_telemetry` (id, flight_id, timestamp_offset, location geometry(PointZ, 4326), altitude_msl, **altitude_barometric_m**, heading, speed, raw_json) — altitude_barometric_m nullable; null means barometric data unavailable
     - `reconstruction_jobs` (id, flight_id, org_id, status, quality_preset, **positioning_mode**, **rtk_corrections_used**, current_stage, progress, error_message, started_at, completed_at)
     - `models` (id, job_id, project_id, name, crs, bbox geometry(Polygon, 4326), center_lat, center_lon, min_altitude, max_altitude, coverage_percent, observed_pct, partial_pct, inferred_pct, dynamic_excluded_pct, unknown_pct, low_confidence_pct, **geometry_confidence**, **depth_confidence**, **pose_confidence**, **geolocation_confidence**, **texture_confidence**, confidence_score, position_rmse, vertical_rmse, **estimated_h_uncertainty_m**, **estimated_v_uncertainty_m**, **positioning_mode**, **scale_source**, **ground_control_used**, s3_prefix, status, **reconstruction_version**, **sensor_config_json**, created_at)
     - `model_assets` (id, model_id, asset_type enum [`GLB`, `OBJ`, `PLY`, `LAS`, `TILES_3D`, `DEM_TIF`, `DSM_TIF`], s3_key, file_size_bytes, lod_level)
     - `measurements` (id, model_id, user_id, type enum [`DISTANCE`, `HEIGHT`, `AREA`, `VOLUME`], geometry_json, value, error_margin, **observation_state_at_point**, created_at)
  3. Execute `alembic upgrade head`.
- **Definition of Done:** Migration applies cleanly against the PostGIS container; spatial indices (`GIST`) are verified on `bbox` and `location` columns; all new nullable columns exist.

---

### TASK-006: S3 Object Storage Client & Key Structure Provisioner [COMPLETED]
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

### TASK-007: Redis Client and Distributed State Tracker [COMPLETED]
- **Prerequisites:** TASK-006 completed.
- **Description:** Implement a high-performance Redis client for transient job state caching, worker heartbeats, and pub/sub message propagation.
- **Implementation Steps:**
  1. In `packages/shared/python/redis_client.py`, create `RedisStateTracker` using `redis-py` async.
  2. Implement atomic job stage updates: `set_job_progress(job_id, stage, progress, frames_processed, total_frames, gpu_stats)`.
  3. Implement pub/sub channel publishing on key `job_events:{job_id}` for WebSocket fan-out.
  4. Implement worker liveness heartbeats with TTL expiration (15 seconds).
- **Definition of Done:** Integration test verifies that a published job progress event updates the cached hash in Redis and emits a message to the pub/sub subscriber.

---

### TASK-008: OpenTelemetry Instrumentation Baseline [COMPLETED]
- **Prerequisites:** TASK-007 completed.
- **Description:** Set up centralized OpenTelemetry tracing and structured logging for both the API and background workers.
- **Implementation Steps:**
  1. In `packages/shared/python/telemetry.py`, configure OpenTelemetry SDK with OTLP exporter support and JSON structured console logging.
  2. Add tracer hooks for tracing async tasks, database transactions, S3 file transfers, and ML inference blocks.
  3. Define standard trace tags: `project_id`, `flight_id`, `job_id`, `worker_stage`.
- **Definition of Done:** A mock script creates a span with attributes and outputs structured JSON logs containing `trace_id` and `span_id`.

---

## Phase 2: Backend Control Plane API & Authentication

### TASK-009: FastAPI Application Bootstrap and Health Probes [COMPLETED]
- **Prerequisites:** TASK-008 completed.
- **Description:** Initialize the FastAPI REST application with standard middleware, CORS, lifecycle management, and health endpoints.
- **Implementation Steps:**
  1. In `apps/api/src/main.py`, instantiate `FastAPI(title="Single-Pass 3D Reconstruction API", version="1.0.0")`.
  2. Attach CORS middleware allowing Next.js frontend origin.
  3. Implement lifespan context manager connecting/disconnecting SQLAlchemy engine pool and Redis pool.
  4. Implement `/health/liveness` and `/health/readiness` (checking PostgreSQL and Redis connections).
- **Definition of Done:** Running `uvicorn apps.api.src.main:app` starts the server; `GET /health/readiness` returns HTTP 200 with DB and Redis statuses `UP`.

---

### TASK-010: Auth0 JWT Authentication and Token Validation Middleware [COMPLETED]
- **Prerequisites:** TASK-009 completed.
- **Description:** Implement security middleware to validate Auth0 RS256 Bearer JWT tokens, extract claims, and map to local user identities.
- **Implementation Steps:**
  1. In `apps/api/src/auth/jwt.py`, implement `JWTVerifier` using `PyJWT` and Auth0 JSON Web Key Sets (`JWKS`).
  2. Implement dependency `get_current_user(token: str = Depends(oauth2_scheme))` that caches JWKS public keys, verifies token expiration/audience/issuer, and queries or creates the user record in PostgreSQL.
- **Definition of Done:** Test suite verifies valid JWT allows access; expired, malformed, or missing tokens return HTTP 401 Unauthorized.

---

### TASK-011: Multi-Tenant RBAC Authorization Guard [COMPLETED]
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

### TASK-012: Organizations and Users Management Endpoints [COMPLETED]
- **Prerequisites:** TASK-011 completed.
- **Description:** Implement REST endpoints for managing organizations, member invitations, and role assignments.
- **Implementation Steps:**
  1. In `apps/api/src/routers/organizations.py`, implement:
     - `GET /organizations/me`: Return caller's organization details.
     - `GET /organizations/me/members`: List users with roles.
     - `PATCH /organizations/me/members/{user_id}`: Update user role (Admin only).
- **Definition of Done:** Verified via integration test: non-admin gets HTTP 403 when updating roles; admin gets HTTP 200 with updated role.

---

### TASK-013: Projects CRUD API Endpoints [COMPLETED]
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

### TASK-014: S3 Pre-Signed Direct Upload URL Service [COMPLETED]
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

### TASK-015: Flights Registration and Metadata Linkage Endpoints [COMPLETED]
- **Prerequisites:** TASK-014 completed.
- **Description:** Implement endpoints to register an uploaded video and GPS file as a cohesive flight asset ready for validation.
- **Implementation Steps:**
  1. In `apps/api/src/routers/flights.py`, implement:
     - `POST /projects/{project_id}/flights`: Links uploaded S3 video key and S3 telemetry key to a new `flights` row.
     - `GET /projects/{project_id}/flights`: List all flights in a project with upload status and duration.
     - `GET /flights/{id}`: Fetch detailed flight metadata.
- **Definition of Done:** Database query confirms flight record is linked to project with initial status `PENDING_VALIDATION`.

---

### TASK-016: Reconstruction Job Submission & State Machine API [COMPLETED]
- **Prerequisites:** TASK-015 completed.
- **Description:** Implement endpoints to start, inspect, and cancel 3D reconstruction jobs.
- **Implementation Steps:**
  1. In `apps/api/src/routers/reconstruction_jobs.py`, implement:
     - `POST /flights/{flight_id}/reconstruction-jobs`: Accepts `quality_preset` (`LOW`, `BALANCED`, `HIGH`), verifies flight is valid, creates `reconstruction_jobs` entry with state `QUEUED`, publishes job ID to SQS/queue.
     - `GET /reconstruction-jobs/{id}`: Returns state, progress percentage, current stage, frames processed, and warnings.
     - `POST /reconstruction-jobs/{id}/cancel`: Sets state to `CANCELLED` if job is still in progress.
- **Definition of Done:** Submitting a job creates a DB record with state `QUEUED` and enqueues the payload into SQS/mock queue.

---

### TASK-017: Real-Time WebSocket Server for Job Status Streaming [COMPLETED]
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

### TASK-018: Reconstruction Worker Dispatcher & SQS Consumer Loop [COMPLETED]
- **Prerequisites:** TASK-017 completed.
- **Description:** Build the asynchronous Python worker daemon that polls SQS, acquires job locks, and orchestrates pipeline execution.
- **Implementation Steps:**
  1. In `workers/orchestrator/runner.py`, implement `JobWorkerDaemon`.
  2. Poll SQS queue with long-polling (20s).
  3. On message receive: parse `job_id`, fetch job and flight records from PostgreSQL, mark job state as `VALIDATING` in DB and Redis.
  4. Provide graceful SIGTERM handling to release or requeue stalled jobs.
- **Definition of Done:** Worker pulls a test message from the queue, updates PostgreSQL job status to `VALIDATING`, and logs the trace.

### TASK-019: Video File Validation, Stream Integrity Probing & Compression Quality Assessment [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-018 completed.
- **Description:** Build the video validation module using FFprobe / OpenCV to inspect video codec, resolution, frame rate, container integrity, and produce a complete VideoQualityReport including compression artifact analysis.
- **Implementation Steps:**
  1. In `workers/preprocessing/video_validator.py`, implement `validate_video_stream(local_video_path: Path) -> VideoQualityReport`.
  2. Verify video format (`MP4`, `MOV`), codec (`H.264`, `H.265/HEVC`), minimum resolution (1080p), duration, and valid frame count.
  3. Detect corrupted frames, missing moov atoms, or variable framerate anomalies.
  4. **[ENHANCED]** Measure bitrate (average and peak) using FFprobe stream analysis; flag very low bitrate (< 8 Mbps for 1080p, < 25 Mbps for 4K) as HIGH_COMPRESSION.
  5. **[ENHANCED]** Detect block/DCT compression artifacts: sample 10 evenly-spaced I-frames, compute blocky patch variance using 8x8 grid analysis; produce `compression_artifact_score` [0..100] where 0=severe artifacts.
  6. **[ENHANCED]** Estimate per-frame noise using Laplacian-based signal-to-noise estimate; produce `noise_score` [0..100].
  7. **[ENHANCED]** Detect I-frame/keyframe spacing from stream metadata; flag if average GOP (Group of Pictures) > 60 frames as poor for reconstruction.
  8. **[ENHANCED]** Produce full `VideoQualityReport`: {resolution, fps, codec, bitrate_mbps, avg_gop_size, blur_score, compression_artifact_score, noise_score, exposure_score, frame_usability_pct, gps_availability, metadata_completeness_score}.
  9. Update `flights` row with detected width, height, fps, total_frames, duration, and `video_quality_report_json`.
- **Definition of Done:** Tested against valid 4K/1080p MP4s and corrupt/truncated/high-compression video files; VideoQualityReport JSON matches schema; high-compression inputs produce compression_artifact_score < 60.

---

### TASK-020: Flight Telemetry, GPS, Barometric Altitude Extractor & Time Synchronizer [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-019 completed.
- **Description:** Parse uploaded GPS and flight metadata files (CSV, JSON, NMEA, DJI SRT/subtitle stream) and synchronize coordinates with video frame timestamps. Enhanced to extract barometric altitude and flag RTK/PPK data presence.
- **Implementation Steps:**
  1. In `workers/preprocessing/telemetry_parser.py`, implement parser for DJI SRT format, CSV, and standard JSON telemetry.
  2. Extract timestamps, latitude, longitude, altitude (MSL and relative), heading, gimbal pitch/roll/yaw, HDOP/VDOP.
  3. **[ENHANCED]** Extract `barometric_altitude_m` where present (DJI SRT field `baro`, CSV column `baro_alt`, or JSON key `baroAlt`); set to `null` if unavailable. Do NOT treat barometric altitude as absolute survey-grade elevation without calibration.
  4. **[ENHANCED]** Detect presence of RTK/PPK correction data fields (DJI RTK fields, `.obs`/`.nav` RINEX files, separate RTK CSV); set `has_rtk_corrections = True` on the flight record.
  5. Perform linear/spline interpolation of GPS fixes to match video keyframe timestamps.
  6. Populate `flight_telemetry` rows including `altitude_barometric_m` column.
- **Definition of Done:** SRT/CSV telemetry sample parsed; barometric altitude populated where available and null where not; RTK flag correctly set; GPS records synchronized per second.

---

### TASK-021: Visual Quality Metric Evaluator (Blur, Exposure, Texture, Shadow, Illumination) [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-020 completed.
- **Description:** Implement computer-vision algorithms to assess frame quality, motion blur, exposure, scene texture, shadow coverage, and illumination variance. All scores feed the Input Quality Score.
- **Implementation Steps:**
  1. In `workers/preprocessing/quality_evaluator.py`, implement:
     - **Blur score:** Laplacian variance across frames (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
     - **Exposure check:** Luminance histogram distribution; flag underexposed (< 15% luminance) or overexposed (> 85% saturation) frames.
     - **Texture score:** Spatial gradient density (Sobel operator) to detect low-texture surfaces.
     - **[ENHANCED] Shadow score:** Compute the ratio of pixels in the shadow luminance band (< 30% of scene median luminance) across sampled frames; output `shadow_coverage_pct` and `shadow_score` [0..100] where 0 = heavy shadows dominating scene.
     - **[ENHANCED] Illumination variance score:** Measure frame-to-frame mean luminance standard deviation to detect rapid auto-exposure transitions; high variance = low `illumination_score`.
- **Definition of Done:** Unit test processes sharp/blurred/dark/overexposed/shadow-heavy frames and outputs normalized scores [0..100] accurately classifying all frame types.

---

### TASK-022: Flight Trajectory Continuity and Overlap Quality Estimator [COMPLETED]
- **Prerequisites:** TASK-021 completed.
- **Description:** Evaluate drone flight trajectory consistency, speed spikes, GPS gaps, and estimated visual overlap between successive views.
- **Implementation Steps:**
  1. In `workers/preprocessing/trajectory_evaluator.py`, calculate drone velocity vectors between consecutive GPS fixes.
  2. Flag GPS outlier jumps (>30 m/s sudden deviation for standard UAVs) and temporal gaps (>2 seconds without position).
  3. Estimate baseline-to-height ratio (B/H) to ensure sufficient parallax for stereoscopic/multi-view reconstruction.
- **Definition of Done:** A simulated trajectory with a 5-second GPS dropout flags a `GPS_GAP_DETECTED` warning and deducts from the trajectory score.

---

### TASK-023: Input Quality Score Synthesizer and Pre-Flight Quality Report [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-022 completed.
- **Description:** Aggregate visual, compression, and trajectory metrics into an authoritative 0-100 Input Quality Score, write results to DB, and notify the frontend. Enhanced formula includes compression and shadow components.
- **Implementation Steps:**
  1. In `workers/preprocessing/quality_synthesizer.py`, compute weighted score:
     ```
     Score = 0.28*Blur + 0.20*GPS + 0.15*Texture + 0.12*Compression + 0.10*Exposure
           + 0.08*Shadow + 0.07*Illumination
     ```
     (weights are configurable; no single factor should silently zero out the score)
  2. Classify expected result: `HIGH` (>= 80), `MODERATE` (60-79), `LOW` (< 60).
  3. **[ENHANCED]** Produce component-level warnings for each sub-score below threshold:
     - blur_score < 60: `"HIGH_MOTION_BLUR"`
     - compression_score < 50: `"HIGH_VIDEO_COMPRESSION — recommend higher bitrate source"`
     - shadow_score < 40: `"HEAVY_SHADOWS — texture and depth confidence will be reduced in shadow regions"`
     - gps_score < 50: `"POOR_GPS — georeferencing accuracy reduced"`
  4. Persist `InputQualityScore` into `flights.video_quality_report_json`.
  5. If overall score < 40 or critical data missing (video corrupt, GPS entirely absent), mark job `FAILED` with specific actionable remediation instructions; otherwise proceed.
- **Definition of Done:** Mock flight run generates exact output schema from PRD FR-005; WebSocket message sent; high-compression + shadow-heavy test video produces appropriate sub-score warnings.

---

## Phase 4: Keyframe Selection, Camera Poses & Sensor Fusion

### TASK-024: Keyframe Selection Engine with Preset Control [COMPLETED]
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

### TASK-025: Keyframe Disk and Storage Caching Pipeline [COMPLETED]
- **Prerequisites:** TASK-024 completed.
- **Description:** Extract selected keyframes at native resolution to local high-speed NVMe scratch storage and upload thumbnails to S3.
- **Implementation Steps:**
  1. In `workers/pose/frame_exporter.py`, use PyAV / OpenCV to extract selected keyframes directly to `/tmp/scratch/{job_id}/frames/frame_%05d.png`.
  2. Generate downscaled 512px WebP thumbnails and upload to `s3://{bucket}/jobs/{job_id}/interim/thumbnails/`.
  3. Update Redis stage: `EXTRACTING_FRAMES`, emitting progress percentage as frames are decoded.
- **Definition of Done:** Keyframe PNGs exist on local disk and thumbnails are viewable from MinIO bucket.

---

### TASK-026: Visual Odometry & Feature Matching Across Keyframes [COMPLETED]
- **Prerequisites:** TASK-025 completed.
- **Description:** Extract robust keypoint features and compute multi-view correspondences across sequential and overlapping keyframes.
- **Implementation Steps:**
  1. In `workers/pose/feature_tracker.py`, implement feature extraction using OpenCV SIFT / learned keypoints (SuperPoint / DISK).
  2. Match features across adjacent frames and loop-closure candidates using LightGlue / FLANN matcher with ratio test.
  3. Filter outlier matches using RANSAC epipolar constraint (essential matrix estimation $E$).
- **Definition of Done:** Unit test matches 10 sequential drone keyframes, identifying >=500 robust inlier correspondences per frame pair.

---

### TASK-027: Sensor Fusion Extended Kalman Filter / Pose Graph Optimization [COMPLETED]
- **Prerequisites:** TASK-026 completed.
- **Description:** Fuse visual odometry relative poses with GPS, RTK/PPK, barometric altitude, and IMU into an optimized, drift-free trajectory. System must gracefully handle each sensor being absent.
- **Implementation Steps:**
  1. In `workers/pose/sensor_fusion.py`, construct a factor graph using GTSAM / SciPy least squares.
  2. Nodes: Camera 6-DoF poses ($R_i, t_i$).
  3. Factors with explicit fallback hierarchy:
     - Visual relative pose constraints (always present).
     - GPS position priors with uncertainty covariance based on HDOP/VDOP (if GPS available).
     - **[ENHANCED] RTK/PPK position priors** with tighter covariance (sigma_h ~0.02m) when `has_rtk_corrections = True`.
     - Barometric altitude prior for vertical stability (if `altitude_barometric_m` not null; do NOT use as absolute elevation without calibration).
     - IMU orientation priors (gimbal pitch/roll/yaw, if available).
  4. **[ENHANCED] Positioning mode selection:** After optimization, classify and store `positioning_mode`:
     - `RTK_PPK` if RTK corrections used with high residual quality
     - `GPS_IMU` if GPS + IMU used without RTK
     - `GPS_ONLY` if IMU absent
     - `GPS_DEGRADED` if GPS dropout > 20% of flight
     - `VISUAL_ONLY` if GPS entirely absent
  5. Solve non-linear optimization; report per-camera pose uncertainty (position sigma x/y/z) as output.
- **Definition of Done:** Trajectory optimization converges; positioning_mode correctly classified across test scenarios (full GPS, GPS dropout, RTK, GPS-only); pose uncertainty estimated.

---

### TASK-028: Camera Extrinsics/Intrinsics Solver & Trajectory Exporter [COMPLETED]
- **Prerequisites:** TASK-027 completed.
- **Description:** Format and validate camera calibration matrices and optimized extrinsics for each keyframe, computing pose confidence metrics.
- **Implementation Steps:**
  1. In `workers/pose/pose_output.py`, calculate reprojection RMSE for each camera pose.
  2. Assign pose confidence score (0-100%) based on inlier count and GPS-visual alignment residual.
  3. Export `cameras.json` to S3 containing camera matrix $K$, world-to-camera matrix, camera-to-world center, pose confidence, and **per-camera position uncertainty sigma (x, y, z)**.
  4. **[ENHANCED] Accuracy claim correction:** Mean reprojection error is a **[BENCHMARK TARGET]** of < 1.5 px on controlled datasets; scene-dependent factors (repetitive textures, low light, heavy occlusion) will increase this. Never report reprojection error as a guaranteed system property.
- **Definition of Done:** `cameras.json` saved in S3; every keyframe has a valid 4x4 transformation matrix and per-camera pose uncertainty; mean reprojection error logged as a measured metric (not a guarantee).

---

### TASK-029: Trajectory Visualization GeoJSON / CZML Generator [COMPLETED]
- **Prerequisites:** TASK-028 completed.
- **Description:** Convert estimated 3D flight trajectory into Cesium-compatible CZML / GeoJSON for interactive rendering in the frontend viewer.
- **Implementation Steps:**
  1. In `workers/pose/czml_generator.py`, generate a CZML document defining the camera flight path, orientation frustums at keyframe intervals, and time-stamped waypoints.
  2. Style path using Electric Cyan `#00F0FF` per design document specifications.
  3. Upload `trajectory.czml` to S3 interim storage.
- **Definition of Done:** Valid CZML file is written; verified to parse cleanly with Cesium CZML parser schema.

---

## Phase 5: Monocular AI Depth Estimation & Confidence Prediction

### TASK-030: PyTorch GPU Worker Runtime & CUDA Model Loader [COMPLETED]
- **Prerequisites:** TASK-029 completed.
- **Description:** Initialize the GPU worker runtime environment, verifying CUDA availability, TensorRT/torch.compile optimization, and GPU memory management.
- **Implementation Steps:**
  1. In `workers/depth/runtime.py`, verify `torch.cuda.is_available()`, log GPU model name and VRAM capacity.
  2. Implement PyTorch memory cache cleaner and FP16 / BF16 mixed-precision context manager.
  3. Download and cache pre-trained monocular metric depth model weights (e.g., Depth Anything V2 Metric / UniDepth) to local model cache directory.
- **Definition of Done:** Script executes on GPU worker, loads model into VRAM under 6GB memory footprint, and runs dummy inference without CUDA out-of-memory errors.

---

### TASK-031: Monocular Metric Depth Map Inference Pipeline [COMPLETED]
- **Prerequisites:** TASK-030 completed.
- **Description:** Process keyframes through the depth network to produce dense, high-resolution metric depth maps.
- **Implementation Steps:**
  1. In `workers/depth/depth_estimator.py`, implement batch inference over extracted keyframes.
  2. Normalize and resize keyframes to model input resolution; run batched model forward pass under `torch.inference_mode()`.
  3. Upsample output depth maps back to original keyframe resolution using bilateral / guided filtering to preserve sharp building edges.
- **Definition of Done:** Running depth estimation across 50 keyframes outputs 16-bit / 32-bit floating-point depth arrays ($Z$ in meters) for every keyframe.

---

### TASK-032: Depth Uncertainty and Confidence Field Estimation [COMPLETED]
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

### TASK-033: Metric Scale Calibration Against Sensor Baseline [COMPLETED]
- **Prerequisites:** TASK-032 completed.
- **Description:** Align and calibrate estimated depth map scale factors against physical GPS baseline travel distance and barometric altitude ground clearance.
- **Implementation Steps:**
  1. In `workers/depth/scale_aligner.py`, compute scale factor $s$ for each depth map such that 3D triangulated sparse feature points match the physical baseline distance between camera centers:
     $$s^* = \arg\min_s \sum \| s \cdot X_{\text{depth}} - X_{\text{triangulated}} \|^2$$
  2. Apply median scale factor across the keyframe sequence to ensure global metric consistency across all frames.
- **Definition of Done:** Scale-aligned depth maps produce ground-truth-consistent elevation deltas matching barometric flight height within ±5%.

---

### TASK-034: Depth and Confidence Map Serialization to NVMe/S3 [COMPLETED]
- **Prerequisites:** TASK-033 completed.
- **Description:** Compress and serialize metric depth arrays and confidence maps to local NVMe scratch storage and archive them to S3.
- **Implementation Steps:**
  1. In `workers/depth/serializer.py`, save depth maps as compressed 16-bit PNG (with millimetric scale factor) or EXR files: `depth_%05d.png`.
  2. Save confidence maps as 8-bit PNG: `confidence_%05d.png`.
  3. Update Redis stage to `ESTIMATING_DEPTH` and publish progress updates after each batch of 10 frames.
- **Definition of Done:** All depth and confidence maps for the keyframe sequence are written to disk; file sizes and checksums validated.

---

## Phase 6: Semantic Scene Understanding & Dynamic Object Removal

### TASK-035: Semantic Segmentation Pipeline Setup [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-034 completed.
- **Description:** Deploy a lightweight, accurate semantic segmentation model (e.g. SegFormer / Mask2Former) classifying scene regions into target geospatial categories. Enhanced to include animal and general dynamic-object fallback classes.
- **Implementation Steps:**
  1. In `workers/segmentation/semantic_classifier.py`, load segmentation model trained on drone/aerial imagery.
  2. Target classes: `BUILDING`, `ROAD`, `TERRAIN`, `VEGETATION`, `VEHICLE`, `PERSON`, `UTILITY_INFRASTRUCTURE`, `WATER`, `UNKNOWN`, **`ANIMAL`**, **`DYNAMIC_OBJECT`**.
     - `ANIMAL`: Detects horses, cattle, dogs, birds, and other animals using auxiliary detector; falls back to `DYNAMIC_OBJECT` when species not confidently classified — the system must never pretend to classify what it cannot.
     - `DYNAMIC_OBJECT`: General fallback for any moving pixel region not matching a specific class; treated as dynamic and excluded from reconstruction.
  3. Run batched inference across all keyframes; generate 8-bit class index masks.
- **Definition of Done:** Semantic masks correctly categorize buildings, roads, trees, vehicles, pedestrians, and animals [BENCHMARK TARGET: >= 80% mean IoU on standard aerial benchmark]; unknown/unclassified moving objects use DYNAMIC_OBJECT fallback.

---

### TASK-036: Dynamic Object Detection & Multi-Frame Motion Tracking [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-035 completed.
- **Description:** Identify moving vehicles, people, animals, and general dynamic objects across sequential frames to prevent phantom/ghost geometry in the 3D reconstruction.
- **Implementation Steps:**
  1. In `workers/segmentation/dynamic_tracker.py`, detect objects from the semantic mask and bounding box detector.
  2. Track objects across sequential frames using ByteTrack / BoT-SORT.
  3. Compute optical flow and reprojected background motion; if an object's motion vector diverges from expected camera epipolar geometry, classify it as `DYNAMIC`.
  4. **[ENHANCED]** Apply dynamic classification to all tracked classes: VEHICLE, PERSON, ANIMAL, and DYNAMIC_OBJECT fallback.
  5. **[ENHANCED]** A stationary animal (e.g. parked cattle) is only classified DYNAMIC if optical flow confirms motion; do not auto-exclude all animals.
- **Definition of Done:** A moving car, walking pedestrian, and moving animal are all flagged `DYNAMIC`; stationary objects of the same classes remain as static scene; DYNAMIC_OBJECT fallback catches unclassified moving pixels.

---

### TASK-037: Dynamic Object Mask Generation & Dilation [COMPLETED]
- **Prerequisites:** TASK-036 completed.
- **Description:** Generate dilated binary exclusion masks for all dynamic entities to ensure complete exclusion during 3D point cloud back-projection.
- **Implementation Steps:**
  1. In `workers/segmentation/mask_generator.py`, isolate all pixels labeled with active dynamic object tracks.
  2. Apply morphological dilation (kernel size 5x5) to prevent border artifact leaking.
  3. Produce binary dynamic mask `dynamic_%05d.png` (255 = dynamic/exclude, 0 = static).
- **Definition of Done:** Dynamic masks cover 100% of moving vehicle pixels plus safe 5px margin around object contours.

---

### TASK-038: Semantic Mask Association & Mask Pyramid Builder [COMPLETED]
- **Prerequisites:** TASK-037 completed.
- **Description:** Merge static semantic classification with dynamic exclusion masks into a unified multi-channel semantic tensor per keyframe.
- **Implementation Steps:**
  1. In `workers/segmentation/mask_combiner.py`, combine class ID (Channel 0), dynamic flag (Channel 1), and classification confidence (Channel 2).
  2. Generate multi-resolution pyramids (1x, 0.5x, 0.25x) to support multi-scale LOD 3D fusion.
- **Definition of Done:** Verified multi-channel semantic mask file created for each keyframe with intact label mappings.

---

### TASK-039: Semantic and Dynamic Mask Serialization to Storage [COMPLETED]
- **Prerequisites:** TASK-038 completed.
- **Description:** Save final semantic masks to scratch NVMe and upload sample previews to S3 interim storage.
- **Implementation Steps:**
  1. In `workers/segmentation/serializer.py`, write mask files to `/tmp/scratch/{job_id}/semantics/`.
  2. Update Redis stage to `SEGMENTING` with progress percentage.
  3. Upload dynamic object statistics (number of moving objects removed) to job metadata.
- **Definition of Done:** Semantic mask files saved; dynamic object contamination metric stored for the final quality report.

---

## Phase 7: 3D Point Cloud Fusion, Surface Completion & Outlier Filtering

### TASK-040: Depth Unprojection and Point Cloud Back-Projection [COMPLETED]
- **Prerequisites:** TASK-039 completed.
- **Description:** Unproject calibrated 2D metric depth maps into 3D camera coordinates and transform them to global coordinates using 6-DoF poses.
- **Implementation Steps:**
  1. In `workers/fusion/unprojector.py`, use PyTorch / NumPy to back-project depth pixels $(u, v, d)$ to 3D points in camera frame:
     $$X_c = \frac{(u - c_x) \cdot d}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad Z_c = d$$
  2. Transform camera points to world frame using extrinsics matrix $[R_w | t_w]$.
  3. Associate each 3D point with RGB color from the source keyframe, depth confidence score, and semantic class ID.
- **Definition of Done:** Back-projecting 10 keyframes creates 3D point clusters whose overlapping regions physically align in 3D coordinate space.

---

### TASK-041: Multi-Frame Point Cloud Fusion with Confidence Weighting [COMPLETED]
- **Prerequisites:** TASK-040 completed.
- **Description:** Fuse individual point sets into a consolidated global point cloud, applying confidence weights and multi-view visibility verification.
- **Implementation Steps:**
  1. In `workers/fusion/point_fusion.py`, implement spatial voxel hashing to merge duplicate observations across overlapping frames.
  2. Where a voxel contains observations from multiple frames, compute the weighted average 3D position based on depth confidence:
     $$P_{\text{fused}} = \frac{\sum w_i \cdot P_i}{\sum w_i}$$
  3. Discard points observed by only one frame if their confidence is below 60%.
- **Definition of Done:** Point cloud fusion reduces redundant overlapping points while increasing surface sharpness and geometric signal-to-noise ratio.

---

### TASK-042: Dynamic Object & Statistical Outlier Removal [COMPLETED]
- **Prerequisites:** TASK-041 completed.
- **Description:** Cleanse the fused point cloud by stripping out points falling inside dynamic masks and removing floating sky/edge artifacts using Open3D.
- **Implementation Steps:**
  1. In `workers/fusion/outlier_filter.py`, filter out any point tagged with a dynamic object mask.
  2. Apply Statistical Outlier Removal (SOR) using Open3D (`nb_neighbors=20, std_ratio=1.5`).
  3. Apply Radius Outlier Removal (ROR) to eliminate sparse flying points far from primary structures.
- **Definition of Done:** Point cloud shows clean structural outlines (crisp building edges, clean ground) with zero floating ghost points from passing vehicles or birds.

---

### TASK-043: Voxel Downsampling and Normal Vector Estimation [COMPLETED]
- **Prerequisites:** TASK-042 completed.
- **Description:** Resample the cleaned point cloud to a uniform spatial density and compute accurate surface normal vectors.
- **Implementation Steps:**
  1. In `workers/fusion/normal_estimator.py`, perform voxel grid downsampling with voxel size tailored to quality preset (`HIGH` = 0.05m, `BALANCED` = 0.10m, `LOW` = 0.20m).
  2. Estimate normals using covariance analysis over local $k$-nearest neighbors ($k=30$).
  3. Orient normals consistently toward the drone camera positions.
- **Definition of Done:** Fused point cloud has uniform point spacing; 100% of surface points have normalized, consistently oriented normal vectors $(n_x, n_y, n_z)$.

---

### TASK-044: Occlusion Analysis and Surface Observation State Tagging [COMPLETED] [ENHANCED]
- **Prerequisites:** TASK-043 completed.
- **Description:** Analyze viewing ray visibility to categorize every region of the 3D scene into the full 6-state observation state per the ObservationState enum from TASK-002.
- **Implementation Steps:**
  1. In `workers/fusion/occlusion_analyzer.py`, trace camera viewing rays against the reconstructed point cloud and voxel grid.
  2. Classify points and spatial cells into the full **6-state** ObservationState:
     - `OBSERVED`: Visual rays directly intersected from 2+ camera angles.
     - `PARTIAL`: Observed from only a single steep angle or near occlusion boundary.
     - `INFERRED`: Occluded shadow/backside surface filled by learned surface prior. **MUST be labeled as inferred; must never be presented as measured geometry.**
     - `UNKNOWN`: Completely unobserved region — no ray intersections from any camera angle.
     - **`DYNAMIC_EXCLUDED`**: Region removed because it belonged to a dynamic object track; point cloud has no geometry here by design.
     - **`LOW_CONFIDENCE`**: Observed but with depth confidence below threshold (< 30%), or in shadow region, or on reflective surface — reconstruction exists but should not be trusted for measurement.
  3. Calculate global percentages: `observed_pct`, `partial_pct`, `inferred_pct`, `unknown_pct`, `dynamic_excluded_pct`, `low_confidence_pct`; sum must equal 100%.
- **Definition of Done:** Point cloud attributes contain `observation_state` with all 6 possible values; statistics verified to sum to 100%; DYNAMIC_EXCLUDED regions match dynamic mask footprint.

---

### TASK-045: LAS/LAZ and PLY Point Cloud Serializer [COMPLETED]
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

### TASK-046: Surface Mesh Reconstruction Engine [COMPLETED]
- **Prerequisites:** TASK-045 completed.
- **Description:** Generate a continuous 3D triangular surface mesh from the oriented point cloud using Screened Poisson Surface Reconstruction.
- **Implementation Steps:**
  1. In `workers/mesh/surface_reconstructor.py`, invoke Open3D Poisson Reconstruction (`depth=10` or `11` depending on preset).
  2. Trim low-density mesh vertices where point cloud support is absent using density thresholding to avoid spurious bubble surfaces.
  3. Segment mesh into terrain surface and architectural building facades using semantic classification from the underlying points.
- **Definition of Done:** Reconstructed polygonal mesh exhibits clean planar roofs, vertical facades, and smooth terrain without non-physical ballooning.

---

### TASK-047: Mesh Topology Cleanup, Non-Manifold Removal & Hole Infilling [COMPLETED]
- **Prerequisites:** TASK-046 completed.
- **Description:** Clean mesh topology to guarantee 2-manifold surface properties and apply localized hole filling with AI inference tagging.
- **Implementation Steps:**
  1. In `workers/mesh/mesh_cleaner.py`, remove duplicate vertices, zero-area degenerate triangles, and non-manifold edges.
  2. Identify boundary holes caused by minor drone occlusions; apply planar/minimal-surface hole filling for small gaps (<2m diameter).
  3. Flag filled hole faces with metadata attribute `is_ai_inferred = True`.
- **Definition of Done:** Open3D mesh analysis reports `is_edge_manifold() == True` and `is_vertex_manifold() == True`; hole-filled triangles are distinctly tagged.

---

### TASK-048: Keyframe Raycasting & Optimal View Selection for Texturing [COMPLETED]
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

### TASK-049: Exposure Correction, Seam Blending & Texture Atlas Baking [COMPLETED]
- **Prerequisites:** TASK-048 completed.
- **Description:** Parameterize UV coordinates, equalize exposure and white balance across differing source frames, and bake seamless texture atlases.
- **Implementation Steps:**
  1. In `workers/texture/atlas_baker.py`, compute UV parameterization using xatlas.
  2. Apply Poisson image blending / multi-band blending across texture boundaries to eliminate visible seam lines caused by drone lighting changes.
  3. Generate 4096x4096 texture atlases (`diffuse_00.png`) and auxiliary confidence texture maps (`confidence_00.png`).
- **Definition of Done:** Textured 3D mesh displays uniform illumination across face seams without visible tiling boundaries or harsh color transitions.

---

### TASK-050: Multi-Resolution Level-of-Detail (LOD) Decimation [COMPLETED]
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

### TASK-051: GLB, glTF, and OBJ Mesh Exporter [COMPLETED]
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

### TASK-053: Geospatial Accuracy Assessment, RMSE Estimator & Positioning Mode Report [ENHANCED]
- **Prerequisites:** TASK-052 completed.
- **Description:** Statistically evaluate reconstruction accuracy against GPS telemetry/GCPs and generate the quantitative Accuracy Report. Enhanced to distinguish three accuracy types, report positioning mode, and label all numbers as measured/benchmark rather than guaranteed.
- **Implementation Steps:**
  1. In `workers/geospatial/accuracy_evaluator.py`, compute residuals:
     - **Relative reconstruction accuracy:** Internal consistency — reprojection error, track length, loop closure residuals. Does not require external reference.
     - **Absolute geospatial accuracy:** RMSE between estimated camera centers and GPS/RTK positions. **[BENCHMARK TARGET, NOT GUARANTEE]** Horizontal RMSE < 0.5m on good GPS; vertical RMSE < 0.75m. Actual measured value always reported.
     - **Measurement accuracy:** Estimated linear measurement error from propagated uncertainty (depth confidence + pose sigma + calibration confidence). **[BENCHMARK TARGET]** < 2% on high-confidence OBSERVED surfaces; UNKNOWN/INFERRED surfaces must display warning before any measurement.
  2. **[ENHANCED]** Include in AccuracyReport:
     - `positioning_mode`: one of `RTK_PPK`, `RTK`, `GPS_IMU`, `GPS_ONLY`, `GPS_DEGRADED`, `VISUAL_ONLY`
     - `ground_control_used`: boolean (GCPs are optional validation aids, never mandatory)
     - `estimated_horizontal_uncertainty_m`: computed from GPS sigma * scale factor
     - `estimated_vertical_uncertainty_m`: computed from GPS vertical sigma + baro error
     - `scale_source`: one of `RTK`, `GPS_BASELINE`, `BAROMETRIC`, `VISUAL_ONLY`
  3. Construct `AccuracyReport` with all fields from TASK-002 schema; never produce fabricated survey-grade claims.
- **Definition of Done:** Accuracy report contains measured RMSE values with clear labels as MEASURED metrics (not guarantees); positioning mode and uncertainty estimates always populated.

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

### TASK-057: Next.js 16 App Directory Setup & Responsive Brand Theme Configuration [ENHANCED]
- **Prerequisites:** TASK-056 completed (backend pipeline functional).
- **Description:** Initialize the web application frontend using Next.js 16 with TypeScript and configure the exact visual design system, color tokens, and responsive mobile viewport foundations from the Design Document.
- **Implementation Steps:**
  1. In `apps/web/`, initialize Next.js with App Router and Tailwind CSS.
  2. Configure viewport meta in `apps/web/app/layout.tsx` with `viewport-fit=cover`, `initial-scale=1`, and `width=device-width`.
  3. Configure `tailwind.config.ts` with brand color palette:
     - `brand-pink`: `#FF69B4` (Bubblegum Pink — primary CTAs, active highlights)
     - `brand-teal`: `#069494` (Deep Teal — technical panels, secondary actions)
     - `brand-cyan`: `#00F0FF` (Electric Cyan — active 3D overlays, telemetry)
     - `brand-white`: `#FFFFFF` (Main canvas, clean cards)
     - Neutral palette: Ink `#111111`, Muted `#6B6B6B`, Soft Gray `#F4F4F4`, Border `#E5E5E5`.
  4. Configure responsive breakpoints (`sm: 640px`, `md: 768px`, `lg: 1024px`, `xl: 1280px`, `2xl: 1536px`), dynamic viewport height utilities (`100dvh`), and safe-area padding utilities (`pt-safe`, `pb-safe`).
  5. Configure Google Font `Inter` in `apps/web/app/layout.tsx` with clean geometric type scale.
- **Definition of Done:** Running `pnpm --filter web dev` launches the frontend; CSS classes `bg-brand-pink`, `text-brand-teal`, `border-brand-cyan`, and safe-area utility classes render correctly across both 375px mobile and 1440px desktop viewports.

---

### TASK-058: Base Primitive UI Components Library with Mobile Touch Ergonomics [ENHANCED]
- **Prerequisites:** TASK-057 completed.
- **Description:** Implement custom UI primitive components matching the design specifications with strict touch ergonomics (minimum 44x44px touch targets) and mobile-responsive drawer primitives.
- **Implementation Steps:**
  1. In `apps/web/components/ui/`:
     - `Button.tsx`: Variants `primary` (pink background, black/white text, rounded), `secondary` (white background, teal border, teal text), `ghost`. Enforce minimum 44px height for mobile touch targets.
     - `Card.tsx`: White background, subtle `#E5E5E5` border, rounded radius (12px), generous padding, responsive spacing.
     - `Badge.tsx`: Compact status and tag pill badges.
     - `Modal.tsx`: Accessible dialog using `@radix-ui/react-dialog` with responsive full-screen behavior on mobile.
     - `Drawer.tsx`: Accessible mobile bottom sheet / drawer primitive with swipe-down dismissal and safe-area insets.
     - `Input.tsx` and `Select.tsx`: Minimal, high-contrast form controls with 44px minimum touch height and accessible labels.
     - `Tooltip.tsx`: Contextual explanation tooltips with mobile tap-to-reveal fallback.
- **Definition of Done:** Storybook / component preview page displays all button variants, cards, modal dialogs, and mobile drawers with correct hover, focus, and touch states; all interactive elements pass the >= 44x44px touch target check.

---

### TASK-059: Responsive Application Shell: Desktop Sidebar & Mobile Bottom Navigation Bar [ENHANCED]
- **Prerequisites:** TASK-058 completed.
- **Description:** Implement the unified responsive application shell featuring a desktop sidebar / top navigation for large screens and a fixed bottom navigation bar for mobile screens per Design Doc Sections 7, 8, 27, and 32.
- **Implementation Steps:**
  1. In `apps/web/components/shell/Sidebar.tsx` and `apps/web/components/shell/Header.tsx`, build desktop/laptop shell (>= 1024px):
     - Left sidebar or top navigation with direct access to: Dashboard, Projects, Upload, Processing, 3D Workspace, Exports, Settings.
     - Minimal geometric brand emblem, global search bar, active jobs pulse badge, and user profile dropdown.
  2. In `apps/web/components/shell/MobileBottomNav.tsx`, build mobile navigation (< 768px):
     - Fixed bottom navigation bar with 5 primary destinations: `[ Dashboard ] [ Projects ] [ Upload ] [ 3D View ] [ More ]`.
     - Fixed to viewport bottom with device safe-area awareness (`padding-bottom: calc(12px + env(safe-area-inset-bottom))`).
     - Clear active-state indicator (brand pink `#FF69B4` underline or pill background).
     - Minimum 44 x 44 px touch targets with icon and text label.
     - "More" opens accessible mobile bottom sheet drawer with links to: Processing, Exports, Settings, Team, API docs, and Logout.
     - Accessible navigation landmark (`<nav aria-label="Mobile Navigation">`).
     - Fully functional across portrait and landscape orientations without obscuring workspace content.
  3. In `apps/web/components/shell/AppLayout.tsx`, compose shell with automatic responsive breakpoint switching and dynamic content padding avoiding bottom bar overlap.
- **Definition of Done:** Desktop viewport (1440px) renders left sidebar / top navigation; mobile viewport (375px) renders header and fixed bottom navigation bar; tapping bottom nav items switches routes; active page is clearly indicated; safe area padding verified on simulated iPhone/Android viewports.

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

### TASK-064: Responsive Project Dashboard View (`/dashboard`) [ENHANCED]
- **Prerequisites:** TASK-063 completed.
- **Description:** Build the central operational dashboard answering the three primary operator questions per Design Doc Section 9, fully responsive across mobile, tablet, and desktop.
- **Implementation Steps:**
  1. In `apps/web/app/dashboard/page.tsx`, create responsive layout:
     - Top section: Active Projects grid (3 columns on desktop, 2 on tablet, single-column swipeable cards on mobile) with `+ New Project` button.
     - Project Cards: Display Project Name, Location, Flight Count, Model Count, Quality Score badge, and 3D thumbnail preview with touch-friendly target areas.
     - Bottom section: "Active Reconstruction Jobs" monitoring card displaying active flights, animated progress bar, percentage, and current stage; reflows into a vertical stack on mobile.
     - Safe-area bottom padding above mobile navigation bar (`pb-safe`).
- **Definition of Done:** Dashboard loads projects from API and renders live progress bars for active jobs in realtime; verified to reflow cleanly without horizontal scrollbars on 375px mobile, 768px tablet, and 1440px desktop viewports.

---

### TASK-065: Responsive Project Creation Modal & Project Details Page [ENHANCED]
- **Prerequisites:** TASK-064 completed.
- **Description:** Build the responsive modal / bottom sheet to create new projects and the detail page displaying project flights, models, and metadata with mobile card reflow.
- **Implementation Steps:**
  1. In `apps/web/components/project/CreateProjectModal.tsx`, implement form: Name, Description, Location, Coordinate Reference System (CRS dropdown: WGS84, UTM zones), and Tags; reflows to a swipeable bottom sheet or full-screen overlay on mobile (< 768px).
  2. In `apps/web/app/projects/[id]/page.tsx`, display project summary cards, flight list (table on desktop, swipeable touch cards on mobile), and reconstructed models list with direct touch links to 3D viewer.
- **Definition of Done:** User creates a new project via modal or mobile bottom sheet; page navigates to `/projects/[id]`; flight list displays correctly on both desktop tables and mobile card layouts.

---

### TASK-066: Mobile-Responsive Resumable Direct-to-S3 Video & GPS Upload Screen [ENHANCED]
- **Prerequisites:** TASK-065 completed.
- **Description:** Implement a robust, mobile-responsive drag-and-drop and native file picker upload workflow for multi-gigabyte drone videos and companion GPS files per Design Doc Sections 11 & 32.3.
- **Implementation Steps:**
  1. In `apps/web/components/upload/UploadZone.tsx`, build upload UI supporting desktop drag-and-drop and mobile native file picker (`input type="file" accept="video/mp4,video/quicktime"`).
  2. Implement file chunking and direct S3 pre-signed upload using `@uppy/core` or custom multi-part upload worker; offload checksum and chunk slicing to a Web Worker so the mobile UI thread never freezes.
  3. Display upload progress HUD: upload speed (MB/s), percentage completed, and estimated remaining upload time.
  4. Implement pause and resume capability for intermittent mobile network connections.
  5. Provide companion upload slot for GPS files (CSV, JSON, SRT) with automatic timestamp alignment validation.
  6. Display clear, actionable upload error banners with single-tap retry.
  7. Upon upload completion, automatically register flight via `POST /projects/{id}/flights` and transition seamlessly to the pre-flight quality report.
- **Definition of Done:** Uploading a test video on mobile and desktop uploads directly to S3 with active progress HUD, does not freeze the UI, supports pause/resume, and navigates to the quality report upon completion.

---

### TASK-067: Responsive Pre-Flight Input Quality Assessment Screen [ENHANCED]
- **Prerequisites:** TASK-066 completed.
- **Description:** Display immediate pre-flight data validation and quality metrics before initiating GPU reconstruction per Design Doc Sections 12 & 32.2, optimized for mobile screens.
- **Implementation Steps:**
  1. In `apps/web/components/flight/InputQualityScreen.tsx`, render:
     - Video Quality score & status badge (e.g. `91%` — Good).
     - GPS Quality score & status badge (e.g. `74%` — Moderate).
     - Motion Blur score (e.g. `88%` — Low).
     - Scene Texture score (e.g. `94%` — Good).
     - Overall Expected Result badge (`HIGH` / `MODERATE` / `LOW`).
     - Actionable warnings list with clear high-contrast typography.
  2. Responsive reflow: Metrics stack in a 2x2 grid on desktop/tablet and single-column cards on mobile.
  3. Sticky bottom action bar with safe-area spacing: `[ Proceed to Reconstruction ]` (brand pink) or `[ Re-upload Data ]`.
- **Definition of Done:** Screen renders quality metrics across desktop and mobile; sticky bottom CTA is accessible above mobile bottom nav; disables reconstruction if critical input failure is detected.

---

### TASK-068: Responsive Reconstruction Launch Configuration Modal [ENHANCED]
- **Prerequisites:** TASK-067 completed.
- **Description:** Implement job parameter modal allowing operators to select processing quality presets and CRS, reflowing into an accessible bottom sheet on mobile screens.
- **Implementation Steps:**
  1. In `apps/web/components/reconstruction/LaunchJobModal.tsx`, build options:
     - Quality Preset: `Low` (Fastest, preview), `Balanced` (Standard, recommended), `High` (Full resolution, max detail) with touch-friendly segmented pills.
     - Feature toggles: Dynamic Object Removal, Semantic Segmentation.
     - Target CRS projection confirmation.
  2. Mobile adaptation: On screens < 768px, render as a swipeable bottom sheet drawer with full-width buttons and safe-area bottom padding.
  3. Primary action button: `[ Start Reconstruction ]` with brand pink accent.
  4. Submits `POST /flights/{id}/reconstruction-jobs`.
- **Definition of Done:** Submitting form launches backend job and redirects browser to Reconstruction Progress view; verified functional in both desktop modal and mobile bottom sheet modes.

---

### TASK-069: Responsive Reconstruction Real-Time Progress View [ENHANCED]
- **Prerequisites:** TASK-068 completed.
- **Description:** Build the live processing view tracking reconstruction stages, progress percentage, frame count, and ETA per Design Doc Sections 13 & 32.2.
- **Implementation Steps:**
  1. In `apps/web/app/jobs/[id]/page.tsx`, implement responsive layout:
     - Headline: `"Building your 3D scene"`.
     - Large percentage display: e.g. `78%`.
     - Horizontal progress bar with brand pink/teal fill.
     - Frame counter: `"8,421 / 11,672"`.
     - Stage checklist with visual status dots (reflows to vertical mobile stepper):
       - `✓ Input validation`
       - `✓ Frame extraction`
       - `✓ Pose estimation`
       - `● Depth estimation` (active animated indicator)
       - `○ Scene fusion`
       - `○ Mesh generation`
       - `○ Texturing`
       - `○ Quality assessment`
  2. Implement resilient WebSocket connection with automatic background reconnect and mobile screen awake / wake-lock hint.
  3. Automatically transition to the 3D Viewer when status becomes `COMPLETED`.
- **Definition of Done:** Progress view subscribes to WebSocket, animates stage transitions in realtime across mobile and desktop viewports, and navigates to the 3D viewer upon completion.

---

## Phase 12: Frontend CesiumJS 3D Viewer & Interactive Tools

### TASK-070: CesiumJS Container Integration with Next.js & Mobile Touch Navigation [ENHANCED]
- **Prerequisites:** TASK-069 completed.
- **Description:** Mount the CesiumJS 3D geospatial globe cleanly inside the Next.js client component lifecycle without SSR/window conflicts, adding multi-touch gesture navigation (rotate, pinch-zoom, pan), mobile fullscreen toggle, and device pixel ratio capping for mobile thermal efficiency.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/CesiumViewer.tsx`, dynamically import Cesium with `ssr: false`.
  2. Initialize `Cesium.Viewer` with custom minimalist options (disable default Bing imagery, timeline, animation, and info box widgets).
  3. Configure high-precision WGS84 globe terrain with neutral/clean ambient lighting.
  4. Implement smooth camera orbit, pan, zoom, and fly-to controls for mouse/pointer.
  5. Implement touch gesture handler:
     - Single-finger drag: camera rotate and tilt.
     - Two-finger pinch: smooth zoom in/out.
     - Two-finger drag: pan across terrain.
  6. Implement one-tap Fullscreen toggle button hiding mobile browser chrome and UI overlays.
  7. Cap `viewer.resolutionScale` to max 1.5 on high-DPI mobile devices to prevent GPU thermal throttling and conserve battery.
- **Definition of Done:** CesiumJS globe mounts inside React view without browser console errors; touch gestures (rotate, pinch zoom, pan) and fullscreen toggle work reliably on mobile touch devices; frame rate maintains >= 60 FPS in empty scene.

---

### TASK-071: 3D Tiles Streaming, Mobile LOD Optimization & Camera Trajectory [ENHANCED]
- **Prerequisites:** TASK-070 completed.
- **Description:** Stream reconstructed 3D Tileset and camera path in Electric Cyan, with dynamic Level-of-Detail (LOD) tuning for mobile network bandwidth and GPU memory limits.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/TileLoader.ts`, load `Cesium3DTileset.fromUrl(tilesetUrl)`.
  2. Adjust `maximumScreenSpaceError` dynamically based on device tier (e.g. 16 for desktop, 24 for mobile) to optimize streaming speed and frame rate on cellular networks.
  3. Position and orient the tileset at its exact geographic coordinate center.
  4. In `apps/web/components/viewer/TrajectoryLayer.ts`, load CZML/GeoJSON flight path and render camera trajectory in Electric Cyan `#00F0FF` with directional camera frustums at keyframe locations.
  5. Support single-tap selection of keyframe camera frustums to fly camera to capture viewpoint on both desktop and mobile.
- **Definition of Done:** Reconstructed 3D tiles stream dynamically; cyan camera trajectory renders accurately; keyframe frustums respond to touch and mouse selection; mobile memory usage remains within browser budget.

---

### TASK-072: Multi-Layer Visibility, Opacity Control & Mobile Observation-State Drawer [ENHANCED]
- **Prerequisites:** TASK-071 completed.
- **Description:** Build the multi-layer and observation-state control panel, operating as a persistent desktop sidebar and reflowing to a swipeable mobile bottom sheet drawer.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/LayerControl.tsx`, implement layer toggles:
     - `Terrain` / `Buildings` / `Roads` / `Vegetation` / `Point Cloud` / `Mesh` / `Camera Path` / `Confidence Heatmap`.
     - Observation State Filter section (all 6 states):
       - `Observed (solid)` — show/hide directly observed geometry
       - `Partially Observed` — show single-angle or near-occlusion geometry
       - `Inferred (AI)` — show AI-completed surfaces (always labeled, never claimed as measured)
       - `Unknown` — highlight completely unobserved regions
       - `Dynamic Excluded` — visualize regions where dynamic objects were removed
       - `Low Confidence` — highlight shadow/reflective/low-depth-confidence regions.
  2. Implement opacity sliders (0-100%) for each layer with touch-friendly slider handles.
  3. Mobile reflow: On screens < 768px, wrap layer controls into a swipeable mobile bottom sheet drawer (`LayerDrawer.tsx`) accessible via a floating action button or bottom nav, respecting safe-area insets.
  4. Connect toggles to Cesium feature styling using per-point/per-face `observation_state` attribute from TASK-094.
- **Definition of Done:** All 6 observation state filters work independently; desktop sidebar renders cleanly; mobile bottom sheet drawer opens smoothly via touch, toggles layers in realtime, and respects safe-area insets.

---

### TASK-073: Interactive Distance & Height Measurement Tool with Mobile Touch Support [ENHANCED]
- **Prerequisites:** TASK-072 completed.
- **Description:** Implement click and touch-to-measure distance and vertical height tools with uncertainty indicators and mobile precision drag loupe per Design Doc Sections 17 & 32.4.
- **Implementation Steps:**
  1. In `apps/web/components/measurement/MeasurementManager.ts`, attach Cesium `ScreenSpaceEventHandler` for mouse and touch inputs.
  2. Desktop & Mobile Distance Mode: User taps/clicks Point A and Point B; on touch devices, provide drag handles and touch-friendly magnifier loupe for sub-meter pin positioning; render cyan measurement guide line.
  3. Height Mode: User clicks/taps ground and roof point; snap vertical line with structure height display.
  4. Floating Measurement HUD:
     - Distance: `12.48 m` (estimated error: `±0.18 m`)
     - Height: `18.2 m` (estimated error: `±0.31 m`)
     - Observation State at point: Displays whether endpoints are `OBSERVED`, `PARTIAL`, or `INFERRED`.
  5. Mobile layout: Position measurement HUD at top of viewport so it does not interfere with bottom navigation or touch gesture areas.
  6. Coordinates HUD: Display live Latitude, Longitude, and Altitude at cursor or touch pin.
- **Definition of Done:** Tapping or clicking two building points displays measured distance and uncertainty margin; touch drag loupe enables precise mobile pin placement; HUD does not overlap mobile bottom nav.

---

### TASK-074: Polygon Area, Surface Volume Measurement & Mobile Touch Vertices [ENHANCED]
- **Prerequisites:** TASK-073 completed.
- **Description:** Implement multi-point polygon selection for area and volumetric calculation (stockpiles, excavation pits) with mobile touch vertex handles.
- **Implementation Steps:**
  1. In `apps/web/components/measurement/PolygonMeasure.ts`, allow users to click or tap multiple vertices to draw a closed polygon.
  2. On mobile, render touch-friendly draggable vertex pins (minimum 44x44px touch footprint) and an `[ Undo Last Point ]` floating button.
  3. Compute 2D surface area in square meters ($m^2$) on the georeferenced plane.
  4. In Volume Mode: compute cut/fill volume between triangulated polygon surface and base reference plane.
  5. Render measurement summary in floating teal card; on mobile, collapses into a compact top card.
- **Definition of Done:** Drawing a polygon on desktop and mobile calculates correct area and stockpile volume; touch vertex placement and undo work cleanly on mobile viewports.

---

### TASK-075: Semantic Object Inspection & Mobile Bottom Sheet Details [ENHANCED]
- **Prerequisites:** TASK-074 completed.
- **Description:** Implement click/tap-to-inspect semantic metadata and confidence visualization mode, reflowing to a swipeable bottom sheet on mobile screens per Design Doc Sections 19, 20 & 32.4.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/ObjectInspector.tsx`, on clicking/tapping an object (e.g. building), display contextual panel:
     - Object Type: Commercial Building
     - Height: 18.2 m | Footprint: 1,420 $m^2$
     - Geometry Confidence: 94% | Texture Confidence: 87%
     - Observation State: Mostly Observed (84% observed, 10% partially observed, 6% AI inferred)
  2. Mobile reflow: On mobile (< 768px), display object details inside a swipeable bottom sheet drawer with safe-area spacing instead of desktop floating sidebar.
  3. In `apps/web/components/viewer/ConfidenceShader.ts`, apply custom shader recoloring geometry:
     - High Confidence: Cyan / Teal (`#00F0FF` / `#069494`)
     - Medium Confidence: Neutral Gray (`#F4F4F4`)
     - Low Confidence: Bubblegum Pink (`#FF69B4`)
- **Definition of Done:** Tapping an object on mobile opens bottom sheet drawer with semantic attributes and confidence breakdown; confidence mode paints model with cyan-to-pink gradient across all viewports.

---

### TASK-076: Model Quality Panel, Accuracy Report & Mobile Full-Screen Sheet [ENHANCED]
- **Prerequisites:** TASK-075 completed.
- **Description:** Build comprehensive Accuracy Report and Model Quality Panel, enhanced to show all component confidence scores, positioning mode, and sensor config, with full responsive reflow for mobile operators.
- **Implementation Steps:**
  1. In `apps/web/components/viewer/ModelQualityPanel.tsx`, display:
     - Overall Quality Score: `87 / 100`
     - Component Confidence Breakdown: Geometry `94%`, Depth `88%`, Pose `91%`, Geolocation `87%`, Texture `82%`
     - Coverage: `93%`
     - Horizontal RMSE: `0.32 m` *(measured — not a guaranteed value)*
     - Vertical RMSE: `0.58 m` *(measured — not a guaranteed value)*
     - Observation breakdown: Observed (84%), Partially Observed (10%), Inferred (4%), Unknown (1%), Dynamic Excluded (1%)
     - Positioning Mode: e.g. `GPS + IMU` or `RTK + IMU` with color badge
     - Scale Source: e.g. `GPS Baseline` or `RTK`
     - Ground Control Used: `No`
     - Estimated Uncertainties: Horizontal `±0.35 m`, Vertical `±0.62 m`
     - Sensor Configuration: Camera model, calibration source, calibration confidence
     - Actionable warnings list with high-contrast text.
  2. Mobile adaptation: Reflow into a scrollable full-screen mobile sheet or collapsible drawer with sticky download buttons and safe-area padding.
  3. Include `[ Download Full Accuracy PDF/JSON ]` button producing machine-readable JSON matching the `AccuracyReport` schema.
- **Definition of Done:** Panel accurately reflects backend AccuracyReport; all numbers labeled as MEASURED; component confidence bars rendered; mobile view scrolls smoothly and buttons are easily tapped.

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

### TASK-078: Responsive Frontend Export Dialog & Download Manager [ENHANCED]
- **Prerequisites:** TASK-077 completed.
- **Description:** Implement the responsive export dialog modal reflowing into a bottom sheet on mobile devices per Design Doc Sections 21 & 32.2.
- **Implementation Steps:**
  1. In `apps/web/components/export/ExportModal.tsx`, build selection modal:
     - 3D Model: Radio buttons for GLB, OBJ, glTF.
     - Point Cloud: Radio buttons for LAS, LAZ, PLY.
     - Terrain: GeoTIFF DEM/DSM.
     - Streaming: 3D Tiles package.
     - Coordinate system selector (Original UTM or WGS84).
  2. Mobile reflow: Render as an accessible swipeable bottom sheet drawer on screens < 768px with full-width radio options and sticky export CTA button.
  3. Submits `POST /projects/{id}/exports`.
  4. Displays active export conversion progress bar and download manager with direct mobile browser download trigger when ready.
- **Definition of Done:** Selecting "GLB" and clicking "Export" on both desktop and mobile triggers export job, displays progress bar, and successfully initiates browser file download upon completion.

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
  2. Validate quantitative acceptance criteria (all values are **[BENCHMARK TARGETS]**, not guarantees; actual measured results always reported):
     - **[BENCHMARK TARGET]** Horizontal RMSE < 0.5 m on standard GPS; RTK achieves better; GPS-degraded will exceed this.
     - **[BENCHMARK TARGET]** Vertical RMSE < 0.75 m; highly dependent on GPS/baro quality and flight altitude.
     - **[BENCHMARK TARGET]** Observable surface coverage >= 60% single nadir pass; planned orbital passes achieve higher.
     - **[BENCHMARK TARGET]** Dynamic object contamination < 5% on moderate-traffic scenes.
     - **[BENCHMARK TARGET]** 5-minute 4K reconstruction <= 15 minutes on g5.xlarge; longer for dense/large scenes.
     - **[BENCHMARK TARGET]** Measurement error <= 2% on high-confidence OBSERVED surfaces.
     - **[BENCHMARK TARGET]** Interactive viewer >= 30 FPS on modern GPU.
  3. Generate final `benchmark_result.json` (machine-readable, matches AccuracyReport schema).
- **Definition of Done:** Benchmark produces measured results with correct [BENCHMARK TARGET] label

---

## New Tasks Added (TASK-088 → TASK-097)

### TASK-088: Photometric Normalization & Illumination Compensation Module
- **Prerequisites:** TASK-021 completed.
- **Phase:** 3 (Ingestion & Quality), inserted after TASK-021.
- **Description:** Implement a photometric normalization pre-processing step that compensates for frame-to-frame exposure changes, auto-exposure transitions, and strong illumination gradients before feature extraction. The goal is to prevent the pipeline from treating illumination differences as geometry differences.
- **Implementation Steps:**
  1. In `workers/preprocessing/photometric_normalizer.py`, implement `PhotometricNormalizer`.
  2. For each keyframe, compute a histogram equalization in LAB color space to normalize luminance without distorting hue.
  3. Apply adaptive histogram equalization (CLAHE) with a clip limit calibrated to scene brightness variance.
  4. Detect auto-exposure transitions between consecutive keyframes (luminance delta > 20% between adjacent frames); flag affected frames with `ILLUMINATION_TRANSITION` warning.
  5. Optionally apply radiometric gain correction using a reference frame from the sequence (first stable frame after takeoff).
  6. Output normalized keyframe images to `/tmp/scratch/{job_id}/frames_normalized/` alongside original frames (originals preserved for texturing).
  7. Normalized images are used for feature extraction and depth estimation; original photometric frames are used for texture baking.
- **Important:** Normalization does not eliminate illumination problems; it reduces their impact. Shadow regions and extreme overexposure remain unreliable. Do not overclaim that normalization guarantees uniform photometry.
- **Definition of Done:** Unit test shows feature match count increases by >= 15% on an illumination-varying keyframe pair after normalization; ILLUMINATION_TRANSITION warnings correctly emitted on sudden exposure-change frames.

### TASK-089: Shadow Detection & Shadow-Aware Processing Pipeline
- **Prerequisites:** TASK-088 completed.
- **Phase:** 3 (Ingestion & Quality), inserted after TASK-088.
- **Description:** Implement a shadow detection module producing per-frame shadow masks. Shadow masks propagate through the pipeline to: (1) down-weight shadow pixels in feature matching, (2) assign LOW_CONFIDENCE observation state to shadow-dominated surfaces in TASK-044, (3) guide shadow-aware texture selection in TASK-049.
- **Implementation Steps:**
  1. In `workers/preprocessing/shadow_detector.py`, implement `ShadowDetector`.
  2. Shadow detection approach: Convert to LAB; shadow pixels are those with low L channel (< 0.35 * scene median L) AND low saturation variation (to distinguish from dark surfaces).
  3. Produce binary shadow mask `shadow_%05d.png` (255 = shadow, 0 = lit) for each keyframe.
  4. Compute `shadow_coverage_pct` per frame; aggregate to scene-level `shadow_score` [0..100].
  5. Write shadow masks to `/tmp/scratch/{job_id}/shadows/`.
  6. **Important limitation:** The system can distinguish probable shadows from clearly dark surfaces in good-lighting conditions. In low-light or overcast scenes, shadow boundaries are ambiguous. Always flag uncertainty; do not claim confident shadow classification in all conditions.
  7. Downstream integration:
     - Feature matching (TASK-026): de-weight shadow pixels in descriptor computation.
     - Observation state (TASK-044): shadow-dominated points tagged `LOW_CONFIDENCE`.
     - Texture selection (TASK-049): prefer non-shadow keyframe for each face.
- **Definition of Done:** Shadow masks produced for all keyframes; shadow_score correctly rates shadow-heavy test sequence as < 50; feature matching improvement tested on shadow-vs-non-shadow frame pairs.

### TASK-090: Camera Intrinsics First-Class Calibration Handler
- **Prerequisites:** TASK-025 (keyframe extraction) completed.
- **Phase:** 4 (Poses & Trajectory), inserted after TASK-025.
- **Description:** Implement a first-class camera calibration module that resolves camera intrinsics through a priority hierarchy: user-provided > known camera profile database > in-flight self-calibration estimate. Calibration source and confidence propagate into reconstruction accuracy.
- **Implementation Steps:**
  1. In `workers/pose/intrinsics_resolver.py`, implement `IntrinsicsResolver`.
  2. **Priority 1 — User-provided:** Accept `CameraCalibration` JSON uploaded alongside video (fields: fx, fy, cx, cy, k1, k2, p1, p2, width, height, sensor_width_mm, sensor_height_mm, camera_model). Source = `USER_PROVIDED`, confidence = 95.
  3. **Priority 2 — Known profile database:** In `workers/pose/camera_profiles.json`, maintain a database of common drone cameras (DJI Mavic 3, DJI Air 2S, DJI Mini 3 Pro, Autel EVO II, etc.) keyed by EXIF model string. Source = `KNOWN_PROFILE`, confidence = 80.
  4. **Priority 3 — Self-calibration estimate:** If no profile matches, estimate fx = fy ≈ image_width * 0.85 (reasonable drone telephoto prior); cx = image_width/2; cy = image_height/2; k1=k2=p1=p2=0. Source = `ESTIMATED`, confidence = 40. Apply bundle adjustment to refine in TASK-027.
  5. Write resolved calibration to `calibration.json` in S3 interim storage.
  6. Propagate `calibration_confidence` into `geolocation_confidence` and measurement error estimates.
  7. Generate user-visible warning if source is ESTIMATED: `"Camera intrinsics were estimated — measurement accuracy may be reduced"`.
- **Definition of Done:** All 3 paths tested; ESTIMATED path produces plausible fx for known test resolutions; KNOWN_PROFILE correctly matches DJI Mavic 3; USER_PROVIDED overrides all other sources.

### TASK-091: RTK/PPK High-Accuracy Positioning Integration
- **Prerequisites:** TASK-027 (Sensor Fusion) completed.
- **Phase:** 4 (Poses & Trajectory), inserted after TASK-027.
- **Description:** Implement optional RTK/PPK correction processing to achieve higher-accuracy trajectory and georeferencing when RTK/PPK data is available. System must function with standard GPS when RTK is absent.
- **Implementation Steps:**
  1. In `workers/pose/rtk_processor.py`, implement `RTKProcessor`.
  2. **Input detection:** Check `flight.has_rtk_corrections`; if True, locate RTK data file (DJI RTK CSV, RINEX `.obs`/`.nav`, separate correction stream).
  3. **RTK processing:** Apply differential corrections to GPS positions; produce corrected camera positions with horizontal sigma ~ 0.02–0.05 m (vs standard GPS sigma ~ 0.5–3 m).
  4. **PPK processing:** If PPK RINEX file provided, process offline with RTKLib / rtkpost; produce post-processed corrected trajectory.
  5. Feed corrected positions into TASK-027 sensor fusion as RTK/PPK position priors with tighter covariance.
  6. **Graceful fallback:** If RTK/PPK data is corrupt or unavailable, log warning and fall back to standard GPS silently; set `positioning_mode = GPS_IMU`.
  7. Update AccuracyReport with `positioning_mode = RTK_PPK` and tighter estimated uncertainties.
- **Definition of Done:** With RTK test data, estimated horizontal uncertainty reported < 0.1 m vs > 0.5 m without RTK; `positioning_mode` correctly set to `RTK_PPK`; system works identically with RTK absent.

### TASK-092: Sensor Uncertainty Propagation & Component-Level Confidence Model
- **Prerequisites:** TASK-032 (Depth Confidence) completed.
- **Phase:** 5 (Monocular Depth), inserted after TASK-032.
- **Description:** Implement an explicit uncertainty propagation chain connecting sensor-level noise through the reconstruction pipeline to final 3D point uncertainty and measurement error estimates. This enables per-point uncertainty rather than a single generic confidence score.
- **Implementation Steps:**
  1. In `workers/depth/uncertainty_propagator.py`, implement `UncertaintyPropagator`.
  2. **GPS uncertainty → pose uncertainty:** Per-camera pose position sigma (x, y, z) = f(GPS_sigma, HDOP, VDOP, trajectory_smoothness). Store in `cameras.json`.
  3. **Calibration uncertainty → projection uncertainty:** If `calibration_source == ESTIMATED`, add calibration error contribution to projection uncertainty (sigma_px += f(1 - calibration_confidence/100) * 2.0 px).
  4. **Depth uncertainty → 3D point uncertainty:** 3D point sigma = f(depth_sigma, pose_sigma, fx, depth_value): `sigma_3d = sqrt((depth_sigma/depth)^2 + (pose_sigma_t/depth)^2) * depth`.
  5. **3D point uncertainty → measurement error:** For a measurement between two points A and B: `sigma_measurement = sqrt(sigma_A^2 + sigma_B^2)`.
  6. Store per-point `depth_uncertainty_m` and `position_uncertainty_m` as additional LAS dimensions and GLB/glTF extensions.
  7. Surface uncertainty in viewer: when user places measurement on a point, show `±X m` error margin derived from propagated uncertainty (not a fabricated generic value).
- **Definition of Done:** Test point at depth=10m with GPS sigma=1m produces plausible position_uncertainty; measurement tool displays non-trivial per-measurement error margins derived from actual propagated uncertainty.

### TASK-093: Near-Real-Time Fast Preview Generation Path
- **Prerequisites:** TASK-033 (Scale Calibration) completed.
- **Phase:** 5 (Monocular Depth) / 7 (Fusion), inserted after TASK-033.
- **Description:** Implement a fast preview pipeline that produces a coarse 3D point cloud preview for the user within minutes of starting reconstruction, before the full high-quality pipeline completes.
- **Implementation Steps:**
  1. In `workers/preview/fast_preview.py`, implement `FastPreviewPipeline`.
  2. **Fast preview path:** Use every Nth keyframe (N=5 for BALANCED, N=10 for HIGH) instead of all keyframes; use half-resolution depth maps; skip semantic segmentation; skip outlier filtering.
  3. Back-project fast depth maps using estimated poses; produce a coarse sparse point cloud.
  4. Upload coarse point cloud to S3 interim at `jobs/{job_id}/interim/preview/`.
  5. Emit WebSocket message `stage = "PREVIEW_READY"` with S3 URL of coarse point cloud.
  6. The fast preview path is marked as `quality = PREVIEW` in all metadata; it is never confused with the final reconstruction.
  7. **Latency benchmarks (targets, not guarantees):**
     - Upload to preview available: [BENCHMARK TARGET] < 3 minutes on g5.xlarge for 5-minute 4K video.
     - Coarse point cloud density: [BENCHMARK TARGET] ~100K-500K points for typical scene.
  8. Continue full pipeline in parallel; replace preview with final reconstruction when complete.
- **Definition of Done:** Test run shows PREVIEW_READY WebSocket event before COMPLETED event; coarse point cloud visible in viewer; preview clearly labeled as PREVIEW quality; final reconstruction replaces it upon completion.

### TASK-094: Observation State Propagation Through Mesh & 3D Tiles
- **Prerequisites:** TASK-047 (Mesh Topology Cleanup) completed.
- **Phase:** 8 (Mesh & Texturing), inserted after TASK-047.
- **Description:** Ensure the full 6-state ObservationState enum propagates from point cloud through the mesh, GLB export, and 3D Tiles feature tables so the viewer can filter by observation state at runtime.
- **Implementation Steps:**
  1. In `workers/mesh/observation_propagator.py`, implement `ObservationStatePropagator`.
  2. **Point cloud → mesh face:** For each triangular face, sample observation states of its supporting points; assign the worst-case state (hierarchy: OBSERVED > PARTIAL > LOW_CONFIDENCE > INFERRED > UNKNOWN > DYNAMIC_EXCLUDED).
  3. **Mesh → GLB:** Store observation state as a custom vertex attribute `_OBSERVATION_STATE` (uint8) in the glTF binary asset. Integer-to-state mapping: 0=OBSERVED, 1=PARTIAL, 2=INFERRED, 3=UNKNOWN, 4=DYNAMIC_EXCLUDED, 5=LOW_CONFIDENCE.
  4. **Mesh → 3D Tiles:** In each B3DM/GLB tile, include observation state as a Batch Table attribute `observationState` per feature (building/surface).
  5. **Viewer integration:** CesiumJS reads `_OBSERVATION_STATE` attribute via custom shader; TASK-072 layer filters use this attribute to show/hide geometry by state.
  6. Write validation check: sum of face counts per state must equal total face count.
- **Definition of Done:** Exported GLB contains `_OBSERVATION_STATE` vertex attribute; Three.js/glTF viewer shows non-zero values for INFERRED faces; 3D Tiles batch table contains observationState field; TASK-072 layer panel correctly controls visibility by state.

### TASK-095: Digital Twin Metadata Schema & Versioned Export
- **Prerequisites:** TASK-056 (Pipeline Completion) completed.
- **Phase:** 9 (Georeferencing & Tiles), inserted after TASK-056.
- **Description:** Implement digital twin readiness by adding stable object identifiers, reconstruction versioning, sensor configuration snapshots, source video references, and semantic metadata into the model export package.
- **Implementation Steps:**
  1. In `workers/orchestrator/digital_twin_builder.py`, implement `DigitalTwinBuilder`.
  2. Produce `digital_twin_manifest.json` alongside the 3D model containing:
     ```json
     {
       "schema_version": "1.0",
       "reconstruction_id": "<uuid>",
       "reconstruction_version": 1,
       "reconstruction_timestamp": "2026-09-08T01:30:00Z",
       "source_video": {"filename": "...", "s3_key": "...", "duration_s": 300},
       "sensor_configuration": {
         "camera_model": "DJI Mavic 3",
         "calibration_source": "KNOWN_PROFILE",
         "calibration_confidence": 80,
         "positioning_mode": "GPS_IMU",
         "has_imu": true,
         "has_barometric_altitude": true,
         "has_rtk": false
       },
       "coordinate_reference_system": "EPSG:4326",
       "bounding_box": {...},
       "accuracy": {
         "positioning_mode": "GPS_IMU",
         "estimated_horizontal_uncertainty_m": 0.45,
         "estimated_vertical_uncertainty_m": 0.70,
         "ground_control_used": false
       },
       "semantic_objects": [
         {"object_id": "bldg-001", "class": "BUILDING", "height_m": 18.2, "confidence": 94, "observation_state": "OBSERVED"}
       ]
     }
     ```
  3. `reconstruction_version` auto-increments on each re-reconstruction of the same flight.
  4. Upload `digital_twin_manifest.json` to S3 model directory alongside GLB/LAS/GeoTIFF.
  5. Store `reconstruction_version` and `sensor_config_json` in the `models` DB table.
- **Definition of Done:** `digital_twin_manifest.json` validates against schema; reconstruction_version increments on re-run; all required fields present including sensor_configuration and source_video.

---

## Phase 15: Ground-Truth Validation Framework

### TASK-096: Ground-Truth Validation Framework & Machine-Readable Benchmark Reports
- **Prerequisites:** TASK-087 completed (complete platform deployed and benchmarked).
- **Description:** Implement a dedicated ground-truth validation capability that compares reconstructed output against known reference data and produces machine-readable benchmark_result.json reports with per-metric pass/fail evaluation. This is distinct from TASK-087 which runs the benchmark; TASK-096 provides the validation infrastructure and reference dataset pipeline.
- **Implementation Steps:**
  1. In `scripts/validation/ground_truth_validator.py`, implement `GroundTruthValidator`.
  2. **Reference data sources (any of):** Survey-grade 3D model, LiDAR scan, RTK-measured check points, known building dimensions, high-accuracy orthophoto.
  3. **Validation metrics:**
     - Horizontal RMSE (camera centers vs reference positions)
     - Vertical RMSE
     - Absolute position error (ATE)
     - Scale error % (reference known distance vs reconstructed)
     - Building height error vs measured heights
     - Surface completeness % (what % of reference surface has reconstruction within 0.5m)
     - Mean point density (pts/m2)
     - Mean reprojection error (px)
     - Trajectory error (path length vs reference path)
     - Dynamic object removal quality (manual annotation vs mask overlap)
  4. Produce machine-readable output:
     ```json
     {
       "benchmark_id": "<uuid>",
       "timestamp": "...",
       "dataset": "urban_test_01",
       "positioning_mode": "GPS_IMU",
       "horizontal_rmse_m": 0.38,
       "vertical_rmse_m": 0.61,
       "scale_error_pct": 1.2,
       "surface_completeness_pct": 72.4,
       "mean_reprojection_error_px": 1.1,
       "dynamic_removal_iou": 0.91,
       "pass": true,
       "metric_gates": {
         "horizontal_rmse_m": {"target": 0.5, "measured": 0.38, "pass": true},
         "vertical_rmse_m": {"target": 0.75, "measured": 0.61, "pass": true}
       }
     }
     ```
  5. All benchmark values are measured, not fabricated. If reference data is unavailable for a metric, that metric is omitted from the report with `"status": "NO_REFERENCE_DATA"`.
  6. Integrate validation into CI: `pytest scripts/validation/` runs validator against reference dataset on merge to `main`.
- **Definition of Done:** Validator produces valid `benchmark_result.json` against provided reference dataset; no metrics fabricated; CI gate fails if measured RMSE exceeds 2x benchmark target.

### TASK-097: Accuracy Claims Audit, Target Documentation & Benchmark CI Gate
- **Prerequisites:** TASK-096 completed.
- **Phase:** 14/15 (Cloud & Production / Validation).
- **Description:** Create a permanent `docs/accuracy_claims.md` document auditing every numerical claim in the codebase and documentation, converting any unvalidated guarantees to labeled [BENCHMARK TARGET] or [ASPIRATIONAL]. Integrate as a mandatory CI check.
- **Implementation Steps:**
  1. In `docs/accuracy_claims.md`, document the full accuracy claims audit table from the requirements audit.
  2. Add CI check in `.github/workflows/ci.yml`: grep codebase for patterns `"guaranteed"`, `"always achieve"`, `"exact"`, `"perfect"` in reconstruction-related docs/comments; fail CI if found without `[BENCHMARK TARGET]` or `[ASPIRATIONAL]` prefix.
  3. Require benchmark_result.json from TASK-096 to be committed to `benchmarks/` directory on each release.
  4. Any claim in the system that cannot be backed by a committed benchmark_result.json must be labeled [ASPIRATIONAL].
  5. Add `ACCURACY_DISCLAIMER.md` at root: explains that reconstruction accuracy depends on input quality (GPS, calibration, video quality, scene complexity) and that all numbers are measured benchmarks on specific datasets, not universal guarantees.
- **Definition of Done:** `docs/accuracy_claims.md` exists and covers all 10 audited claims; CI grep check active; `ACCURACY_DISCLAIMER.md` created at repo root; any future PR adding unqualified accuracy guarantees fails CI.

---

## Task Verification & Progression Matrix (Updated)

| Phase | Tasks | Key Deliverable | Primary Tech | Verification Criteria |
|---|---|---|---|---|
| **1. Scaffolding & Schemas** | `TASK-001` - `TASK-008` | Monorepo, 7 schema files (ObservationState 6-state, BarometerRecord, CameraCalibration, PositioningMode), local DB/Redis/MinIO | pnpm, Turborepo, Pydantic, Docker | `docker compose up` healthy; all new schema files tested |
| **2. Control Plane API** | `TASK-009` - `TASK-017` | REST API, Auth0 RBAC, S3 upload URLs, WebSockets | FastAPI, SQLAlchemy, PostGIS, Auth0 | Auth enforced, signed URLs work, WS streams |
| **3. Flight Ingestion** | `TASK-018` - `TASK-023`, `TASK-088`, `TASK-089` | VideoQualityReport (7 components), photometric normalization, shadow masks, quality gate | OpenCV, FFprobe, CLAHE | Full report generated; shadow masks produced; compression artifacts scored; bad inputs flagged |
| **4. Pose & Keyframes** | `TASK-024` - `TASK-029`, `TASK-090`, `TASK-091` | Blur-free keyframes, calibration hierarchy (3-tier), RTK/PPK integration, 6-DoF poses + uncertainty, PositioningMode, CZML | SuperPoint, GTSAM, RTKLib | PositioningMode classified; calibration_source set; pose sigma per camera estimated |
| **5. Monocular Depth** | `TASK-030` - `TASK-034`, `TASK-092`, `TASK-093` | Metric depth maps + uncertainty chain, fast preview point cloud | PyTorch, CUDA, Depth Anything V2 | Uncertainty propagated GPS→pose→depth→3D; preview available < 3 min |
| **6. Semantics & Dynamic** | `TASK-035` - `TASK-039` | 11-class semantic masks (incl. ANIMAL, DYNAMIC_OBJECT), moving object removal | SegFormer, ByteTrack | All 11 classes detected; animals/DYNAMIC_OBJECT handled |
| **7. 3D Fusion & Point Cloud** | `TASK-040` - `TASK-045` | Fused 3D point cloud with 6-state ObservationState attributes, LAS/LAZ/PLY export | Open3D, PDAL | 6 states on all points; DYNAMIC_EXCLUDED matches mask footprint; no ghost points |
| **8. Mesh & Texturing** | `TASK-046` - `TASK-051`, `TASK-094` | Poisson mesh, 6-state observation_state on faces/GLB/3D Tiles, shadow-aware textures | Open3D, xatlas, Draco | `_OBSERVATION_STATE` in GLB; state in 3D Tiles batch table; shadow-aware seam blending |
| **9. Georeferencing & Tiles** | `TASK-052` - `TASK-056`, `TASK-095` | WGS84/UTM, GeoTIFF, 3D Tiles, AccuracyReport (measured not guaranteed), digital_twin_manifest.json | GDAL, PostGIS, OGC 3D Tiles | RMSE measured; positioning_mode in report; DT manifest valid JSON |
| **10. UI Design System** | `TASK-057` - `TASK-063` | Bubblegum Pink & Deep Teal component library | Next.js 16, Tailwind, Inter font | Brand colors active, primitives accessible |
| **11. Frontend Core Views** | `TASK-064` - `TASK-069` | Dashboard, Upload, Pre-Flight Quality (7 scores incl. compression/shadow), Progress + preview | React Query, Radix UI | All 7 quality sub-scores shown; fast preview visible before final |
| **12. CesiumJS 3D Viewer** | `TASK-070` - `TASK-076` | 3D Tiles streaming, 6-state observation filter panel, component confidence display, positioning mode badge | CesiumJS, WebGL | >= 30 FPS; all 6 states filterable; accuracy panel shows MEASURED labels |
| **13. Exports & Lifecycle** | `TASK-077` - `TASK-080` | GLB/LAS/GeoTIFF exports with DT metadata, secure downloads | S3 presigned URLs, Python zip | digital_twin_manifest.json in export zip; audit logged; scratch pruned |
| **14. Cloud & Benchmarks** | `TASK-081` - `TASK-087`, `TASK-097` | Terraform, EKS, Karpenter GPU, accuracy claims audit CI gate, ACCURACY_DISCLAIMER.md | Terraform, Karpenter, OpenTelemetry | CI rejects unqualified accuracy guarantees; benchmark_result.json committed |
| **15. Validation** | `TASK-096` | Ground-truth validator, machine-readable benchmark_result.json, per-metric pass/fail | PDAL, NumPy, pytest | RMSE computed vs reference; no fabricated values; CI gate on RMSE regression |

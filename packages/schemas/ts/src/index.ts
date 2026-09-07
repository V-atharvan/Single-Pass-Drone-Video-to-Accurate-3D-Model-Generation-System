/**
 * Auto-generated TypeScript definitions for Single-Pass 3D Reconstruction Platform.
 * Generated directly from authoritative Python Pydantic V2 schemas.
 * DO NOT EDIT DIRECTLY. Run 'pnpm build:types' to regenerate.
 */

// ============================================================================
// ENUMS
// ============================================================================

export enum JobState {
  QUEUED = 'QUEUED',
  VALIDATING = 'VALIDATING',
  EXTRACTING_FRAMES = 'EXTRACTING_FRAMES',
  ESTIMATING_POSE = 'ESTIMATING_POSE',
  ESTIMATING_DEPTH = 'ESTIMATING_DEPTH',
  SEGMENTING = 'SEGMENTING',
  FUSING = 'FUSING',
  RECONSTRUCTING = 'RECONSTRUCTING',
  TEXTURING = 'TEXTURING',
  GEOREFERENCING = 'GEOREFERENCING',
  QUALITY_CHECK = 'QUALITY_CHECK',
  GENERATING_TILES = 'GENERATING_TILES',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
}

export enum QualityPreset {
  LOW = 'LOW',
  BALANCED = 'BALANCED',
  HIGH = 'HIGH',
}

export enum QualityClassification {
  HIGH = 'HIGH',
  MODERATE = 'MODERATE',
  LOW = 'LOW',
}

export enum ObservationState {
  OBSERVED = 'OBSERVED',
  PARTIALLY_OBSERVED = 'PARTIALLY_OBSERVED',
  AI_INFERRED = 'AI_INFERRED',
  UNKNOWN = 'UNKNOWN',
}

export enum ConfidenceLevel {
  HIGH = 'HIGH',
  MEDIUM = 'MEDIUM',
  LOW = 'LOW',
}

export enum MeasurementType {
  DISTANCE = 'DISTANCE',
  HEIGHT = 'HEIGHT',
  AREA = 'AREA',
  VOLUME = 'VOLUME',
}

export enum ExportFormat {
  GLB = 'GLB',
  OBJ = 'OBJ',
  PLY = 'PLY',
  LAS = 'LAS',
  LAZ = 'LAZ',
  TILES_3D = 'TILES_3D',
  DEM_TIF = 'DEM_TIF',
  DSM_TIF = 'DSM_TIF',
}

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Raw and interpolated GPS coordinate records.
 */
export interface GPSRecord {
  /** Latitude in decimal degrees (WGS84) */
  latitude: number;
  /** Longitude in decimal degrees (WGS84) */
  longitude: number;
  /** Altitude above Mean Sea Level in meters */
  altitude_msl: number;
  /** Takeoff-relative altitude in meters */
  altitude_relative?: number | null;
  /** Ground speed in meters per second */
  speed_mps?: number | null;
  /** Compass heading in degrees [0, 360) */
  heading_deg?: number | null;
  /** Elapsed seconds from video start */
  timestamp_offset_seconds: number;
  /** Horizontal Dilution of Precision */
  hdop?: number | null;
  /** Vertical Dilution of Precision */
  vdop?: number | null;
  /** Locked satellite count */
  num_satellites?: number | null;
}

/**
 * Unit quaternion representing 3D spatial rotation.
 */
export interface Quaternion {
  /** Scalar component */
  w: number;
  /** Vector component x */
  x: number;
  /** Vector component y */
  y: number;
  /** Vector component z */
  z: number;
}

/**
 * Inertial Measurement Unit telemetry record.
 */
export interface IMURecord {
  /** Roll in degrees */
  roll_deg: number;
  /** Pitch in degrees */
  pitch_deg: number;
  /** Yaw in degrees */
  yaw_deg: number;
  /** Orientation quaternion */
  quaternion?: Quaternion | null;
  /** Elapsed seconds from video start */
  timestamp_offset_seconds: number;
}

/**
 * Pinhole camera intrinsic calibration model with Brown-Conrady radial distortion.
 */
export interface CameraIntrinsics {
  /** Focal length x in pixels */
  fx: number;
  /** Focal length y in pixels */
  fy: number;
  /** Principal point x in pixels */
  cx: number;
  /** Principal point y in pixels */
  cy: number;
  /** Image sensor width in pixels */
  width: number;
  /** Image sensor height in pixels */
  height: number;
  /** First radial distortion coefficient */
  k1?: number;
  /** Second radial distortion coefficient */
  k2?: number;
  /** First tangential distortion coefficient */
  p1?: number;
  /** Second tangential distortion coefficient */
  p2?: number;
  /** Camera/sensor model identifier */
  sensor_name?: string | null;
}

/**
 * Optimized 6-Degrees-of-Freedom camera extrinsic pose.
 */
export interface CameraPose {
  /** Extracted keyframe sequential index */
  frame_index: number;
  /** Video timestamp offset */
  timestamp_offset_seconds: number;
  /** 3x3 orthonormal rotation matrix from camera to world frame */
  rotation_matrix: number[][];
  /** 3-element translation vector [x, y, z] in world coordinates (meters) */
  translation_vector: number[];
  /** Feature reprojection error RMSE in pixels */
  reprojection_error_px: number;
  /** Pose certainty score (0-100%) */
  pose_confidence: number;
}

/**
 * Metadata for an extracted video keyframe.
 */
export interface KeyframeMetadata {
  frame_index: number;
  source_video_timestamp_sec: number;
  keyframe_s3_key: string;
  /** Laplacian sharpness metric (higher = sharper) */
  blur_score: number;
  /** Luminance balance metric */
  exposure_score: number;
  gps?: GPSRecord | null;
  pose?: CameraPose | null;
  /** Whether selected for 3D reconstruction pipeline */
  is_selected?: number;
}

/**
 * Real-time GPU worker performance snapshot.
 */
export interface GPUStatistics {
  gpu_utilization_pct?: number;
  vram_used_mb?: number;
  vram_total_mb?: number;
  temperature_celsius?: number | null;
}

/**
 * Real-time progress telemetry streamed over WebSocket to frontend clients.
 */
export interface JobProgressUpdate {
  job_id: string;
  state: JobState;
  /** Human-readable active stage description */
  stage: string;
  /** Overall job progress percentage (0-100) */
  progress_percent: number;
  frames_processed?: number;
  frames_total?: number;
  gpu_stats?: GPUStatistics | null;
  estimated_seconds_remaining?: number | null;
  warnings?: string[];
  timestamp?: string;
}

/**
 * Payload for submitting a new reconstruction job.
 */
export interface ReconstructionJobCreate {
  flight_id: string;
  quality_preset?: QualityPreset;
  /** Target Coordinate Reference System (e.g. EPSG:4326 or UTM) */
  target_crs?: string;
  /** Detect and mask moving vehicles/pedestrians */
  enable_dynamic_removal?: number;
  /** Classify scene elements (buildings, terrain, roads) */
  enable_semantics?: number;
}

/**
 * API response model for a reconstruction job.
 */
export interface ReconstructionJobResponse {
  id: string;
  flight_id: string;
  project_id: string;
  org_id: string;
  status: JobState;
  quality_preset: QualityPreset;
  current_stage: string;
  progress_percent?: number;
  frames_processed?: number;
  frames_total?: number;
  error_message?: string | null;
  warnings?: string[];
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

/**
 * Computer-vision visual quality indicators.
 */
export interface VisualQualityMetrics {
  /** Laplacian variance sharpness metric (0-100) */
  blur_score: number;
  /** Luminance balance and exposure metric (0-100) */
  exposure_score: number;
  /** Spatial gradient detail density (0-100) */
  texture_score: number;
  /** Compression artifact absence score (0-100) */
  compression_score?: number;
}

/**
 * Flight path continuity and baseline overlap indicators.
 */
export interface TrajectoryQualityMetrics {
  /** GPS gap and dropout absence score (0-100) */
  gps_continuity_score: number;
  /** Velocity vector smoothness score (0-100) */
  speed_variance_score: number;
  /** Inter-frame baseline overlap score (0-100) */
  baseline_overlap_score: number;
}

/**
 * Comprehensive pre-flight data validation report matching PRD FR-005 and Design Doc Section 12.
 */
export interface InputQualityScore {
  /** Weighted aggregate quality score (0-100) */
  overall_score: number;
  classification: QualityClassification;
  /** Video quality score (0-100%) */
  video_quality_percent: number;
  /** GPS quality score (0-100%) */
  gps_quality_percent: number;
  /** Sharpness score (0-100%) */
  motion_blur_percent: number;
  /** Texture density score (0-100%) */
  scene_texture_percent: number;
  /** Whether camera parameters were detected */
  camera_metadata_complete?: number;
  expected_quality: QualityClassification;
  visual_metrics: VisualQualityMetrics;
  trajectory_metrics: TrajectoryQualityMetrics;
  /** Actionable pre-flight warnings */
  warnings?: string[];
  /** Whether dataset meets minimum threshold for processing */
  is_reconstructible: number;
}

/**
 * Quantitative reconstruction accuracy report matching PRD Section 39.
 */
export interface AccuracyReport {
  /** Overall model score (0-100) */
  overall_quality_score: number;
  /** Horizontal RMSE in meters (PRD target <= 0.5m) */
  horizontal_rmse_meters: number;
  /** Vertical RMSE in meters (PRD target <= 0.75m) */
  vertical_rmse_meters: number;
  /** Observable surface coverage % (PRD target >= 90%) */
  coverage_percent: number;
  /** Moving-object residual % (PRD target < 5%) */
  dynamic_contamination_percent: number;
  observed_surface_percent: number;
  partially_observed_percent: number;
  ai_inferred_percent: number;
  pose_confidence?: ConfidenceLevel;
  depth_confidence?: ConfidenceLevel;
  geolocation_confidence?: ConfidenceLevel;
  warnings?: string[];
  reference_data_used?: string[];
}

/**
 * User-created geospatial measurement.
 */
export interface MeasurementResult {
  id: string;
  model_id: string;
  user_id: string;
  type: MeasurementType;
  /** Measured quantity value */
  value: number;
  /** Measurement unit (e.g. 'm', 'm²', 'm³') */
  unit: string;
  /** Uncertainty margin (±value) */
  estimated_error_margin: number;
  /** GeoJSON geometry of points/polygon */
  points_geojson: Record<string, any>;
  created_at?: string;
}

/**
 * Registered 3D file asset associated with a model.
 */
export interface ModelAsset {
  id: string;
  model_id: string;
  asset_type: ExportFormat;
  s3_key: string;
  file_size_bytes: number;
  lod_level?: number | null;
}

/**
 * Geographic bounding box in decimal degrees and elevation.
 */
export interface BoundingBox {
  min_latitude: number;
  max_latitude: number;
  min_longitude: number;
  max_longitude: number;
  min_altitude: number;
  max_altitude: number;
}

/**
 * Authoritative metadata record for a reconstructed 3D scene.
 */
export interface ModelMetadata {
  id: string;
  project_id: string;
  job_id: string;
  name: string;
  crs?: string;
  bbox: BoundingBox;
  center_latitude: number;
  center_longitude: number;
  accuracy_report: AccuracyReport;
  assets?: ModelAsset[];
  created_at?: string;
}

/**
 * Verification test to confirm @single-pass-3d/schemas types are correctly imported and typed.
 */

import {
  JobState,
  QualityPreset,
  QualityClassification,
  ObservationState,
  ConfidenceLevel,
  MeasurementType,
  ExportFormat,
  GPSRecord,
  CameraIntrinsics,
  CameraPose,
  JobProgressUpdate,
  AccuracyReport,
  MeasurementResult,
  ModelMetadata,
} from '@single-pass-3d/schemas';

// Verify enums
const state: JobState = JobState.ESTIMATING_DEPTH;
const preset: QualityPreset = QualityPreset.BALANCED;
const classification: QualityClassification = QualityClassification.HIGH;
const obsState: ObservationState = ObservationState.OBSERVED;
const confLevel: ConfidenceLevel = ConfidenceLevel.HIGH;
const measureType: MeasurementType = MeasurementType.HEIGHT;
const exportFormat: ExportFormat = ExportFormat.GLB;

// Verify interfaces
const sampleGPS: GPSRecord = {
  latitude: 37.7749,
  longitude: -122.4194,
  altitude_msl: 120.4,
  timestamp_offset_seconds: 1.5,
};

const sampleIntrinsics: CameraIntrinsics = {
  fx: 2850.5,
  fy: 2850.5,
  cx: 1920.0,
  cy: 1080.0,
  width: 3840,
  height: 2160,
};

const sampleAccuracy: AccuracyReport = {
  overall_quality_score: 87.0,
  horizontal_rmse_meters: 0.32,
  vertical_rmse_meters: 0.58,
  coverage_percent: 93.0,
  dynamic_contamination_percent: 2.1,
  observed_surface_percent: 84.0,
  partially_observed_percent: 10.0,
  ai_inferred_percent: 6.0,
  pose_confidence: ConfidenceLevel.HIGH,
  depth_confidence: ConfidenceLevel.HIGH,
  geolocation_confidence: ConfidenceLevel.MEDIUM,
  warnings: [],
  reference_data_used: [],
};

const sampleProgress: JobProgressUpdate = {
  job_id: '123e4567-e89b-12d3-a456-426614174000',
  state: JobState.COMPLETED,
  stage: 'Processing completed',
  progress_percent: 100.0,
  frames_processed: 1200,
  frames_total: 1200,
  warnings: [],
  timestamp: new Date().toISOString(),
};

console.log('TypeScript type validation successful:', {
  state,
  preset,
  sampleGPS,
  sampleIntrinsics,
  sampleAccuracy,
  sampleProgress,
});

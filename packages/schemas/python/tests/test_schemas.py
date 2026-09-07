"""Test suite for single_pass_schemas data contracts."""

import uuid
import pytest
from pydantic import ValidationError

from single_pass_schemas import (
    GPSRecord,
    Quaternion,
    IMURecord,
    CameraIntrinsics,
    CameraPose,
    KeyframeMetadata,
    JobState,
    QualityPreset,
    GPUStatistics,
    JobProgressUpdate,
    ReconstructionJobCreate,
    ReconstructionJobResponse,
    QualityClassification,
    VisualQualityMetrics,
    TrajectoryQualityMetrics,
    InputQualityScore,
    ObservationState,
    ConfidenceLevel,
    AccuracyReport,
    MeasurementType,
    MeasurementResult,
    ExportFormat,
    ModelAsset,
    BoundingBox,
    ModelMetadata,
)


def test_gps_record_valid():
    gps = GPSRecord(
        latitude=37.7749,
        longitude=-122.4194,
        altitude_msl=150.5,
        altitude_relative=45.2,
        speed_mps=12.5,
        heading_deg=185.0,
        timestamp_offset_seconds=14.2,
        hdop=0.8,
        vdop=1.1,
        num_satellites=18,
    )
    assert gps.latitude == 37.7749
    assert gps.longitude == -122.4194


def test_gps_record_bounds_violation():
    # Latitude > 90
    with pytest.raises(ValidationError):
        GPSRecord(
            latitude=95.0,
            longitude=0.0,
            altitude_msl=100.0,
            timestamp_offset_seconds=0.0,
        )

    # Longitude < -180
    with pytest.raises(ValidationError):
        GPSRecord(
            latitude=0.0,
            longitude=-185.0,
            altitude_msl=100.0,
            timestamp_offset_seconds=0.0,
        )


def test_camera_intrinsics_validation():
    intrinsics = CameraIntrinsics(
        fx=2850.5,
        fy=2850.5,
        cx=1920.0,
        cy=1080.0,
        width=3840,
        height=2160,
        k1=-0.05,
        k2=0.01,
        sensor_name="Sony IMX586",
    )
    assert intrinsics.width == 3840
    assert intrinsics.height == 2160


def test_camera_pose_dimension_checks():
    valid_rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    valid_translation = [10.0, 20.0, 30.0]

    pose = CameraPose(
        frame_index=1,
        timestamp_offset_seconds=0.5,
        rotation_matrix=valid_rotation,
        translation_vector=valid_translation,
        reprojection_error_px=0.45,
        pose_confidence=95.5,
    )
    assert pose.reprojection_error_px == 0.45

    # Invalid rotation matrix dimensions (2x3 instead of 3x3)
    with pytest.raises(ValidationError):
        CameraPose(
            frame_index=1,
            timestamp_offset_seconds=0.5,
            rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            translation_vector=valid_translation,
            reprojection_error_px=0.45,
            pose_confidence=95.5,
        )


def test_job_state_enum_has_all_15_states():
    expected_states = {
        "QUEUED",
        "VALIDATING",
        "EXTRACTING_FRAMES",
        "ESTIMATING_POSE",
        "ESTIMATING_DEPTH",
        "SEGMENTING",
        "FUSING",
        "RECONSTRUCTING",
        "TEXTURING",
        "GEOREFERENCING",
        "QUALITY_CHECK",
        "GENERATING_TILES",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }
    actual_states = {state.value for state in JobState}
    assert expected_states == actual_states


def test_job_progress_update_serialization():
    job_id = uuid.uuid4()
    progress = JobProgressUpdate(
        job_id=job_id,
        state=JobState.ESTIMATING_DEPTH,
        stage="Depth estimation",
        progress_percent=72.0,
        frames_processed=8421,
        frames_total=11672,
        gpu_stats=GPUStatistics(
            gpu_utilization_pct=91.0,
            vram_used_mb=12450.0,
            vram_total_mb=16384.0,
        ),
        estimated_seconds_remaining=192.0,
        warnings=["Low texture detected on roof plane"],
    )

    json_str = progress.model_dump_json()
    deserialized = JobProgressUpdate.model_validate_json(json_str)
    assert deserialized.job_id == job_id
    assert deserialized.state == JobState.ESTIMATING_DEPTH
    assert deserialized.frames_processed == 8421


def test_input_quality_score():
    score = InputQualityScore(
        overall_score=82.0,
        classification=QualityClassification.HIGH,
        video_quality_percent=91.0,
        gps_quality_percent=74.0,
        motion_blur_percent=88.0,
        scene_texture_percent=94.0,
        camera_metadata_complete=True,
        expected_quality=QualityClassification.HIGH,
        visual_metrics=VisualQualityMetrics(
            blur_score=88.0,
            exposure_score=92.0,
            texture_score=94.0,
            compression_score=90.0,
        ),
        trajectory_metrics=TrajectoryQualityMetrics(
            gps_continuity_score=85.0,
            speed_variance_score=88.0,
            baseline_overlap_score=90.0,
        ),
        warnings=["Slight GPS dilution in turns"],
        is_reconstructible=True,
    )
    assert score.overall_score == 82.0
    assert score.is_reconstructible is True


def test_accuracy_report_validation():
    report = AccuracyReport(
        overall_quality_score=87.0,
        horizontal_rmse_meters=0.32,
        vertical_rmse_meters=0.58,
        coverage_percent=93.0,
        dynamic_contamination_percent=2.1,
        observed_surface_percent=84.0,
        partially_observed_percent=10.0,
        ai_inferred_percent=6.0,
        pose_confidence=ConfidenceLevel.HIGH,
        depth_confidence=ConfidenceLevel.HIGH,
        geolocation_confidence=ConfidenceLevel.MEDIUM,
        warnings=["North-facing facade has limited observations"],
        reference_data_used=["RTK Base Station", "DJI Gimbal IMU"],
    )
    assert report.horizontal_rmse_meters == 0.32
    assert report.vertical_rmse_meters == 0.58

    # Percentage sum violation (>100%)
    with pytest.raises(ValidationError):
        AccuracyReport(
            overall_quality_score=87.0,
            horizontal_rmse_meters=0.32,
            vertical_rmse_meters=0.58,
            coverage_percent=93.0,
            dynamic_contamination_percent=2.1,
            observed_surface_percent=85.0,
            partially_observed_percent=20.0,
            ai_inferred_percent=10.0,  # 85 + 20 + 10 = 115%
        )


def test_model_metadata_and_measurement():
    model_id = uuid.uuid4()
    project_id = uuid.uuid4()
    job_id = uuid.uuid4()
    user_id = uuid.uuid4()

    model = ModelMetadata(
        id=model_id,
        project_id=project_id,
        job_id=job_id,
        name="Survey Zone Alpha",
        crs="EPSG:4326",
        bbox=BoundingBox(
            min_latitude=37.77,
            max_latitude=37.78,
            min_longitude=-122.42,
            max_longitude=-122.41,
            min_altitude=10.0,
            max_altitude=120.0,
        ),
        center_latitude=37.775,
        center_longitude=-122.415,
        accuracy_report=AccuracyReport(
            overall_quality_score=85.0,
            horizontal_rmse_meters=0.42,
            vertical_rmse_meters=0.65,
            coverage_percent=91.0,
            dynamic_contamination_percent=1.5,
            observed_surface_percent=80.0,
            partially_observed_percent=12.0,
            ai_inferred_percent=8.0,
        ),
        assets=[
            ModelAsset(
                id=uuid.uuid4(),
                model_id=model_id,
                asset_type=ExportFormat.GLB,
                s3_key=f"models/{model_id}/model.glb",
                file_size_bytes=45120890,
                lod_level=0,
            )
        ],
    )
    assert model.name == "Survey Zone Alpha"
    assert len(model.assets) == 1

    measurement = MeasurementResult(
        id=uuid.uuid4(),
        model_id=model_id,
        user_id=user_id,
        type=MeasurementType.HEIGHT,
        value=18.2,
        unit="m",
        estimated_error_margin=0.31,
        points_geojson={"type": "LineString", "coordinates": [[-122.415, 37.775, 10.0], [-122.415, 37.775, 28.2]]},
    )
    assert measurement.value == 18.2
    assert measurement.estimated_error_margin == 0.31


def test_json_schema_generation():
    # Every schema must successfully produce standard JSON Schema
    models = [
        GPSRecord,
        IMURecord,
        CameraIntrinsics,
        CameraPose,
        KeyframeMetadata,
        JobProgressUpdate,
        ReconstructionJobCreate,
        InputQualityScore,
        AccuracyReport,
        MeasurementResult,
        ModelMetadata,
    ]
    for model in models:
        schema = model.model_json_schema()
        assert "properties" in schema
        assert "title" in schema

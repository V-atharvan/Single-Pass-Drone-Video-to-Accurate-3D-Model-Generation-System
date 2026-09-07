"""
Unit tests for Trajectory Continuity & Overlap Quality Estimator – TASK-022.
"""
from __future__ import annotations

import pytest

from packages.schemas.python.single_pass_schemas.quality import TrajectoryQualityMetrics
from workers.preprocessing.telemetry_parser import ParsedTelemetryPoint
from workers.preprocessing.trajectory_evaluator import evaluate_trajectory_quality


def test_smooth_trajectory_evaluation():
    """Verify clean 1 Hz drone flight scores high across continuity, speed, and overlap."""
    records = []
    # 10 fixes at 1 Hz, moving north ~5 m/s (~0.000045 degrees lat per second)
    for i in range(10):
        records.append(
            ParsedTelemetryPoint(
                timestamp_offset=float(i),
                latitude=37.7749 + i * 0.000045,
                longitude=-122.4194,
                altitude_msl=150.0,
                altitude_relative=50.0,
                speed=5.0,
            )
        )

    report = evaluate_trajectory_quality(records)

    assert isinstance(report.metrics, TrajectoryQualityMetrics)
    assert report.metrics.gps_continuity_score == 100.0
    assert report.metrics.speed_variance_score > 90.0
    assert report.metrics.baseline_overlap_score > 80.0
    assert "GPS_GAP_DETECTED" not in "".join(report.warnings)
    assert report.max_time_gap_seconds == 1.0


def test_gps_gap_detected_warning_and_score_deduction():
    """Verify a 5-second GPS dropout flags GPS_GAP_DETECTED and deducts from continuity score."""
    records = [
        ParsedTelemetryPoint(
            timestamp_offset=0.0,
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=150.0,
            altitude_relative=50.0,
            speed=5.0,
        ),
        # 5.5-second temporal dropout!
        ParsedTelemetryPoint(
            timestamp_offset=5.5,
            latitude=37.7752,
            longitude=-122.4194,
            altitude_msl=150.0,
            altitude_relative=50.0,
            speed=5.0,
        ),
        ParsedTelemetryPoint(
            timestamp_offset=6.5,
            latitude=37.7753,
            longitude=-122.4194,
            altitude_msl=150.0,
            altitude_relative=50.0,
            speed=5.0,
        ),
    ]

    report = evaluate_trajectory_quality(records)

    assert any("GPS_GAP_DETECTED" in w for w in report.warnings)
    assert report.max_time_gap_seconds == 5.5
    assert report.metrics.gps_continuity_score < 70.0


def test_speed_spike_outlier_detection():
    """Verify speed jump > 30 m/s triggers SPEED_SPIKE_DETECTED warning and deducts from score."""
    records = [
        ParsedTelemetryPoint(
            timestamp_offset=0.0,
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=150.0,
            speed=5.0,
        ),
        # 1-second later drone jumps 60 meters (~0.00054 deg -> ~60 m/s!)
        ParsedTelemetryPoint(
            timestamp_offset=1.0,
            latitude=37.77544,
            longitude=-122.4194,
            altitude_msl=150.0,
            speed=60.0,
        ),
    ]

    report = evaluate_trajectory_quality(records)

    assert any("SPEED_SPIKE_DETECTED" in w for w in report.warnings)
    assert report.metrics.speed_variance_score < 80.0


def test_insufficient_fixes_rejection():
    """Verify trajectory with fewer than 2 fixes returns zeroed metrics and warning."""
    records = [
        ParsedTelemetryPoint(
            timestamp_offset=0.0,
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=150.0,
        )
    ]

    report = evaluate_trajectory_quality(records)

    assert report.metrics.gps_continuity_score == 0.0
    assert any("INSUFFICIENT_TELEMETRY_POINTS" in w for w in report.warnings)

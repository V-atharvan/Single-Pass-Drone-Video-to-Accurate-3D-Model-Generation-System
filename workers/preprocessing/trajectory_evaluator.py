"""
Flight Trajectory Continuity & Overlap Quality Estimator – TASK-022.
Analyzes velocity consistency, temporal GPS gaps, speed spikes,
baseline-to-height ratio (B/H), and visual overlap quality.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ensure packages are resolvable
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.schemas.python.single_pass_schemas.quality import TrajectoryQualityMetrics
from workers.preprocessing.telemetry_parser import ParsedTelemetryPoint

logger = logging.getLogger("preprocessing.trajectory_evaluator")

MAX_PERMISSIBLE_SPEED_MPS = 30.0
MAX_NORMAL_GAP_SECONDS = 2.0
EARTH_RADIUS_METERS = 6371000.0


@dataclass
class TrajectoryEvaluationReport:
    """Detailed evaluation report for trajectory continuity and multi-view overlap."""

    metrics: TrajectoryQualityMetrics
    max_speed_mps: float
    avg_speed_mps: float
    max_time_gap_seconds: float
    estimated_overlap_pct: float
    avg_baseline_meters: float
    avg_height_meters: float
    warnings: list[str] = field(default_factory=list)


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle horizontal distance in meters between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def evaluate_trajectory_quality(
    records: List[ParsedTelemetryPoint],
    camera_fov_deg: float = 84.0,
) -> TrajectoryEvaluationReport:
    """
    Evaluates trajectory continuity, velocity smoothness, and stereoscopic overlap.
    Flags temporal dropouts (>2.0s), sudden speed spikes (>30 m/s), and low parallax.
    """
    if len(records) < 2:
        return TrajectoryEvaluationReport(
            metrics=TrajectoryQualityMetrics(
                gps_continuity_score=0.0,
                speed_variance_score=0.0,
                baseline_overlap_score=0.0,
            ),
            max_speed_mps=0.0,
            avg_speed_mps=0.0,
            max_time_gap_seconds=999.0,
            estimated_overlap_pct=0.0,
            avg_baseline_meters=0.0,
            avg_height_meters=0.0,
            warnings=["INSUFFICIENT_TELEMETRY_POINTS - minimum 2 fixes required"],
        )

    warnings: list[str] = []
    time_gaps: list[float] = []
    speeds: list[float] = []
    baselines: list[float] = []
    heights: list[float] = []

    has_long_gap = False
    speed_spike_count = 0

    min_alt = min(r.altitude_msl for r in records)

    for i in range(1, len(records)):
        p1 = records[i - 1]
        p2 = records[i]

        dt = p2.timestamp_offset - p1.timestamp_offset
        if dt <= 0:
            continue

        time_gaps.append(dt)
        if dt > MAX_NORMAL_GAP_SECONDS:
            has_long_gap = True

        dist = calculate_haversine_distance(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
        baselines.append(dist)

        step_speed = dist / dt
        speeds.append(step_speed)
        if step_speed > MAX_PERMISSIBLE_SPEED_MPS:
            speed_spike_count += 1

        # Determine operating height above takeoff/ground
        if p2.altitude_relative is not None and p2.altitude_relative > 0:
            h = p2.altitude_relative
        elif p2.altitude_barometric_m is not None and p2.altitude_barometric_m > 0:
            h = p2.altitude_barometric_m
        else:
            h = max(10.0, p2.altitude_msl - min_alt)
        heights.append(h)

    max_gap = max(time_gaps) if time_gaps else 0.0
    max_speed = max(speeds) if speeds else 0.0
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    avg_baseline = float(np.mean(baselines)) if baselines else 0.0
    avg_height = float(np.mean(heights)) if heights else 50.0

    # 1. GPS Continuity Score [0..100]
    # Penalize temporal gaps (>2s)
    continuity_score = 100.0
    for dt in time_gaps:
        if dt > 5.0:
            continuity_score -= 35.0
        elif dt > MAX_NORMAL_GAP_SECONDS:
            continuity_score -= 15.0
    continuity_score = max(0.0, min(100.0, continuity_score))

    if has_long_gap:
        warnings.append("GPS_GAP_DETECTED - temporal dropout in flight path exceeds threshold")

    # 2. Speed Smoothness Score [0..100]
    speed_smoothness = 100.0
    if speeds:
        speed_std = float(np.std(speeds))
        # Large variance indicates erratic flight or GPS multipath jumps
        speed_smoothness = max(0.0, min(100.0, 100.0 - (speed_std * 3.0) - (speed_spike_count * 25.0)))
    if speed_spike_count > 0:
        warnings.append("SPEED_SPIKE_DETECTED - sudden velocity jump exceeds UAV flight dynamics limits")

    # 3. Baseline Overlap Estimation
    # Ground footprint width: W = 2 * H * tan(FOV/2)
    ground_footprint = 2.0 * avg_height * math.tan(math.radians(camera_fov_deg / 2.0))
    if ground_footprint > 0:
        overlap_pct = max(0.0, min(100.0, (1.0 - (avg_baseline / ground_footprint)) * 100.0))
    else:
        overlap_pct = 70.0

    # Photogrammetry baseline overlap score: optimal is 65% to 85%
    if 65.0 <= overlap_pct <= 88.0:
        overlap_score = 100.0
    elif overlap_pct > 88.0:
        # High overlap has minor parallax penalty but good reconstruction
        overlap_score = max(75.0, 100.0 - (overlap_pct - 88.0) * 2.0)
    else:
        # Low overlap < 60% severely impacts SfM feature matching
        overlap_score = max(0.0, overlap_pct * 1.5)

    if overlap_pct < 55.0:
        warnings.append("LOW_VIEW_OVERLAP - estimated visual overlap between successive views is under 55%")

    metrics = TrajectoryQualityMetrics(
        gps_continuity_score=round(continuity_score, 2),
        speed_variance_score=round(speed_smoothness, 2),
        baseline_overlap_score=round(overlap_score, 2),
    )

    return TrajectoryEvaluationReport(
        metrics=metrics,
        max_speed_mps=round(max_speed, 2),
        avg_speed_mps=round(avg_speed, 2),
        max_time_gap_seconds=round(max_gap, 2),
        estimated_overlap_pct=round(overlap_pct, 2),
        avg_baseline_meters=round(avg_baseline, 2),
        avg_height_meters=round(avg_height, 2),
        warnings=warnings,
    )

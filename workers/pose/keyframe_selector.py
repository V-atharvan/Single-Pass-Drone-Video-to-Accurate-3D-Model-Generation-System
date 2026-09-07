"""
Keyframe Selection Engine with Quality Preset Control – TASK-024.
Filters raw drone video frames into an optimal keyframe sequence for multi-view SfM.
Enforces distance/time thresholds according to LOW, BALANCED, and HIGH presets,
and performs adjacent sharp frame recovery to eliminate motion blur outliers.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import cv2
import numpy as np

# Ensure packages are resolvable
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.schemas.python.single_pass_schemas.jobs import QualityPreset
from workers.preprocessing.quality_evaluator import compute_blur_score
from workers.preprocessing.telemetry_parser import ParsedTelemetryPoint
from workers.preprocessing.trajectory_evaluator import calculate_haversine_distance

logger = logging.getLogger("pose.keyframe_selector")


@dataclass
class KeyframePresetThresholds:
    """Distance and temporal constraints per quality preset."""

    min_distance_meters: float
    max_time_gap_seconds: float
    min_blur_score: float


PRESET_CONFIGS: dict[str, KeyframePresetThresholds] = {
    QualityPreset.LOW.value: KeyframePresetThresholds(
        min_distance_meters=1.5,
        max_time_gap_seconds=1.5,
        min_blur_score=45.0,
    ),
    QualityPreset.BALANCED.value: KeyframePresetThresholds(
        min_distance_meters=0.8,
        max_time_gap_seconds=0.8,
        min_blur_score=55.0,
    ),
    QualityPreset.HIGH.value: KeyframePresetThresholds(
        min_distance_meters=0.4,
        max_time_gap_seconds=0.4,
        min_blur_score=65.0,
    ),
}


@dataclass
class SelectedKeyframe:
    """Metadata record for a chosen keyframe."""

    frame_index: int
    source_video_timestamp_sec: float
    blur_score: float
    distance_from_last_m: float
    time_from_last_sec: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_msl: Optional[float] = None


@dataclass
class KeyframeSelectionResult:
    """Aggregate result of the keyframe selection pipeline."""

    keyframes: list[SelectedKeyframe]
    total_raw_frames: int
    total_keyframes: int
    selection_ratio: float
    avg_keyframe_interval_sec: float
    blur_outliers_recovered: int


class KeyframeSelector:
    """
    Intelligent keyframe selection engine with preset control and motion blur recovery.
    """

    def __init__(
        self,
        preset: Union[QualityPreset, str] = QualityPreset.BALANCED,
        search_window_frames: int = 5,
        blur_evaluator: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        preset_key = preset.value if isinstance(preset, QualityPreset) else str(preset).upper()
        if preset_key not in PRESET_CONFIGS:
            logger.warning("Unrecognized preset '%s'. Defaulting to BALANCED.", preset)
            preset_key = QualityPreset.BALANCED.value

        self.preset_key: str = preset_key
        self.thresholds: KeyframePresetThresholds = PRESET_CONFIGS[preset_key]
        self.search_window_frames: int = search_window_frames
        self.blur_evaluator: Callable[[np.ndarray], float] = blur_evaluator or compute_blur_score

    def _get_frame_blur(
        self,
        cap: Optional[cv2.VideoCapture],
        frame_idx: int,
        precomputed_blur_map: Optional[dict[int, float]] = None,
    ) -> float:
        """Retrieves or calculates the sharpness metric for a specific frame index."""
        if precomputed_blur_map is not None and frame_idx in precomputed_blur_map:
            return precomputed_blur_map[frame_idx]

        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                score = self.blur_evaluator(gray)
                if precomputed_blur_map is not None:
                    precomputed_blur_map[frame_idx] = score
                return score

        return 75.0  # Default fallback if frame cannot be decoded

    def _find_best_sharp_neighbor(
        self,
        candidate_idx: int,
        total_frames: int,
        cap: Optional[cv2.VideoCapture],
        precomputed_blur_map: Optional[dict[int, float]] = None,
    ) -> tuple[int, float]:
        """
        Scans adjacent frames within [-search_window_frames, +search_window_frames]
        to identify the local maximum sharpness frame.
        """
        start_idx = max(0, candidate_idx - self.search_window_frames)
        end_idx = min(total_frames - 1, candidate_idx + self.search_window_frames)

        best_idx = candidate_idx
        best_score = -1.0

        for idx in range(start_idx, end_idx + 1):
            score = self._get_frame_blur(cap, idx, precomputed_blur_map)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score

    def select_keyframes(
        self,
        total_frames: int,
        fps: float = 30.0,
        video_path: Optional[Union[Path, str]] = None,
        telemetry_points: Optional[List[ParsedTelemetryPoint]] = None,
        precomputed_blur_map: Optional[dict[int, float]] = None,
    ) -> KeyframeSelectionResult:
        """
        Selects keyframes matching the configured quality preset thresholds.
        Ensures zero blur outliers via local window search.
        """
        if total_frames <= 0:
            return KeyframeSelectionResult([], 0, 0, 0.0, 0.0, 0)

        cap = None
        if video_path is not None:
            v_path = Path(video_path).resolve()
            if v_path.exists():
                cap = cv2.VideoCapture(str(v_path))

        blur_cache: dict[int, float] = dict(precomputed_blur_map or {})
        blur_outliers_recovered = 0
        selected: list[SelectedKeyframe] = []

        try:
            # 1. Anchor the first sharp frame near the start
            first_idx, first_blur = self._find_best_sharp_neighbor(0, total_frames, cap, blur_cache)
            p0_lat, p0_lon, p0_alt = None, None, None
            if telemetry_points and len(telemetry_points) > 0:
                p0 = telemetry_points[0]
                p0_lat, p0_lon, p0_alt = p0.latitude, p0.longitude, p0.altitude_msl

            selected.append(
                SelectedKeyframe(
                    frame_index=first_idx,
                    source_video_timestamp_sec=round(first_idx / fps, 3),
                    blur_score=first_blur,
                    distance_from_last_m=0.0,
                    time_from_last_sec=0.0,
                    latitude=p0_lat,
                    longitude=p0_lon,
                    altitude_msl=p0_alt,
                )
            )

            last_selected_idx = first_idx
            last_timestamp = first_idx / fps
            last_lat = p0_lat
            last_lon = p0_lon

            # Map telemetry by second/frame for rapid lookup if available
            telemetry_by_frame: dict[int, ParsedTelemetryPoint] = {}
            if telemetry_points:
                for pt in telemetry_points:
                    pt_frame = int(round(pt.timestamp_offset * fps))
                    if 0 <= pt_frame < total_frames:
                        telemetry_by_frame[pt_frame] = pt

            # 2. Iterate and evaluate candidate steps
            min_dist = self.thresholds.min_distance_meters
            max_time = self.thresholds.max_time_gap_seconds
            min_blur = self.thresholds.min_blur_score

            step_frame_stride = max(1, int(round(fps * 0.1)))  # Check every ~100ms
            current_frame = last_selected_idx + step_frame_stride

            while current_frame < total_frames:
                current_time = current_frame / fps
                dt = current_time - last_timestamp

                # Determine distance travelled from last selected keyframe
                dist_travelled = 0.0
                curr_pt = telemetry_by_frame.get(current_frame)
                has_gps_fix = curr_pt is not None and last_lat is not None and last_lon is not None
                if has_gps_fix:
                    dist_travelled = calculate_haversine_distance(
                        last_lat, last_lon, curr_pt.latitude, curr_pt.longitude
                    )

                # Check if threshold criteria met (distance or time)
                is_threshold_met = dt >= max_time or (has_gps_fix and dist_travelled >= min_dist)

                if is_threshold_met:
                    # Candidate frame reached: inspect sharpness
                    cand_blur = self._get_frame_blur(cap, current_frame, blur_cache)

                    chosen_idx = current_frame
                    chosen_blur = cand_blur

                    # If blurry, perform adjacent sharp frame recovery
                    if cand_blur < min_blur:
                        sharp_idx, sharp_blur = self._find_best_sharp_neighbor(
                            current_frame, total_frames, cap, blur_cache
                        )
                        # Only shift if neighbor is demonstrably sharper
                        if sharp_blur > cand_blur and sharp_idx > last_selected_idx:
                            chosen_idx = sharp_idx
                            chosen_blur = sharp_blur
                            blur_outliers_recovered += 1

                    # Record selected keyframe
                    c_pt = telemetry_by_frame.get(chosen_idx)
                    chosen_time = chosen_idx / fps
                    d_from_last = dist_travelled
                    if c_pt and last_lat is not None and last_lon is not None:
                        d_from_last = calculate_haversine_distance(
                            last_lat, last_lon, c_pt.latitude, c_pt.longitude
                        )

                    selected.append(
                        SelectedKeyframe(
                            frame_index=chosen_idx,
                            source_video_timestamp_sec=round(chosen_time, 3),
                            blur_score=chosen_blur,
                            distance_from_last_m=round(d_from_last, 2),
                            time_from_last_sec=round(chosen_time - last_timestamp, 3),
                            latitude=c_pt.latitude if c_pt else None,
                            longitude=c_pt.longitude if c_pt else None,
                            altitude_msl=c_pt.altitude_msl if c_pt else None,
                        )
                    )

                    last_selected_idx = chosen_idx
                    last_timestamp = chosen_time
                    if c_pt:
                        last_lat = c_pt.latitude
                        last_lon = c_pt.longitude

                    current_frame = last_selected_idx + step_frame_stride
                else:
                    current_frame += step_frame_stride

            # 3. Ensure last frame anchor if sufficiently separated
            if (total_frames - 1) - last_selected_idx > int(fps * 0.3):
                end_idx, end_blur = self._find_best_sharp_neighbor(
                    total_frames - 1, total_frames, cap, blur_cache
                )
                end_time = end_idx / fps
                e_pt = telemetry_by_frame.get(end_idx)
                selected.append(
                    SelectedKeyframe(
                        frame_index=end_idx,
                        source_video_timestamp_sec=round(end_time, 3),
                        blur_score=end_blur,
                        distance_from_last_m=0.5,
                        time_from_last_sec=round(end_time - last_timestamp, 3),
                        latitude=e_pt.latitude if e_pt else None,
                        longitude=e_pt.longitude if e_pt else None,
                        altitude_msl=e_pt.altitude_msl if e_pt else None,
                    )
                )

        finally:
            if cap is not None:
                cap.release()

        total_kf = len(selected)
        ratio = round((total_kf / total_frames) * 100.0, 2) if total_frames > 0 else 0.0
        avg_dt = (
            round(sum(k.time_from_last_sec for k in selected[1:]) / max(1, total_kf - 1), 3)
            if total_kf > 1
            else 0.0
        )

        return KeyframeSelectionResult(
            keyframes=selected,
            total_raw_frames=total_frames,
            total_keyframes=total_kf,
            selection_ratio=ratio,
            avg_keyframe_interval_sec=avg_dt,
            blur_outliers_recovered=blur_outliers_recovered,
        )

"""
Visual Quality Metric Evaluator – TASK-021.
Assesses frame sharpness/blur, exposure balance, spatial texture density,
shadow coverage, and sequence illumination stability.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

# Ensure packages are resolvable
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.schemas.python.single_pass_schemas.quality import VisualQualityMetrics

logger = logging.getLogger("preprocessing.quality_evaluator")


def compute_blur_score(gray_frame: np.ndarray) -> float:
    """
    Computes sharpness score using Laplacian variance.
    Returns normalized [0..100] score where >= 60 indicates sharp feature visibility.
    """
    lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
    lap_var = float(lap.var())
    score = min(100.0, max(0.0, (lap_var / 3.0)))
    return round(float(score), 2)


def compute_exposure_score(gray_frame: np.ndarray) -> float:
    """
    Evaluates dynamic range and exposure distribution.
    Penalizes underexposed (< 30) and overexposed/saturated (> 225) pixels.
    """
    under_ratio = float(np.mean(gray_frame < 30))
    over_ratio = float(np.mean(gray_frame > 225))
    total_clipped = under_ratio + over_ratio
    score = max(0.0, min(100.0, (1.0 - total_clipped * 1.6) * 100.0))
    return round(float(score), 2)


def compute_texture_score(gray_frame: np.ndarray) -> float:
    """
    Evaluates spatial gradient detail density using Sobel operators.
    Low scores indicate featureless surfaces (calm water, clouds, asphalt).
    """
    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_mag = float(np.mean(mag))
    score = min(100.0, max(0.0, (mean_mag / 25.0) * 100.0))
    return round(float(score), 2)


def compute_shadow_score(gray_frame: np.ndarray) -> Tuple[float, float]:
    """
    Detects deep shadow regions (< 35% scene median luminance).
    Returns (shadow_score [0..100], shadow_coverage_pct [0..100]).
    High shadow coverage results in low shadow score (< 40).
    """
    median_val = float(np.median(gray_frame))
    shadow_thresh = max(20.0, median_val * 0.35)
    shadow_pixels = float(np.sum(gray_frame < shadow_thresh))
    coverage_pct = round((shadow_pixels / gray_frame.size) * 100.0, 2)

    # Clean scene has < 8% shadows (score > 85)
    # Heavy shadows (e.g. > 35%) drop score below 40
    score = max(0.0, min(100.0, 100.0 - coverage_pct * 1.8))
    return round(float(score), 2), coverage_pct


def compute_illumination_score(sampled_gray_frames: List[np.ndarray]) -> float:
    """
    Measures frame-to-frame mean luminance standard deviation across the video sequence.
    Detects aggressive auto-exposure shifts or flickering.
    """
    if not sampled_gray_frames:
        return 100.0
    mean_luminances = [float(np.mean(f)) for f in sampled_gray_frames]
    std_lum = float(np.std(mean_luminances))
    # std_lum < 4.0 is very stable (> 85 score); std_lum > 18.0 is erratic (< 40 score)
    score = max(0.0, min(100.0, 100.0 - (std_lum * 3.5)))
    return round(float(score), 2)


def evaluate_visual_quality(
    frames: List[np.ndarray],
    compression_score: float = 100.0,
) -> VisualQualityMetrics:
    """
    Synthesizes frame quality indicators across sampled video frames:
    blur, exposure, texture, shadow, and illumination stability.
    """
    if not frames:
        raise ValueError("Cannot evaluate visual quality on an empty list of frames.")

    gray_frames: list[np.ndarray] = []
    blur_list: list[float] = []
    exposure_list: list[float] = []
    texture_list: list[float] = []
    shadow_scores: list[float] = []
    shadow_coverages: list[float] = []

    for f in frames:
        if len(f.shape) == 3 and f.shape[2] == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = f

        gray_frames.append(gray)
        blur_list.append(compute_blur_score(gray))
        exposure_list.append(compute_exposure_score(gray))
        texture_list.append(compute_texture_score(gray))

        s_score, s_cov = compute_shadow_score(gray)
        shadow_scores.append(s_score)
        shadow_coverages.append(s_cov)

    avg_blur = round(float(np.mean(blur_list)), 2)
    avg_exposure = round(float(np.mean(exposure_list)), 2)
    avg_texture = round(float(np.mean(texture_list)), 2)
    avg_shadow = round(float(np.mean(shadow_scores)), 2)
    avg_coverage = round(float(np.mean(shadow_coverages)), 2)
    illumination_score = compute_illumination_score(gray_frames)

    return VisualQualityMetrics(
        blur_score=avg_blur,
        exposure_score=avg_exposure,
        texture_score=avg_texture,
        compression_score=compression_score,
        shadow_score=avg_shadow,
        illumination_score=illumination_score,
        shadow_coverage_pct=avg_coverage,
    )

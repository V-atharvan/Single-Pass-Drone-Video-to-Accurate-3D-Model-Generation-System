"""
Unit tests for Visual Quality Metric Evaluator – TASK-021.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from packages.schemas.python.single_pass_schemas.quality import VisualQualityMetrics
from workers.preprocessing.quality_evaluator import (
    compute_blur_score,
    compute_exposure_score,
    compute_illumination_score,
    compute_shadow_score,
    compute_texture_score,
    evaluate_visual_quality,
)


def test_blur_score_sharp_vs_blurred():
    """Verify sharp frame produces high blur_score (>70) and blurred produces low (<40)."""
    # Create sharp grid pattern
    sharp = np.zeros((400, 400), dtype=np.uint8)
    for y in range(0, 400, 20):
        sharp[y : y + 2, :] = 255
    for x in range(0, 400, 20):
        sharp[:, x : x + 2] = 255

    blurred = cv2.GaussianBlur(sharp, (45, 45), 0)

    score_sharp = compute_blur_score(sharp)
    score_blurred = compute_blur_score(blurred)

    assert score_sharp > 70.0
    assert score_blurred < 40.0


def test_exposure_score_balanced_vs_clipped():
    """Verify balanced exposure scores high (>80) while dark and blown-out score low (<40)."""
    balanced = np.full((300, 300), 128, dtype=np.uint8)
    dark = np.full((300, 300), 15, dtype=np.uint8)  # Underexposed
    overexposed = np.full((300, 300), 245, dtype=np.uint8)  # Saturated

    assert compute_exposure_score(balanced) > 80.0
    assert compute_exposure_score(dark) < 40.0
    assert compute_exposure_score(overexposed) < 40.0


def test_texture_score_rich_vs_featureless():
    """Verify textured scene produces high texture_score (>70) and flat surface scores low (<10)."""
    # Rich texture: random checkerboard / noise pattern
    np.random.seed(42)
    textured = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    flat = np.full((300, 300), 120, dtype=np.uint8)

    assert compute_texture_score(textured) > 70.0
    assert compute_texture_score(flat) < 10.0


def test_shadow_score_clean_vs_heavy_shadows():
    """Verify clean scene produces high shadow_score (>85) and shadow-heavy produces low (<40)."""
    # Clean daytime scene: median ~140, minimal pixels < 40
    clean = np.full((400, 400), 140, dtype=np.uint8)
    # 5% light shadows
    clean[:50, :50] = 30
    clean_score, clean_cov = compute_shadow_score(clean)
    assert clean_score > 85.0
    assert clean_cov < 10.0

    # Heavy shadows: median ~120, 50% under deep shadow (< 25)
    shadow_heavy = np.full((400, 400), 130, dtype=np.uint8)
    shadow_heavy[:200, :] = 15  # 50% of the image is deep shadow
    heavy_score, heavy_cov = compute_shadow_score(shadow_heavy)
    assert heavy_score < 40.0
    assert heavy_cov >= 45.0


def test_illumination_stability_score():
    """Verify stable illumination scores high (>90) and flickering scores low (<50)."""
    # Stable sequence
    stable_seq = [np.full((100, 100), 120 + i, dtype=np.uint8) for i in range(5)]
    assert compute_illumination_score(stable_seq) > 90.0

    # Wild auto-exposure hunting sequence (jumps between 40 and 220)
    erratic_seq = [
        np.full((100, 100), 40, dtype=np.uint8),
        np.full((100, 100), 220, dtype=np.uint8),
        np.full((100, 100), 50, dtype=np.uint8),
        np.full((100, 100), 210, dtype=np.uint8),
    ]
    assert compute_illumination_score(erratic_seq) < 50.0


def test_evaluate_visual_quality_full_pipeline():
    """Verify full evaluation returns valid VisualQualityMetrics within [0..100] bounds."""
    frames = [
        np.full((200, 200, 3), 128, dtype=np.uint8),
        np.full((200, 200, 3), 132, dtype=np.uint8),
    ]
    metrics = evaluate_visual_quality(frames, compression_score=88.0)

    assert isinstance(metrics, VisualQualityMetrics)
    assert 0.0 <= metrics.blur_score <= 100.0
    assert 0.0 <= metrics.exposure_score <= 100.0
    assert 0.0 <= metrics.texture_score <= 100.0
    assert 0.0 <= metrics.shadow_score <= 100.0
    assert 0.0 <= metrics.illumination_score <= 100.0
    assert metrics.compression_score == 88.0

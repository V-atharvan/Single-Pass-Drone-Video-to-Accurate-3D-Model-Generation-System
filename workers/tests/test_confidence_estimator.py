"""
Unit tests for Depth Uncertainty and Confidence Field Estimation — TASK-032.

Covers:
  - Gradient consistency scoring (smooth surfaces vs sharp spikes).
  - Depth plausibility scoring (realistic range vs sky / noise floor).
  - Photometric reprojection consistency warping between adjacent frames.
  - Fused confidence map generation (0..255 uint8).
  - Sky and infinite-depth region penalization.
  - Multi-frame confidence estimation batch pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from workers.depth.confidence_estimator import (
    ConfidenceEstimationResult,
    ConfidenceEstimator,
    ConfidenceMap,
    _gradient_consistency_score,
    _plausibility_score,
)
from workers.depth.depth_estimator import DepthMap


@pytest.fixture
def sample_depth_map():
    """Generates a realistic 360x640 depth map with a ground plane, building, and sky."""
    h, w = 360, 640
    depth = np.full((h, w), 25.0, dtype=np.float32)  # Ground / facade
    # Sky region at top
    depth[:80, :] = 280.0  # Above plausibility max
    # Noise floor at bottom corner
    depth[340:, :50] = 0.1  # Below plausibility min
    mask = (depth >= 0.3) & (depth <= 250.0)

    return DepthMap(
        frame_index=1,
        depth_m=depth,
        valid_mask=mask,
        original_height=h,
        original_width=w,
        min_depth_m=0.1,
        max_depth_m=280.0,
        median_depth_m=25.0,
        inference_time_ms=12.5,
    )


# ---------------------------------------------------------------------------
# Individual Component Tests
# ---------------------------------------------------------------------------


def test_gradient_consistency_score_smooth_vs_noisy():
    """Smooth depth yields high confidence; noisy depth yields lower confidence."""
    smooth = np.full((100, 100), 10.0, dtype=np.float32)
    score_smooth = _gradient_consistency_score(smooth)
    assert np.mean(score_smooth) > 250  # High confidence on smooth surface

    # Add realistic step edges and noise
    noisy = smooth.copy()
    noisy[20:60, 20:60] += 20.0
    score_noisy = _gradient_consistency_score(noisy)
    assert np.mean(score_noisy) < np.mean(score_smooth)


def test_plausibility_score_penalizes_sky_and_floor():
    """Plausibility score penalizes sky and sensor noise floor."""
    depth = np.array([
        [15.0, 20.0],    # Normal outdoor depth
        [0.1, 280.0],    # Noise floor (0.1m) and sky (280m)
    ], dtype=np.float32)

    score = _plausibility_score(depth)
    assert score[0, 0] > score[1, 0]  # Normal > noise floor
    assert score[0, 1] > score[1, 1]  # Normal > sky


# ---------------------------------------------------------------------------
# Confidence Estimator Pipeline Tests
# ---------------------------------------------------------------------------


def test_confidence_estimator_single_and_batch(sample_depth_map):
    """Confidence estimator produces uint8 map matching depth map dimensions."""
    estimator = ConfidenceEstimator(use_reprojection=False)
    result = estimator.estimate([sample_depth_map])

    assert isinstance(result, ConfidenceEstimationResult)
    assert result.total_keyframes == 1
    assert len(result.confidence_maps) == 1

    cm = result.confidence_maps[0]
    assert isinstance(cm, ConfidenceMap)
    assert cm.frame_index == 1
    assert cm.confidence.shape == (360, 640)
    assert cm.confidence.dtype == np.uint8
    assert 0 <= cm.mean_confidence <= 255

    # Sky pixels (top 80 rows) must have significantly lower confidence than ground plane (middle)
    sky_conf = np.mean(cm.confidence[:80, :])
    ground_conf = np.mean(cm.confidence[100:200, :])
    assert ground_conf > sky_conf


def test_confidence_estimator_multiple_frames(sample_depth_map):
    """Estimates confidence over multiple depth maps."""
    dm2 = DepthMap(
        frame_index=2,
        depth_m=sample_depth_map.depth_m.copy(),
        valid_mask=sample_depth_map.valid_mask.copy(),
        original_height=360,
        original_width=640,
        min_depth_m=0.1,
        max_depth_m=280.0,
        median_depth_m=25.0,
        inference_time_ms=10.0,
    )
    estimator = ConfidenceEstimator()
    result = estimator.estimate([sample_depth_map, dm2])

    assert len(result.confidence_maps) == 2
    assert result.confidence_maps[0].frame_index == 1
    assert result.confidence_maps[1].frame_index == 2

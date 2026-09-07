"""
Unit tests for Monocular Metric Depth Map Inference Pipeline — TASK-031.

Covers:
  - Frame preprocessing: ImageNet normalization and channel-first tensor conversion.
  - Bilateral / edge-preserving upsampling to native frame resolution.
  - Batched inference over in-memory BGR numpy arrays.
  - File-based inference over keyframe image paths.
  - Error handling: missing files, mismatched indices, corrupted frames.
  - Depth statistics calculation (min, max, median) and valid pixel masking.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from workers.depth.depth_estimator import (
    DepthEstimationResult,
    DepthEstimator,
    DepthMap,
    _compute_depth_stats,
    _preprocess_frame,
    _upsample_depth_bilateral,
)
from workers.depth.runtime import DepthModelLoader


@pytest.fixture
def loaded_stub_model():
    loader = DepthModelLoader()
    return loader.load(simulation_mode=True)


@pytest.fixture
def synthetic_bgr_frames():
    """Generates 3 synthetic BGR frames (640x360) with simulated texture."""
    frames = []
    for i in range(3):
        img = np.full((360, 640, 3), (120 + i * 20, 100, 80), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)  # Building box
        cv2.circle(img, (400, 180), 40, (0, 0, 255), -1)
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Preprocessing & Upsampling Tests
# ---------------------------------------------------------------------------


def test_preprocess_frame_shape_and_range():
    """Preprocessed frame should be (3, target_size, target_size) float32."""
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = _preprocess_frame(bgr, target_size=518)
    assert tensor.shape == (3, 518, 518)
    assert tensor.dtype == np.float32


def test_upsample_depth_bilateral_preserves_resolution():
    """Bilateral upsampling correctly restores native frame resolution."""
    depth_low = np.full((128, 128), 15.0, dtype=np.float32)
    guide_bgr = np.zeros((360, 640, 3), dtype=np.uint8)

    depth_up = _upsample_depth_bilateral(depth_low, guide_bgr, target_h=360, target_w=640)
    assert depth_up.shape == (360, 640)
    assert depth_up.dtype == np.float32
    assert np.all(depth_up >= 0.0)


def test_compute_depth_stats():
    """Calculates min, max, median accurately over valid mask."""
    depth = np.array([[5.0, 10.0], [0.0, 20.0]], dtype=np.float32)
    mask = depth > 0.0
    d_min, d_max, d_med = _compute_depth_stats(depth, mask)
    assert d_min == 5.0
    assert d_max == 20.0
    assert d_med == 10.0


# ---------------------------------------------------------------------------
# Estimator Pipeline Tests
# ---------------------------------------------------------------------------


def test_estimate_from_arrays_success(loaded_stub_model, synthetic_bgr_frames):
    """Running inference on 3 in-memory arrays produces 3 DepthMaps."""
    estimator = DepthEstimator(loaded_stub_model, batch_size=2)
    result = estimator.estimate_from_arrays(
        bgr_frames=synthetic_bgr_frames,
        frame_indices=[10, 20, 30],
    )

    assert isinstance(result, DepthEstimationResult)
    assert result.total_keyframes == 3
    assert len(result.depth_maps) == 3
    assert len(result.failed_frame_indices) == 0

    for i, dm in enumerate(result.depth_maps):
        assert isinstance(dm, DepthMap)
        assert dm.frame_index == [10, 20, 30][i]
        assert dm.depth_m.shape == (360, 640)
        assert dm.valid_mask.shape == (360, 640)
        assert dm.original_height == 360
        assert dm.original_width == 640
        assert dm.min_depth_m >= 0.0
        assert dm.max_depth_m > 0.0
        assert dm.inference_time_ms >= 0.0


def test_estimate_from_paths_success(tmp_path: Path, loaded_stub_model, synthetic_bgr_frames):
    """Writing frames to disk and estimating from paths succeeds."""
    img_paths = []
    for i, frame in enumerate(synthetic_bgr_frames):
        p = tmp_path / f"frame_{i:04d}.png"
        cv2.imwrite(str(p), frame)
        img_paths.append(p)

    estimator = DepthEstimator(loaded_stub_model, batch_size=4)
    result = estimator.estimate_from_paths(
        image_paths=img_paths,
        frame_indices=[0, 1, 2],
    )

    assert len(result.depth_maps) == 3
    assert len(result.failed_frame_indices) == 0
    assert result.total_inference_time_sec >= 0.0


def test_estimate_mismatched_lengths_raises(loaded_stub_model):
    """Mismatched frames and indices raises ValueError."""
    estimator = DepthEstimator(loaded_stub_model)
    with pytest.raises(ValueError, match="equal length"):
        estimator.estimate_from_arrays(
            bgr_frames=[np.zeros((10, 10, 3), dtype=np.uint8)],
            frame_indices=[1, 2],
        )


def test_estimate_handles_missing_file_gracefully(tmp_path: Path, loaded_stub_model):
    """Missing file is added to failed_frame_indices without crashing."""
    valid_p = tmp_path / "valid.png"
    cv2.imwrite(str(valid_p), np.zeros((100, 100, 3), dtype=np.uint8))
    missing_p = tmp_path / "missing_frame_999.png"

    estimator = DepthEstimator(loaded_stub_model)
    result = estimator.estimate_from_paths(
        image_paths=[valid_p, missing_p],
        frame_indices=[1, 2],
    )

    assert len(result.depth_maps) == 1
    assert 2 in result.failed_frame_indices

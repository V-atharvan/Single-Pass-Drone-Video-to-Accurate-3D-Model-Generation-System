"""
Unit tests for Visual Odometry and Feature Matching — TASK-026.

Definition of Done: Match 10 sequential drone keyframes, identifying >= 500 robust
inlier correspondences per frame pair (on controlled synthetic imagery).

Coverage:
  - SIFT feature extraction from disk images.
  - SIFT extraction from in-memory arrays.
  - FLANN match with Lowe ratio test.
  - RANSAC inlier filtering and fundamental matrix estimation.
  - FeatureTracker sequential + loop-closure orchestration.
  - Degenerate cases: empty frames, insufficient keypoints, missing files.
  - Definition-of-Done: >= 500 inliers per pair on high-texture sequential images.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from workers.pose.feature_tracker import (
    FeatureTracker,
    FLANNFeatureMatcher,
    FeatureTrackingResult,
    FramePairMatches,
    KeyframeFeatures,
    SIFTFeatureExtractor,
    _MIN_INLIER_CORRESPONDENCES,
)


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------


def _make_high_texture_frame(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """
    Generate a high-frequency noise image that SIFT can reliably detect features in.
    Two consecutive frames are rendered with a sub-pixel homography offset (pure translation)
    to produce a known, reproducible match pattern.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    # Draw a grid of high-contrast checkerboard patches for extra texture
    block = 16
    for by in range(0, height, block):
        for bx in range(0, width, block):
            if (by // block + bx // block) % 2 == 0:
                base[by:by+block, bx:bx+block] = np.clip(
                    base[by:by+block, bx:bx+block].astype(np.int16) + 80, 0, 255
                ).astype(np.uint8)
    return base


def _make_translated_frame(source: np.ndarray, tx: float = 5.0, ty: float = 3.0) -> np.ndarray:
    """Translate a frame by (tx, ty) pixels to simulate camera motion."""
    h, w = source.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(source, M, (w, h))


def _write_frames_to_disk(frames: List[np.ndarray], out_dir: Path) -> List[Path]:
    """Write BGR frames as PNG files and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        p = out_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# SIFTFeatureExtractor tests
# ---------------------------------------------------------------------------


def test_sift_extract_from_array_returns_features():
    """SIFT must detect features on a high-texture in-memory frame."""
    extractor = SIFTFeatureExtractor(n_features=2000)
    frame = _make_high_texture_frame()
    result = extractor.extract_from_array(frame, frame_index=42)

    assert result is not None
    assert result.frame_index == 42
    assert result.num_keypoints > 50  # Expect many features on a noise+checkerboard image
    assert result.descriptors.dtype == np.float32
    assert result.descriptors.shape[1] == 128  # SIFT descriptor dimensionality


def test_sift_extract_from_disk(tmp_path: Path):
    """SIFT must successfully load and extract features from a PNG on disk."""
    extractor = SIFTFeatureExtractor()
    frame = _make_high_texture_frame(seed=5)
    img_path = tmp_path / "test_frame.png"
    cv2.imwrite(str(img_path), frame)

    result = extractor.extract(image_path=img_path, frame_index=0)
    assert result is not None
    assert result.image_path == str(img_path)
    assert result.num_keypoints > 0


def test_sift_extract_missing_file_returns_none():
    """Extraction from a non-existent path must return None (no exception)."""
    extractor = SIFTFeatureExtractor()
    result = extractor.extract(image_path="/nonexistent/frame.png", frame_index=99)
    assert result is None


def test_sift_extract_blank_frame_returns_none_or_sparse():
    """A completely uniform frame may yield no SIFT features; must not raise."""
    extractor = SIFTFeatureExtractor()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)  # Uniform grey
    result = extractor.extract_from_array(blank, frame_index=0)
    # Either None (0 keypoints) or very few; must not raise an exception
    if result is not None:
        assert isinstance(result.num_keypoints, int)


# ---------------------------------------------------------------------------
# FLANNFeatureMatcher tests
# ---------------------------------------------------------------------------


def test_flann_matcher_produces_inliers_for_translated_pair():
    """
    Matching two slightly translated frames of the same scene must yield many inliers.
    The Lowe ratio test + RANSAC combination should keep a high inlier count.
    """
    extractor = SIFTFeatureExtractor(n_features=5000)
    matcher = FLANNFeatureMatcher()

    frame_a = _make_high_texture_frame(seed=1)
    frame_b = _make_translated_frame(frame_a, tx=8.0, ty=5.0)

    feat_a = extractor.extract_from_array(frame_a, frame_index=0)
    feat_b = extractor.extract_from_array(frame_b, frame_index=1)

    assert feat_a is not None and feat_b is not None

    pair_match = matcher.match(feat_a, feat_b)

    assert pair_match.is_valid, (
        f"Expected valid match; inlier_count={pair_match.inlier_count}, "
        f"raw={pair_match.raw_match_count}"
    )
    assert pair_match.inlier_count >= _MIN_INLIER_CORRESPONDENCES
    assert pair_match.pts_a.shape[1] == 2
    assert pair_match.pts_b.shape[1] == 2
    assert pair_match.fundamental_matrix is not None
    assert pair_match.fundamental_matrix.shape == (3, 3)


def test_flann_matcher_inlier_ratio_bounded():
    """Inlier ratio must be in [0.0, 1.0]."""
    extractor = SIFTFeatureExtractor()
    matcher = FLANNFeatureMatcher()
    frame_a = _make_high_texture_frame(seed=2)
    frame_b = _make_translated_frame(frame_a, tx=4.0, ty=2.0)
    feat_a = extractor.extract_from_array(frame_a, frame_index=0)
    feat_b = extractor.extract_from_array(frame_b, frame_index=1)
    if feat_a and feat_b:
        m = matcher.match(feat_a, feat_b)
        assert 0.0 <= m.inlier_ratio <= 1.0


def test_flann_matcher_unrelated_frames_may_fail():
    """
    Two completely unrelated noise images may produce few inliers.
    The is_valid flag must correctly reflect below-threshold cases.
    """
    extractor = SIFTFeatureExtractor()
    matcher = FLANNFeatureMatcher(min_inliers=_MIN_INLIER_CORRESPONDENCES)

    # Independent random seeds => largely unrelated content
    frame_a = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    frame_b = np.random.default_rng(999).integers(0, 256, (480, 640, 3), dtype=np.uint8)

    feat_a = extractor.extract_from_array(frame_a, frame_index=0)
    feat_b = extractor.extract_from_array(frame_b, frame_index=1)

    if feat_a is not None and feat_b is not None:
        m = matcher.match(feat_a, feat_b)
        # We only verify the match object is structurally correct; pass/fail is content-dependent
        assert isinstance(m.is_valid, bool)
        assert m.inlier_count >= 0


# ---------------------------------------------------------------------------
# FeatureTracker orchestration tests
# ---------------------------------------------------------------------------


def test_feature_tracker_sequential_pairs(tmp_path: Path):
    """
    FeatureTracker must process N keyframes and produce N-1 sequential pair matches.
    """
    n_frames = 5
    frame_a = _make_high_texture_frame(seed=10)
    frames = [_make_translated_frame(frame_a, tx=float(i * 3), ty=float(i * 2)) for i in range(n_frames)]
    paths = _write_frames_to_disk(frames, tmp_path / "kf")
    path_strs = [str(p) for p in paths]
    frame_indices = list(range(n_frames))

    tracker = FeatureTracker(loop_closure_window=0)
    result = tracker.track(image_paths=path_strs, frame_indices=frame_indices)

    assert result.total_keyframes == n_frames
    assert len(result.extracted_features) == n_frames
    # With loop_closure_window=0, only sequential pairs: n_frames - 1
    assert len(result.frame_pair_matches) == n_frames - 1

    for pair in result.frame_pair_matches:
        assert pair.frame_b_index > pair.frame_a_index


def test_feature_tracker_loop_closure_pairs(tmp_path: Path):
    """
    With loop_closure_window=2, tracker should produce additional non-adjacent pairs.
    """
    n_frames = 6
    frame_a = _make_high_texture_frame(seed=20)
    frames = [_make_translated_frame(frame_a, tx=float(i * 2), ty=0) for i in range(n_frames)]
    paths = _write_frames_to_disk(frames, tmp_path / "kf")
    path_strs = [str(p) for p in paths]

    tracker = FeatureTracker(loop_closure_window=2)
    result = tracker.track(image_paths=path_strs)

    # Sequential: 5 pairs. Loop-closure adds (i, i+2) and (i, i+3) for each i:
    # (0,2),(0,3), (1,3),(1,4), (2,4),(2,5), (3,5) = 7 extra = 12 total (some may be deduped)
    assert len(result.frame_pair_matches) > n_frames - 1, (
        f"Expected more than {n_frames - 1} pairs with loop closure; got {len(result.frame_pair_matches)}"
    )


def test_feature_tracker_missing_image_counted_as_failure(tmp_path: Path):
    """
    A path pointing to a non-existent file must result in failed_extraction_indices entry.
    """
    frame = _make_high_texture_frame(seed=30)
    good_path = tmp_path / "frame_00000.png"
    cv2.imwrite(str(good_path), frame)

    tracker = FeatureTracker()
    result = tracker.track(
        image_paths=[str(good_path), "/nonexistent/missing.png"],
        frame_indices=[0, 1],
    )

    assert 1 in result.failed_extraction_indices


def test_feature_tracker_mismatched_lengths_raises():
    """Mismatched image_paths and frame_indices lengths must raise ValueError."""
    tracker = FeatureTracker()
    with pytest.raises(ValueError, match="equal length"):
        tracker.track(image_paths=["a.png", "b.png"], frame_indices=[0])


# ---------------------------------------------------------------------------
# Definition of Done: >= 500 inlier correspondences on 10 sequential keyframes
# ---------------------------------------------------------------------------


def test_definition_of_done_500_inliers_per_pair(tmp_path: Path):
    """
    Definition of Done for TASK-026:
    Match 10 sequential drone keyframes and identify >= 500 robust RANSAC inlier
    correspondences per valid frame pair on controlled synthetic imagery.
    """
    n_frames = 10
    # High-texture base frame ensures many stable SIFT features
    base = _make_high_texture_frame(width=1280, height=720, seed=42)
    frames = [_make_translated_frame(base, tx=float(i * 6), ty=float(i * 3)) for i in range(n_frames)]
    paths = _write_frames_to_disk(frames, tmp_path / "kf_dod")
    path_strs = [str(p) for p in paths]

    tracker = FeatureTracker(
        extractor=SIFTFeatureExtractor(n_features=8000),
        loop_closure_window=0,
    )
    result = tracker.track(image_paths=path_strs)

    valid_pairs = [m for m in result.frame_pair_matches if m.is_valid]
    assert len(valid_pairs) >= n_frames - 1, (
        f"Expected all {n_frames - 1} sequential pairs to be valid; got {len(valid_pairs)}"
    )

    low_inlier_pairs = [(m.frame_a_index, m.frame_b_index, m.inlier_count) for m in valid_pairs if m.inlier_count < 500]
    assert not low_inlier_pairs, (
        f"Some pairs have fewer than 500 inliers: {low_inlier_pairs}"
    )

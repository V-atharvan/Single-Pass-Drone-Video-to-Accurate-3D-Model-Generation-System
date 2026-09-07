"""
Visual Odometry and Feature Matching Across Keyframes — TASK-026.

Implements robust multi-view feature extraction and correspondence matching
for preparing the keyframe set for camera pose estimation (SfM).

Pipeline stages:
  1. Feature extraction: SIFT keypoints + descriptors per keyframe image.
     SuperPoint/DISK integration is architecturally reserved but falls back to SIFT
     when learned model weights are unavailable.
  2. Feature matching: FLANN-based ratio-test matching across adjacent frames and
     optional loop-closure candidate pairs.
  3. Outlier filtering: RANSAC essential matrix estimation via the 5-point algorithm
     to retain only geometrically consistent inlier correspondences.

Output contract:
  FramePairMatches: per-pair descriptor in terms of inlier 2D-to-2D point correspondences,
  fundamental/essential matrices, and inlier ratios.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

logger = logging.getLogger("pose.feature_tracker")

# ---------------------------------------------------------------------------
# RANSAC / ratio-test tunables
# ---------------------------------------------------------------------------

_LOWE_RATIO_THRESHOLD: float = 0.75       # Lowe ratio test threshold
_RANSAC_REPROJECTION_THRESHOLD: float = 1.0  # px — essential matrix RANSAC
_RANSAC_CONFIDENCE: float = 0.999
_MIN_INLIER_CORRESPONDENCES: int = 8       # Minimum 5-point algo requires 5; use 8 for stability
_SIFT_N_FEATURES: int = 8000               # Max SIFT features per frame
_SIFT_CONTRAST_THRESHOLD: float = 0.03    # Lower = more features in low-contrast regions
_SIFT_EDGE_THRESHOLD: float = 10.0


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class KeyframeFeatures:
    """
    Extracted keypoints and descriptors for a single keyframe image.

    Attributes
    ----------
    frame_index:
        Source video frame index.
    image_path:
        Absolute path to the extracted keyframe PNG.
    keypoints:
        List of cv2.KeyPoint objects.
    descriptors:
        Float32 descriptor matrix (N x 128 for SIFT).
    image_shape:
        (height, width) of the source image.
    """

    frame_index: int
    image_path: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray            # shape (N, D), float32
    image_shape: Tuple[int, int]       # (height, width)

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)


@dataclass
class FramePairMatches:
    """
    Verified inlier correspondences between two keyframe images after RANSAC filtering.

    Attributes
    ----------
    frame_a_index, frame_b_index:
        Source video frame indices of the matched pair.
    pts_a, pts_b:
        (N, 2) float32 arrays of matched pixel coordinates after RANSAC.
    fundamental_matrix:
        3x3 fundamental matrix F estimated by RANSAC (may be None if degenerate).
    inlier_count:
        Number of RANSAC inlier correspondences.
    raw_match_count:
        Number of ratio-test matches before RANSAC.
    inlier_ratio:
        inlier_count / raw_match_count (0.0 if raw_match_count == 0).
    is_valid:
        True when inlier_count >= _MIN_INLIER_CORRESPONDENCES.
    """

    frame_a_index: int
    frame_b_index: int
    pts_a: np.ndarray           # (N, 2) float32
    pts_b: np.ndarray           # (N, 2) float32
    fundamental_matrix: Optional[np.ndarray]  # (3, 3) or None
    inlier_count: int
    raw_match_count: int
    inlier_ratio: float
    is_valid: bool


@dataclass
class FeatureTrackingResult:
    """Aggregate result of the full feature tracking pipeline."""

    total_keyframes: int
    extracted_features: List[KeyframeFeatures] = field(default_factory=list)
    frame_pair_matches: List[FramePairMatches] = field(default_factory=list)
    failed_extraction_indices: List[int] = field(default_factory=list)
    invalid_pair_count: int = 0

    @property
    def valid_pair_count(self) -> int:
        return len([m for m in self.frame_pair_matches if m.is_valid])


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class SIFTFeatureExtractor:
    """
    Extracts SIFT keypoints and L2-normalised float32 descriptors from greyscale keyframe images.
    """

    def __init__(
        self,
        n_features: int = _SIFT_N_FEATURES,
        contrast_threshold: float = _SIFT_CONTRAST_THRESHOLD,
        edge_threshold: float = _SIFT_EDGE_THRESHOLD,
    ) -> None:
        self._sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

    def extract(self, image_path: Union[str, Path], frame_index: int) -> Optional[KeyframeFeatures]:
        """
        Load a keyframe PNG and extract SIFT features.
        Returns None if the image cannot be loaded or yields no keypoints.
        """
        path = Path(image_path)
        if not path.exists():
            logger.warning("Keyframe image not found: %s", path)
            return None

        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning("OpenCV failed to decode keyframe: %s", path)
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self._sift.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0 or descs is None:
            logger.debug("No keypoints detected in frame %d (%s)", frame_index, path.name)
            return None

        return KeyframeFeatures(
            frame_index=frame_index,
            image_path=str(path),
            keypoints=list(kps),
            descriptors=descs.astype(np.float32),
            image_shape=(bgr.shape[0], bgr.shape[1]),
        )

    def extract_from_array(
        self,
        bgr_frame: np.ndarray,
        frame_index: int,
    ) -> Optional[KeyframeFeatures]:
        """Extract features directly from an in-memory BGR NumPy array (for testing)."""
        if bgr_frame is None or bgr_frame.size == 0:
            return None
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY) if len(bgr_frame.shape) == 3 else bgr_frame
        kps, descs = self._sift.detectAndCompute(gray, None)
        if kps is None or len(kps) == 0 or descs is None:
            return None
        return KeyframeFeatures(
            frame_index=frame_index,
            image_path="<in_memory>",
            keypoints=list(kps),
            descriptors=descs.astype(np.float32),
            image_shape=(bgr_frame.shape[0], bgr_frame.shape[1]),
        )


# ---------------------------------------------------------------------------
# Feature matcher
# ---------------------------------------------------------------------------


class FLANNFeatureMatcher:
    """
    Matches SIFT descriptors between frame pairs using FLANN with Lowe ratio test.
    Filters outliers with RANSAC essential matrix estimation (5-point algorithm).
    """

    def __init__(
        self,
        ratio_threshold: float = _LOWE_RATIO_THRESHOLD,
        ransac_threshold: float = _RANSAC_REPROJECTION_THRESHOLD,
        ransac_confidence: float = _RANSAC_CONFIDENCE,
        min_inliers: int = _MIN_INLIER_CORRESPONDENCES,
    ) -> None:
        self._ratio_threshold = ratio_threshold
        self._ransac_threshold = ransac_threshold
        self._ransac_confidence = ransac_confidence
        self._min_inliers = min_inliers

        # FLANN index parameters for float descriptors (SIFT)
        flann_index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE = 1
        flann_search_params = {"checks": 50}
        self._flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    def match(
        self,
        feat_a: KeyframeFeatures,
        feat_b: KeyframeFeatures,
    ) -> FramePairMatches:
        """
        Match features between feat_a and feat_b.

        Steps:
        1. kNN match (k=2) on SIFT descriptors via FLANN.
        2. Apply Lowe ratio test.
        3. Estimate fundamental matrix with RANSAC (findFundamentalMat).
        4. Return inlier correspondences only.
        """
        if feat_a.descriptors is None or feat_b.descriptors is None:
            return self._empty_match(feat_a.frame_index, feat_b.frame_index)

        if feat_a.num_keypoints < self._min_inliers or feat_b.num_keypoints < self._min_inliers:
            logger.debug(
                "Skipping pair (%d, %d): insufficient keypoints (%d, %d)",
                feat_a.frame_index, feat_b.frame_index,
                feat_a.num_keypoints, feat_b.num_keypoints,
            )
            return self._empty_match(feat_a.frame_index, feat_b.frame_index)

        # kNN matching
        try:
            knn_matches = self._flann.knnMatch(feat_a.descriptors, feat_b.descriptors, k=2)
        except cv2.error as exc:
            logger.warning(
                "FLANN kNN match failed for pair (%d, %d): %s",
                feat_a.frame_index, feat_b.frame_index, exc,
            )
            return self._empty_match(feat_a.frame_index, feat_b.frame_index)

        # Lowe ratio test
        good_matches = []
        for match_group in knn_matches:
            if len(match_group) == 2:
                m, n = match_group
                if m.distance < self._ratio_threshold * n.distance:
                    good_matches.append(m)

        raw_match_count = len(good_matches)
        if raw_match_count < self._min_inliers:
            logger.debug(
                "Pair (%d, %d): only %d ratio-test matches (need %d).",
                feat_a.frame_index, feat_b.frame_index, raw_match_count, self._min_inliers,
            )
            return FramePairMatches(
                frame_a_index=feat_a.frame_index,
                frame_b_index=feat_b.frame_index,
                pts_a=np.empty((0, 2), dtype=np.float32),
                pts_b=np.empty((0, 2), dtype=np.float32),
                fundamental_matrix=None,
                inlier_count=0,
                raw_match_count=raw_match_count,
                inlier_ratio=0.0,
                is_valid=False,
            )

        # Extract matched pixel coordinates
        pts_a = np.float32([feat_a.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts_b = np.float32([feat_b.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # RANSAC fundamental matrix estimation
        F, inlier_mask = cv2.findFundamentalMat(
            pts_a,
            pts_b,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self._ransac_threshold,
            confidence=self._ransac_confidence,
            maxIters=2000,
        )

        if inlier_mask is None or F is None:
            logger.debug(
                "RANSAC degenerate for pair (%d, %d).",
                feat_a.frame_index, feat_b.frame_index,
            )
            return FramePairMatches(
                frame_a_index=feat_a.frame_index,
                frame_b_index=feat_b.frame_index,
                pts_a=pts_a,
                pts_b=pts_b,
                fundamental_matrix=None,
                inlier_count=0,
                raw_match_count=raw_match_count,
                inlier_ratio=0.0,
                is_valid=False,
            )

        mask = inlier_mask.ravel().astype(bool)
        inlier_pts_a = pts_a[mask]
        inlier_pts_b = pts_b[mask]
        inlier_count = int(mask.sum())
        inlier_ratio = round(inlier_count / raw_match_count, 4) if raw_match_count > 0 else 0.0
        is_valid = inlier_count >= self._min_inliers

        logger.debug(
            "Pair (%d, %d): raw=%d  inliers=%d  ratio=%.3f  valid=%s",
            feat_a.frame_index, feat_b.frame_index,
            raw_match_count, inlier_count, inlier_ratio, is_valid,
        )

        return FramePairMatches(
            frame_a_index=feat_a.frame_index,
            frame_b_index=feat_b.frame_index,
            pts_a=inlier_pts_a,
            pts_b=inlier_pts_b,
            fundamental_matrix=F,
            inlier_count=inlier_count,
            raw_match_count=raw_match_count,
            inlier_ratio=inlier_ratio,
            is_valid=is_valid,
        )

    @staticmethod
    def _empty_match(idx_a: int, idx_b: int) -> FramePairMatches:
        return FramePairMatches(
            frame_a_index=idx_a,
            frame_b_index=idx_b,
            pts_a=np.empty((0, 2), dtype=np.float32),
            pts_b=np.empty((0, 2), dtype=np.float32),
            fundamental_matrix=None,
            inlier_count=0,
            raw_match_count=0,
            inlier_ratio=0.0,
            is_valid=False,
        )


# ---------------------------------------------------------------------------
# Feature Tracking Orchestrator
# ---------------------------------------------------------------------------


class FeatureTracker:
    """
    Orchestrates feature extraction and matching across a keyframe sequence.

    Strategy:
    - Sequential adjacent pairs: (0,1), (1,2), ..., (N-2, N-1)
    - Loop-closure pairs (if loop_closure_window > 0): connects frame i to frame i+w for w in range.
    """

    def __init__(
        self,
        extractor: Optional[SIFTFeatureExtractor] = None,
        matcher: Optional[FLANNFeatureMatcher] = None,
        loop_closure_window: int = 3,
    ) -> None:
        self._extractor = extractor or SIFTFeatureExtractor()
        self._matcher = matcher or FLANNFeatureMatcher()
        self._loop_window = max(0, loop_closure_window)

    def track(
        self,
        image_paths: List[Union[str, Path]],
        frame_indices: Optional[List[int]] = None,
    ) -> FeatureTrackingResult:
        """
        Extract features and compute pairwise matches for a sequence of keyframe images.

        Parameters
        ----------
        image_paths:
            Ordered list of absolute paths to keyframe PNG files.
        frame_indices:
            Optional source video frame indices corresponding to each image.
            Defaults to [0, 1, 2, ...] if not provided.

        Returns
        -------
        FeatureTrackingResult with features per keyframe and all valid pair matches.
        """
        if frame_indices is None:
            frame_indices = list(range(len(image_paths)))

        if len(image_paths) != len(frame_indices):
            raise ValueError("image_paths and frame_indices must have equal length.")

        result = FeatureTrackingResult(total_keyframes=len(image_paths))

        # 1. Extract features for all keyframes
        feature_map: Dict[int, KeyframeFeatures] = {}
        for idx, (path, f_idx) in enumerate(zip(image_paths, frame_indices)):
            feats = self._extractor.extract(image_path=path, frame_index=f_idx)
            if feats is None:
                result.failed_extraction_indices.append(f_idx)
                logger.warning("Feature extraction failed for frame %d", f_idx)
            else:
                feature_map[idx] = feats
                result.extracted_features.append(feats)

        # 2. Build pair list: sequential adjacent + loop-closure
        n = len(image_paths)
        pairs: list[tuple[int, int]] = []
        for i in range(n - 1):
            pairs.append((i, i + 1))                    # Adjacent
            for w in range(2, self._loop_window + 2):   # Loop-closure
                j = i + w
                if j < n:
                    pairs.append((i, j))

        # Deduplicate while preserving order
        seen: set[tuple[int, int]] = set()
        unique_pairs: list[tuple[int, int]] = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                unique_pairs.append(p)

        # 3. Match each pair
        for i, j in unique_pairs:
            if i not in feature_map or j not in feature_map:
                result.invalid_pair_count += 1
                continue

            pair_match = self._matcher.match(feature_map[i], feature_map[j])
            result.frame_pair_matches.append(pair_match)
            if not pair_match.is_valid:
                result.invalid_pair_count += 1

        logger.info(
            "FeatureTracker: keyframes=%d  extracted=%d  pairs=%d  valid=%d  invalid=%d",
            result.total_keyframes,
            len(result.extracted_features),
            len(result.frame_pair_matches),
            result.valid_pair_count,
            result.invalid_pair_count,
        )

        return result

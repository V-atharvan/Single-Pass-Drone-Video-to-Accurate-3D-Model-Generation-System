"""
Depth Uncertainty and Confidence Field Estimation — TASK-032.

Produces per-pixel confidence maps [0..255] for each keyframe depth map.
High confidence (255) = reliable depth. Low confidence (0) = unreliable.

Three independent confidence signals are computed and fused:

  1. Gradient consistency score:
     Local depth gradient magnitude. Smooth depth regions (low gradient)
     → high confidence. Abrupt isolated spikes (noise) → low confidence.

  2. Depth range plausibility score:
     Pixels within the expected outdoor depth range [0.5m, 200m] are valid.
     Sky pixels (near max depth) and sensor floor pixels (near min) penalized.

  3. Photometric reprojection consistency score (optional, when adjacent
     depth maps and camera poses are provided):
     Warp frame t+1 into frame t using depth D_t and relative pose.
     Pixels with high photometric error are flagged as low confidence.
     I_warp = warp(I_{t+1}, D_t, R_{t->t+1}, t_{t->t+1})

Fusion: confidence = mean(gradient_score, plausibility_score [, reprojection_score])
Output: uint8 [0..255] confidence map, same spatial resolution as depth map.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.depth.depth_estimator import DepthMap

logger = logging.getLogger("depth.confidence_estimator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Depth plausibility range (metres)
_PLAUSIBILITY_MIN_M: float = 0.3
_PLAUSIBILITY_MAX_M: float = 250.0

# Gradient noise threshold: depth gradient magnitude below this → high confidence
_GRADIENT_SMOOTH_THRESHOLD: float = 0.5    # metres per pixel

# Photometric reprojection error threshold (absolute pixel intensity difference)
_PHOTOMETRIC_ERROR_THRESHOLD: float = 30.0   # 0-255 scale


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceMap:
    """
    Per-pixel depth confidence map for a single keyframe.

    Attributes
    ----------
    frame_index:
        Source video frame index.
    confidence:
        (H, W) uint8 array where 255 = highest confidence, 0 = lowest.
    gradient_score:
        (H, W) uint8 gradient consistency component.
    plausibility_score:
        (H, W) uint8 depth plausibility component.
    reprojection_score:
        (H, W) uint8 photometric reprojection component (None if not computed).
    mean_confidence:
        Scalar mean confidence value over all pixels.
    """

    frame_index: int
    confidence: np.ndarray            # (H, W) uint8
    gradient_score: np.ndarray        # (H, W) uint8
    plausibility_score: np.ndarray    # (H, W) uint8
    reprojection_score: Optional[np.ndarray]  # (H, W) uint8 or None
    mean_confidence: float


@dataclass
class ConfidenceEstimationResult:
    """Aggregate result of confidence estimation over all keyframes."""

    total_keyframes: int
    confidence_maps: List[ConfidenceMap] = field(default_factory=list)
    failed_frame_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual signal estimators
# ---------------------------------------------------------------------------


def _gradient_consistency_score(depth_m: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel gradient consistency score.

    Smooth depth regions → low gradient → high confidence (255).
    Noisy isolated spikes → high gradient → low confidence (0).

    Returns uint8 (H, W) score array.
    """
    # Sobel gradients in x and y
    grad_x = cv2.Sobel(depth_m, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_m, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)

    # Normalize: 0 gradient = 255 confidence; high gradient = 0 confidence
    max_thresh = _GRADIENT_SMOOTH_THRESHOLD * 10.0
    clipped = np.clip(grad_mag, 0.0, max_thresh)
    score = 1.0 - (clipped / max_thresh)
    score = np.clip(score, 0.0, 1.0)
    return (score * 255.0).astype(np.uint8)


def _plausibility_score(depth_m: np.ndarray) -> np.ndarray:
    """
    Per-pixel depth range plausibility score.

    Pixels within [_PLAUSIBILITY_MIN_M, _PLAUSIBILITY_MAX_M] → high confidence.
    Pixels at near-min (noise floor) or near-max (sky) → penalized.

    Returns uint8 (H, W) score array.
    """
    # Core validity: within plausible outdoor range
    valid = (depth_m >= _PLAUSIBILITY_MIN_M) & (depth_m <= _PLAUSIBILITY_MAX_M)

    # Sky penalty: pixels > 90% of max get low confidence
    sky_threshold = _PLAUSIBILITY_MAX_M * 0.9
    sky_penalty = depth_m > sky_threshold

    # Near-floor penalty
    floor_threshold = _PLAUSIBILITY_MIN_M * 2.0
    floor_penalty = depth_m < floor_threshold

    score = np.where(valid, 220, 50).astype(np.float32)
    score[sky_penalty] = 30.0
    score[floor_penalty] = 80.0

    return np.clip(score, 0, 255).astype(np.uint8)


def _photometric_reprojection_score(
    bgr_frame_t: np.ndarray,
    bgr_frame_t1: np.ndarray,
    depth_t: np.ndarray,
    K: np.ndarray,
    R_relative: np.ndarray,
    t_relative: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel photometric reprojection consistency confidence.

    Warps frame t+1 into frame t using the depth D_t and relative pose,
    then measures per-pixel absolute photometric error.

    I_warp = warp(I_{t+1}, D_t, R_{t->t+1}, t_{t->t+1})
    error = |I_t - I_warp|

    Low error → high confidence. High error → depth unreliable (moving object,
    occlusion, texture-less region).

    Returns uint8 (H, W) score array.
    """
    h, w = depth_t.shape[:2]

    # Build pixel grid (u, v)
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    uv1 = np.stack([u_grid, v_grid, np.ones_like(u_grid)], axis=0).reshape(3, -1).astype(np.float64)

    # Back-project into 3D using depth
    K_inv = np.linalg.inv(K.astype(np.float64))
    depth_flat = depth_t.flatten().astype(np.float64)
    X_cam = K_inv @ uv1 * depth_flat  # (3, N)

    # Transform to frame t+1
    X_cam_t1 = (R_relative.astype(np.float64) @ X_cam) + t_relative.astype(np.float64).reshape(3, 1)

    # Project into frame t+1
    X_proj = K.astype(np.float64) @ X_cam_t1
    z = X_proj[2, :]
    valid_z = z > 0.1

    u_proj = np.where(valid_z, X_proj[0] / (z + 1e-9), -1).reshape(h, w).astype(np.float32)
    v_proj = np.where(valid_z, X_proj[1] / (z + 1e-9), -1).reshape(h, w).astype(np.float32)

    # Sample frame t+1 at projected coordinates
    warped = cv2.remap(
        bgr_frame_t1.astype(np.float32),
        u_proj,
        v_proj,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Per-pixel absolute photometric error (mean across channels)
    error = np.abs(bgr_frame_t.astype(np.float32) - warped).mean(axis=2)

    # Convert to confidence: low error → 255, high error → 0
    score = np.clip(1.0 - (error / _PHOTOMETRIC_ERROR_THRESHOLD), 0.0, 1.0)
    return (score * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main confidence estimator
# ---------------------------------------------------------------------------


class ConfidenceEstimator:
    """
    Computes pixel-wise depth confidence maps for each keyframe.

    Optionally incorporates photometric reprojection consistency when
    adjacent frames and camera poses are provided.
    """

    def __init__(
        self,
        use_reprojection: bool = False,
    ) -> None:
        self._use_reprojection = use_reprojection

    def estimate(
        self,
        depth_maps: List[DepthMap],
        bgr_frames: Optional[List[np.ndarray]] = None,
        intrinsic_K: Optional[np.ndarray] = None,
        relative_rotations: Optional[List[np.ndarray]] = None,
        relative_translations: Optional[List[np.ndarray]] = None,
    ) -> ConfidenceEstimationResult:
        """
        Compute confidence maps for all depth maps.

        Parameters
        ----------
        depth_maps:
            Output of DepthEstimator from TASK-031.
        bgr_frames:
            Corresponding BGR images (required for reprojection score).
        intrinsic_K:
            3x3 camera intrinsic matrix (required for reprojection score).
        relative_rotations:
            List of 3x3 relative rotation matrices R_{t->t+1} (length = n-1).
        relative_translations:
            List of 3-vectors t_{t->t+1} in metres (length = n-1).

        Returns
        -------
        ConfidenceEstimationResult with one ConfidenceMap per depth map.
        """
        result = ConfidenceEstimationResult(total_keyframes=len(depth_maps))

        for i, dm in enumerate(depth_maps):
            try:
                conf_map = self._estimate_single(
                    dm=dm,
                    idx_in_sequence=i,
                    bgr_frames=bgr_frames,
                    K=intrinsic_K,
                    rel_rotations=relative_rotations,
                    rel_translations=relative_translations,
                )
                result.confidence_maps.append(conf_map)
            except Exception:
                logger.exception("Confidence estimation failed for frame %d", dm.frame_index)
                result.failed_frame_indices.append(dm.frame_index)

        logger.info(
            "Confidence estimation: %d maps computed, %d failed",
            len(result.confidence_maps), len(result.failed_frame_indices),
        )
        return result

    def _estimate_single(
        self,
        dm: DepthMap,
        idx_in_sequence: int,
        bgr_frames: Optional[List[np.ndarray]],
        K: Optional[np.ndarray],
        rel_rotations: Optional[List[np.ndarray]],
        rel_translations: Optional[List[np.ndarray]],
    ) -> ConfidenceMap:
        """Compute confidence for a single depth map."""
        depth = dm.depth_m.astype(np.float32)

        grad_score = _gradient_consistency_score(depth)
        plaus_score = _plausibility_score(depth)

        repro_score: Optional[np.ndarray] = None
        use_repro = (
            self._use_reprojection
            and bgr_frames is not None
            and K is not None
            and rel_rotations is not None
            and rel_translations is not None
            and idx_in_sequence < len(bgr_frames) - 1
            and idx_in_sequence < len(rel_rotations)
        )

        if use_repro:
            try:
                repro_score = _photometric_reprojection_score(
                    bgr_frame_t=bgr_frames[idx_in_sequence],
                    bgr_frame_t1=bgr_frames[idx_in_sequence + 1],
                    depth_t=depth,
                    K=K,
                    R_relative=rel_rotations[idx_in_sequence],
                    t_relative=rel_translations[idx_in_sequence],
                )
            except Exception:
                logger.warning(
                    "Reprojection score failed for frame %d; using 2-signal fusion.",
                    dm.frame_index,
                )
                repro_score = None

        # Fuse signals
        signals = [grad_score.astype(np.float32), plaus_score.astype(np.float32)]
        if repro_score is not None:
            signals.append(repro_score.astype(np.float32))

        fused = np.mean(signals, axis=0)
        confidence = np.clip(fused, 0, 255).astype(np.uint8)
        mean_conf = float(np.mean(confidence))

        return ConfidenceMap(
            frame_index=dm.frame_index,
            confidence=confidence,
            gradient_score=grad_score,
            plausibility_score=plaus_score,
            reprojection_score=repro_score,
            mean_confidence=round(mean_conf, 2),
        )

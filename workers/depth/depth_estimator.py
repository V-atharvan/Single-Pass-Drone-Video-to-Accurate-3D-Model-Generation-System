"""
Monocular Metric Depth Map Inference Pipeline — TASK-031.

Processes batches of keyframe images through the depth estimation network
to produce dense, high-resolution metric depth maps in meters.

Pipeline stages per batch:
  1. Load keyframe PNG from disk (or accept numpy array for testing).
  2. Resize to model input resolution (default 518x518 for DepthAnythingV2).
  3. Normalize using ImageNet statistics.
  4. Run batched forward pass under torch.inference_mode() and autocast.
  5. Upsample output depth map back to original keyframe resolution using
     bilateral filtering to preserve sharp edges at building facades and roads.
  6. Return float32 depth array (meters) per keyframe.

Output contract:
  DepthMap dataclass with frame_index, depth_m (H x W float32 array),
  valid_mask (H x W bool), and min/max/median depth statistics.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.depth.runtime import (
    DeviceInfo,
    LoadedModel,
    autocast_context,
    clear_gpu_cache,
    detect_device,
)

logger = logging.getLogger("depth.depth_estimator")

# Lazy PyTorch import
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DepthAnythingV2 native input resolution
_MODEL_INPUT_SIZE: int = int(os.environ.get("DEPTH_INPUT_SIZE", "518")) if False else 518

# ImageNet normalization statistics
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Inference batch size (reduce if OOM)
_DEFAULT_BATCH_SIZE: int = int(
    __import__("os").environ.get("DEPTH_BATCH_SIZE", "4")
)

# Minimum valid depth threshold (metres) — below this is treated as invalid
_MIN_VALID_DEPTH_M: float = 0.1
# Maximum depth clip (metres) — sky pixels often produce unrealistically large values
_MAX_VALID_DEPTH_M: float = 300.0

# Bilateral filter parameters for edge-preserving upsampling
_BILATERAL_D: int = 5
_BILATERAL_SIGMA_COLOR: float = 75.0
_BILATERAL_SIGMA_SPACE: float = 75.0


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class DepthMap:
    """
    Metric depth map for a single keyframe.

    Attributes
    ----------
    frame_index:
        Source video frame index.
    depth_m:
        (H, W) float32 array of metric depth values in metres.
        Invalid/occluded pixels are set to 0.0.
    valid_mask:
        (H, W) bool array where True indicates a reliable depth estimate.
    original_height, original_width:
        Source keyframe resolution in pixels.
    min_depth_m, max_depth_m, median_depth_m:
        Depth statistics over the valid pixel set.
    inference_time_ms:
        Forward pass wall-clock time in milliseconds.
    """

    frame_index: int
    depth_m: np.ndarray          # (H, W) float32, metres
    valid_mask: np.ndarray       # (H, W) bool
    original_height: int
    original_width: int
    min_depth_m: float
    max_depth_m: float
    median_depth_m: float
    inference_time_ms: float


@dataclass
class DepthEstimationResult:
    """Aggregate output of the depth estimation pipeline over all keyframes."""

    total_keyframes: int
    depth_maps: List[DepthMap] = field(default_factory=list)
    failed_frame_indices: List[int] = field(default_factory=list)
    total_inference_time_sec: float = 0.0
    mean_inference_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------


def _load_keyframe_bgr(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load a keyframe PNG from disk as a BGR numpy array."""
    path = Path(image_path)
    if not path.exists():
        logger.warning("Keyframe image not found: %s", path)
        return None
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("OpenCV failed to decode: %s", path)
        return None
    return img


def _preprocess_frame(bgr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Preprocess a BGR frame for depth model inference.

    Steps:
      1. Convert BGR -> RGB.
      2. Resize to (target_size, target_size) with INTER_LINEAR.
      3. Normalize with ImageNet mean/std.
      4. Transpose to (C, H, W) channel-first format.

    Returns float32 numpy array shape (3, target_size, target_size).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return normalized.transpose(2, 0, 1)  # (C, H, W)


def _upsample_depth_bilateral(
    depth_low: np.ndarray,
    guide_bgr: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Upsample a low-resolution depth map to target resolution using bilateral filtering.

    The guide image preserves sharp depth discontinuities at texture edges
    (building facades, road markings), preventing depth bleeding across boundaries.

    Parameters
    ----------
    depth_low:
        (H_low, W_low) float32 depth array.
    guide_bgr:
        (H_orig, W_orig, 3) BGR guide image at target resolution.
    target_h, target_w:
        Output resolution.

    Returns
    -------
    (target_h, target_w) float32 depth array.
    """
    # Step 1: Bilinear upsample to target resolution
    depth_up = cv2.resize(
        depth_low,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Step 2: Bilateral filter for edge preservation
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
    guide_norm = guide_gray.astype(np.float32) / 255.0

    # Apply joint bilateral filter: depth guided by image structure
    # cv2.ximgproc.jointBilateralFilter is not always available;
    # use standard bilateral filter as fallback.
    try:
        import cv2.ximgproc as ximgproc
        depth_refined = ximgproc.jointBilateralFilter(
            joint=guide_bgr,
            src=depth_up,
            d=_BILATERAL_D,
            sigmaColor=_BILATERAL_SIGMA_COLOR,
            sigmaSpace=_BILATERAL_SIGMA_SPACE,
        )
    except (AttributeError, ImportError):
        # Fallback: standard bilateral filter on the depth map
        depth_u8 = np.clip(depth_up / max(depth_up.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(
            depth_u8, _BILATERAL_D, _BILATERAL_SIGMA_COLOR, _BILATERAL_SIGMA_SPACE
        )
        depth_refined = (depth_filtered.astype(np.float32) / 255.0) * max(depth_up.max(), 1e-6)

    return depth_refined.astype(np.float32)


def _build_valid_mask(depth_m: np.ndarray) -> np.ndarray:
    """Return a boolean mask where depth values are within the valid range."""
    return (depth_m >= _MIN_VALID_DEPTH_M) & (depth_m <= _MAX_VALID_DEPTH_M)


def _compute_depth_stats(depth_m: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float, float]:
    """Return (min, max, median) depth statistics over valid pixels."""
    valid_depths = depth_m[valid_mask]
    if len(valid_depths) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.min(valid_depths)),
        float(np.max(valid_depths)),
        float(np.median(valid_depths)),
    )


# ---------------------------------------------------------------------------
# Depth estimator
# ---------------------------------------------------------------------------


class DepthEstimator:
    """
    Batched monocular metric depth map inference engine.

    Accepts a pre-loaded model from TASK-030 and processes keyframe images
    to produce dense float32 depth arrays in metres.
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        input_size: int = _MODEL_INPUT_SIZE,
    ) -> None:
        self._model = loaded_model
        self._batch_size = max(1, batch_size)
        self._input_size = input_size
        self._device_info: DeviceInfo = loaded_model.device_info

    def estimate_from_paths(
        self,
        image_paths: List[Union[str, Path]],
        frame_indices: List[int],
    ) -> DepthEstimationResult:
        """
        Estimate metric depth for a list of keyframe image files.

        Parameters
        ----------
        image_paths:
            Ordered list of absolute paths to extracted keyframe PNGs.
        frame_indices:
            Source video frame indices corresponding to each path.

        Returns
        -------
        DepthEstimationResult with a DepthMap per keyframe.
        """
        if len(image_paths) != len(frame_indices):
            raise ValueError("image_paths and frame_indices must have equal length.")

        result = DepthEstimationResult(total_keyframes=len(image_paths))
        t_pipeline_start = time.perf_counter()

        # Load images into memory
        bgr_frames: List[Optional[np.ndarray]] = []
        for path, fi in zip(image_paths, frame_indices):
            bgr = _load_keyframe_bgr(path)
            if bgr is None:
                result.failed_frame_indices.append(fi)
            bgr_frames.append(bgr)

        # Batch inference
        self._run_batched_inference(bgr_frames, frame_indices, result)

        result.total_inference_time_sec = round(time.perf_counter() - t_pipeline_start, 3)
        if result.depth_maps:
            result.mean_inference_time_ms = round(
                sum(d.inference_time_ms for d in result.depth_maps) / len(result.depth_maps), 2
            )

        logger.info(
            "Depth estimation: %d keyframes  |  %d successful  |  %d failed  |  "
            "total=%.2fs  mean=%.1fms/frame",
            result.total_keyframes,
            len(result.depth_maps),
            len(result.failed_frame_indices),
            result.total_inference_time_sec,
            result.mean_inference_time_ms,
        )
        return result

    def estimate_from_arrays(
        self,
        bgr_frames: List[np.ndarray],
        frame_indices: List[int],
    ) -> DepthEstimationResult:
        """
        Estimate depth from in-memory BGR numpy arrays (for testing / streaming).

        Parameters
        ----------
        bgr_frames:
            List of (H, W, 3) uint8 BGR arrays.
        frame_indices:
            Source frame indices.

        Returns
        -------
        DepthEstimationResult with a DepthMap per frame.
        """
        if len(bgr_frames) != len(frame_indices):
            raise ValueError("bgr_frames and frame_indices must have equal length.")

        result = DepthEstimationResult(total_keyframes=len(bgr_frames))
        t_start = time.perf_counter()
        self._run_batched_inference(
            [f for f in bgr_frames],  # All valid
            frame_indices,
            result,
        )
        result.total_inference_time_sec = round(time.perf_counter() - t_start, 3)
        if result.depth_maps:
            result.mean_inference_time_ms = round(
                sum(d.inference_time_ms for d in result.depth_maps) / len(result.depth_maps), 2
            )
        return result

    def _run_batched_inference(
        self,
        bgr_frames: List[Optional[np.ndarray]],
        frame_indices: List[int],
        result: DepthEstimationResult,
    ) -> None:
        """Process frames in batches through the depth model."""
        n = len(bgr_frames)
        net = self._model.model

        for batch_start in range(0, n, self._batch_size):
            batch_bgrs = bgr_frames[batch_start: batch_start + self._batch_size]
            batch_indices = frame_indices[batch_start: batch_start + self._batch_size]

            # Build batch tensor
            tensors = []
            valid_items = []  # (idx_in_batch, fi, bgr)
            for idx_in_batch, (bgr, fi) in enumerate(zip(batch_bgrs, batch_indices)):
                if bgr is None:
                    if fi not in result.failed_frame_indices:
                        result.failed_frame_indices.append(fi)
                    continue
                preprocessed = _preprocess_frame(bgr, self._input_size)
                tensors.append(preprocessed)
                valid_items.append((idx_in_batch, fi, bgr))

            if not tensors:
                continue

            t_infer_start = time.perf_counter()
            depth_outputs = self._forward_batch(tensors)
            t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            per_frame_ms = t_infer_ms / len(tensors)

            for local_idx, (_, fi, bgr_orig) in enumerate(valid_items):
                if local_idx >= len(depth_outputs):
                    result.failed_frame_indices.append(fi)
                    continue

                depth_low = depth_outputs[local_idx]   # (H_model, W_model) float32
                h_orig, w_orig = bgr_orig.shape[:2]

                # Upsample with bilateral edge preservation
                depth_full = _upsample_depth_bilateral(depth_low, bgr_orig, h_orig, w_orig)

                # Clip to valid range
                depth_full = np.clip(depth_full, 0.0, _MAX_VALID_DEPTH_M)
                valid_mask = _build_valid_mask(depth_full)
                d_min, d_max, d_med = _compute_depth_stats(depth_full, valid_mask)

                result.depth_maps.append(DepthMap(
                    frame_index=fi,
                    depth_m=depth_full,
                    valid_mask=valid_mask,
                    original_height=h_orig,
                    original_width=w_orig,
                    min_depth_m=round(d_min, 3),
                    max_depth_m=round(d_max, 3),
                    median_depth_m=round(d_med, 3),
                    inference_time_ms=round(per_frame_ms, 2),
                ))

    def _forward_batch(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Stack tensors, run model forward pass, unstack outputs.

        Returns list of (H_model, W_model) float32 depth arrays.
        """
        if _TORCH_AVAILABLE:
            batch = torch.from_numpy(np.stack(tensors, axis=0))
            if self._device_info.device_type == "cuda":
                device = torch.device(f"cuda:{self._device_info.device_index}")
                batch = batch.to(device)

            with autocast_context(self._device_info):
                with torch.inference_mode():
                    output = self._model.model(batch)

            # Extract depth from model output (dict or tensor)
            if isinstance(output, dict):
                depth_tensor = output.get("depth", output.get("pred_depth", None))
            elif isinstance(output, (list, tuple)):
                depth_tensor = output[0]
            else:
                depth_tensor = output

            if depth_tensor is None:
                logger.error("Model output does not contain a recognized depth key.")
                return []

            # Move to CPU and convert to numpy
            depth_np = depth_tensor.squeeze(1).cpu().float().numpy()  # (B, H, W)
            return [depth_np[i] for i in range(depth_np.shape[0])]

        else:
            # Simulation: return constant depth maps at model resolution
            return [
                np.full((self._input_size, self._input_size), 10.0, dtype=np.float32)
                for _ in tensors
            ]

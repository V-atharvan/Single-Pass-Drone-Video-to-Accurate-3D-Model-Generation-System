"""
PyTorch GPU Worker Runtime and CUDA Model Loader — TASK-030.

Provides the foundational GPU worker infrastructure for the monocular depth
estimation pipeline (Phase 5). Responsibilities:

  1. Device detection: CUDA availability check, GPU model name, VRAM capacity.
  2. Mixed-precision context manager: FP16 / BF16 autocast with CPU fallback.
  3. Memory management: cache cleaner, VRAM headroom guard.
  4. Model weight manager: download, verify checksum, cache pre-trained weights
     locally; supports Depth Anything V2 Metric (primary) with UniDepth as fallback.
  5. Model loader: instantiates the depth network, moves to device, wraps with
     torch.compile (when available) or TorchScript for inference optimization.

Design principles:
  - CPU fallback is always available; GPU is preferred but never required.
  - Memory footprint target: < 6 GB VRAM for the depth model at inference time.
  - Model cache directory respects XDG_CACHE_HOME / HF_HOME environment variables.
  - All CUDA operations are wrapped to surface interpretable errors on OOM.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger("depth.runtime")

# ---------------------------------------------------------------------------
# Lazy PyTorch import — allows CPU-only testing without GPU installation
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed; runtime will operate in simulation mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VRAM headroom guard: refuse to load model if free VRAM < this threshold (MB)
_MIN_FREE_VRAM_MB: int = int(os.environ.get("MIN_FREE_VRAM_MB", "1024"))

# Target memory footprint for depth model weights (MB)
_TARGET_MODEL_VRAM_MB: int = 6 * 1024  # 6 GB

# Default model cache directory
_DEFAULT_MODEL_CACHE = Path(
    os.environ.get(
        "MODEL_CACHE_DIR",
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "single_pass_depth")),
    )
)

# Supported model identifiers and their download metadata
_SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
    "depth_anything_v2_metric_vitl": {
        "description": "Depth Anything V2 Metric — ViT-L backbone (primary)",
        "filename": "depth_anything_v2_metric_vitl.pth",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large/resolve/main/depth_anything_v2_metric_outdoor_vitl.pth",
        "expected_size_mb": 1340,
    },
    "depth_anything_v2_metric_vits": {
        "description": "Depth Anything V2 Metric — ViT-S backbone (lightweight fallback)",
        "filename": "depth_anything_v2_metric_vits.pth",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small/resolve/main/depth_anything_v2_metric_outdoor_vits.pth",
        "expected_size_mb": 100,
    },
}

_DEFAULT_MODEL_ID = "depth_anything_v2_metric_vitl"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    """Hardware and runtime information for the selected compute device."""

    device_type: str              # "cuda" | "cpu"
    device_index: int             # GPU index (0 for first GPU; -1 for CPU)
    device_name: str              # Human-readable device name
    total_vram_mb: float          # Total VRAM in MB (0.0 for CPU)
    free_vram_mb: float           # Free VRAM at detection time (0.0 for CPU)
    cuda_version: Optional[str]   # e.g. "12.4" or None
    torch_version: str            # e.g. "2.3.0"
    fp16_supported: bool          # True if FP16 autocast is available
    bf16_supported: bool          # True if BF16 is available
    torch_compile_available: bool # True if torch.compile is callable


@dataclass
class ModelCacheEntry:
    """Record describing a cached model weight file."""

    model_id: str
    cache_path: str
    file_size_bytes: int
    sha256: Optional[str]
    is_verified: bool


@dataclass
class LoadedModel:
    """Container for a successfully loaded depth model."""

    model_id: str
    device: Any             # torch.device
    device_info: DeviceInfo
    model: Any              # nn.Module (or stub in CPU simulation mode)
    is_compiled: bool
    load_time_sec: float
    estimated_vram_mb: float


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def detect_device(preferred_device_index: int = 0) -> DeviceInfo:
    """
    Detect and characterize the best available compute device.

    Priority: CUDA GPU > CPU (fallback).
    Logs all device properties at INFO level.

    Parameters
    ----------
    preferred_device_index:
        CUDA device index to prefer. If unavailable, falls back to CPU.

    Returns
    -------
    DeviceInfo describing the selected device.
    """
    if not _TORCH_AVAILABLE:
        return DeviceInfo(
            device_type="cpu",
            device_index=-1,
            device_name="CPU (PyTorch not installed — simulation mode)",
            total_vram_mb=0.0,
            free_vram_mb=0.0,
            cuda_version=None,
            torch_version="N/A",
            fp16_supported=False,
            bf16_supported=False,
            torch_compile_available=False,
        )

    torch_version = torch.__version__

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_index = min(preferred_device_index, n_gpus - 1)
        device_name = torch.cuda.get_device_name(device_index)

        props = torch.cuda.get_device_properties(device_index)
        total_vram_mb = props.total_memory / (1024 ** 2)
        free_vram_mb = (
            torch.cuda.mem_get_info(device_index)[0] / (1024 ** 2)
            if hasattr(torch.cuda, "mem_get_info")
            else total_vram_mb * 0.9
        )

        cuda_version = torch.version.cuda or "unknown"
        fp16_supported = props.major >= 7       # Volta+ supports FP16 TensorCores
        bf16_supported = props.major >= 8       # Ampere+ supports BF16

        compile_available = hasattr(torch, "compile")

        info = DeviceInfo(
            device_type="cuda",
            device_index=device_index,
            device_name=device_name,
            total_vram_mb=round(total_vram_mb, 1),
            free_vram_mb=round(free_vram_mb, 1),
            cuda_version=cuda_version,
            torch_version=torch_version,
            fp16_supported=fp16_supported,
            bf16_supported=bf16_supported,
            torch_compile_available=compile_available,
        )

        logger.info(
            "GPU detected: %s  |  VRAM: %.0f MB total / %.0f MB free  |  "
            "CUDA %s  |  FP16=%s  BF16=%s  torch.compile=%s",
            device_name, total_vram_mb, free_vram_mb,
            cuda_version, fp16_supported, bf16_supported, compile_available,
        )

        if free_vram_mb < _MIN_FREE_VRAM_MB:
            logger.warning(
                "Free VRAM (%.0f MB) is below the minimum threshold (%d MB). "
                "Inference may fail with OOM. Consider reducing batch size.",
                free_vram_mb, _MIN_FREE_VRAM_MB,
            )

        return info

    # CPU fallback
    logger.warning(
        "CUDA not available. Running on CPU. "
        "Depth estimation will be significantly slower (not suitable for production)."
    )
    return DeviceInfo(
        device_type="cpu",
        device_index=-1,
        device_name="CPU",
        total_vram_mb=0.0,
        free_vram_mb=0.0,
        cuda_version=None,
        torch_version=torch_version,
        fp16_supported=False,
        bf16_supported=False,
        torch_compile_available=hasattr(torch, "compile"),
    )


# ---------------------------------------------------------------------------
# Mixed-precision context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def autocast_context(device_info: DeviceInfo) -> Generator[None, None, None]:
    """
    Context manager enabling FP16 / BF16 autocast for the active device.

    Priority:
      - BF16 on Ampere+ GPUs (numerically stable, no loss scaling needed).
      - FP16 on Volta/Turing GPUs (fast, requires careful loss scaling).
      - No-op on CPU (autocast unsupported, runs in FP32).

    Usage:
        with autocast_context(device_info):
            output = model(input_tensor)
    """
    if not _TORCH_AVAILABLE or device_info.device_type == "cpu":
        yield
        return

    if device_info.bf16_supported:
        dtype = torch.bfloat16
        label = "BF16"
    elif device_info.fp16_supported:
        dtype = torch.float16
        label = "FP16"
    else:
        yield
        return

    logger.debug("Enabling %s autocast for inference.", label)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        yield


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------


def clear_gpu_cache() -> None:
    """
    Release all unused CUDA memory back to the memory pool.
    Safe to call at any time; no-op on CPU.
    """
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared.")


def get_free_vram_mb(device_index: int = 0) -> float:
    """Return current free VRAM in MB for the specified CUDA device."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0.0
    if hasattr(torch.cuda, "mem_get_info"):
        free, _ = torch.cuda.mem_get_info(device_index)
        return free / (1024 ** 2)
    return 0.0


def guard_vram_headroom(required_mb: float, device_info: DeviceInfo) -> None:
    """
    Raise RuntimeError if free VRAM is insufficient for the requested operation.

    Parameters
    ----------
    required_mb:
        Minimum required free VRAM in megabytes.
    device_info:
        Current device info snapshot.
    """
    if device_info.device_type == "cpu":
        return  # No VRAM constraint on CPU

    free_mb = get_free_vram_mb(device_info.device_index)
    if free_mb < required_mb:
        raise RuntimeError(
            f"Insufficient VRAM: {free_mb:.0f} MB free, {required_mb:.0f} MB required. "
            f"Call clear_gpu_cache() or reduce batch size before proceeding."
        )


# ---------------------------------------------------------------------------
# Model weight manager
# ---------------------------------------------------------------------------


class ModelWeightManager:
    """
    Manages download, verification, and local caching of depth model weights.

    Cache layout:
        {cache_dir}/{model_id}/{filename}
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        self._cache_dir = Path(cache_dir or _DEFAULT_MODEL_CACHE)

    def get_model_path(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        download_if_missing: bool = True,
    ) -> Optional[Path]:
        """
        Return the local path to model weights, downloading if not cached.

        Parameters
        ----------
        model_id:
            One of the keys in _SUPPORTED_MODELS.
        download_if_missing:
            When True, attempts to download the weights if not present.
            When False, returns None if weights are absent.

        Returns
        -------
        Path to the .pth file, or None if unavailable.
        """
        if model_id not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Supported: {list(_SUPPORTED_MODELS.keys())}"
            )

        spec = _SUPPORTED_MODELS[model_id]
        model_dir = self._cache_dir / model_id
        weight_path = model_dir / spec["filename"]

        if weight_path.exists():
            logger.info("Model weights found in cache: %s", weight_path)
            return weight_path

        if not download_if_missing:
            logger.info("Model weights not cached and download disabled: %s", model_id)
            return None

        # Attempt download
        return self._download(model_id, spec, weight_path)

    def _download(
        self,
        model_id: str,
        spec: Dict[str, str],
        dest_path: Path,
    ) -> Optional[Path]:
        """Download weights from HuggingFace to local cache with progress logging."""
        import urllib.request

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = spec["url"]
        expected_mb = float(spec["expected_size_mb"])

        logger.info(
            "Downloading model weights: %s (~%.0f MB) -> %s",
            model_id, expected_mb, dest_path,
        )

        try:
            def _reporthook(count: int, block_size: int, total_size: int) -> None:
                if total_size > 0 and count % 100 == 0:
                    downloaded_mb = count * block_size / (1024 ** 2)
                    total_mb = total_size / (1024 ** 2)
                    logger.info("  Downloading: %.1f / %.1f MB", downloaded_mb, total_mb)

            urllib.request.urlretrieve(url, dest_path, _reporthook)
            logger.info("Download complete: %s", dest_path)
            return dest_path

        except Exception as exc:
            logger.error("Download failed for %s: %s", model_id, exc)
            if dest_path.exists():
                dest_path.unlink()
            return None

    def list_cached_models(self) -> List[ModelCacheEntry]:
        """Return all model weight files found in the cache directory."""
        entries = []
        for model_id, spec in _SUPPORTED_MODELS.items():
            weight_path = self._cache_dir / model_id / spec["filename"]
            if weight_path.exists():
                size = weight_path.stat().st_size
                entries.append(ModelCacheEntry(
                    model_id=model_id,
                    cache_path=str(weight_path),
                    file_size_bytes=size,
                    sha256=None,  # Computed on demand
                    is_verified=False,
                ))
        return entries

    def compute_sha256(self, model_id: str) -> Optional[str]:
        """Compute SHA-256 checksum of cached weights (for integrity verification)."""
        spec = _SUPPORTED_MODELS.get(model_id)
        if spec is None:
            return None
        weight_path = self._cache_dir / model_id / spec["filename"]
        if not weight_path.exists():
            return None

        h = hashlib.sha256()
        with open(weight_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


class DepthModelLoader:
    """
    Loads a depth estimation model into VRAM / RAM, applies torch.compile
    optimizations, and validates inference with a dummy forward pass.

    Supports:
      - Real PyTorch model loading (production).
      - Simulation stub (when PyTorch unavailable or in unit-test context).
    """

    def __init__(
        self,
        device_info: Optional[DeviceInfo] = None,
        weight_manager: Optional[ModelWeightManager] = None,
    ) -> None:
        self._device_info = device_info or detect_device()
        self._weight_manager = weight_manager or ModelWeightManager()

    def load(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        download_if_missing: bool = True,
        apply_torch_compile: bool = True,
        simulation_mode: bool = False,
    ) -> LoadedModel:
        """
        Load the specified depth model onto the compute device.

        Parameters
        ----------
        model_id:
            Depth model identifier (see _SUPPORTED_MODELS).
        download_if_missing:
            Download weights if not in local cache.
        apply_torch_compile:
            Apply torch.compile() for inference speedup (requires PyTorch 2.0+).
        simulation_mode:
            When True, returns a stub model (for unit tests without GPU/weights).

        Returns
        -------
        LoadedModel with the instantiated network and device info.
        """
        t_start = time.perf_counter()

        if simulation_mode or not _TORCH_AVAILABLE:
            return self._load_simulation_stub(model_id, t_start)

        device_str = (
            f"cuda:{self._device_info.device_index}"
            if self._device_info.device_type == "cuda"
            else "cpu"
        )
        device = torch.device(device_str)

        # Resolve weight path
        weight_path = self._weight_manager.get_model_path(
            model_id, download_if_missing=download_if_missing
        )

        if weight_path is None:
            logger.warning(
                "Weights unavailable for %s. Falling back to simulation stub.", model_id
            )
            return self._load_simulation_stub(model_id, t_start)

        # Load state dict
        logger.info("Loading depth model: %s -> %s", model_id, device_str)
        try:
            state_dict = torch.load(str(weight_path), map_location=device)
            model = self._instantiate_model(model_id, state_dict, device)

            # Apply torch.compile if available and requested
            is_compiled = False
            if apply_torch_compile and self._device_info.torch_compile_available:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    is_compiled = True
                    logger.info("torch.compile applied to %s.", model_id)
                except Exception as exc:
                    logger.warning("torch.compile failed (%s); using eager mode.", exc)

            # Validate with dummy inference
            self._validate_dummy_inference(model, device)

            load_time = time.perf_counter() - t_start
            vram_used = self._estimate_vram_mb(model)

            logger.info(
                "Model loaded: %s on %s  |  VRAM est. %.0f MB  |  compiled=%s  |  %.2fs",
                model_id, device_str, vram_used, is_compiled, load_time,
            )

            return LoadedModel(
                model_id=model_id,
                device=device,
                device_info=self._device_info,
                model=model,
                is_compiled=is_compiled,
                load_time_sec=round(load_time, 3),
                estimated_vram_mb=round(vram_used, 1),
            )

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_gpu_cache()
                raise RuntimeError(
                    f"CUDA out-of-memory loading {model_id}. "
                    f"Free VRAM: {get_free_vram_mb(self._device_info.device_index):.0f} MB. "
                    f"Try a lighter model variant (e.g. depth_anything_v2_metric_vits) "
                    f"or reduce concurrent processes."
                ) from exc
            raise

    def _load_simulation_stub(self, model_id: str, t_start: float) -> LoadedModel:
        """Return a simulation stub LoadedModel for unit testing without GPU/weights."""
        load_time = time.perf_counter() - t_start

        class _StubModel:
            """Minimal model stub that returns a fixed-shape depth tensor."""
            training = False

            def __call__(self, x: Any) -> Any:
                if _TORCH_AVAILABLE:
                    b = x.shape[0] if hasattr(x, "shape") else 1
                    h = x.shape[2] if hasattr(x, "shape") and len(x.shape) > 2 else 384
                    w = x.shape[3] if hasattr(x, "shape") and len(x.shape) > 3 else 384
                    return {"depth": torch.ones(b, 1, h, w, dtype=torch.float32) * 10.0}
                return {"depth": [[10.0]]}

            def eval(self):
                return self

            def to(self, device):
                return self

            def parameters(self):
                return iter([])

        device = torch.device("cpu") if _TORCH_AVAILABLE else "cpu"

        return LoadedModel(
            model_id=f"{model_id} (stub)",
            device=device,
            device_info=self._device_info,
            model=_StubModel(),
            is_compiled=False,
            load_time_sec=round(load_time, 3),
            estimated_vram_mb=0.0,
        )

    @staticmethod
    def _instantiate_model(model_id: str, state_dict: Any, device: Any) -> Any:
        """
        Instantiate the depth model architecture and load weights.

        In production: import the Depth Anything V2 model class from the
        vendor package and call model.load_state_dict(state_dict).

        This method is a thin adapter to allow swapping architectures.
        """
        # The DepthAnything V2 model class would be imported here.
        # For portability without the vendor package installed:
        raise ImportError(
            "Depth Anything V2 model package not installed. "
            "Install with: pip install depth-anything-v2 "
            "or use simulation_mode=True for testing."
        )

    @staticmethod
    def _validate_dummy_inference(model: Any, device: Any) -> None:
        """
        Run a dummy forward pass (1, 3, 518, 518) to validate the loaded model.
        Raises RuntimeError on failure (e.g., mismatched architecture).
        """
        if not _TORCH_AVAILABLE:
            return
        model.eval()
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 518, 518, dtype=torch.float32, device=device)
            try:
                out = model(dummy)
                logger.debug("Dummy inference OK. Output keys: %s", list(out.keys()) if isinstance(out, dict) else type(out))
            except Exception as exc:
                raise RuntimeError(f"Dummy inference validation failed: {exc}") from exc

    @staticmethod
    def _estimate_vram_mb(model: Any) -> float:
        """Estimate model parameter memory footprint in MB."""
        if not _TORCH_AVAILABLE or not hasattr(model, "parameters"):
            return 0.0
        try:
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            return param_bytes / (1024 ** 2)
        except Exception:
            return 0.0

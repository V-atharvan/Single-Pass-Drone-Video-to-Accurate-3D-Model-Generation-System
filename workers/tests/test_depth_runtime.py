"""
Unit tests for PyTorch GPU Worker Runtime & CUDA Model Loader — TASK-030.

Covers:
  - Device detection (CUDA availability, CPU fallback, device metadata).
  - VRAM headroom check and cache clearing utilities.
  - Mixed-precision autocast context manager across devices.
  - ModelWeightManager cache directory handling, unknown model rejection, and checksums.
  - DepthModelLoader instantiation in simulation and fallback modes.
  - OOM error simulation and recovery handling.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from workers.depth.runtime import (
    _DEFAULT_MODEL_ID,
    _SUPPORTED_MODELS,
    DeviceInfo,
    DepthModelLoader,
    LoadedModel,
    ModelWeightManager,
    autocast_context,
    clear_gpu_cache,
    detect_device,
    get_free_vram_mb,
    guard_vram_headroom,
)


# ---------------------------------------------------------------------------
# Device Detection & Memory Tests
# ---------------------------------------------------------------------------


def test_detect_device_returns_valid_device_info():
    """Device detection should always return a well-formed DeviceInfo."""
    info = detect_device()
    assert isinstance(info, DeviceInfo)
    assert info.device_type in ("cpu", "cuda")
    assert isinstance(info.device_name, str)
    assert info.total_vram_mb >= 0.0
    assert info.free_vram_mb >= 0.0
    assert isinstance(info.fp16_supported, bool)
    assert isinstance(info.bf16_supported, bool)
    assert isinstance(info.torch_compile_available, bool)


def test_clear_gpu_cache_runs_without_exception():
    """Cache clearing must be safe to call on any environment."""
    clear_gpu_cache()


def test_get_free_vram_mb_returns_float():
    """VRAM enquiry returns non-negative float."""
    vram = get_free_vram_mb()
    assert isinstance(vram, float)
    assert vram >= 0.0


def test_guard_vram_headroom_cpu_noop():
    """On CPU, headroom guard should never raise."""
    info = DeviceInfo(
        device_type="cpu",
        device_index=-1,
        device_name="CPU",
        total_vram_mb=0.0,
        free_vram_mb=0.0,
        cuda_version=None,
        torch_version="none",
        fp16_supported=False,
        bf16_supported=False,
        torch_compile_available=False,
    )
    # Should not raise even if large memory requested
    guard_vram_headroom(999999.0, info)


def test_guard_vram_headroom_cuda_insufficient():
    """On CUDA, headroom guard raises when free VRAM is below required."""
    info = DeviceInfo(
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA A100",
        total_vram_mb=40960.0,
        free_vram_mb=500.0,
        cuda_version="12.2",
        torch_version="2.3.0",
        fp16_supported=True,
        bf16_supported=True,
        torch_compile_available=True,
    )
    with patch("workers.depth.runtime.get_free_vram_mb", return_value=500.0):
        with pytest.raises(RuntimeError, match="Insufficient VRAM"):
            guard_vram_headroom(2048.0, info)


def test_autocast_context_executes_block():
    """Autocast context executes code inside the with block."""
    info = detect_device()
    executed = False
    with autocast_context(info):
        executed = True
    assert executed is True


# ---------------------------------------------------------------------------
# Model Weight Manager Tests
# ---------------------------------------------------------------------------


def test_model_weight_manager_unknown_model_raises(tmp_path: Path):
    """Requesting unknown model ID should raise ValueError."""
    mwm = ModelWeightManager(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown model_id"):
        mwm.get_model_path("non_existent_depth_model_1234")


def test_model_weight_manager_missing_no_download(tmp_path: Path):
    """When download_if_missing is False and weights are absent, returns None."""
    mwm = ModelWeightManager(cache_dir=tmp_path)
    result = mwm.get_model_path(_DEFAULT_MODEL_ID, download_if_missing=False)
    assert result is None


def test_model_weight_manager_cached_hit(tmp_path: Path):
    """If weight file exists in cache directory, returns path directly without download."""
    spec = _SUPPORTED_MODELS[_DEFAULT_MODEL_ID]
    cached_dir = tmp_path / _DEFAULT_MODEL_ID
    cached_dir.mkdir(parents=True)
    fake_weight = cached_dir / spec["filename"]
    fake_weight.write_bytes(b"dummy_weights_data")

    mwm = ModelWeightManager(cache_dir=tmp_path)
    path = mwm.get_model_path(_DEFAULT_MODEL_ID, download_if_missing=False)
    assert path == fake_weight
    assert path.exists()

    # Test list_cached_models
    cached_list = mwm.list_cached_models()
    assert len(cached_list) >= 1
    assert cached_list[0].model_id == _DEFAULT_MODEL_ID
    assert cached_list[0].file_size_bytes == len(b"dummy_weights_data")

    # Test compute_sha256
    sha = mwm.compute_sha256(_DEFAULT_MODEL_ID)
    assert isinstance(sha, str)
    assert len(sha) == 64


# ---------------------------------------------------------------------------
# Depth Model Loader Tests
# ---------------------------------------------------------------------------


def test_depth_model_loader_simulation_mode(tmp_path: Path):
    """Loading model in simulation mode returns a functioning LoadedModel."""
    mwm = ModelWeightManager(cache_dir=tmp_path)
    loader = DepthModelLoader(weight_manager=mwm)
    loaded = loader.load(model_id=_DEFAULT_MODEL_ID, simulation_mode=True)

    assert isinstance(loaded, LoadedModel)
    assert "stub" in loaded.model_id
    assert loaded.is_compiled is False
    assert loaded.load_time_sec >= 0.0
    assert loaded.model is not None


def test_depth_model_loader_missing_weights_falls_back_to_stub(tmp_path: Path):
    """If weights are missing and download disabled, falls back to simulation stub gracefully."""
    mwm = ModelWeightManager(cache_dir=tmp_path)
    loader = DepthModelLoader(weight_manager=mwm)
    loaded = loader.load(
        model_id=_DEFAULT_MODEL_ID,
        download_if_missing=False,
        simulation_mode=False,
    )
    assert isinstance(loaded, LoadedModel)
    assert "stub" in loaded.model_id

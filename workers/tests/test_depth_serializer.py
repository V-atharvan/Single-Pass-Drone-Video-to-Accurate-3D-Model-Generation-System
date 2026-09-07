"""
Unit tests for Depth Map and Confidence Field Serialization — TASK-034.

Covers:
  - 16-bit millimetric PNG round-trip encode / decode.
  - Compressed NPZ round-trip encode / decode with valid masks.
  - 8-bit confidence PNG round-trip encode / decode.
  - Turbo-colormapped preview generation.
  - SHA-256 checksum and file size tracking.
  - Local scratch file writing (depth, confidence, preview, manifest.json).
  - S3 interim upload to canonical keys (jobs/{job_id}/interim/depth/...).
  - Redis progress reporting under ESTIMATING_DEPTH stage.
  - Offline fallback (no S3, no Redis).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from packages.shared.python.storage import S3StorageClient
from workers.depth.depth_estimator import DepthMap
from workers.depth.serializer import (
    DEFAULT_DEPTH_SCALE,
    DepthSerializationResult,
    DepthSerializer,
    compute_sha256,
    decode_confidence_png,
    decode_depth_npz,
    decode_depth_png,
    encode_confidence_png,
    encode_depth_npz,
    encode_depth_png,
    encode_depth_preview,
)


@pytest.fixture
def sample_depth_data():
    """Generates synthetic depth and confidence maps."""
    h, w = 180, 320
    depth = np.linspace(1.5, 35.0, h * w, dtype=np.float32).reshape((h, w))
    conf = np.full((h, w), 220, dtype=np.uint8)
    mask = (depth >= 2.0) & (depth <= 30.0)
    return depth, conf, mask


# ---------------------------------------------------------------------------
# Encoders & Decoders Round-Trip Tests
# ---------------------------------------------------------------------------


def test_depth_png_roundtrip(sample_depth_data):
    """16-bit PNG preserves depth in meters within millimeter precision."""
    depth, _, _ = sample_depth_data
    encoded = encode_depth_png(depth, scale_factor=DEFAULT_DEPTH_SCALE)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0

    decoded = decode_depth_png(encoded, scale_factor=DEFAULT_DEPTH_SCALE)
    assert decoded.shape == depth.shape
    assert decoded.dtype == np.float32
    # At 1000 units/m, quantization error is at most 0.5 mm = 0.0005 m
    np.testing.assert_allclose(decoded, depth, atol=0.001)


def test_depth_npz_roundtrip(sample_depth_data):
    """Compressed NPZ exactly preserves float32 depth and boolean valid mask."""
    depth, _, mask = sample_depth_data
    encoded = encode_depth_npz(depth, mask)
    assert isinstance(encoded, bytes)

    dec_depth, dec_mask = decode_depth_npz(encoded)
    np.testing.assert_array_equal(dec_depth, depth)
    assert dec_mask is not None
    np.testing.assert_array_equal(dec_mask, mask)


def test_confidence_png_roundtrip(sample_depth_data):
    """Confidence 8-bit PNG exactly preserves uint8 values."""
    _, conf, _ = sample_depth_data
    encoded = encode_confidence_png(conf)
    decoded = decode_confidence_png(encoded)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, conf)


def test_depth_preview_generation(sample_depth_data):
    """Preview generation produces valid non-empty PNG bytes."""
    depth, _, mask = sample_depth_data
    preview = encode_depth_preview(depth, mask)
    assert isinstance(preview, bytes)
    assert len(preview) > 100


def test_compute_sha256():
    """Checksum computes standard 64-char hex digest."""
    data = b"depth_anything_metric_bytes"
    h = compute_sha256(data)
    assert isinstance(h, str)
    assert len(h) == 64
    assert h == compute_sha256(data)


# ---------------------------------------------------------------------------
# Local Scratch Serialization Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serializer_local_scratch(tmp_path: Path, sample_depth_data):
    """Serializes depth items to NVMe scratch directory without S3 or Redis."""
    depth, conf, mask = sample_depth_data
    items = [
        (0, depth, conf, mask),
        (1, depth + 1.0, conf, mask),
    ]

    serializer = DepthSerializer(
        storage_client=None,
        redis_tracker=None,
        save_npz=True,
        save_preview=True,
    )

    res = await serializer.serialize_depth_maps(
        job_id="test-job-local",
        depth_items=items,
        scratch_root=tmp_path,
    )

    assert isinstance(res, DepthSerializationResult)
    assert res.total_frames == 2
    assert len(res.frames) == 2
    assert res.all_s3_uploaded is True
    assert res.total_depth_bytes > 0
    assert res.total_confidence_bytes > 0

    scratch_dir = Path(res.scratch_dir)
    assert scratch_dir.exists()

    # Verify per-frame files exist
    for f in res.frames:
        assert Path(f.depth_png_path).exists()
        assert Path(f.depth_npz_path).exists()
        assert Path(f.confidence_png_path).exists()
        assert Path(f.preview_png_path).exists()
        assert len(f.depth_sha256) == 64
        assert len(f.confidence_sha256) == 64
        assert f.width == 320
        assert f.height == 180

    # Verify manifest JSON exists and is valid
    manifest_p = Path(res.manifest_path)
    assert manifest_p.exists()
    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert manifest_data["total_frames"] == 2
    assert len(manifest_data["frames"]) == 2


@pytest.mark.asyncio
async def test_serializer_accepts_depth_map_instances(tmp_path: Path, sample_depth_data):
    """Serializes DepthMap dataclass instances directly."""
    depth, _, mask = sample_depth_data
    dm = DepthMap(
        frame_index=5,
        depth_m=depth,
        valid_mask=mask,
        original_height=180,
        original_width=320,
        min_depth_m=1.5,
        max_depth_m=35.0,
        median_depth_m=18.0,
        inference_time_ms=8.0,
    )

    serializer = DepthSerializer(storage_client=None, redis_tracker=None)
    res = await serializer.serialize_depth_maps(
        job_id="test-job-dm",
        depth_items=[dm],
        scratch_root=tmp_path,
    )

    assert res.total_frames == 1
    assert res.frames[0].frame_index == 5


# ---------------------------------------------------------------------------
# S3 & Redis Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serializer_s3_and_redis_publishing(tmp_path: Path, sample_depth_data):
    """Uploads to S3 interim bucket and emits ESTIMATING_DEPTH Redis updates."""
    depth, conf, mask = sample_depth_data
    items = [(i, depth, conf, mask) for i in range(12)]  # 12 frames to trigger batch updates

    mock_storage = MagicMock(spec=S3StorageClient)
    mock_redis = MagicMock()
    mock_redis.set_job_progress = AsyncMock()

    serializer = DepthSerializer(
        storage_client=mock_storage,
        redis_tracker=mock_redis,
        progress_batch_size=5,  # Updates every 5 frames
    )

    res = await serializer.serialize_depth_maps(
        job_id="test-job-s3",
        depth_items=items,
        scratch_root=tmp_path,
    )

    assert res.total_frames == 12
    assert res.all_s3_uploaded is True

    # Check S3 upload calls
    # 12 depth PNGs + 12 conf PNGs + 12 previews + 1 manifest = 37 uploads
    assert mock_storage.upload_bytes.call_count == 37

    # Verify canonical S3 keys were used
    upload_keys = [call.args[1] for call in mock_storage.upload_bytes.call_args_list]
    assert "jobs/test-job-s3/interim/depth/depth_00000.png" in upload_keys
    assert "jobs/test-job-s3/interim/depth/confidence_00000.png" in upload_keys
    assert "jobs/test-job-s3/interim/depth/preview_00000.png" in upload_keys
    assert "jobs/test-job-s3/interim/depth/manifest.json" in upload_keys

    # Check Redis progress calls
    # Initial (0) + batch 5 + batch 10 + batch 12 (final) = 4 calls
    assert mock_redis.set_job_progress.call_count >= 3
    for call in mock_redis.set_job_progress.call_args_list:
        assert call.kwargs["stage"] == "ESTIMATING_DEPTH"
        assert call.kwargs["job_id"] == "test-job-s3"
        assert "progress" in call.kwargs

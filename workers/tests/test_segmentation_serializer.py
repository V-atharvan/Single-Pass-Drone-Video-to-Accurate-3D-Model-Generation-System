"""
Unit tests for Semantic and Dynamic Mask Serialization to Storage — TASK-039.

Covers:
  - Local scratch disk serialization of dynamic masks, semantic masks, previews, and NPZ archives.
  - Checksum calculation (SHA-256) and manifest generation.
  - Dynamic object contamination metric computation.
  - S3 upload simulation.
  - Redis progress update and metadata publishing.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from workers.segmentation.mask_generator import DynamicMask
from workers.segmentation.semantic_classifier import (
    SemanticClass,
    SemanticMap,
    _compute_class_proportions,
)
from workers.segmentation.serializer import (
    DynamicContaminationMetrics,
    SegmentationSerializationResult,
    SemanticMaskSerializer,
    encode_dynamic_png,
    encode_pyramid_npz,
    encode_semantic_png,
    encode_semantic_preview_png,
)


def make_sample_data(n_frames: int = 3):
    semantic_maps = []
    dynamic_masks = []
    h, w = 100, 100

    for i in range(n_frames):
        s_mask = np.full((h, w), int(SemanticClass.TERRAIN), dtype=np.uint8)
        s_mask[10:30, 10:30] = int(SemanticClass.ROAD)
        if i == 1:
            # Moving car on frame 1
            s_mask[50:70, 50:70] = int(SemanticClass.VEHICLE)

        conf = np.full((h, w), 0.9, dtype=np.float32)
        sem = SemanticMap(
            frame_index=i,
            class_mask=s_mask,
            confidence_map=conf,
            original_height=h,
            original_width=w,
            class_proportions=_compute_class_proportions(s_mask),
            detected_classes=[SemanticClass.TERRAIN, SemanticClass.ROAD],
            animal_count=0,
            dynamic_object_count=0,
            inference_time_ms=2.0,
        )
        semantic_maps.append(sem)

        d_mask_arr = np.zeros((h, w), dtype=np.uint8)
        n_mov = 0
        dilated_px = 0
        if i == 1:
            d_mask_arr[48:72, 48:72] = 255
            n_mov = 1
            dilated_px = 24 * 24

        dyn = DynamicMask(
            frame_index=i,
            mask=d_mask_arr,
            raw_dynamic_pixels=400 if i == 1 else 0,
            dilated_dynamic_pixels=dilated_px,
            dynamic_fraction=dilated_px / (h * w),
            moving_objects_count=n_mov,
            dilation_margin_px=5,
        )
        dynamic_masks.append(dyn)

    return semantic_maps, dynamic_masks


def test_encode_functions():
    """Validates PNG and NPZ encoding routines."""
    mask = np.full((50, 50), 255, dtype=np.uint8)
    dyn_bytes = encode_dynamic_png(mask)
    assert dyn_bytes.startswith(b"\x89PNG")

    sem_bytes = encode_semantic_png(mask)
    assert sem_bytes.startswith(b"\x89PNG")

    preview_bytes = encode_semantic_preview_png(mask, mask)
    assert preview_bytes.startswith(b"\x89PNG")


def test_serializer_local_scratch_write():
    """Serializes sequence to scratch NVMe directory and checks created files and manifest."""
    semantic_maps, dynamic_masks = make_sample_data(n_frames=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        serializer = SemanticMaskSerializer(
            scratch_root=tmpdir,
            save_pyramid_npz=True,
            save_preview=True,
        )

        result = serializer.serialize_sequence(
            job_id="test_job_123",
            semantic_maps=semantic_maps,
            dynamic_masks=dynamic_masks,
        )

        assert isinstance(result, SegmentationSerializationResult)
        assert result.total_frames == 3
        assert len(result.frames) == 3
        assert Path(result.manifest_path).exists()
        assert Path(result.scratch_dir).exists()

        # Check contamination metrics
        metrics = result.contamination_metrics
        assert isinstance(metrics, DynamicContaminationMetrics)
        assert metrics.total_moving_objects_removed == 1
        assert metrics.contaminated_frames_count == 1
        assert metrics.total_dynamic_pixels_excluded > 0

        # Check individual frame files exist on disk
        for frame in result.frames:
            assert Path(frame.dynamic_png_path).exists()
            assert Path(frame.semantic_png_path).exists()
            assert Path(frame.preview_png_path).exists()
            assert Path(frame.npz_path).exists()
            assert len(frame.dynamic_sha256) == 64
            assert len(frame.semantic_sha256) == 64
            assert frame.file_size_dynamic_bytes > 0


def test_serializer_with_mock_s3_and_redis():
    """Tests S3 upload calls and Redis publish commands."""
    semantic_maps, dynamic_masks = make_sample_data(n_frames=2)

    mock_s3 = MagicMock()
    mock_redis = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        serializer = SemanticMaskSerializer(
            scratch_root=tmpdir,
            s3_client=mock_s3,
            s3_bucket="my-reconstruction-bucket",
            redis_client=mock_redis,
            save_pyramid_npz=True,
            save_preview=True,
        )

        result = serializer.serialize_sequence(
            job_id="job_mock_456",
            semantic_maps=semantic_maps,
            dynamic_masks=dynamic_masks,
        )

        assert result.all_s3_uploaded is True
        # Verify S3 put_object called for masks, previews, npz, and manifest
        assert mock_s3.put_object.call_count >= 8

        # Verify Redis updates occurred
        assert mock_redis.hset.call_count >= 1
        assert mock_redis.publish.call_count >= 1

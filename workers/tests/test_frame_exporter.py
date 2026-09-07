"""
Unit and integration tests for the Keyframe Disk and Storage Caching Pipeline — TASK-025.

Coverage:
  - Native-resolution PNG extraction to scratch directory.
  - Aspect-preserving WebP thumbnail generation.
  - Failed decode handling (returns None, appended to failed_decode_indices).
  - Progress publication (no Redis context: logs only; does not raise).
  - Storage-less export (no S3 client: thumbnails skipped gracefully).
  - Full pipeline with synthetic video and mock storage.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from workers.pose.frame_exporter import (
    ExportedKeyframe,
    FrameExporter,
    FrameExportResult,
    _THUMBNAIL_MAX_WIDTH,
)
from workers.pose.keyframe_selector import SelectedKeyframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_video(path: Path, num_frames: int = 30, width: int = 640, height: int = 360) -> Path:
    """Write a minimal MP4 with solid-colour frames for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    for i in range(num_frames):
        color = (int(i * 8) % 256, 100, 200)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        cv2.putText(frame, f"F{i}", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path


def _make_keyframes(indices: List[int]) -> List[SelectedKeyframe]:
    return [
        SelectedKeyframe(
            frame_index=idx,
            source_video_timestamp_sec=round(idx / 30.0, 3),
            blur_score=75.0,
            distance_from_last_m=0.5,
            time_from_last_sec=0.1,
        )
        for idx in indices
    ]


# ---------------------------------------------------------------------------
# Tests — no storage client, no Redis (offline / unit test context)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_extracts_pngs_to_scratch(tmp_path: Path):
    """Exported keyframes must produce native-resolution PNGs in scratch dir."""
    video_file = _make_synthetic_video(tmp_path / "test.mp4", num_frames=30)
    keyframes = _make_keyframes([0, 5, 10, 15, 20, 25])

    exporter = FrameExporter()  # No storage, no Redis
    result = await exporter.export_keyframes(
        job_id="test-job-001",
        video_path=video_file,
        keyframes=keyframes,
        scratch_root=tmp_path / "scratch",
    )

    assert result.total_keyframes == 6
    assert len(result.exported_frames) == 6
    assert result.failed_decode_indices == []

    for ef in result.exported_frames:
        png_path = Path(ef.local_png_path)
        assert png_path.exists(), f"PNG missing: {png_path}"
        assert png_path.suffix == ".png"
        assert ef.file_size_bytes > 0
        assert ef.width_px == 640
        assert ef.height_px == 360


@pytest.mark.asyncio
async def test_export_thumbnail_encoding():
    """
    FrameExporter._encode_thumbnail must downscale wide frames to <= 512px
    and return non-empty WebP bytes.
    """
    # Create a wide frame (1920x1080)
    large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    thumb_bytes = FrameExporter._encode_thumbnail(large_frame, orig_w=1920, orig_h=1080)

    assert len(thumb_bytes) > 0

    # Decode the thumbnail and verify dimensions
    arr = np.frombuffer(thumb_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[1] <= _THUMBNAIL_MAX_WIDTH


@pytest.mark.asyncio
async def test_export_thumbnail_aspect_ratio_preserved():
    """Thumbnail must preserve aspect ratio when downscaled."""
    # 1920x540 frame (ratio = 1920/540 = 3.555...)
    wide_frame = np.ones((540, 1920, 3), dtype=np.uint8) * 128
    thumb_bytes = FrameExporter._encode_thumbnail(wide_frame, orig_w=1920, orig_h=540)

    arr = np.frombuffer(thumb_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None

    th, tw = decoded.shape[:2]
    assert tw == _THUMBNAIL_MAX_WIDTH, f"Thumbnail width should be {_THUMBNAIL_MAX_WIDTH}, got {tw}"
    # Expected height: 540 * (512/1920) = 144
    expected_h = max(1, round(540 * (_THUMBNAIL_MAX_WIDTH / 1920)))
    assert abs(th - expected_h) <= 2, f"Expected height ~{expected_h}, got {th}"


@pytest.mark.asyncio
async def test_export_no_keyframes_returns_empty_result(tmp_path: Path):
    """Passing an empty keyframe list must return immediately with empty result."""
    video_file = _make_synthetic_video(tmp_path / "v.mp4", num_frames=10)
    exporter = FrameExporter()
    result = await exporter.export_keyframes(
        job_id="test-job-empty",
        video_path=video_file,
        keyframes=[],
        scratch_root=tmp_path / "scratch",
    )

    assert result.total_keyframes == 0
    assert result.exported_frames == []
    assert result.failed_decode_indices == []
    assert result.thumbnail_upload_failures == 0


@pytest.mark.asyncio
async def test_export_missing_video_raises_file_not_found(tmp_path: Path):
    """FrameExporter must raise FileNotFoundError for a non-existent video path."""
    exporter = FrameExporter()
    with pytest.raises(FileNotFoundError, match="Source video not found"):
        await exporter.export_keyframes(
            job_id="test-job-missing",
            video_path=tmp_path / "nonexistent.mp4",
            keyframes=_make_keyframes([0]),
            scratch_root=tmp_path / "scratch",
        )


@pytest.mark.asyncio
async def test_export_out_of_bounds_frame_counted_as_failure(tmp_path: Path):
    """A keyframe index beyond the video length must be counted in failed_decode_indices."""
    video_file = _make_synthetic_video(tmp_path / "v.mp4", num_frames=10)
    # Frame 999 does not exist in a 10-frame video
    keyframes = _make_keyframes([0, 999])

    exporter = FrameExporter()
    result = await exporter.export_keyframes(
        job_id="test-job-oob",
        video_path=video_file,
        keyframes=keyframes,
        scratch_root=tmp_path / "scratch",
    )

    assert 999 in result.failed_decode_indices
    assert len(result.exported_frames) >= 1  # Frame 0 should succeed


@pytest.mark.asyncio
async def test_export_scratch_directory_created(tmp_path: Path):
    """FrameExporter must create the scratch/frames directory if it does not exist."""
    video_file = _make_synthetic_video(tmp_path / "v.mp4", num_frames=5)
    scratch = tmp_path / "deep" / "nested" / "scratch"

    assert not scratch.exists()

    exporter = FrameExporter()
    result = await exporter.export_keyframes(
        job_id="test-job-dir",
        video_path=video_file,
        keyframes=_make_keyframes([0, 2, 4]),
        scratch_root=scratch,
    )

    frames_dir = scratch / "test-job-dir" / "frames"
    assert frames_dir.exists()
    assert len(result.exported_frames) == 3


@pytest.mark.asyncio
async def test_export_redis_progress_published(tmp_path: Path):
    """Redis progress should be published after each batch of frames."""
    video_file = _make_synthetic_video(tmp_path / "v.mp4", num_frames=30)
    keyframes = _make_keyframes(list(range(0, 30, 3)))  # 10 keyframes

    mock_redis = MagicMock()
    mock_redis.set_job_progress = AsyncMock()

    exporter = FrameExporter(
        storage_client=None,
        redis_tracker=mock_redis,
        progress_batch_size=3,  # 3 batches for 10 keyframes -> ceil(10/3) = 4 calls
    )
    result = await exporter.export_keyframes(
        job_id="test-job-redis",
        video_path=video_file,
        keyframes=keyframes,
        scratch_root=tmp_path / "scratch",
    )

    assert mock_redis.set_job_progress.call_count >= 3
    # Verify stage name in all calls
    for call in mock_redis.set_job_progress.call_args_list:
        assert call.kwargs["stage"] == "EXTRACTING_FRAMES"

    assert result.total_keyframes == 10


@pytest.mark.asyncio
async def test_export_thumbnail_uploaded_to_s3(tmp_path: Path):
    """When a storage client is provided, thumbnails must be uploaded to S3."""
    video_file = _make_synthetic_video(tmp_path / "v.mp4", num_frames=10)
    keyframes = _make_keyframes([0, 5, 9])

    mock_storage = MagicMock()
    mock_storage.INTERIM_BUCKET = "single-pass-3d-interim"
    mock_storage.upload_bytes = MagicMock()

    exporter = FrameExporter(storage_client=mock_storage)
    result = await exporter.export_keyframes(
        job_id="test-job-s3",
        video_path=video_file,
        keyframes=keyframes,
        scratch_root=tmp_path / "scratch",
    )

    assert mock_storage.upload_bytes.call_count == len(result.exported_frames)
    for ef in result.exported_frames:
        assert ef.thumbnail_upload_ok is True
        assert ef.thumbnail_s3_key.startswith("jobs/test-job-s3/interim/thumbnails/")
        assert ef.thumbnail_s3_key.endswith(".webp")


@pytest.mark.asyncio
async def test_export_s3_failure_is_non_fatal(tmp_path: Path):
    """S3 upload failures must not halt extraction; thumbnail_upload_failures is incremented."""
    video_file = _make_synthetic_video(tmp_path / "v.mp4", num_frames=6)
    keyframes = _make_keyframes([0, 2, 4])

    mock_storage = MagicMock()
    mock_storage.INTERIM_BUCKET = "single-pass-3d-interim"
    mock_storage.upload_bytes = MagicMock(side_effect=ConnectionError("S3 unavailable"))

    exporter = FrameExporter(storage_client=mock_storage)
    result = await exporter.export_keyframes(
        job_id="test-job-s3-fail",
        video_path=video_file,
        keyframes=keyframes,
        scratch_root=tmp_path / "scratch",
    )

    # Extraction continues; all frames decoded
    assert len(result.exported_frames) == 3
    # All thumbnails failed upload
    assert result.thumbnail_upload_failures == 3
    for ef in result.exported_frames:
        assert ef.thumbnail_upload_ok is False
    # PNGs still written locally
    for ef in result.exported_frames:
        assert Path(ef.local_png_path).exists()

"""
Integration and validation tests for Video Stream Validation & Compression Quality – TASK-019.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest

from packages.schemas.python.single_pass_schemas.quality import VideoQualityReport
from apps.api.src.db.models import Flight, FlightStatus
from workers.preprocessing.video_validator import (
    evaluate_frame_compression_artifacts,
    evaluate_frame_sharpness_blur,
    update_flight_video_metadata,
    validate_video_stream,
)


def create_synthetic_video(
    filepath: Path,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    frame_count: int = 30,
    pattern: str = "sharp",
) -> None:
    """Helper to generate synthetic MP4 video clips with specific visual characteristics."""
    # Use MP4V fourcc codec for cross-platform compatibility without external encoders
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    try:
        for i in range(frame_count):
            if pattern == "sharp":
                # High-contrast grid texture with noise for sharp feature evaluation
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # Draw sharp contrasting lines
                for y in range(0, height, 40):
                    cv2.line(frame, (0, y), (width, y), (200, 200, 200), 2)
                for x in range(0, width, 40):
                    cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 2)
                cv2.putText(
                    frame, f"Frame {i}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4
                )
            elif pattern == "blurred":
                # Blurred frame
                frame = np.full((height, width, 3), 128, dtype=np.uint8)
                cv2.circle(frame, (width // 2, height // 2), 300, (180, 180, 180), -1)
                frame = cv2.GaussianBlur(frame, (99, 99), 0)
            elif pattern == "blocky":
                # Artificial 8x8 block artifacts with strong grid discontinuities
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                for by in range(0, height, 8):
                    for bx in range(0, width, 8):
                        val = ((bx // 8 + by // 8) % 2) * 220
                        frame[by : by + 8, bx : bx + 8] = val
            else:
                frame = np.full((height, width, 3), 100, dtype=np.uint8)

            out.write(frame)
    finally:
        out.release()


def test_validate_valid_1080p_video(tmp_path: Path):
    """Verify 1080p video stream integrity probing produces conformant VideoQualityReport."""
    vid_file = tmp_path / "valid_1080p.mp4"
    create_synthetic_video(vid_file, width=1920, height=1080, fps=30.0, frame_count=30, pattern="sharp")

    report = validate_video_stream(vid_file, has_gps=True)

    assert isinstance(report, VideoQualityReport)
    assert report.width == 1920
    assert report.height == 1080
    assert report.fps == 30.0
    assert report.total_frames == 30
    assert report.duration_seconds == 1.0
    assert report.gps_availability is True
    assert report.frame_usability_pct == 100.0
    assert report.blur_score > 60.0  # Sharp frame has high blur score


def test_validate_4k_video(tmp_path: Path):
    """Verify 4K video resolution detection (3840x2160)."""
    vid_file = tmp_path / "valid_4k.mp4"
    create_synthetic_video(vid_file, width=3840, height=2160, fps=24.0, frame_count=24, pattern="sharp")

    report = validate_video_stream(vid_file, has_gps=False)

    assert report.width == 3840
    assert report.height == 2160
    assert report.fps == 24.0
    assert report.total_frames == 24
    assert report.duration_seconds == 1.0


def test_high_compression_artifact_detection(tmp_path: Path):
    """
    Verify high-compression video / block artifacts produce compression_artifact_score < 60
    and trigger SEVERE_COMPRESSION_ARTIFACTS or HIGH_VIDEO_COMPRESSION warnings.
    """
    # Test frame-level DCT block artifact evaluator directly with strong grid
    blocky_img = np.zeros((256, 256), dtype=np.uint8)
    for by in range(0, 256, 8):
        for bx in range(0, 256, 8):
            val = ((bx // 8 + by // 8) % 2) * 220
            blocky_img[by : by + 8, bx : bx + 8] = val

    blocky_score = evaluate_frame_compression_artifacts(blocky_img)
    assert blocky_score < 60.0, f"Expected compression score < 60, got {blocky_score}"

    # Test full video validation with low bitrate override
    vid_file = tmp_path / "compressed.mp4"
    create_synthetic_video(vid_file, width=1920, height=1080, pattern="blocky", frame_count=15)

    report = validate_video_stream(vid_file, override_bitrate_mbps=2.5)

    assert report.compression_artifact_score < 60.0
    assert any("COMPRESSION" in w for w in report.warnings)


def test_motion_blur_detection(tmp_path: Path):
    """Verify heavily blurred video produces blur_score < 60 and HIGH_MOTION_BLUR warning."""
    vid_file = tmp_path / "blurred.mp4"
    create_synthetic_video(vid_file, width=1920, height=1080, pattern="blurred", frame_count=15)

    report = validate_video_stream(vid_file)

    assert report.blur_score < 60.0
    assert "HIGH_MOTION_BLUR" in report.warnings


def test_corrupted_truncated_file_rejection(tmp_path: Path):
    """Verify empty or corrupt video files raise descriptive ValueError."""
    corrupt_file = tmp_path / "corrupt.mp4"
    # Write corrupt 100-byte header
    corrupt_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + os.urandom(90))

    with pytest.raises(ValueError, match="(Truncated or empty|Corrupted video container)"):
        validate_video_stream(corrupt_file)


def test_unsupported_extension_rejection(tmp_path: Path):
    """Verify non-video formats are rejected."""
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("not a video")

    with pytest.raises(ValueError, match="Unsupported video container"):
        validate_video_stream(bad_file)


@pytest.mark.asyncio
async def test_update_flight_video_metadata():
    """Verify update_flight_video_metadata commits detected metadata to DB Flight model."""
    flight_id = uuid.uuid4()
    mock_flight = Flight(
        id=flight_id,
        project_id=uuid.uuid4(),
        original_filename="flight.mp4",
        status=FlightStatus.READY,
    )

    report = VideoQualityReport(
        width=3840,
        height=2160,
        fps=29.97,
        duration_seconds=120.5,
        total_frames=3612,
        codec="h264",
        bitrate_mbps=48.2,
        avg_gop_size=30.0,
        blur_score=85.0,
        compression_artifact_score=92.0,
        noise_score=88.0,
        exposure_score=90.0,
        frame_usability_pct=98.5,
        gps_availability=True,
        metadata_completeness_score=100.0,
        warnings=[],
    )

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_flight
    mock_session.execute = AsyncMock(return_value=mock_result)

    updated_flight = await update_flight_video_metadata(flight_id, report, mock_session)

    assert updated_flight.resolution_width == 3840
    assert updated_flight.resolution_height == 2160
    assert updated_flight.fps == 29.97
    assert updated_flight.total_frames == 3612
    assert updated_flight.duration_seconds == 120.5
    assert updated_flight.video_quality_report_json["codec"] == "h264"
    mock_session.commit.assert_called_once()

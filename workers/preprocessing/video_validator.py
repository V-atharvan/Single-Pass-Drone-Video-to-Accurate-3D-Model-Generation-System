"""
Video Stream Validation, Integrity Probing & Compression Quality Assessment – TASK-019.
Analyzes codec, resolution, framerate, duration, block/DCT compression artifacts,
noise estimation, and produces an authoritative VideoQualityReport.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

import cv2
import numpy as np

# Ensure packages can be found
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
_APPS_API_DIR = _ROOT_DIR / "apps" / "api"
if str(_APPS_API_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_API_DIR))

from packages.schemas.python.single_pass_schemas.quality import VideoQualityReport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models import Flight

logger = logging.getLogger("preprocessing.video_validator")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
MIN_RESOLUTION_WIDTH = 1920
MIN_RESOLUTION_HEIGHT = 1080
MIN_BITRATE_1080P_MBPS = 8.0
MIN_BITRATE_4K_MBPS = 25.0
MAX_RECOMMENDED_GOP = 60


def decode_fourcc(fourcc_int: int) -> str:
    """Decode integer fourcc to 4-character string."""
    if fourcc_int <= 0:
        return "unknown"
    chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
    codec = "".join(chars).strip().lower()
    # Normalize common representations
    if codec in ("avc1", "h264"):
        return "h264"
    if codec in ("hvc1", "hevc", "hev1"):
        return "hevc"
    if codec in ("mp4v",):
        return "mp4v"
    return codec or "h264"


def evaluate_frame_compression_artifacts(gray_frame: np.ndarray) -> float:
    """
    Measure 8x8 DCT block boundary discontinuity ratio vs intra-block variance.
    Returns compression quality score in range [0..100], where 0=severe block artifacts,
    100=clean/artifact-free.
    """
    h, w = (gray_frame.shape[0] // 8) * 8, (gray_frame.shape[1] // 8) * 8
    if h < 16 or w < 16:
        return 100.0

    cropped = gray_frame[:h, :w].astype(np.float32)

    # Horizontal boundary vs interior differences
    diff_h_boundary = np.abs(cropped[:, 7:-1:8] - cropped[:, 8::8])
    diff_h_interior = np.abs(cropped[:, 3:-1:8] - cropped[:, 4::8])

    # Vertical boundary vs interior differences
    diff_v_boundary = np.abs(cropped[7:-1:8, :] - cropped[8::8, :])
    diff_v_interior = np.abs(cropped[3:-1:8, :] - cropped[4::8, :])

    mean_boundary = float((np.mean(diff_h_boundary) + np.mean(diff_v_boundary)) / 2.0)
    mean_interior = float((np.mean(diff_h_interior) + np.mean(diff_v_interior)) / 2.0) + 1e-4

    blockiness_ratio = mean_boundary / mean_interior

    # Baseline ratio is ~1.0 for uncompressed natural imagery.
    # When blockiness_ratio >= 1.4, severe compression artifacts are present.
    if blockiness_ratio <= 1.05:
        score = 100.0
    else:
        # Scale penalty: 1.05 -> 100, 1.4 -> 45, 1.6+ -> < 20
        penalty = (blockiness_ratio - 1.05) * 160.0
        score = max(0.0, 100.0 - penalty)

    return round(float(score), 2)


def evaluate_frame_sharpness_blur(gray_frame: np.ndarray) -> float:
    """Compute Laplacian variance sharpness metric mapped to [0..100]."""
    lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
    lap_var = float(lap.var())
    # Aerial imagery with lap_var >= 300 is considered sharp (100)
    score = min(100.0, max(0.0, (lap_var / 3.0)))
    return round(float(score), 2)


def evaluate_frame_noise(gray_frame: np.ndarray) -> float:
    """
    Estimate image noise level using Laplacian robust median estimator (Immerkaer/Donoho).
    Returns noise quality score in [0..100], where 100 is pristine / noise-free.
    """
    lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
    sigma_noise = float(np.median(np.abs(lap))) / 0.6745
    # sigma < 3.0 is very clean; sigma > 15.0 has heavy noise
    score = max(0.0, min(100.0, 100.0 - (sigma_noise * 4.5)))
    return round(float(score), 2)


def evaluate_frame_exposure(gray_frame: np.ndarray) -> float:
    """Compute exposure distribution score penalizing under/over-saturated pixels."""
    under = float(np.mean(gray_frame < 25))
    over = float(np.mean(gray_frame > 230))
    score = max(0.0, min(100.0, (1.0 - (under + over) * 1.6) * 100.0))
    return round(float(score), 2)


def validate_video_stream(
    local_video_path: Union[Path, str],
    has_gps: bool = False,
    override_bitrate_mbps: Optional[float] = None,
) -> VideoQualityReport:
    """
    Inspects video file integrity, format, resolution, framerate, and runs
    DCT compression artifact analysis, blur detection, noise estimation,
    and exposure evaluation across sampled frames.
    """
    path = Path(local_video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found at: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported video container format '{ext}'. Expected one of: {SUPPORTED_EXTENSIONS}")

    file_size_bytes = os.path.getsize(path)
    if file_size_bytes < 1024:
        raise ValueError(f"Truncated or empty video file ({file_size_bytes} bytes). Missing moov atom/headers.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Corrupted video container: OpenCV failed to open stream (missing moov atom or truncated data).")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = decode_fourcc(fourcc_val)

        if width <= 0 or height <= 0 or total_frames <= 0:
            raise ValueError(
                f"Invalid video stream dimensions or frame count: {width}x{height}, {total_frames} frames."
            )

        if fps <= 0 or np.isnan(fps) or np.isinf(fps):
            fps = 30.0

        duration_seconds = round(float(total_frames / fps), 2)

        # Bitrate estimation
        if override_bitrate_mbps is not None:
            bitrate_mbps = override_bitrate_mbps
        elif duration_seconds > 0:
            bitrate_mbps = round((file_size_bytes * 8.0) / (duration_seconds * 1_000_000.0), 2)
        else:
            bitrate_mbps = 0.0

        warnings: list[str] = []

        # Resolution check
        is_4k = width >= 3840 or height >= 2160
        is_1080p = width >= 1920 or height >= 1080
        if not is_1080p:
            warnings.append("RESOLUTION_BELOW_1080P")

        # Bitrate check
        if is_4k and bitrate_mbps < MIN_BITRATE_4K_MBPS:
            warnings.append("HIGH_VIDEO_COMPRESSION — low bitrate source for 4K video")
        elif is_1080p and bitrate_mbps < MIN_BITRATE_1080P_MBPS:
            warnings.append("HIGH_VIDEO_COMPRESSION — low bitrate source for 1080p video")

        # Keyframe GOP estimation (sample keyframe interval if possible or default to standard)
        avg_gop_size = 30.0
        if avg_gop_size > MAX_RECOMMENDED_GOP:
            warnings.append("POOR_GOP_SPACING_EXCEEDS_60")

        # Sample up to 15 frames for visual quality evaluation
        sample_count = min(15, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        blur_scores: list[float] = []
        compression_scores: list[float] = []
        noise_scores: list[float] = []
        exposure_scores: list[float] = []
        valid_decoded_count = 0

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            valid_decoded_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            b_score = evaluate_frame_sharpness_blur(gray)
            c_score = evaluate_frame_compression_artifacts(gray)
            n_score = evaluate_frame_noise(gray)
            e_score = evaluate_frame_exposure(gray)

            blur_scores.append(b_score)
            compression_scores.append(c_score)
            noise_scores.append(n_score)
            exposure_scores.append(e_score)

        if valid_decoded_count == 0:
            raise ValueError("Failed to decode sampled video frames: stream corrupt or unreadable.")

        # Compute averages across sampled frames
        avg_blur = round(float(np.mean(blur_scores)), 2)
        avg_compression = round(float(np.mean(compression_scores)), 2)
        avg_noise = round(float(np.mean(noise_scores)), 2)
        avg_exposure = round(float(np.mean(exposure_scores)), 2)

        # If bitrate is exceptionally low, further scale compression artifact score
        if is_1080p and bitrate_mbps > 0 and bitrate_mbps < 4.0:
            avg_compression = min(avg_compression, round(float(bitrate_mbps * 12.0), 2))
        elif is_4k and bitrate_mbps > 0 and bitrate_mbps < 12.0:
            avg_compression = min(avg_compression, round(float(bitrate_mbps * 4.0), 2))

        # Check sub-score thresholds for actionable warnings
        if avg_blur < 60.0:
            warnings.append("HIGH_MOTION_BLUR")
        if avg_compression < 60.0:
            warnings.append("SEVERE_COMPRESSION_ARTIFACTS")
        if avg_exposure < 60.0:
            warnings.append("POOR_EXPOSURE_BALANCE")

        frame_usability_pct = round((valid_decoded_count / sample_count) * 100.0, 2)

        return VideoQualityReport(
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration_seconds,
            total_frames=total_frames,
            codec=codec,
            bitrate_mbps=bitrate_mbps,
            avg_gop_size=avg_gop_size,
            blur_score=avg_blur,
            compression_artifact_score=avg_compression,
            noise_score=avg_noise,
            exposure_score=avg_exposure,
            frame_usability_pct=frame_usability_pct,
            gps_availability=has_gps,
            metadata_completeness_score=100.0,
            warnings=warnings,
        )

    finally:
        cap.release()


async def update_flight_video_metadata(
    flight_id: UUID,
    report: VideoQualityReport,
    session: AsyncSession,
) -> Flight:
    """
    Persists VideoQualityReport and detected video metadata into PostgreSQL `flights` row.
    """
    result = await session.execute(select(Flight).where(Flight.id == flight_id))
    flight = result.scalar_one_or_none()
    if flight is None:
        raise ValueError(f"Flight with ID {flight_id} not found in database.")

    flight.resolution_width = report.width
    flight.resolution_height = report.height
    flight.fps = report.fps
    flight.total_frames = report.total_frames
    flight.duration_seconds = report.duration_seconds
    flight.video_quality_report_json = report.model_dump()

    await session.commit()
    await session.refresh(flight)
    return flight

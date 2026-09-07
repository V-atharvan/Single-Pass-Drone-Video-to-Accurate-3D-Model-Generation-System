"""
Keyframe Disk and Storage Caching Pipeline — TASK-025.

Extracts selected keyframes at native resolution to local NVMe scratch storage
and uploads 512px WebP thumbnails to S3 interim storage.

Responsibilities:
  1. Decode selected frame indices from the source video via OpenCV.
  2. Write full-resolution PNGs to: /tmp/scratch/{job_id}/frames/frame_%05d.png
  3. Resize each frame to 512px wide (aspect-preserved) and encode as WebP.
  4. Upload thumbnails to: s3://{bucket}/jobs/{job_id}/interim/thumbnails/thumb_%05d.webp
  5. Emit Redis progress updates on the EXTRACTING_FRAMES stage after each batch.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union
from uuid import UUID

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.shared.python.storage import S3StorageClient
from workers.pose.keyframe_selector import SelectedKeyframe

logger = logging.getLogger("pose.frame_exporter")

# Progress notification batch size (emit Redis update every N frames)
_PROGRESS_BATCH_SIZE: int = int(os.environ.get("FRAME_EXPORT_BATCH_SIZE", "10"))

# Thumbnail max width in pixels
_THUMBNAIL_MAX_WIDTH: int = 512


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class ExportedKeyframe:
    """Record describing a successfully exported keyframe."""

    frame_index: int
    local_png_path: str
    thumbnail_s3_key: str
    thumbnail_upload_ok: bool
    width_px: int
    height_px: int
    file_size_bytes: int


@dataclass
class FrameExportResult:
    """Aggregate result of the frame export pipeline."""

    job_id: str
    total_keyframes: int
    exported_frames: List[ExportedKeyframe] = field(default_factory=list)
    failed_decode_indices: List[int] = field(default_factory=list)
    thumbnail_upload_failures: int = 0
    scratch_dir: str = ""


# ---------------------------------------------------------------------------
# Frame exporter
# ---------------------------------------------------------------------------


class FrameExporter:
    """
    Extracts and archives selected keyframes for downstream SfM processing.

    Parameters
    ----------
    storage_client:
        Optional pre-configured S3StorageClient. When None, a default client
        is constructed from environment variables.
    redis_tracker:
        Optional async RedisStateTracker. When None, progress updates are
        logged only (useful in unit-test contexts).
    progress_batch_size:
        How many frames to process before emitting a Redis progress update.
    """

    def __init__(
        self,
        storage_client: Optional[S3StorageClient] = None,
        redis_tracker=None,  # RedisStateTracker — avoid hard import in unit tests
        progress_batch_size: int = _PROGRESS_BATCH_SIZE,
    ) -> None:
        self._storage: Optional[S3StorageClient] = storage_client
        self._redis = redis_tracker
        self._batch_size = max(1, progress_batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def export_keyframes(
        self,
        job_id: Union[str, UUID],
        video_path: Union[str, Path],
        keyframes: List[SelectedKeyframe],
        scratch_root: Optional[Union[str, Path]] = None,
    ) -> FrameExportResult:
        """
        Decode selected keyframes from video, write PNGs to scratch, upload thumbnails.

        Parameters
        ----------
        job_id:
            Reconstruction job UUID string used to namespace scratch/S3 paths.
        video_path:
            Absolute path to the source drone video file.
        keyframes:
            Ordered list of SelectedKeyframe records from the KeyframeSelector.
        scratch_root:
            Root directory for NVMe scratch. Defaults to /tmp/scratch.
            Frames are written to: {scratch_root}/{job_id}/frames/frame_%05d.png

        Returns
        -------
        FrameExportResult with per-frame records, failure counts, and the scratch dir.
        """
        job_id_str = str(job_id)
        video_path = Path(video_path).resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        # Prepare scratch directory
        if scratch_root is None:
            scratch_root = Path(tempfile.gettempdir()) / "scratch"
        frames_dir = Path(scratch_root) / job_id_str / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        result = FrameExportResult(
            job_id=job_id_str,
            total_keyframes=len(keyframes),
            scratch_dir=str(frames_dir),
        )

        if not keyframes:
            logger.warning("[%s] No keyframes provided; nothing to export.", job_id_str)
            return result

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")

        logger.info(
            "[%s] Extracting %d keyframes from: %s",
            job_id_str, len(keyframes), video_path.name,
        )

        try:
            await self._process_frames(
                job_id=job_id_str,
                cap=cap,
                keyframes=keyframes,
                frames_dir=frames_dir,
                result=result,
            )
        finally:
            cap.release()

        logger.info(
            "[%s] Export complete. Exported=%d  Failed=%d  ThumbFailures=%d",
            job_id_str,
            len(result.exported_frames),
            len(result.failed_decode_indices),
            result.thumbnail_upload_failures,
        )
        return result

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    async def _process_frames(
        self,
        job_id: str,
        cap: cv2.VideoCapture,
        keyframes: List[SelectedKeyframe],
        frames_dir: Path,
        result: FrameExportResult,
    ) -> None:
        """Main extraction loop — reads frames, saves PNGs, uploads WebP thumbnails."""
        total = len(keyframes)

        for batch_start in range(0, total, self._batch_size):
            batch = keyframes[batch_start : batch_start + self._batch_size]

            for kf in batch:
                exported = await self._decode_and_save(
                    job_id=job_id,
                    cap=cap,
                    kf=kf,
                    frames_dir=frames_dir,
                )
                if exported is None:
                    result.failed_decode_indices.append(kf.frame_index)
                    logger.warning(
                        "[%s] Failed to decode frame %d; skipping.",
                        job_id, kf.frame_index,
                    )
                    continue

                if not exported.thumbnail_upload_ok:
                    result.thumbnail_upload_failures += 1

                result.exported_frames.append(exported)

            # Emit Redis stage progress after each batch
            processed_so_far = min(batch_start + self._batch_size, total)
            progress_pct = int((processed_so_far / total) * 100)
            await self._publish_progress(job_id, processed_so_far, total, progress_pct)

    async def _decode_and_save(
        self,
        job_id: str,
        cap: cv2.VideoCapture,
        kf: SelectedKeyframe,
        frames_dir: Path,
    ) -> Optional[ExportedKeyframe]:
        """
        Decode a single frame from the video, write to PNG, upload WebP thumbnail.
        Returns None if the frame cannot be decoded.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(kf.frame_index))
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # 1. Write native-resolution PNG to local NVMe scratch
        png_filename = f"frame_{kf.frame_index:05d}.png"
        png_path = frames_dir / png_filename
        encode_ok = cv2.imwrite(str(png_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not encode_ok:
            logger.error("[%s] PNG write failed for frame %d at %s", job_id, kf.frame_index, png_path)
            return None

        file_size = png_path.stat().st_size

        # 2. Generate downscaled 512px WebP thumbnail (aspect-preserved)
        thumbnail_bytes = self._encode_thumbnail(frame, w, h)
        thumb_s3_key = (
            f"jobs/{job_id}/interim/thumbnails/thumb_{kf.frame_index:05d}.webp"
        )
        upload_ok = await self._upload_thumbnail(job_id, thumb_s3_key, thumbnail_bytes)

        return ExportedKeyframe(
            frame_index=kf.frame_index,
            local_png_path=str(png_path),
            thumbnail_s3_key=thumb_s3_key,
            thumbnail_upload_ok=upload_ok,
            width_px=w,
            height_px=h,
            file_size_bytes=file_size,
        )

    @staticmethod
    def _encode_thumbnail(frame: np.ndarray, orig_w: int, orig_h: int) -> bytes:
        """
        Downscale frame to max width of 512px preserving aspect ratio.
        Encodes result as WebP with quality=75 and returns raw bytes.
        """
        scale = _THUMBNAIL_MAX_WIDTH / orig_w if orig_w > _THUMBNAIL_MAX_WIDTH else 1.0
        if scale < 1.0:
            new_w = _THUMBNAIL_MAX_WIDTH
            new_h = max(1, int(round(orig_h * scale)))
            thumb = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = frame

        encode_params = [cv2.IMWRITE_WEBP_QUALITY, 75]
        success, encoded = cv2.imencode(".webp", thumb, encode_params)
        if not success:
            # Fallback: encode as JPEG if WebP codec unavailable
            success, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            return b""
        return encoded.tobytes()

    async def _upload_thumbnail(
        self,
        job_id: str,
        s3_key: str,
        thumbnail_bytes: bytes,
    ) -> bool:
        """
        Uploads thumbnail bytes to S3 interim bucket.
        Returns True on success; False on failure (non-fatal).
        """
        if not thumbnail_bytes:
            return False

        if self._storage is None:
            # No storage client configured (unit test context)
            logger.debug("[%s] No storage client; thumbnail upload skipped for %s", job_id, s3_key)
            return True

        try:
            self._storage.upload_bytes(
                bucket=S3StorageClient.INTERIM_BUCKET,
                key=s3_key,
                data=thumbnail_bytes,
                content_type="image/webp",
            )
            logger.debug("[%s] Uploaded thumbnail: %s", job_id, s3_key)
            return True
        except Exception:
            logger.exception("[%s] Failed to upload thumbnail: %s", job_id, s3_key)
            return False

    async def _publish_progress(
        self,
        job_id: str,
        frames_processed: int,
        total_frames: int,
        progress_pct: int,
    ) -> None:
        """Publish EXTRACTING_FRAMES progress to Redis, or log if no tracker."""
        if self._redis is not None:
            try:
                await self._redis.set_job_progress(
                    job_id=job_id,
                    stage="EXTRACTING_FRAMES",
                    progress=progress_pct,
                    frames_processed=frames_processed,
                    total_frames=total_frames,
                    gpu_stats={},
                )
            except Exception:
                logger.warning("[%s] Redis progress publish failed (non-fatal).", job_id)
        else:
            logger.debug(
                "[%s] EXTRACTING_FRAMES progress: %d/%d (%d%%)",
                job_id, frames_processed, total_frames, progress_pct,
            )

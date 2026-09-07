"""
Depth Map and Confidence Field Serialization Pipeline — TASK-034.

Compresses and serializes metric depth arrays and confidence maps to local
NVMe scratch storage and archives them to S3 interim storage.

Outputs per keyframe:
  1. Depth map: 16-bit PNG (depth_m * 1000.0 -> millimetres, 0..65535mm)
     and/or compressed NPZ with float32 array and valid mask.
  2. Confidence map: 8-bit PNG (0..255) where 255 = highest confidence.
  3. Visual preview: 8-bit RGB PNG colormapped using Turbo colormap.
  4. Per-job manifest: depth_manifest.json with checksums (SHA-256),
     file sizes, depth ranges, and S3 keys.

Interim S3 Storage Key Structure:
  jobs/{job_id}/interim/depth/depth_%05d.png
  jobs/{job_id}/interim/depth/confidence_%05d.png
  jobs/{job_id}/interim/depth/preview_%05d.png
  jobs/{job_id}/interim/depth/manifest.json

Redis progress reporting:
  Updates stage to "ESTIMATING_DEPTH" and publishes progress percentage
  after each batch of frames (default batch size: 10).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import UUID

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from packages.shared.python.storage import S3StorageClient
from workers.depth.depth_estimator import DepthMap

logger = logging.getLogger("depth.serializer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default millimetric depth scale factor: 1.0 metre = 1000.0 units in 16-bit PNG
# Range: 0.001 m to 65.535 m at 1 mm precision, or scaled for larger scenes
DEFAULT_DEPTH_SCALE: float = 1000.0  # units per metre

# Progress notification batch size (emit Redis update every N frames)
_PROGRESS_BATCH_SIZE: int = int(os.environ.get("DEPTH_PROGRESS_BATCH_SIZE", "10"))

# Default scratch directory for NVMe depth caching
_DEFAULT_SCRATCH_ROOT: Path = Path(os.environ.get("NVME_SCRATCH_ROOT", "/tmp/scratch"))


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SerializedFrame:
    """Metadata record for a serialized depth and confidence map pair."""
    frame_index: int
    depth_png_path: Optional[str] = None
    depth_npz_path: Optional[str] = None
    confidence_png_path: Optional[str] = None
    preview_png_path: Optional[str] = None
    depth_s3_key: Optional[str] = None
    confidence_s3_key: Optional[str] = None
    preview_s3_key: Optional[str] = None
    depth_sha256: str = ""
    confidence_sha256: str = ""
    file_size_depth_bytes: int = 0
    file_size_confidence_bytes: int = 0
    min_depth_m: float = 0.0
    max_depth_m: float = 0.0
    median_depth_m: float = 0.0
    mean_confidence: float = 0.0
    width: int = 0
    height: int = 0
    scale_factor: float = DEFAULT_DEPTH_SCALE
    s3_uploaded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DepthSerializationResult:
    """Complete summary of serialized depth and confidence maps for a job."""
    job_id: str
    frames: List[SerializedFrame] = field(default_factory=list)
    total_frames: int = 0
    manifest_path: str = ""
    manifest_s3_key: Optional[str] = None
    scratch_dir: str = ""
    total_depth_bytes: int = 0
    total_confidence_bytes: int = 0
    all_s3_uploaded: bool = False
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_frames": self.total_frames,
            "manifest_path": self.manifest_path,
            "manifest_s3_key": self.manifest_s3_key,
            "scratch_dir": self.scratch_dir,
            "total_depth_bytes": self.total_depth_bytes,
            "total_confidence_bytes": self.total_confidence_bytes,
            "all_s3_uploaded": self.all_s3_uploaded,
            "duration_sec": self.duration_sec,
            "frames": [f.to_dict() for f in self.frames],
        }


# ---------------------------------------------------------------------------
# Encoders & Decoders
# ---------------------------------------------------------------------------


def encode_depth_png(
    depth_m: np.ndarray,
    scale_factor: float = DEFAULT_DEPTH_SCALE,
    compression: int = 3,
) -> bytes:
    """
    Encodes a floating-point metric depth map (meters) to 16-bit PNG bytes.

    Values are clipped to [0, 65535 / scale_factor] and quantized:
      depth_uint16 = clip(round(depth_m * scale_factor), 0, 65535)
    """
    depth_m_clean = np.nan_to_num(depth_m, nan=0.0, posinf=65.535, neginf=0.0)
    scaled = np.clip(np.round(depth_m_clean * scale_factor), 0, 65535).astype(np.uint16)
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    success, encoded = cv2.imencode(".png", scaled, encode_params)
    if not success:
        raise ValueError("Failed to encode depth map as 16-bit PNG")
    return encoded.tobytes()


def decode_depth_png(
    data_or_path: Union[bytes, str, Path],
    scale_factor: float = DEFAULT_DEPTH_SCALE,
) -> np.ndarray:
    """
    Decodes 16-bit PNG bytes or file path back to float32 depth map in meters.
    """
    if isinstance(data_or_path, (str, Path)):
        img = cv2.imread(str(data_or_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read depth PNG from {data_or_path}")
    else:
        arr = np.frombuffer(data_or_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode 16-bit PNG bytes")

    if img.dtype != np.uint16:
        # Handle 8-bit fallback if accidentally decoded as 8-bit
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32)

    return img / scale_factor


def encode_depth_npz(
    depth_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> bytes:
    """
    Compresses depth map float32 array and optional valid mask to .npz bytes.
    """
    bio = BytesIO()
    kw: Dict[str, np.ndarray] = {"depth": depth_m.astype(np.float32)}
    if valid_mask is not None:
        kw["valid_mask"] = valid_mask.astype(bool)
    np.savez_compressed(bio, **kw)
    return bio.getvalue()


def decode_depth_npz(
    data_or_path: Union[bytes, str, Path],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Decodes .npz bytes or file path into (depth_m, valid_mask).
    """
    if isinstance(data_or_path, (str, Path)):
        npz = np.load(str(data_or_path))
    else:
        bio = BytesIO(data_or_path)
        npz = np.load(bio)

    depth = npz["depth"].astype(np.float32)
    valid_mask = npz["valid_mask"] if "valid_mask" in npz else None
    return depth, valid_mask


def encode_confidence_png(
    confidence: np.ndarray,
    compression: int = 3,
) -> bytes:
    """
    Encodes 8-bit confidence map [0..255] to PNG bytes.
    """
    if confidence.dtype != np.uint8:
        conf_u8 = np.clip(np.round(confidence), 0, 255).astype(np.uint8)
    else:
        conf_u8 = confidence

    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    success, encoded = cv2.imencode(".png", conf_u8, encode_params)
    if not success:
        raise ValueError("Failed to encode confidence map as PNG")
    return encoded.tobytes()


def decode_confidence_png(
    data_or_path: Union[bytes, str, Path],
) -> np.ndarray:
    """
    Decodes 8-bit confidence PNG bytes or file path to uint8 numpy array.
    """
    if isinstance(data_or_path, (str, Path)):
        img = cv2.imread(str(data_or_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read confidence PNG from {data_or_path}")
    else:
        arr = np.frombuffer(data_or_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode confidence PNG bytes")
    return img


def encode_depth_preview(
    depth_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> bytes:
    """
    Generates an 8-bit RGB Turbo-colormapped preview image for visual inspection.
    Near depth = warm/red, far depth = cool/blue, invalid/zero = black.
    """
    depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        valid_depth = depth_clean[valid_mask]
    else:
        valid_depth = depth_clean[depth_clean > 0]

    if valid_depth.size > 0:
        d_min = float(np.percentile(valid_depth, 2.0))
        d_max = float(np.percentile(valid_depth, 98.0))
        if d_max <= d_min:
            d_max = d_min + 1.0
    else:
        d_min, d_max = 0.0, 1.0

    norm = np.clip((depth_clean - d_min) / (d_max - d_min), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)

    # Invert so near is red/bright and far is dark/blue in TURBO
    colored = cv2.applyColorMap(255 - norm_u8, colormap)

    # Mask out invalid pixels with black
    if valid_mask is not None:
        colored[~valid_mask] = 0
    else:
        colored[depth_clean <= 0] = 0

    success, encoded = cv2.imencode(".png", colored, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        return b""
    return encoded.tobytes()


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hex digest of given byte array."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Serializer Class
# ---------------------------------------------------------------------------


class DepthSerializer:
    """
    Compresses and archives metric depth maps and confidence fields to NVMe scratch
    and S3 interim storage.

    Parameters
    ----------
    storage_client:
        Optional S3StorageClient for pushing interim files to S3.
    redis_tracker:
        Optional async RedisStateTracker for publishing stage & progress.
    progress_batch_size:
        Number of frames processed per Redis progress update (default 10).
    scale_factor:
        Millimetric scale factor for 16-bit PNGs (default 1000.0 units/m).
    save_npz:
        Whether to also save compressed .npz archive alongside 16-bit PNG.
    save_preview:
        Whether to generate colorized 8-bit preview PNGs.
    """

    def __init__(
        self,
        storage_client: Optional[S3StorageClient] = None,
        redis_tracker: Any = None,
        progress_batch_size: int = _PROGRESS_BATCH_SIZE,
        scale_factor: float = DEFAULT_DEPTH_SCALE,
        save_npz: bool = True,
        save_preview: bool = True,
    ) -> None:
        self._storage = storage_client
        self._redis = redis_tracker
        self._batch_size = max(1, progress_batch_size)
        self.scale_factor = scale_factor
        self.save_npz = save_npz
        self.save_preview = save_preview

    # ------------------------------------------------------------------
    # S3 Key Builders
    # ------------------------------------------------------------------

    @staticmethod
    def get_depth_s3_key(job_id: Union[str, UUID], frame_idx: int) -> str:
        """jobs/{job_id}/interim/depth/depth_{idx:05d}.png"""
        return f"jobs/{job_id}/interim/depth/depth_{frame_idx:05d}.png"

    @staticmethod
    def get_confidence_s3_key(job_id: Union[str, UUID], frame_idx: int) -> str:
        """jobs/{job_id}/interim/depth/confidence_{idx:05d}.png"""
        return f"jobs/{job_id}/interim/depth/confidence_{frame_idx:05d}.png"

    @staticmethod
    def get_preview_s3_key(job_id: Union[str, UUID], frame_idx: int) -> str:
        """jobs/{job_id}/interim/depth/preview_{idx:05d}.png"""
        return f"jobs/{job_id}/interim/depth/preview_{frame_idx:05d}.png"

    @staticmethod
    def get_manifest_s3_key(job_id: Union[str, UUID]) -> str:
        """jobs/{job_id}/interim/depth/manifest.json"""
        return f"jobs/{job_id}/interim/depth/manifest.json"

    # ------------------------------------------------------------------
    # Main Serialization API
    # ------------------------------------------------------------------

    async def serialize_depth_maps(
        self,
        job_id: Union[str, UUID],
        depth_items: Sequence[Union[DepthMap, Tuple[int, np.ndarray, np.ndarray]]],
        scratch_root: Optional[Union[str, Path]] = None,
    ) -> DepthSerializationResult:
        """
        Serializes depth and confidence maps for all given keyframes.

        Parameters
        ----------
        job_id:
            Reconstruction job ID.
        depth_items:
            List of either:
              - DepthMap instances (with confidence field or separate tuple)
              - Tuples: (frame_index, depth_m_array, confidence_array [, valid_mask])
        scratch_root:
            Scratch root directory. Defaults to _DEFAULT_SCRATCH_ROOT.

        Returns
        -------
        DepthSerializationResult containing frame records and summary stats.
        """
        start_time = time.perf_counter()
        job_id_str = str(job_id)

        # Setup local NVMe scratch directories
        base_scratch = Path(scratch_root) if scratch_root else _DEFAULT_SCRATCH_ROOT
        job_scratch_depth = base_scratch / job_id_str / "depth"
        job_scratch_depth.mkdir(parents=True, exist_ok=True)

        total_frames = len(depth_items)
        logger.info(
            "[%s] Starting depth serialization for %d frames into %s",
            job_id_str, total_frames, job_scratch_depth,
        )

        # Emit initial Redis progress
        await self._publish_progress(job_id_str, 0, total_frames, 0)

        serialized_frames: List[SerializedFrame] = []
        total_depth_bytes = 0
        total_confidence_bytes = 0
        all_uploaded = True

        for i, item in enumerate(depth_items):
            # Parse input item
            frame_idx, depth_m, conf_u8, valid_mask = self._unpack_item(item)

            h, w = depth_m.shape[:2]
            valid_depths = depth_m[valid_mask] if valid_mask is not None else depth_m[depth_m > 0]
            d_min = float(np.min(valid_depths)) if valid_depths.size > 0 else 0.0
            d_max = float(np.max(valid_depths)) if valid_depths.size > 0 else 0.0
            d_med = float(np.median(valid_depths)) if valid_depths.size > 0 else 0.0
            mean_conf = float(np.mean(conf_u8)) if conf_u8.size > 0 else 0.0

            # 1. Encode depth PNG
            depth_png_bytes = encode_depth_png(depth_m, self.scale_factor)
            depth_sha = compute_sha256(depth_png_bytes)
            depth_png_path = job_scratch_depth / f"depth_{frame_idx:05d}.png"
            depth_png_path.write_bytes(depth_png_bytes)
            file_size_depth = len(depth_png_bytes)
            total_depth_bytes += file_size_depth

            # Optional NPZ
            depth_npz_path_str: Optional[str] = None
            if self.save_npz:
                depth_npz_bytes = encode_depth_npz(depth_m, valid_mask)
                depth_npz_path = job_scratch_depth / f"depth_{frame_idx:05d}.npz"
                depth_npz_path.write_bytes(depth_npz_bytes)
                depth_npz_path_str = str(depth_npz_path)

            # 2. Encode confidence PNG
            conf_png_bytes = encode_confidence_png(conf_u8)
            conf_sha = compute_sha256(conf_png_bytes)
            conf_png_path = job_scratch_depth / f"confidence_{frame_idx:05d}.png"
            conf_png_path.write_bytes(conf_png_bytes)
            file_size_conf = len(conf_png_bytes)
            total_confidence_bytes += file_size_conf

            # 3. Optional Preview PNG
            preview_png_path_str: Optional[str] = None
            preview_bytes: Optional[bytes] = None
            if self.save_preview:
                preview_bytes = encode_depth_preview(depth_m, valid_mask)
                if preview_bytes:
                    preview_png_path = job_scratch_depth / f"preview_{frame_idx:05d}.png"
                    preview_png_path.write_bytes(preview_bytes)
                    preview_png_path_str = str(preview_png_path)

            # 4. Upload to S3 if configured
            depth_s3_key = self.get_depth_s3_key(job_id_str, frame_idx)
            conf_s3_key = self.get_confidence_s3_key(job_id_str, frame_idx)
            preview_s3_key = self.get_preview_s3_key(job_id_str, frame_idx) if preview_bytes else None

            s3_ok = True
            if self._storage is not None:
                d_ok = await self._upload_to_s3(depth_s3_key, depth_png_bytes, "image/png")
                c_ok = await self._upload_to_s3(conf_s3_key, conf_png_bytes, "image/png")
                p_ok = True
                if preview_bytes and preview_s3_key:
                    p_ok = await self._upload_to_s3(preview_s3_key, preview_bytes, "image/png")
                s3_ok = d_ok and c_ok and p_ok
                if not s3_ok:
                    all_uploaded = False

            rec = SerializedFrame(
                frame_index=frame_idx,
                depth_png_path=str(depth_png_path),
                depth_npz_path=depth_npz_path_str,
                confidence_png_path=str(conf_png_path),
                preview_png_path=preview_png_path_str,
                depth_s3_key=depth_s3_key if self._storage else None,
                confidence_s3_key=conf_s3_key if self._storage else None,
                preview_s3_key=preview_s3_key if (self._storage and preview_bytes) else None,
                depth_sha256=depth_sha,
                confidence_sha256=conf_sha,
                file_size_depth_bytes=file_size_depth,
                file_size_confidence_bytes=file_size_conf,
                min_depth_m=round(d_min, 3),
                max_depth_m=round(d_max, 3),
                median_depth_m=round(d_med, 3),
                mean_confidence=round(mean_conf, 2),
                width=w,
                height=h,
                scale_factor=self.scale_factor,
                s3_uploaded=s3_ok,
            )
            serialized_frames.append(rec)

            # Periodic progress reporting
            frames_done = i + 1
            if frames_done % self._batch_size == 0 or frames_done == total_frames:
                progress_pct = int(round(100.0 * frames_done / max(1, total_frames)))
                await self._publish_progress(job_id_str, frames_done, total_frames, progress_pct)

        # 5. Write Manifest
        manifest_path = job_scratch_depth / "manifest.json"
        manifest_data = {
            "job_id": job_id_str,
            "total_frames": total_frames,
            "created_at_sec": time.time(),
            "scale_factor": self.scale_factor,
            "total_depth_bytes": total_depth_bytes,
            "total_confidence_bytes": total_confidence_bytes,
            "frames": [f.to_dict() for f in serialized_frames],
        }
        manifest_json = json.dumps(manifest_data, indent=2)
        manifest_path.write_text(manifest_json, encoding="utf-8")

        manifest_s3_key = self.get_manifest_s3_key(job_id_str)
        if self._storage is not None:
            m_ok = await self._upload_to_s3(
                manifest_s3_key,
                manifest_json.encode("utf-8"),
                "application/json",
            )
            if not m_ok:
                all_uploaded = False

        duration = time.perf_counter() - start_time
        logger.info(
            "[%s] Serialized %d frames in %.2fs (depth: %d bytes, confidence: %d bytes)",
            job_id_str, total_frames, duration, total_depth_bytes, total_confidence_bytes,
        )

        return DepthSerializationResult(
            job_id=job_id_str,
            frames=serialized_frames,
            total_frames=total_frames,
            manifest_path=str(manifest_path),
            manifest_s3_key=manifest_s3_key if self._storage else None,
            scratch_dir=str(job_scratch_depth),
            total_depth_bytes=total_depth_bytes,
            total_confidence_bytes=total_confidence_bytes,
            all_s3_uploaded=all_uploaded if self._storage else True,
            duration_sec=round(duration, 3),
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _unpack_item(
        self,
        item: Union[DepthMap, Tuple[int, np.ndarray, np.ndarray]],
    ) -> Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extracts (frame_index, depth_m, confidence_u8, valid_mask) from various inputs."""
        if isinstance(item, DepthMap):
            frame_idx = item.frame_index
            depth_m = item.depth_m
            valid_mask = item.valid_mask
            # Synthetic confidence if not attached
            conf = (valid_mask.astype(np.uint8) * 255) if valid_mask is not None else np.full(depth_m.shape, 255, dtype=np.uint8)
            return frame_idx, depth_m, conf, valid_mask
        elif isinstance(item, (tuple, list)):
            frame_idx = int(item[0])
            depth_m = np.asarray(item[1], dtype=np.float32)
            conf = np.asarray(item[2], dtype=np.uint8)
            valid_mask = np.asarray(item[3], dtype=bool) if len(item) > 3 else (depth_m > 0)
            return frame_idx, depth_m, conf, valid_mask
        else:
            raise TypeError(f"Unsupported item type for depth serialization: {type(item)}")

    async def _upload_to_s3(
        self,
        s3_key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Uploads bytes to S3 interim bucket asynchronously via thread executor."""
        if self._storage is None:
            return True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._storage.upload_bytes,
                S3StorageClient.INTERIM_BUCKET,
                s3_key,
                data,
                content_type,
            )
            return True
        except Exception:
            logger.exception("Failed to upload %s to S3 interim bucket", s3_key)
            return False

    async def _publish_progress(
        self,
        job_id: str,
        frames_processed: int,
        total_frames: int,
        progress_pct: int,
    ) -> None:
        """Updates stage to ESTIMATING_DEPTH and publishes progress to Redis."""
        if self._redis is not None:
            try:
                await self._redis.set_job_progress(
                    job_id=job_id,
                    stage="ESTIMATING_DEPTH",
                    progress=progress_pct,
                    frames_processed=frames_processed,
                    total_frames=total_frames,
                    gpu_stats={},
                )
            except Exception:
                logger.warning("[%s] Redis progress publish failed (non-fatal).", job_id)
        else:
            logger.debug(
                "[%s] ESTIMATING_DEPTH progress: %d/%d (%d%%)",
                job_id, frames_processed, total_frames, progress_pct,
            )

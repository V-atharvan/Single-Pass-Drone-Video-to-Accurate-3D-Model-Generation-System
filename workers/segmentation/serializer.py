"""
Semantic and Dynamic Mask Serialization to Storage — TASK-039.

Serializes semantic segmentation outputs and dilated dynamic exclusion masks to
NVMe scratch storage and S3 interim storage:
  - `dynamic_%05d.png`: Binary dynamic exclusion mask (255 = dynamic, 0 = static).
  - `semantic_%05d.png`: 8-bit discrete semantic class index mask.
  - `semantic_preview_%05d.png`: 24-bit RGB colorized visual preview.
  - `semantics_%05d.npz`: Compressed multi-scale pyramid archive (1x, 0.5x, 0.25x).
  - `segmentation_manifest.json`: Full manifest with SHA256 checksums and contamination metrics.

Updates Redis stage to `SEGMENTING` with progress percentage and stores dynamic
object contamination metrics for the final quality report.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.segmentation.mask_combiner import (
    MaskPyramid,
    MultiChannelSemanticMask,
    SemanticMaskCombiner,
)
from workers.segmentation.mask_generator import DynamicMask
from workers.segmentation.semantic_classifier import (
    CLASS_COLOR_PALETTE,
    SemanticClass,
    SemanticMap,
)

logger = logging.getLogger("segmentation.serializer")

_DEFAULT_SCRATCH_ROOT: Path = Path(os.environ.get("NVME_SCRATCH_ROOT", "/tmp/scratch"))
_PROGRESS_BATCH_SIZE: int = int(os.environ.get("SEGMENTATION_PROGRESS_BATCH_SIZE", "10"))


def compute_sha256(data: bytes) -> str:
    """Computes hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class SerializedSemanticFrame:
    """Metadata record for a serialized semantic and dynamic mask set."""

    frame_index: int
    dynamic_png_path: Optional[str] = None
    semantic_png_path: Optional[str] = None
    preview_png_path: Optional[str] = None
    npz_path: Optional[str] = None
    dynamic_s3_key: Optional[str] = None
    semantic_s3_key: Optional[str] = None
    preview_s3_key: Optional[str] = None
    npz_s3_key: Optional[str] = None
    dynamic_sha256: str = ""
    semantic_sha256: str = ""
    npz_sha256: str = ""
    file_size_dynamic_bytes: int = 0
    file_size_semantic_bytes: int = 0
    file_size_npz_bytes: int = 0
    dynamic_pixel_count: int = 0
    dynamic_pixel_fraction: float = 0.0
    moving_objects_count: int = 0
    width: int = 0
    height: int = 0
    s3_uploaded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicContaminationMetrics:
    """Dynamic object contamination metrics stored for quality dossiers."""

    total_moving_objects_removed: int = 0
    total_dynamic_pixels_excluded: int = 0
    mean_dynamic_pixel_fraction: float = 0.0
    max_frame_dynamic_fraction: float = 0.0
    contaminated_frames_count: int = 0
    total_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationSerializationResult:
    """Complete summary of serialized semantic and dynamic masks for a job."""

    job_id: str
    frames: List[SerializedSemanticFrame] = field(default_factory=list)
    total_frames: int = 0
    manifest_path: str = ""
    manifest_s3_key: Optional[str] = None
    scratch_dir: str = ""
    contamination_metrics: DynamicContaminationMetrics = field(
        default_factory=DynamicContaminationMetrics
    )
    total_bytes_written: int = 0
    all_s3_uploaded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_frames": self.total_frames,
            "manifest_path": self.manifest_path,
            "manifest_s3_key": self.manifest_s3_key,
            "scratch_dir": self.scratch_dir,
            "contamination_metrics": self.contamination_metrics.to_dict(),
            "total_bytes_written": self.total_bytes_written,
            "all_s3_uploaded": self.all_s3_uploaded,
            "frames": [f.to_dict() for f in self.frames],
        }


def encode_dynamic_png(mask: np.ndarray) -> bytes:
    """Encodes binary dynamic mask to PNG bytes."""
    success, buf = cv2.imencode(".png", mask.astype(np.uint8))
    if not success:
        raise ValueError("Failed to encode dynamic mask to PNG")
    return bytes(buf)


def encode_semantic_png(class_mask: np.ndarray) -> bytes:
    """Encodes 8-bit discrete semantic class mask to PNG bytes."""
    success, buf = cv2.imencode(".png", class_mask.astype(np.uint8))
    if not success:
        raise ValueError("Failed to encode semantic class mask to PNG")
    return bytes(buf)


def encode_semantic_preview_png(
    class_mask: np.ndarray,
    dynamic_mask: Optional[np.ndarray] = None,
) -> bytes:
    """
    Renders an RGB colorized preview with dynamic exclusion overlay.
    """
    h, w = class_mask.shape[:2]
    bgr = np.zeros((h, w, 3), dtype=np.uint8)

    for s_class, rgb in CLASS_COLOR_PALETTE.items():
        # Palette is RGB, cv2 expects BGR
        bgr_color = (rgb[2], rgb[1], rgb[0])
        match_pixels = class_mask == int(s_class)
        if np.any(match_pixels):
            bgr[match_pixels] = bgr_color

    # If dynamic mask is present, highlight dynamic exclusion regions with bright diagonal hatch/tint
    if dynamic_mask is not None:
        dynamic_pixels = dynamic_mask == 255
        if np.any(dynamic_pixels):
            # Overlay bright red/pink tint with alpha blending
            overlay = bgr.copy()
            overlay[dynamic_pixels] = (0, 0, 255)  # pure red BGR
            cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0.0, bgr)

    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise ValueError("Failed to encode semantic preview to PNG")
    return bytes(buf)


def encode_pyramid_npz(pyramid: MaskPyramid) -> bytes:
    """Serializes 3-level MaskPyramid into compressed NPZ bytes."""
    buf = BytesIO()
    np.savez_compressed(
        buf,
        level_0=pyramid.level_0,
        level_1=pyramid.level_1,
        level_2=pyramid.level_2,
        frame_index=pyramid.frame_index,
    )
    return buf.getvalue()


class SemanticMaskSerializer:
    """
    Coordinates local NVMe caching, S3 interim uploading, and Redis stage updates.
    """

    def __init__(
        self,
        scratch_root: Union[str, Path] = _DEFAULT_SCRATCH_ROOT,
        s3_client: Optional[Any] = None,
        s3_bucket: Optional[str] = None,
        redis_client: Optional[Any] = None,
        save_pyramid_npz: bool = True,
        save_preview: bool = True,
    ):
        self.scratch_root = Path(scratch_root)
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.redis_client = redis_client
        self.save_pyramid_npz = save_pyramid_npz
        self.save_preview = save_preview
        self.combiner = SemanticMaskCombiner()

    def get_job_scratch_dir(self, job_id: str) -> Path:
        """Returns the local scratch directory for semantic masks."""
        return self.scratch_root / str(job_id) / "semantics"

    def get_s3_prefix(self, job_id: str) -> str:
        """Standard S3 prefix for interim segmentation artifacts."""
        return f"jobs/{job_id}/interim/semantics/"

    def serialize_sequence(
        self,
        job_id: str,
        semantic_maps: Sequence[SemanticMap],
        dynamic_masks: Sequence[DynamicMask],
    ) -> SegmentationSerializationResult:
        """
        Serializes semantic maps and dynamic masks for a complete sequence of keyframes.
        """
        if len(semantic_maps) != len(dynamic_masks):
            raise ValueError(
                f"Count mismatch: {len(semantic_maps)} semantic maps vs {len(dynamic_masks)} dynamic masks"
            )

        job_id_str = str(job_id)
        scratch_dir = self.get_job_scratch_dir(job_id_str)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        serialized_frames: List[SerializedSemanticFrame] = []
        total_bytes = 0
        all_s3_uploaded = True

        total_moving_objects = 0
        total_dynamic_pixels = 0
        fractions: List[float] = []
        contaminated_count = 0

        total_frames = len(semantic_maps)

        for i, (sem_map, dyn_mask) in enumerate(zip(semantic_maps, dynamic_masks)):
            frame_idx = sem_map.frame_index
            h = sem_map.original_height
            w = sem_map.original_width

            # 1. Dynamic exclusion mask PNG
            dyn_bytes = encode_dynamic_png(dyn_mask.mask)
            dyn_sha = compute_sha256(dyn_bytes)
            dyn_path = scratch_dir / f"dynamic_{frame_idx:05d}.png"
            dyn_path.write_bytes(dyn_bytes)
            file_size_dyn = len(dyn_bytes)
            total_bytes += file_size_dyn

            # 2. Semantic class mask PNG
            sem_bytes = encode_semantic_png(sem_map.class_mask)
            sem_sha = compute_sha256(sem_bytes)
            sem_path = scratch_dir / f"semantic_{frame_idx:05d}.png"
            sem_path.write_bytes(sem_bytes)
            file_size_sem = len(sem_bytes)
            total_bytes += file_size_sem

            # 3. Optional Preview PNG
            preview_path_str: Optional[str] = None
            preview_bytes: Optional[bytes] = None
            if self.save_preview:
                preview_bytes = encode_semantic_preview_png(sem_map.class_mask, dyn_mask.mask)
                preview_path = scratch_dir / f"semantic_preview_{frame_idx:05d}.png"
                preview_path.write_bytes(preview_bytes)
                preview_path_str = str(preview_path)
                total_bytes += len(preview_bytes)

            # 4. Optional 3-Channel Pyramid NPZ
            npz_path_str: Optional[str] = None
            npz_sha = ""
            file_size_npz = 0
            if self.save_pyramid_npz:
                _, pyramid = self.combiner.combine_and_pyramid(sem_map, dyn_mask)
                npz_bytes = encode_pyramid_npz(pyramid)
                npz_sha = compute_sha256(npz_bytes)
                npz_path = scratch_dir / f"semantics_{frame_idx:05d}.npz"
                npz_path.write_bytes(npz_bytes)
                npz_path_str = str(npz_path)
                file_size_npz = len(npz_bytes)
                total_bytes += file_size_npz

            # Contamination tracking
            total_moving_objects += dyn_mask.moving_objects_count
            total_dynamic_pixels += dyn_mask.dilated_dynamic_pixels
            fractions.append(dyn_mask.dynamic_fraction)
            if dyn_mask.dilated_dynamic_pixels > 0:
                contaminated_count += 1

            # 5. S3 Uploads
            s3_dyn_key = None
            s3_sem_key = None
            s3_prev_key = None
            s3_npz_key = None
            uploaded = False

            if self.s3_client and self.s3_bucket:
                try:
                    pfx = self.get_s3_prefix(job_id_str)
                    s3_dyn_key = f"{pfx}dynamic_{frame_idx:05d}.png"
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=s3_dyn_key,
                        Body=dyn_bytes,
                        ContentType="image/png",
                    )
                    s3_sem_key = f"{pfx}semantic_{frame_idx:05d}.png"
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=s3_sem_key,
                        Body=sem_bytes,
                        ContentType="image/png",
                    )
                    if preview_bytes:
                        s3_prev_key = f"{pfx}semantic_preview_{frame_idx:05d}.png"
                        self.s3_client.put_object(
                            Bucket=self.s3_bucket,
                            Key=s3_prev_key,
                            Body=preview_bytes,
                            ContentType="image/png",
                        )
                    if npz_path_str:
                        s3_npz_key = f"{pfx}semantics_{frame_idx:05d}.npz"
                        self.s3_client.put_object(
                            Bucket=self.s3_bucket,
                            Key=s3_npz_key,
                            Body=(scratch_dir / f"semantics_{frame_idx:05d}.npz").read_bytes(),
                            ContentType="application/octet-stream",
                        )
                    uploaded = True
                except Exception as exc:
                    logger.error(f"Failed to upload masks for frame {frame_idx} to S3: {exc}")
                    all_s3_uploaded = False
            else:
                all_s3_uploaded = False

            serialized_frames.append(
                SerializedSemanticFrame(
                    frame_index=frame_idx,
                    dynamic_png_path=str(dyn_path),
                    semantic_png_path=str(sem_path),
                    preview_png_path=preview_path_str,
                    npz_path=npz_path_str,
                    dynamic_s3_key=s3_dyn_key,
                    semantic_s3_key=s3_sem_key,
                    preview_s3_key=s3_prev_key,
                    npz_s3_key=s3_npz_key,
                    dynamic_sha256=dyn_sha,
                    semantic_sha256=sem_sha,
                    npz_sha256=npz_sha,
                    file_size_dynamic_bytes=file_size_dyn,
                    file_size_semantic_bytes=file_size_sem,
                    file_size_npz_bytes=file_size_npz,
                    dynamic_pixel_count=dyn_mask.dilated_dynamic_pixels,
                    dynamic_pixel_fraction=dyn_mask.dynamic_fraction,
                    moving_objects_count=dyn_mask.moving_objects_count,
                    width=w,
                    height=h,
                    s3_uploaded=uploaded,
                )
            )

            # Redis progress notification
            if (i + 1) % _PROGRESS_BATCH_SIZE == 0 or (i + 1) == total_frames:
                pct = ((i + 1) / total_frames) * 100.0
                self._publish_redis_progress(job_id_str, i + 1, total_frames, pct)

        # Compute summary contamination metrics
        mean_frac = float(np.mean(fractions)) if fractions else 0.0
        max_frac = float(np.max(fractions)) if fractions else 0.0

        contamination_metrics = DynamicContaminationMetrics(
            total_moving_objects_removed=total_moving_objects,
            total_dynamic_pixels_excluded=total_dynamic_pixels,
            mean_dynamic_pixel_fraction=mean_frac,
            max_frame_dynamic_fraction=max_frac,
            contaminated_frames_count=contaminated_count,
            total_frames=total_frames,
        )

        # Write manifest
        manifest = {
            "job_id": job_id_str,
            "total_frames": total_frames,
            "scratch_dir": str(scratch_dir),
            "contamination_metrics": contamination_metrics.to_dict(),
            "total_bytes_written": total_bytes,
            "frames": [f.to_dict() for f in serialized_frames],
        }
        manifest_path = scratch_dir / "segmentation_manifest.json"
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        total_bytes += len(manifest_bytes)

        manifest_s3_key = None
        if self.s3_client and self.s3_bucket:
            try:
                manifest_s3_key = f"{self.get_s3_prefix(job_id_str)}segmentation_manifest.json"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=manifest_s3_key,
                    Body=manifest_bytes,
                    ContentType="application/json",
                )
            except Exception as exc:
                logger.error(f"Failed to upload segmentation manifest to S3: {exc}")

        # Update Redis metadata with dynamic object statistics
        self._publish_redis_metadata(job_id_str, contamination_metrics)

        return SegmentationSerializationResult(
            job_id=job_id_str,
            frames=serialized_frames,
            total_frames=total_frames,
            manifest_path=str(manifest_path),
            manifest_s3_key=manifest_s3_key,
            scratch_dir=str(scratch_dir),
            contamination_metrics=contamination_metrics,
            total_bytes_written=total_bytes,
            all_s3_uploaded=all_s3_uploaded,
        )

    def _publish_redis_progress(
        self,
        job_id: str,
        frames_done: int,
        total_frames: int,
        progress_pct: float,
    ) -> None:
        """Publishes progress update to Redis stage SEGMENTING."""
        if not self.redis_client:
            return
        try:
            update = {
                "job_id": job_id,
                "state": "SEGMENTING",
                "stage": "Segmenting semantic classes and masking dynamic objects",
                "progress_percent": round(progress_pct, 1),
                "frames_processed": frames_done,
                "frames_total": total_frames,
            }
            self.redis_client.hset(f"job:{job_id}", "progress", json.dumps(update))
            self.redis_client.publish(f"job:{job_id}:progress", json.dumps(update))
        except Exception as exc:
            logger.warning(f"Failed to publish Redis progress for job {job_id}: {exc}")

    def _publish_redis_metadata(
        self,
        job_id: str,
        metrics: DynamicContaminationMetrics,
    ) -> None:
        """Stores dynamic contamination metrics in Redis."""
        if not self.redis_client:
            return
        try:
            self.redis_client.hset(
                f"job:{job_id}:metadata",
                "dynamic_contamination",
                json.dumps(metrics.to_dict()),
            )
        except Exception as exc:
            logger.warning(f"Failed to store dynamic contamination in Redis: {exc}")

"""
Dynamic Object Mask Generation & Dilation Pipeline — TASK-037.

Generates dilated binary exclusion masks for all dynamic entities (moving vehicles,
pedestrians, moving animals, unclassified moving objects) to ensure complete exclusion
during 3D point cloud back-projection.

Prevents phantom/ghost geometry and boundary edge smearing in 3D reconstruction.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.segmentation.dynamic_tracker import (
    FrameTrackingResult,
    MotionState,
    TrackedObject,
)
from workers.segmentation.semantic_classifier import (
    SemanticClass,
    SemanticMap,
)

logger = logging.getLogger("segmentation.mask_generator")

_DEFAULT_DILATION_KERNEL_SIZE = 5  # 5x5 morphological dilation


@dataclass
class DynamicMask:
    """
    Binary dynamic exclusion mask for a single keyframe.

    Attributes
    ----------
    frame_index:
        Keyframe index.
    mask:
        (H, W) uint8 array (255 = dynamic/exclude from 3D, 0 = static/keep).
    raw_dynamic_pixels:
        Number of dynamic pixels prior to dilation.
    dilated_dynamic_pixels:
        Total excluded pixels including dilation margin.
    dynamic_fraction:
        Fraction of total frame area excluded [0.0, 1.0].
    moving_objects_count:
        Number of active moving objects in this frame.
    dilation_margin_px:
        Kernel diameter used for morphological dilation.
    """

    frame_index: int
    mask: np.ndarray                   # (H, W) uint8
    raw_dynamic_pixels: int
    dilated_dynamic_pixels: int
    dynamic_fraction: float
    moving_objects_count: int
    dilation_margin_px: int


class DynamicMaskGenerator:
    """
    Generates dilated binary exclusion masks for dynamic entities.
    """

    def __init__(self, dilation_kernel_size: int = _DEFAULT_DILATION_KERNEL_SIZE):
        self.dilation_kernel_size = max(1, dilation_kernel_size)
        # Construct elliptical structuring element
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.dilation_kernel_size, self.dilation_kernel_size),
        )

    def generate_mask(
        self,
        height: int,
        width: int,
        moving_objects: Sequence[TrackedObject],
        semantic_map: Optional[SemanticMap] = None,
        frame_index: int = 0,
    ) -> DynamicMask:
        """
        Creates a dilated binary dynamic mask from moving objects.

        If a semantic_map is provided, pixels matching the moving object's semantic class
        inside the bounding box are extracted with pixel-level precision. Otherwise, the
        bounding box region is filled.
        """
        raw_mask = np.zeros((height, width), dtype=np.uint8)

        for obj in moving_objects:
            if obj.motion_state != MotionState.DYNAMIC:
                continue

            x1, y1, x2, y2 = obj.bbox_xyxy
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            if semantic_map is not None:
                # Precise pixel-level mask matching object semantic class inside bbox
                sem_patch = semantic_map.class_mask[y1:y2, x1:x2]
                class_val = int(obj.semantic_class)
                obj_pixels = sem_patch == class_val
                if np.any(obj_pixels):
                    raw_mask[y1:y2, x1:x2][obj_pixels] = 255
                else:
                    # Fallback to bbox if semantic patch has discrepancy
                    raw_mask[y1:y2, x1:x2] = 255
            else:
                raw_mask[y1:y2, x1:x2] = 255

        raw_pixels = int(np.count_nonzero(raw_mask))

        # Morphological dilation to create safe margin around object boundaries
        if raw_pixels > 0 and self.dilation_kernel_size > 1:
            dilated_mask = cv2.dilate(raw_mask, self._kernel, iterations=1)
        else:
            dilated_mask = raw_mask.copy()

        dilated_pixels = int(np.count_nonzero(dilated_mask))
        total_pixels = height * width
        dynamic_frac = float(dilated_pixels / total_pixels) if total_pixels > 0 else 0.0

        return DynamicMask(
            frame_index=frame_index,
            mask=dilated_mask,
            raw_dynamic_pixels=raw_pixels,
            dilated_dynamic_pixels=dilated_pixels,
            dynamic_fraction=dynamic_frac,
            moving_objects_count=len(moving_objects),
            dilation_margin_px=self.dilation_kernel_size,
        )

    def generate_from_tracking_result(
        self,
        tracking_result: FrameTrackingResult,
        semantic_map: SemanticMap,
    ) -> DynamicMask:
        """Helper that generates DynamicMask directly from FrameTrackingResult."""
        return self.generate_mask(
            height=semantic_map.original_height,
            width=semantic_map.original_width,
            moving_objects=tracking_result.moving_objects,
            semantic_map=semantic_map,
            frame_index=tracking_result.frame_index,
        )

    def generate_sequence_masks(
        self,
        tracking_results: Sequence[FrameTrackingResult],
        semantic_maps: Sequence[SemanticMap],
    ) -> List[DynamicMask]:
        """Generates dilated dynamic masks for an entire sequence of frames."""
        masks: List[DynamicMask] = []
        for track_res, sem_map in zip(tracking_results, semantic_maps):
            mask = self.generate_from_tracking_result(track_res, sem_map)
            masks.append(mask)
        return masks

"""
Semantic Mask Association & Mask Pyramid Builder — TASK-038.

Merges static semantic classification with dynamic exclusion masks into a unified
multi-channel semantic tensor per keyframe:
  - Channel 0: Semantic Class ID (0..10 uint8)
  - Channel 1: Dynamic Exclusion Flag (0 = static, 255 = dynamic/exclude uint8)
  - Channel 2: Classification Confidence (0..255 uint8, quantized from float [0, 1])

Builds multi-resolution pyramids (1x, 0.5x, 0.25x) to support multi-scale LOD 3D fusion.
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

from workers.segmentation.mask_generator import DynamicMask
from workers.segmentation.semantic_classifier import SemanticClass, SemanticMap

logger = logging.getLogger("segmentation.mask_combiner")


@dataclass
class MultiChannelSemanticMask:
    """
    Unified 3-channel semantic tensor for a single keyframe.

    Channels:
      Channel 0 (uint8): SemanticClass index (0..10)
      Channel 1 (uint8): Dynamic exclusion flag (0 = static, 255 = dynamic/exclude)
      Channel 2 (uint8): Quantized confidence score (0..255)
    """

    frame_index: int
    tensor: np.ndarray                 # (H, W, 3) uint8
    original_height: int
    original_width: int

    @property
    def class_mask(self) -> np.ndarray:
        """(H, W) uint8 class index mask."""
        return self.tensor[:, :, 0]

    @property
    def dynamic_mask(self) -> np.ndarray:
        """(H, W) uint8 dynamic exclusion mask (255 = dynamic, 0 = static)."""
        return self.tensor[:, :, 1]

    @property
    def confidence_map(self) -> np.ndarray:
        """(H, W) float32 confidence map in [0.0, 1.0]."""
        return self.tensor[:, :, 2].astype(np.float32) / 255.0

    @property
    def static_class_mask(self) -> np.ndarray:
        """
        (H, W) uint8 class mask where dynamic pixels are replaced with UNKNOWN (0)
        or explicitly masked out.
        """
        mask = self.class_mask.copy()
        mask[self.dynamic_mask == 255] = int(SemanticClass.UNKNOWN)
        return mask


@dataclass
class MaskPyramid:
    """
    Multi-resolution pyramid hierarchy for 3D multi-scale Level-of-Detail (LOD) fusion.
    """

    frame_index: int
    level_0: np.ndarray                # 1.0x (H, W, 3) uint8
    level_1: np.ndarray                # 0.5x (H//2, W//2, 3) uint8
    level_2: np.ndarray                # 0.25x (H//4, W//4, 3) uint8
    scale_factors: Tuple[float, float, float] = (1.0, 0.5, 0.25)


class SemanticMaskCombiner:
    """
    Combines semantic maps and dynamic exclusion masks, and generates LOD pyramids.
    """

    def combine(
        self,
        semantic_map: SemanticMap,
        dynamic_mask: DynamicMask,
    ) -> MultiChannelSemanticMask:
        """
        Merges SemanticMap with DynamicMask into a unified (H, W, 3) uint8 tensor.
        """
        h = semantic_map.original_height
        w = semantic_map.original_width

        tensor = np.zeros((h, w, 3), dtype=np.uint8)

        # Channel 0: Semantic Class ID
        tensor[:, :, 0] = semantic_map.class_mask.astype(np.uint8)

        # Channel 1: Dynamic Exclusion Flag
        tensor[:, :, 1] = dynamic_mask.mask.astype(np.uint8)

        # Channel 2: Quantized Confidence (float in [0, 1] -> uint8 in [0, 255])
        conf_float = np.clip(semantic_map.confidence_map, 0.0, 1.0)
        tensor[:, :, 2] = (conf_float * 255.0).astype(np.uint8)

        return MultiChannelSemanticMask(
            frame_index=semantic_map.frame_index,
            tensor=tensor,
            original_height=h,
            original_width=w,
        )

    def build_pyramid(
        self,
        multi_channel_mask: MultiChannelSemanticMask,
    ) -> MaskPyramid:
        """
        Builds a 3-level pyramid (1.0x, 0.5x, 0.25x).
        Uses nearest-neighbor downsampling for Channels 0 and 1 to prevent creating
        invalid fractional class IDs or diluted exclusion boundaries.
        Uses area downsampling for Channel 2 (confidence).
        """
        level_0 = multi_channel_mask.tensor
        h, w = level_0.shape[:2]

        # Level 1: 0.5x
        h1, w1 = max(1, h // 2), max(1, w // 2)
        level_1 = np.zeros((h1, w1, 3), dtype=np.uint8)
        level_1[:, :, 0] = cv2.resize(level_0[:, :, 0], (w1, h1), interpolation=cv2.INTER_NEAREST)
        level_1[:, :, 1] = cv2.resize(level_0[:, :, 1], (w1, h1), interpolation=cv2.INTER_NEAREST)
        level_1[:, :, 2] = cv2.resize(level_0[:, :, 2], (w1, h1), interpolation=cv2.INTER_AREA)

        # Level 2: 0.25x
        h2, w2 = max(1, h // 4), max(1, w // 4)
        level_2 = np.zeros((h2, w2, 3), dtype=np.uint8)
        level_2[:, :, 0] = cv2.resize(level_0[:, :, 0], (w2, h2), interpolation=cv2.INTER_NEAREST)
        level_2[:, :, 1] = cv2.resize(level_0[:, :, 1], (w2, h2), interpolation=cv2.INTER_NEAREST)
        level_2[:, :, 2] = cv2.resize(level_0[:, :, 2], (w2, h2), interpolation=cv2.INTER_AREA)

        return MaskPyramid(
            frame_index=multi_channel_mask.frame_index,
            level_0=level_0,
            level_1=level_1,
            level_2=level_2,
        )

    def combine_and_pyramid(
        self,
        semantic_map: SemanticMap,
        dynamic_mask: DynamicMask,
    ) -> Tuple[MultiChannelSemanticMask, MaskPyramid]:
        """Convenience method combining and building pyramid in one call."""
        combined = self.combine(semantic_map, dynamic_mask)
        pyramid = self.build_pyramid(combined)
        return combined, pyramid

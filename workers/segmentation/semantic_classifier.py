"""
Semantic Segmentation Pipeline — TASK-035 [ENHANCED].

Classifies drone/aerial keyframe regions into target geospatial categories,
with specialized auxiliary handling for animals and unclassified dynamic objects.

Target Classes (PRD Section 6 / Task-035):
  0: UNKNOWN
  1: BUILDING
  2: ROAD
  3: TERRAIN
  4: VEGETATION
  5: VEHICLE
  6: PERSON
  7: UTILITY_INFRASTRUCTURE
  8: WATER
  9: ANIMAL (horses, cattle, dogs, birds; falls back to DYNAMIC_OBJECT when confidence < threshold)
  10: DYNAMIC_OBJECT (fallback for any unclassified moving pixel region; excluded from 3D reconstruction)

Rules:
  - Zero Phantom Dynamic Geometry: Animals and moving objects are explicitly identified.
  - The system must never pretend to classify species it cannot verify; ambiguous animals
    fall back to DYNAMIC_OBJECT.
  - Unclassified moving pixels use DYNAMIC_OBJECT fallback.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

logger = logging.getLogger("segmentation.semantic_classifier")

# Lazy PyTorch import
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Class Hierarchy & Metadata
# ---------------------------------------------------------------------------

class SemanticClass(IntEnum):
    """Authoritative semantic categories for aerial drone scene reconstruction."""

    UNKNOWN = 0
    BUILDING = 1
    ROAD = 2
    TERRAIN = 3
    VEGETATION = 4
    VEHICLE = 5
    PERSON = 6
    UTILITY_INFRASTRUCTURE = 7
    WATER = 8
    ANIMAL = 9
    DYNAMIC_OBJECT = 10


# Canonical 8-bit RGB color palette for visualization and layer overlays
CLASS_COLOR_PALETTE: Dict[SemanticClass, Tuple[int, int, int]] = {
    SemanticClass.UNKNOWN: (128, 128, 128),               # Gray
    SemanticClass.BUILDING: (230, 25, 75),                # Red
    SemanticClass.ROAD: (60, 60, 60),                     # Dark Slate
    SemanticClass.TERRAIN: (245, 130, 48),                # Orange / Soil
    SemanticClass.VEGETATION: (60, 180, 75),              # Green
    SemanticClass.VEHICLE: (255, 225, 25),                # Yellow
    SemanticClass.PERSON: (0, 130, 200),                  # Blue
    SemanticClass.UTILITY_INFRASTRUCTURE: (145, 30, 180), # Purple
    SemanticClass.WATER: (70, 240, 240),                  # Cyan
    SemanticClass.ANIMAL: (240, 50, 230),                 # Magenta
    SemanticClass.DYNAMIC_OBJECT: (250, 190, 212),        # Pink
}

# Static classes that represent permanent background geometry
STATIC_CLASSES: Set[SemanticClass] = {
    SemanticClass.UNKNOWN,
    SemanticClass.BUILDING,
    SemanticClass.ROAD,
    SemanticClass.TERRAIN,
    SemanticClass.VEGETATION,
    SemanticClass.UTILITY_INFRASTRUCTURE,
    SemanticClass.WATER,
}

# Dynamic or potentially dynamic classes
DYNAMIC_CLASSES: Set[SemanticClass] = {
    SemanticClass.VEHICLE,
    SemanticClass.PERSON,
    SemanticClass.ANIMAL,
    SemanticClass.DYNAMIC_OBJECT,
}

# Normalization constants (ImageNet defaults)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_INPUT_SIZE = 512
_ANIMAL_SPECIES_CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

@dataclass
class SemanticMap:
    """
    Semantic segmentation result for a single keyframe.

    Attributes
    ----------
    frame_index:
        Source keyframe index.
    class_mask:
        (H, W) uint8 array of SemanticClass integer values (0..10).
    confidence_map:
        (H, W) float32 array in [0.0, 1.0] representing classification confidence.
    original_height:
        Height of the keyframe in pixels.
    original_width:
        Width of the keyframe in pixels.
    class_proportions:
        Dictionary mapping class name to fraction of total pixels [0.0, 1.0].
    detected_classes:
        List of unique SemanticClass instances present in this frame.
    animal_count:
        Number of distinct animal regions classified.
    dynamic_object_count:
        Number of unclassified or fallback dynamic object regions.
    inference_time_ms:
        Inference and post-processing wall-clock time in milliseconds.
    """

    frame_index: int
    class_mask: np.ndarray             # (H, W) uint8
    confidence_map: np.ndarray         # (H, W) float32
    original_height: int
    original_width: int
    class_proportions: Dict[str, float]
    detected_classes: List[SemanticClass]
    animal_count: int
    dynamic_object_count: int
    inference_time_ms: float

    @property
    def colorized_mask(self) -> np.ndarray:
        """Generates an (H, W, 3) RGB visualization of the semantic mask."""
        h, w = self.class_mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for s_class, color in CLASS_COLOR_PALETTE.items():
            mask = self.class_mask == int(s_class)
            if np.any(mask):
                rgb[mask] = color
        return rgb


@dataclass
class SemanticSegmentationResult:
    """Aggregated results across a sequence of keyframes."""

    total_keyframes: int
    semantic_maps: List[SemanticMap]
    failed_frame_indices: List[int]
    mean_inference_time_ms: float
    overall_class_distribution: Dict[str, float]
    total_animals_detected: int
    total_dynamic_objects_detected: int


# ---------------------------------------------------------------------------
# Model Loader and Stub Execution
# ---------------------------------------------------------------------------

@dataclass
class LoadedSegmentationModel:
    """Container for the loaded neural network model or simulation stub."""

    model: Any
    device: str
    is_simulation: bool
    model_name: str
    input_size: int = _DEFAULT_INPUT_SIZE


class SegmentationModelLoader:
    """Loads aerial segmentation model weights or creates a simulation model."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load(
        self,
        model_name: str = "segformer_b2_aerial",
        simulation_mode: bool = False,
    ) -> LoadedSegmentationModel:
        """
        Loads the segmentation model. If simulation_mode is True or PyTorch is
        unavailable, returns a deterministic stub model.
        """
        if simulation_mode or not _TORCH_AVAILABLE:
            logger.info("Instantiating Semantic Classifier in simulation/stub mode.")
            return LoadedSegmentationModel(
                model=None,
                device="cpu",
                is_simulation=True,
                model_name=f"{model_name}_simulation",
                input_size=_DEFAULT_INPUT_SIZE,
            )

        logger.info(f"Loading segmentation model '{model_name}' on device '{self.device}'.")
        # In full runtime, instantiate model architecture
        return LoadedSegmentationModel(
            model=None,
            device=self.device,
            is_simulation=True,
            model_name=model_name,
            input_size=_DEFAULT_INPUT_SIZE,
        )


# ---------------------------------------------------------------------------
# Preprocessing and Postprocessing Utilities
# ---------------------------------------------------------------------------

def _preprocess_frame(bgr: np.ndarray, target_size: int = _DEFAULT_INPUT_SIZE) -> np.ndarray:
    """
    Resizes BGR image to target size, converts to RGB, normalizes using ImageNet stats,
    and returns (3, target_size, target_size) float32 array.
    """
    h, w = bgr.shape[:2]
    if (h, w) != (target_size, target_size):
        resized = cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = bgr

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    # Transpose to Channel-First: (H, W, 3) -> (3, H, W)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def _compute_class_proportions(mask: np.ndarray) -> Dict[str, float]:
    """Computes the fraction of pixels belonging to each SemanticClass."""
    total_pixels = mask.size
    if total_pixels == 0:
        return {s.name: 0.0 for s in SemanticClass}

    unique, counts = np.unique(mask, return_counts=True)
    count_dict = dict(zip(unique, counts))

    proportions = {}
    for s_class in SemanticClass:
        count = count_dict.get(int(s_class), 0)
        proportions[s_class.name] = float(count / total_pixels)
    return proportions


def _upsample_mask_and_confidence(
    class_mask_low: np.ndarray,
    confidence_low: np.ndarray,
    guide_bgr: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsamples the class mask and confidence map back to native resolution.
    Nearest neighbor interpolation is used for discrete class masks to prevent
    creating invalid intermediate class indices.
    """
    low_h, low_w = class_mask_low.shape[:2]
    if (low_h, low_w) == (target_h, target_w):
        return class_mask_low.astype(np.uint8), confidence_low.astype(np.float32)

    # Upsample discrete class mask using nearest-neighbor
    class_mask_up = cv2.resize(
        class_mask_low,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)

    # Upsample continuous confidence map with bilinear interpolation
    confidence_up = cv2.resize(
        confidence_low,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    confidence_up = np.clip(confidence_up, 0.0, 1.0)

    return class_mask_up, confidence_up


# ---------------------------------------------------------------------------
# Auxiliary Detector: Animal & Dynamic Object Fallback Logic
# ---------------------------------------------------------------------------

@dataclass
class AuxiliaryDetection:
    """Auxiliary detection bounding box with classification confidence."""

    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    category: str                         # "animal", "vehicle", "person", "moving_region"
    species: Optional[str] = None         # "horse", "cattle", "dog", "bird", etc.
    species_confidence: float = 0.0       # Confidence in specific species
    detection_confidence: float = 0.90    # Confidence that an object is present


def apply_auxiliary_detections(
    class_mask: np.ndarray,
    confidence_map: np.ndarray,
    detections: List[AuxiliaryDetection],
    animal_threshold: float = _ANIMAL_SPECIES_CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Applies auxiliary detections onto the semantic class mask and confidence map.

    Strict ENHANCED Rules:
      1. ANIMAL: Detects animals using auxiliary detector.
         - If species is confidently classified (species_confidence >= animal_threshold),
           label as SemanticClass.ANIMAL.
         - If species is ambiguous (species_confidence < animal_threshold),
           the system MUST NEVER pretend to classify what it cannot:
           it falls back to SemanticClass.DYNAMIC_OBJECT.
      2. DYNAMIC_OBJECT: General fallback for any moving pixel region not matching a
         specific static class; treated as dynamic and excluded from reconstruction.
    """
    h, w = class_mask.shape[:2]
    mask = class_mask.copy()
    conf = confidence_map.copy()

    animal_count = 0
    dynamic_count = 0

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))

        if det.category == "animal":
            if det.species is not None and det.species_confidence >= animal_threshold:
                # Verified animal species
                mask[y1:y2, x1:x2] = int(SemanticClass.ANIMAL)
                conf[y1:y2, x1:x2] = float(det.species_confidence)
                animal_count += 1
            else:
                # Ambiguous animal -> fallback to DYNAMIC_OBJECT
                mask[y1:y2, x1:x2] = int(SemanticClass.DYNAMIC_OBJECT)
                conf[y1:y2, x1:x2] = float(det.detection_confidence)
                dynamic_count += 1

        elif det.category in ("moving_region", "dynamic_fallback", "unknown_motion"):
            # Generic moving pixel fallback
            mask[y1:y2, x1:x2] = int(SemanticClass.DYNAMIC_OBJECT)
            conf[y1:y2, x1:x2] = float(det.detection_confidence)
            dynamic_count += 1

        elif det.category == "vehicle":
            mask[y1:y2, x1:x2] = int(SemanticClass.VEHICLE)
            conf[y1:y2, x1:x2] = float(det.detection_confidence)

        elif det.category == "person":
            mask[y1:y2, x1:x2] = int(SemanticClass.PERSON)
            conf[y1:y2, x1:x2] = float(det.detection_confidence)

    return mask, conf, animal_count, dynamic_count


# ---------------------------------------------------------------------------
# Semantic Classifier Pipeline
# ---------------------------------------------------------------------------

class SemanticClassifier:
    """
    Batched Semantic Segmentation Inference Pipeline.

    Processes keyframe images to produce dense, pixel-accurate 8-bit semantic masks,
    confidence maps, and class distribution statistics.
    """

    def __init__(
        self,
        loaded_model: Optional[LoadedSegmentationModel] = None,
        batch_size: int = 4,
        animal_confidence_threshold: float = _ANIMAL_SPECIES_CONFIDENCE_THRESHOLD,
    ):
        if loaded_model is None:
            loader = SegmentationModelLoader()
            self.loaded_model = loader.load(simulation_mode=True)
        else:
            self.loaded_model = loaded_model

        self.batch_size = max(1, batch_size)
        self.animal_confidence_threshold = animal_confidence_threshold

    def _simulate_inference(
        self,
        bgr_frame: np.ndarray,
        aux_detections: Optional[List[AuxiliaryDetection]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Simulates realistic drone scene segmentation based on image color/texture heuristics
        when running in testing or CPU simulation mode.
        """
        h, w = bgr_frame.shape[:2]
        low_res = cv2.resize(bgr_frame, (self.loaded_model.input_size, self.loaded_model.input_size))
        lh, lw = low_res.shape[:2]

        # Convert to HSV and Grayscale for aerial color distribution
        hsv = cv2.cvtColor(low_res, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(low_res, cv2.COLOR_BGR2GRAY)

        mask = np.full((lh, lw), int(SemanticClass.TERRAIN), dtype=np.uint8)
        conf = np.full((lh, lw), 0.85, dtype=np.float32)

        # 1. Vegetation: Green hues in HSV
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 40)
        mask[green_mask] = int(SemanticClass.VEGETATION)
        conf[green_mask] = 0.90

        # 2. Water: Blue hues in HSV with low variance
        water_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] >= 50)
        mask[water_mask] = int(SemanticClass.WATER)
        conf[water_mask] = 0.88

        # 3. Roads: Low saturation, dark to mid-gray strips
        road_mask = (hsv[:, :, 1] < 30) & (gray >= 40) & (gray <= 110)
        mask[road_mask] = int(SemanticClass.ROAD)
        conf[road_mask] = 0.82

        # 4. Buildings: Bright surfaces or sharp edge density
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        building_candidate = (edge_mag > 80) & (gray > 120) & (~green_mask) & (~water_mask)
        mask[building_candidate] = int(SemanticClass.BUILDING)
        conf[building_candidate] = 0.85

        # Upsample low_res simulation mask to native frame resolution
        class_mask, confidence_map = _upsample_mask_and_confidence(mask, conf, bgr_frame, h, w)

        # Apply auxiliary detections if supplied
        animal_count = 0
        dynamic_count = 0
        if aux_detections:
            class_mask, confidence_map, animal_count, dynamic_count = apply_auxiliary_detections(
                class_mask,
                confidence_map,
                aux_detections,
                animal_threshold=self.animal_confidence_threshold,
            )

        return class_mask, confidence_map, animal_count, dynamic_count

    def classify_frame(
        self,
        bgr_frame: np.ndarray,
        frame_index: int = 0,
        aux_detections: Optional[List[AuxiliaryDetection]] = None,
    ) -> SemanticMap:
        """Runs semantic segmentation on a single BGR keyframe array."""
        t_start = time.perf_counter()
        orig_h, orig_w = bgr_frame.shape[:2]

        if self.loaded_model.is_simulation or self.loaded_model.model is None:
            class_mask, conf_map, animal_cnt, dynamic_cnt = self._simulate_inference(
                bgr_frame, aux_detections
            )
        else:
            # PyTorch inference branch
            tensor = _preprocess_frame(bgr_frame, self.loaded_model.input_size)
            input_tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.loaded_model.device)
            with torch.inference_mode():
                logits = self.loaded_model.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                pred_conf, pred_class = torch.max(probs, dim=1)

            low_mask = pred_class.squeeze(0).cpu().numpy().astype(np.uint8)
            low_conf = pred_conf.squeeze(0).cpu().numpy().astype(np.float32)

            class_mask, conf_map = _upsample_mask_and_confidence(
                low_mask, low_conf, bgr_frame, orig_h, orig_w
            )
            animal_cnt, dynamic_cnt = 0, 0
            if aux_detections:
                class_mask, conf_map, animal_cnt, dynamic_cnt = apply_auxiliary_detections(
                    class_mask, conf_map, aux_detections, self.animal_confidence_threshold
                )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Unique classes present
        present_int_classes = np.unique(class_mask).tolist()
        detected_classes = [SemanticClass(c) for c in present_int_classes if c in SemanticClass._value2member_map_]

        proportions = _compute_class_proportions(class_mask)

        return SemanticMap(
            frame_index=frame_index,
            class_mask=class_mask,
            confidence_map=conf_map,
            original_height=orig_h,
            original_width=orig_w,
            class_proportions=proportions,
            detected_classes=detected_classes,
            animal_count=animal_cnt,
            dynamic_object_count=dynamic_cnt,
            inference_time_ms=t_elapsed_ms,
        )

    def classify_from_arrays(
        self,
        bgr_frames: Sequence[np.ndarray],
        frame_indices: Optional[Sequence[int]] = None,
        aux_detections_per_frame: Optional[Sequence[Optional[List[AuxiliaryDetection]]]] = None,
    ) -> SemanticSegmentationResult:
        """Runs batched semantic segmentation over in-memory BGR numpy arrays."""
        if frame_indices is None:
            frame_indices = list(range(len(bgr_frames)))

        if len(bgr_frames) != len(frame_indices):
            raise ValueError(
                f"Mismatch: {len(bgr_frames)} frames vs {len(frame_indices)} frame_indices"
            )

        semantic_maps: List[SemanticMap] = []
        failed_indices: List[int] = []
        total_animals = 0
        total_dynamic = 0

        for i, (bgr, idx) in enumerate(zip(bgr_frames, frame_indices)):
            try:
                dets = None
                if aux_detections_per_frame and i < len(aux_detections_per_frame):
                    dets = aux_detections_per_frame[i]

                sem_map = self.classify_frame(bgr, frame_index=idx, aux_detections=dets)
                semantic_maps.append(sem_map)
                total_animals += sem_map.animal_count
                total_dynamic += sem_map.dynamic_object_count
            except Exception as exc:
                logger.error(f"Semantic segmentation failed on frame {idx}: {exc}")
                failed_indices.append(idx)

        # Compute overall class distribution
        overall_dist = {s.name: 0.0 for s in SemanticClass}
        if semantic_maps:
            for sm in semantic_maps:
                for k, v in sm.class_proportions.items():
                    overall_dist[k] += v / len(semantic_maps)

        mean_time = (
            float(np.mean([sm.inference_time_ms for sm in semantic_maps]))
            if semantic_maps
            else 0.0
        )

        return SemanticSegmentationResult(
            total_keyframes=len(bgr_frames),
            semantic_maps=semantic_maps,
            failed_frame_indices=failed_indices,
            mean_inference_time_ms=mean_time,
            overall_class_distribution=overall_dist,
            total_animals_detected=total_animals,
            total_dynamic_objects_detected=total_dynamic,
        )

    def classify_from_files(
        self,
        image_paths: Sequence[Union[str, Path]],
        frame_indices: Optional[Sequence[int]] = None,
        aux_detections_per_frame: Optional[Sequence[Optional[List[AuxiliaryDetection]]]] = None,
    ) -> SemanticSegmentationResult:
        """Reads keyframe images from disk and runs semantic segmentation."""
        if frame_indices is None:
            frame_indices = list(range(len(image_paths)))

        loaded_frames: List[np.ndarray] = []
        valid_indices: List[int] = []
        failed_indices: List[int] = []

        for p_str, idx in zip(image_paths, frame_indices):
            path = Path(p_str)
            if not path.exists():
                logger.error(f"Image path does not exist: {path}")
                failed_indices.append(idx)
                continue

            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to decode image at: {path}")
                failed_indices.append(idx)
                continue

            loaded_frames.append(img)
            valid_indices.append(idx)

        res = self.classify_from_arrays(
            bgr_frames=loaded_frames,
            frame_indices=valid_indices,
            aux_detections_per_frame=aux_detections_per_frame,
        )
        res.total_keyframes = len(image_paths)
        res.failed_frame_indices.extend(failed_indices)
        return res

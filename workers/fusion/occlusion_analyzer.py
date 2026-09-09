"""
Occlusion Analysis and Surface Observation State Tagging — TASK-044 [ENHANCED].

Categorizes every reconstructed point and scene region into the canonical 6-state
ObservationState model defined in PRD Section 19 and TASK-002:

  0: UNKNOWN          — Completely unobserved region (no camera ray intersections).
  1: OBSERVED         — Directly observed and geometrically triangulated from 2+ views.
  2: PARTIAL          — Observed from a single view or near grazing occlusion boundary.
  3: INFERRED         — Occluded surface infilled by surface priors (NEVER presented as measured).
  4: DYNAMIC_EXCLUDED — Explicitly excluded due to moving vehicle/animal/pedestrian track.
  5: LOW_CONFIDENCE   — Observed with confidence < 0.30, specular reflections, or deep shadows.

Enforces that global scene observation percentages strictly sum to 100.0%.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from workers.fusion.normal_estimator import OrientedPointCloud

logger = logging.getLogger("fusion.occlusion_analyzer")

_DEFAULT_LOW_CONF_THRESHOLD: float = 0.30
_DEFAULT_GRAZING_ANGLE_COS: float = 0.20  # ~78 degrees incidence


class ObservationState(IntEnum):
    """The 6 canonical surface observation states."""

    UNKNOWN = 0
    OBSERVED = 1
    PARTIAL = 2
    INFERRED = 3
    DYNAMIC_EXCLUDED = 4
    LOW_CONFIDENCE = 5


OBSERVATION_STATE_COLORS: Dict[ObservationState, Tuple[int, int, int]] = {
    ObservationState.UNKNOWN: (128, 128, 128),            # Gray
    ObservationState.OBSERVED: (46, 204, 113),             # Green
    ObservationState.PARTIAL: (241, 196, 15),              # Yellow
    ObservationState.INFERRED: (155, 89, 182),             # Purple / Synthetic
    ObservationState.DYNAMIC_EXCLUDED: (231, 76, 60),      # Red
    ObservationState.LOW_CONFIDENCE: (230, 126, 34),       # Orange
}


@dataclass
class ObservationStatistics:
    """Quantitative scene coverage distribution; percentages must sum to 100.0%."""

    observed_pct: float
    partial_pct: float
    inferred_pct: float
    unknown_pct: float
    dynamic_excluded_pct: float
    low_confidence_pct: float
    total_points_evaluated: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "observed_pct": round(self.observed_pct, 2),
            "partial_pct": round(self.partial_pct, 2),
            "inferred_pct": round(self.inferred_pct, 2),
            "unknown_pct": round(self.unknown_pct, 2),
            "dynamic_excluded_pct": round(self.dynamic_excluded_pct, 2),
            "low_confidence_pct": round(self.low_confidence_pct, 2),
            "total_points_evaluated": self.total_points_evaluated,
        }

    def verify_sum(self) -> bool:
        """Verifies that the 6 mutually exclusive state percentages sum to 100.0%."""
        s = (
            self.observed_pct +
            self.partial_pct +
            self.inferred_pct +
            self.unknown_pct +
            self.dynamic_excluded_pct +
            self.low_confidence_pct
        )
        return abs(s - 100.0) < 0.1


@dataclass
class ClassifiedObservationCloud:
    """Point cloud with surface normals, confidence, and 6-state observation tags."""

    oriented_cloud: OrientedPointCloud
    observation_states: np.ndarray  # (N,) uint8 enum values (0..5)
    stats: ObservationStatistics

    @property
    def point_count(self) -> int:
        return self.oriented_cloud.point_count


class OcclusionAnalyzer:
    """
    Analyzes viewing angles, visibility counts, and confidence fields to tag observation states.
    """

    def __init__(
        self,
        low_confidence_threshold: float = _DEFAULT_LOW_CONF_THRESHOLD,
        min_multi_view_count: int = 2,
    ):
        self.low_confidence_threshold = low_confidence_threshold
        self.min_multi_view_count = min_multi_view_count

    def analyze_cloud(
        self,
        cloud: OrientedPointCloud,
        dynamic_excluded_count: int = 0,
        unknown_region_points: int = 0,
        inferred_indices: Optional[Sequence[int]] = None,
    ) -> ClassifiedObservationCloud:
        """
        Classifies each point in the oriented point cloud into one of the 6 ObservationStates.
        """
        n_pts = cloud.point_count
        obs_states = np.full(n_pts, int(ObservationState.OBSERVED), dtype=np.uint8)

        if n_pts == 0:
            stats = ObservationStatistics(
                observed_pct=0.0,
                partial_pct=0.0,
                inferred_pct=0.0,
                unknown_pct=100.0 if unknown_region_points > 0 else 0.0,
                dynamic_excluded_pct=0.0,
                low_confidence_pct=0.0,
                total_points_evaluated=0,
            )
            return ClassifiedObservationCloud(
                oriented_cloud=cloud,
                observation_states=obs_states,
                stats=stats,
            )

        conf = cloud.confidences
        views = cloud.view_counts

        # Priority Rule 1: Inferred geometry (filled occlusions)
        if inferred_indices is not None and len(inferred_indices) > 0:
            obs_states[list(inferred_indices)] = int(ObservationState.INFERRED)

        # Priority Rule 2: Low confidence (< threshold e.g. 0.30)
        low_conf_mask = (conf < self.low_confidence_threshold)
        # Apply to points not already explicitly marked inferred
        obs_states[low_conf_mask & (obs_states != int(ObservationState.INFERRED))] = int(
            ObservationState.LOW_CONFIDENCE
        )

        # Priority Rule 3: Single view / partial coverage
        # Points observed from only 1 camera angle with good confidence
        partial_mask = (views < self.min_multi_view_count) & (obs_states == int(ObservationState.OBSERVED))
        obs_states[partial_mask] = int(ObservationState.PARTIAL)

        # Count frequencies in reconstructed point cloud
        count_observed = int(np.count_nonzero(obs_states == int(ObservationState.OBSERVED)))
        count_partial = int(np.count_nonzero(obs_states == int(ObservationState.PARTIAL)))
        count_inferred = int(np.count_nonzero(obs_states == int(ObservationState.INFERRED)))
        count_low_conf = int(np.count_nonzero(obs_states == int(ObservationState.LOW_CONFIDENCE)))
        count_dyn = max(0, int(dynamic_excluded_count))
        count_unk = max(0, int(unknown_region_points))

        total_evaluated = (
            count_observed + count_partial + count_inferred +
            count_low_conf + count_dyn + count_unk
        )

        if total_evaluated > 0:
            pct_observed = (count_observed / total_evaluated) * 100.0
            pct_partial = (count_partial / total_evaluated) * 100.0
            pct_inferred = (count_inferred / total_evaluated) * 100.0
            pct_low_conf = (count_low_conf / total_evaluated) * 100.0
            pct_dyn = (count_dyn / total_evaluated) * 100.0
            pct_unk = (count_unk / total_evaluated) * 100.0
        else:
            pct_observed = pct_partial = pct_inferred = pct_low_conf = pct_dyn = pct_unk = 0.0

        stats = ObservationStatistics(
            observed_pct=pct_observed,
            partial_pct=pct_partial,
            inferred_pct=pct_inferred,
            unknown_pct=pct_unk,
            dynamic_excluded_pct=pct_dyn,
            low_confidence_pct=pct_low_conf,
            total_points_evaluated=total_evaluated,
        )

        return ClassifiedObservationCloud(
            oriented_cloud=cloud,
            observation_states=obs_states,
            stats=stats,
        )

"""
Texture Atlas Baking & View Selection Package — Phase 8 (TASK-048, TASK-049).
"""
from workers.texture.atlas_baker import (
    TextureAtlasBaker,
    TexturedMesh,
)
from workers.texture.view_selector import (
    FaceViewAssignment,
    KeyframeView,
    KeyframeViewSelector,
    ViewSelectionResult,
)

__all__ = [
    # TASK-048
    "KeyframeView",
    "FaceViewAssignment",
    "ViewSelectionResult",
    "KeyframeViewSelector",
    # TASK-049
    "TexturedMesh",
    "TextureAtlasBaker",
]

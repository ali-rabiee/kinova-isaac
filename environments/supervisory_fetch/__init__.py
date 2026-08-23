"""The supervisory-fetch scene: a target, a blocker, and a controllable clearance gap.

Isaac-free at import time. The USD paths and simulator handles are resolved inside
:mod:`environments.supervisory_fetch.scene`, in the functions that need them, so the offline
test suite and the Tier-1 study stay runnable on a machine with no simulator.
"""

from .config import (
    DEFAULT_SUP_SCENE,
    DEFAULT_SUP_TOPDOWN_CAMERA,
    DEFAULT_SUP_WRIST_CAMERA,
    SupSceneConfig,
    layout_for_margin,
)

__all__ = [
    "SupSceneConfig",
    "DEFAULT_SUP_SCENE",
    "DEFAULT_SUP_TOPDOWN_CAMERA",
    "DEFAULT_SUP_WRIST_CAMERA",
    "layout_for_margin",
]

"""Digital twin for the ARCHIVED arm-choice study (see vla_lab/old_direction/).

Kept runnable, not maintained: the live direction uses
``environments/supervisory_fetch/`` instead.
"""

"""W10 — the Isaac Lab scene for the Phase 0 bilateral-choice apparatus (``rehab.md`` §8).

Mirrors the structure of ``environments/reach_to_grasp_VLA/`` (config + scene + demo) but is
a *different environment for a different job*: the robot presents standardized reach targets
across a bilateral tabletop workspace and never manipulates anything, and a seated-participant
proxy volume is part of the scene.

Nothing here is imported by the VLA track, and ``environments/reach_to_grasp_VLA/config.py``
is untouched — the Phase 0 wrist mount lives in :class:`~environments.bilateral_choice.config.TwinWristCameraConfig`
so the VLA camera contract is not disturbed.

Run the dry-run with::

    ./vla_lab/scripts/rehab_twin_dryrun.sh
"""

from __future__ import annotations

from .config import (
    DEFAULT_TWIN,
    TwinConfig,
    TwinFrontCameraConfig,
    TwinSceneConfig,
    TwinWristCameraConfig,
)
from .twin import BilateralChoiceTwin

__all__ = [
    "BilateralChoiceTwin",
    "TwinConfig",
    "TwinSceneConfig",
    "TwinFrontCameraConfig",
    "TwinWristCameraConfig",
    "DEFAULT_TWIN",
]

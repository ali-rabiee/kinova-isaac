from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EnvSpec:
    """Specifies where to import scene construction utilities for an environment."""

    name: str
    module_base: str  # e.g. "environments.ycb_reach_to_grasp"


def get_envs() -> Dict[str, EnvSpec]:
    # Add new environments here.
    envs = [
        EnvSpec(name="ycb_reach_to_grasp", module_base="environments.ycb_reach_to_grasp"),
        EnvSpec(name="cubes", module_base="environments.cubes"),
        # Backwards-compatible aliases for older profile/CLI names.
        EnvSpec(name="reach_to_grasp", module_base="environments.ycb_reach_to_grasp"),
        EnvSpec(name="reach_to_grasp_VLA", module_base="environments.ycb_reach_to_grasp"),
    ]
    return {e.name: e for e in envs}



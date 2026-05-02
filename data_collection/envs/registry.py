from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EnvSpec:
    """Specifies where to import scene construction utilities for an environment."""

    name: str
    module_base: str  # e.g. "environments.reach_to_grasp"


def get_envs() -> Dict[str, EnvSpec]:
    # Add new environments here.
    envs = [
        EnvSpec(name="reach_to_grasp", module_base="environments.reach_to_grasp"),
        EnvSpec(name="reach_to_grasp_VLA", module_base="environments.reach_to_grasp_VLA"),
    ]
    return {e.name: e for e in envs}



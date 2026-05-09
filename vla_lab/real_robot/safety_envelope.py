"""Safety envelope helpers for real-robot policy execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class WorkspaceAABB:
    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]


@dataclass
class SafetyEnvelope:
    workspace: WorkspaceAABB
    max_linear_vel_m_s: float = 0.25
    max_angular_vel_rad_s: float = 0.8
    max_wrench_N: float = 80.0  # placeholder; use real driver thresholds
    estop_topic: str = "/estop"

    def clamp_delta_pose(self, delta_xyz_rpy: Tuple[float, ...]) -> Tuple[float, ...]:
        # Placeholder: clamp translation magnitude component-wise (improve with true velocity limits).
        if len(delta_xyz_rpy) < 6:
            return delta_xyz_rpy
        mx = float(self.max_linear_vel_m_s)
        mr = float(self.max_angular_vel_rad_s)
        x, y, z, rx, ry, rz = delta_xyz_rpy[:6]
        def _c(v: float, lim: float) -> float:
            return max(-lim, min(lim, v))
        return (_c(x, mx), _c(y, mx), _c(z, mx), _c(rx, mr), _c(ry, mr), _c(rz, mr), *delta_xyz_rpy[6:])

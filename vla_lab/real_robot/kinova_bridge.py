"""Real-robot deployment stubs (Kinova Gen 3 class naming in LeRobot export metadata).

This is a **baseline wiring** surface: plug in your ROS/pyKinova stack where marked.
`new_changes.md` calls for safety bounds + policy stepping; we keep imports light so
the module imports even without ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch


class _PolicyFn(Protocol):
    def __call__(self, image_chw: torch.Tensor, state: torch.Tensor, instruction: str) -> torch.Tensor:
        """Return action chunk (T, 7)."""


@dataclass
class KinovaPolicyServerConfig:
    control_rate_hz: float = 5.0
    camera_topic: str = "/camera/color/image_raw"
    instruction: str = "pick up the target object"


class KinovaPolicyServer:
    """High-level stepping loop: camera + proprio → policy chunk → controller.

    Replace `_read_image` / `_read_eef` / `_execute_delta` with your robot I/O.
    """

    def __init__(
        self,
        *,
        policy: _PolicyFn,
        cfg: KinovaPolicyServerConfig,
        controller: Any = None,
        camera: Any = None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.controller = controller
        self.camera = camera

    def _read_image_chw(self) -> torch.Tensor:
        raise NotImplementedError("Wire this to your camera subscriber / latest frame buffer.")

    def _read_state(self) -> torch.Tensor:
        raise NotImplementedError("Wire this to your end-effector / joint state encoder.")

    def _execute_delta(self, delta_7: torch.Tensor) -> None:
        raise NotImplementedError("Wire this to your IK + velocity/position controller.")

    def step(self, instruction: Optional[str] = None) -> torch.Tensor:
        task = instruction or self.cfg.instruction
        img = self._read_image_chw()
        st = self._read_state()
        action_chunk = self.policy(img, st, task)
        if action_chunk.dim() != 2:
            raise ValueError(f"Expected (T,A) chunk; got {tuple(action_chunk.shape)}")
        for a in action_chunk:
            self._execute_delta(a)
        return action_chunk

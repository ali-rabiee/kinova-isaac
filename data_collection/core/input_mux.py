from __future__ import annotations

from typing import List, Optional

import torch

from controllers.base import InputProvider


class CommandMuxInputProvider(InputProvider):
    """Multiplex base input with an injected action command stream.

    - When idle, returns base_provider.advance().
    - When an action is active, returns the next precomputed command tensor.
    """

    def __init__(self, base_provider: Optional[InputProvider] = None) -> None:
        self._base = base_provider
        self._action_stream: List[torch.Tensor] = []
        self._cursor: int = 0
        self._last_cmd: Optional[torch.Tensor] = None

    def set_base(self, base_provider: Optional[InputProvider]) -> None:
        self._base = base_provider

    def reset(self) -> None:
        if self._base is not None:
            self._base.reset()
        self._action_stream = []
        self._cursor = 0
        self._last_cmd = None

    def run_action(self, stream: List[torch.Tensor]) -> None:
        self._action_stream = stream
        self._cursor = 0

    def cancel_action(self) -> None:
        self._action_stream = []
        self._cursor = 0

    def is_action_active(self) -> bool:
        return self._cursor < len(self._action_stream)

    def advance(self) -> torch.Tensor:
        if self.is_action_active():
            cmd = self._action_stream[self._cursor]
            self._cursor += 1
            self._last_cmd = cmd
            return cmd
        if self._base is None:
            cmd = torch.zeros(1, 6)
            self._last_cmd = cmd
            return cmd
        cmd = self._base.advance()
        self._last_cmd = cmd
        return cmd

    @property
    def last_cmd(self) -> Optional[torch.Tensor]:
        return self._last_cmd


class SampleAndHoldInputProvider(InputProvider):
    """Sample an underlying provider on demand, otherwise return the last sample.

    This is useful when you want a *policy-rate* command to be held across multiple
    physics steps, while keeping the controller loop unchanged (it still calls
    `advance()` every physics tick).
    """

    def __init__(
        self,
        base_provider: Optional[InputProvider] = None,
        *,
        default_dim: int = 6,
        device: Optional[str] = None,
    ) -> None:
        self._base = base_provider
        self._default_dim = int(default_dim)
        self._device = torch.device(device) if device is not None else None
        self._held: Optional[torch.Tensor] = None

    def set_base(self, base_provider: Optional[InputProvider]) -> None:
        self._base = base_provider

    def reset(self) -> None:
        if self._base is not None:
            self._base.reset()
        self._held = None

    def set_held(self, cmd: torch.Tensor) -> None:
        cmd_t = cmd
        if cmd_t.ndim == 1:
            cmd_t = cmd_t.view(1, -1)
        if self._device is not None and str(cmd_t.device) != str(self._device):
            cmd_t = cmd_t.to(self._device)
        self._held = cmd_t

    def sample(self) -> torch.Tensor:
        """Advance the underlying provider once and hold that command."""
        if self._base is None:
            cmd = torch.zeros(1, self._default_dim, dtype=torch.float32, device=self._device)
            self._held = cmd
            return cmd
        cmd = self._base.advance()
        if cmd.ndim == 1:
            cmd = cmd.view(1, -1)
        if self._device is not None and str(cmd.device) != str(self._device):
            cmd = cmd.to(self._device)
        self._held = cmd
        return cmd

    def advance(self) -> torch.Tensor:
        if self._held is None:
            return torch.zeros(1, self._default_dim, dtype=torch.float32, device=self._device)
        return self._held

    @property
    def last_cmd(self) -> Optional[torch.Tensor]:
        return self._held


class SharedAutonomyBlendInputProvider(InputProvider):
    """Blend human and autonomous commands via a convex combination.

    Outputs: u_exec = (1-gamma)*u_human + gamma*u_auto, where gamma in [0,1].

    Notes:
    - This provider assumes both command streams are in the same frame and scaling.
    - Commands are padded/truncated to 7D: [dx, dy, dz, rx, ry, rz, g].
    """

    def __init__(
        self,
        *,
        human_provider: Optional[InputProvider] = None,
        auto_provider: Optional[InputProvider] = None,
        gamma: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        self._human = human_provider
        self._auto = auto_provider
        self._gamma: float = float(gamma)
        self._device = torch.device(device) if device is not None else None

        self._last_human: Optional[torch.Tensor] = None
        self._last_auto: Optional[torch.Tensor] = None
        self._last_exec: Optional[torch.Tensor] = None
        self._last_gamma: Optional[float] = None

    def set_human(self, human_provider: Optional[InputProvider]) -> None:
        self._human = human_provider

    def set_auto(self, auto_provider: Optional[InputProvider]) -> None:
        self._auto = auto_provider

    def set_gamma(self, gamma: float) -> None:
        self._gamma = float(gamma)

    def reset(self) -> None:
        if self._human is not None:
            self._human.reset()
        if self._auto is not None:
            self._auto.reset()
        self._last_human = None
        self._last_auto = None
        self._last_exec = None
        self._last_gamma = None

    @staticmethod
    def _as_2d(cmd: torch.Tensor) -> torch.Tensor:
        return cmd.view(1, -1) if cmd.ndim == 1 else cmd

    def _normalize_cmd7(self, cmd: torch.Tensor) -> torch.Tensor:
        cmd2 = self._as_2d(cmd)
        if self._device is not None and str(cmd2.device) != str(self._device):
            cmd2 = cmd2.to(self._device)
        # Pad or truncate to 7D
        if cmd2.shape[-1] < 7:
            pad = torch.zeros(cmd2.shape[0], 7 - cmd2.shape[-1], device=cmd2.device, dtype=cmd2.dtype)
            cmd2 = torch.cat([cmd2, pad], dim=-1)
        elif cmd2.shape[-1] > 7:
            cmd2 = cmd2[..., :7]
        return cmd2

    def advance(self) -> torch.Tensor:
        gamma = float(max(0.0, min(1.0, float(self._gamma))))

        if self._human is None:
            u_h = torch.zeros(1, 7, dtype=torch.float32, device=self._device)
        else:
            u_h = self._normalize_cmd7(self._human.advance())

        if self._auto is None:
            u_a = torch.zeros(1, 7, dtype=torch.float32, device=self._device)
        else:
            u_a = self._normalize_cmd7(self._auto.advance())

        u_exec = (1.0 - gamma) * u_h + gamma * u_a

        self._last_human = u_h
        self._last_auto = u_a
        self._last_exec = u_exec
        self._last_gamma = gamma

        return u_exec

    @property
    def last_human_cmd(self) -> Optional[torch.Tensor]:
        return self._last_human

    @property
    def last_auto_cmd(self) -> Optional[torch.Tensor]:
        return self._last_auto

    @property
    def last_exec_cmd(self) -> Optional[torch.Tensor]:
        return self._last_exec

    @property
    def last_gamma(self) -> Optional[float]:
        return self._last_gamma


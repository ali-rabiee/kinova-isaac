"""Uncertainty-*type* transparency — the HRI condition that isolates §5.2 cond. 5 vs 4.

The study's key manipulation (memo §5.2): does communicating the **type** of uncertainty
("I'm thinking" vs "I can't see — which one?"), rather than just its **magnitude** ("I'm
unsure"), change human behaviour and improve appropriate reliance? This module renders the
human-facing signal for a decision; the controller is identical across conditions, only the
message differs:

- ``cond4`` (magnitude only): a single generic "unsure" signal regardless of cause.
- ``cond5`` (type): distinct messages for *reducible* (compute / "thinking") vs
  *irreducible* (query / "can't see") uncertainty.

Keeping this isolated means the same allocator drives every interaction condition; only
``TransparencyConfig`` changes between cells of the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TransparencyConfig:
    """How (and whether) to surface uncertainty to the human.

    enabled:
        If False, the robot never signals uncertainty (autonomy / silent conditions).
    signal_type:
        If True, communicate the *type* of uncertainty (cond. 5). If False, communicate
        only that it is uncertain (cond. 5's control, cond. 4).
    """

    enabled: bool = True
    signal_type: bool = True
    msg_act: str = ""
    msg_thinking: str = "Thinking…"
    msg_unsure_generic: str = "I'm not sure about this step."
    msg_cant_see: str = "I can't see the target clearly — which one should I pick?"

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "signal_type": bool(self.signal_type),
            "msg_act": self.msg_act,
            "msg_thinking": self.msg_thinking,
            "msg_unsure_generic": self.msg_unsure_generic,
            "msg_cant_see": self.msg_cant_see,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TransparencyConfig":
        base = cls()
        return cls(
            enabled=bool(d.get("enabled", base.enabled)),
            signal_type=bool(d.get("signal_type", base.signal_type)),
            msg_act=str(d.get("msg_act", base.msg_act)),
            msg_thinking=str(d.get("msg_thinking", base.msg_thinking)),
            msg_unsure_generic=str(d.get("msg_unsure_generic", base.msg_unsure_generic)),
            msg_cant_see=str(d.get("msg_cant_see", base.msg_cant_see)),
        )


def uncertainty_type_for(decision: str) -> str:
    """Map an allocator decision to an uncertainty *type* label.

    ``compute`` -> reducible (the policy represents the ambiguity and may resolve it);
    ``query`` -> irreducible (information absent from the view); ``act`` -> none.
    """

    d = str(decision).lower()
    if d == "compute":
        return "reducible"
    if d == "query":
        return "irreducible"
    return "none"


def render_message(decision: str, cfg: TransparencyConfig) -> str:
    """Human-facing string for a decision under a transparency condition."""

    if not cfg.enabled:
        return ""
    utype = uncertainty_type_for(decision)
    if utype == "none":
        return cfg.msg_act
    if not cfg.signal_type:
        # Magnitude-only: one generic message for any non-act step.
        return cfg.msg_unsure_generic
    # Type-aware messaging.
    if utype == "reducible":
        return cfg.msg_thinking
    return cfg.msg_cant_see

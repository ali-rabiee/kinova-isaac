"""Shared datatypes for the real-time semantic feedback channel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class Interjection:
    """One unsolicited human utterance delivered to the robot mid-execution."""

    text: str
    tick_idx: int = -1  # policy tick the utterance was DELIVERED at (-1 = unknown)
    source: str = "human"  # human | sim
    # Sim-only ground truth for logging/experiments: did the utterance point at
    # the true target (None when unknowable, e.g. chatter or real humans).
    intended_correct: Optional[bool] = None


# Base-frame direction conventions used by the parser AND the simulated human
# (self-consistent in sim; document to real operators): x forward (into the
# workspace), y left, z up.
DIRECTION_VECTORS: Dict[str, Tuple[float, float, float]] = {
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "forward": (1.0, 0.0, 0.0),
    "ahead": (1.0, 0.0, 0.0),
    "away": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "closer": (-1.0, 0.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "higher": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "lower": (0.0, 0.0, -1.0),
}


@dataclass
class Refinement:
    """Parsed meaning of an utterance.

    kinds:
      target       — override the believed target ("no, the red one")
      avoid        — exclude a candidate ("not the red one")
      nudge        — bias motion along `direction` ("a bit to the left")
      stop/resume  — pause / continue execution
      confirm_yes / confirm_no — answer to a robot confirmation question
      unknown      — unparseable / chatter
    """

    kind: str
    target_label: Optional[str] = None
    direction: Optional[Tuple[float, float, float]] = None  # unit-ish base-frame vector
    magnitude: float = 1.0  # 0.5 = "slightly", 1.5 = "much more"
    raw_text: str = ""
    matched_tokens: Tuple[str, ...] = field(default_factory=tuple)

"""Simulated human with explicit INPUT-QUALITY knobs.

One model serves both feedback directions so quality is a single controlled
variable in experiments:

- unsolicited corrections (``observe`` -> Interjections) fire when the robot
  visibly heads for a wrong object;
- answers to robot-initiated queries (``ask``) reuse the same accuracy knob,
  mirroring :class:`vla_lab.allocation.query.ScriptedQueryInterface`.

Quality axes (SimulatedHumanConfig):
  accuracy      P(the human refers to the TRUE target)     — correctness
  latency_ticks delivery delay in policy ticks             — timeliness
  specificity   color | directional | vague                — information content
  noise_rate    P(chatter instead of a correction)         — signal-to-noise

Everything is seeded and deterministic given (seed, episode).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..allocation.query import QueryAnswer
from .types import Interjection

_CHATTER = ["hmm let me think", "interesting", "keep going i guess", "the weather is nice today"]
_VAGUE = ["not that one", "no not this one", "wrong one"]


@dataclass
class SimulatedHumanConfig:
    accuracy: float = 0.95
    latency_ticks: int = 3
    specificity: str = "color"  # color | directional | vague
    noise_rate: float = 0.0
    react_cos: float = 0.85  # EE heading alignment with a wrong object that triggers a correction
    react_dist_m: float = 0.30  # only react when EE is within this XY distance of that object
    min_gap_ticks: int = 20  # refractory period between interjections
    query_latency_s: float = 3.0  # simulated interruption latency for robot-initiated queries
    seed: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "accuracy": float(self.accuracy),
            "latency_ticks": int(self.latency_ticks),
            "specificity": str(self.specificity),
            "noise_rate": float(self.noise_rate),
            "react_cos": float(self.react_cos),
            "react_dist_m": float(self.react_dist_m),
            "min_gap_ticks": int(self.min_gap_ticks),
            "query_latency_s": float(self.query_latency_s),
            "seed": int(self.seed),
        }


class SimulatedHuman:
    """Watches the rollout and interjects; also answers explicit queries."""

    def __init__(
        self,
        cfg: SimulatedHumanConfig,
        *,
        true_target_label: str,
        candidate_labels: Sequence[str],
        episode_idx: int = 0,
    ) -> None:
        self.cfg = cfg
        self.true_target = str(true_target_label)
        self.labels = [str(l) for l in candidate_labels]
        self.rng = random.Random(int(cfg.seed) * 100003 + int(episode_idx))
        self._pending: List[Tuple[int, Interjection]] = []  # (deliver_at_tick, interjection)
        self._last_fire_tick = -(10 ** 9)
        self._prev_ee: Optional[Tuple[float, float, float]] = None
        self.n_triggers = 0

    # ------------------------------------------------------------- utterances
    def _referred_label(self) -> Tuple[str, bool]:
        """The label the human refers to; correct with P(accuracy)."""
        if self.rng.random() < float(self.cfg.accuracy) or len(self.labels) <= 1:
            return self.true_target, True
        others = [l for l in self.labels if l != self.true_target]
        return self.rng.choice(others), False

    def _correction_text(
        self,
        referred: str,
        wrong_obj_pos: Optional[Sequence[float]],
        obj_pos_b: Dict[str, Sequence[float]],
    ) -> str:
        spec = str(self.cfg.specificity)
        if spec == "color":
            return self.rng.choice([f"no, the {referred} one", f"the {referred} box", f"go for the {referred} one"])
        if spec == "directional" and wrong_obj_pos is not None and referred in obj_pos_b:
            tgt = obj_pos_b[referred]
            d = [float(tgt[k]) - float(wrong_obj_pos[k]) for k in range(3)]
            # Dominant axis -> word (base frame: x forward, y left, z up).
            words = [("forward", "back"), ("left", "right"), ("up", "down")]
            ax = max(range(3), key=lambda k: abs(d[k]))
            word = words[ax][0 if d[ax] >= 0 else 1]
            prefix = self.rng.choice(["more", "a bit", ""])
            return f"{prefix} {word}".strip()
        return self.rng.choice(_VAGUE)

    # ------------------------------------------------------------- observation
    def observe(
        self,
        *,
        tick_idx: int,
        ee_pos_b: Sequence[float],
        object_pos_b: Dict[str, Sequence[float]],
    ) -> List[Interjection]:
        """Feed one policy tick; returns interjections DUE at this tick."""

        ee = (float(ee_pos_b[0]), float(ee_pos_b[1]), float(ee_pos_b[2]))
        heading = None
        if self._prev_ee is not None:
            h = [ee[k] - self._prev_ee[k] for k in range(3)]
            n = (h[0] ** 2 + h[1] ** 2 + h[2] ** 2) ** 0.5
            if n > 1e-4:
                heading = [x / n for x in h]
        self._prev_ee = ee

        # Reactive trigger: heading at a WRONG object while near it.
        if heading is not None and (tick_idx - self._last_fire_tick) >= int(self.cfg.min_gap_ticks):
            for lbl, pos in object_pos_b.items():
                if lbl == self.true_target:
                    continue
                to_obj = [float(pos[k]) - ee[k] for k in range(3)]
                dist = (to_obj[0] ** 2 + to_obj[1] ** 2 + to_obj[2] ** 2) ** 0.5
                if dist > float(self.cfg.react_dist_m) or dist < 1e-6:
                    continue
                cos = sum(heading[k] * to_obj[k] for k in range(3)) / dist
                if cos >= float(self.cfg.react_cos):
                    self._last_fire_tick = tick_idx
                    self.n_triggers += 1
                    if self.rng.random() < float(self.cfg.noise_rate):
                        text, correct = self.rng.choice(_CHATTER), None
                    else:
                        referred, correct = self._referred_label()
                        text = self._correction_text(referred, pos, object_pos_b)
                    deliver_at = tick_idx + max(0, int(self.cfg.latency_ticks))
                    self._pending.append(
                        (deliver_at, Interjection(text=text, tick_idx=deliver_at, source="sim", intended_correct=correct))
                    )
                    break

        due = [i for at, i in self._pending if at <= tick_idx]
        if due:
            self._pending = [(at, i) for at, i in self._pending if at > tick_idx]
        return due

    # ------------------------------------------------------------------ query
    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        """Robot-initiated query; same accuracy model as corrections."""
        referred, correct = self._referred_label()
        opt_idx: Optional[int] = None
        if options:
            lowered = [str(o).lower() for o in options]
            if referred.lower() in lowered:
                opt_idx = lowered.index(referred.lower())
        return QueryAnswer(
            text=referred,
            option_idx=opt_idx,
            latency_s=float(self.cfg.query_latency_s),
            source="oracle",
            correct=bool(correct),
        )

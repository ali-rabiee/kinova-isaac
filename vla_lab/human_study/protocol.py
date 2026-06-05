"""Study design: independent variables, the 5 interaction conditions, and counterbalancing.

IV1 — **Observability** (memo §5.2): clean → graded occlusion, reusing the same dial as the
robot experiments (``bottom_strip(ρ)`` for ρ∈{.15,.25,.35,.50}, plus center/random patches).

IV2 — **Controller / interaction condition**:
  1. Autonomy (K=1, never asks)                              — transparency off
  2. Fixed compute (best-of-N, never asks)                   — transparency off
  3. Compute-gated (the existing method, never asks)         — transparency off
  4. Compute-or-query allocator (proposed)                   — asks; magnitude-only signal
  5. Allocator + uncertainty-**type** transparency           — asks; signals *type*

Condition 5 vs. 4 isolates whether communicating the *type* (not just magnitude) of
uncertainty changes human behaviour. Order is counterbalanced with a balanced Latin square
(within-subjects) or assigned round-robin (between-subjects).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class Condition:
    idx: int
    name: str
    controller_kind: str
    transparency_enabled: bool
    transparency_signal_type: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservabilityLevel:
    mode: str  # none | bottom_strip | top_strip | center_box | random_patch
    fraction: float

    @property
    def label(self) -> str:
        if self.mode in ("none", "off", ""):
            return "clean"
        return f"{self.mode}_{self.fraction:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "fraction": float(self.fraction), "label": self.label}


def default_conditions() -> List[Condition]:
    return [
        Condition(1, "autonomy", "autonomy", False, False),
        Condition(2, "fixed_compute", "fixed_compute", False, False),
        Condition(3, "compute_gated", "compute_gated", False, False),
        Condition(4, "allocator", "allocator", True, False),
        Condition(5, "allocator_type", "allocator", True, True),
    ]


def default_observability() -> List[ObservabilityLevel]:
    return [
        ObservabilityLevel("none", 0.0),
        ObservabilityLevel("bottom_strip", 0.15),
        ObservabilityLevel("bottom_strip", 0.25),
        ObservabilityLevel("bottom_strip", 0.35),
        ObservabilityLevel("bottom_strip", 0.50),
        ObservabilityLevel("center_box", 0.35),
        ObservabilityLevel("random_patch", 0.35),
    ]


@dataclass
class Trial:
    trial_idx: int
    block_idx: int
    condition_idx: int
    condition_name: str
    controller_kind: str
    transparency_enabled: bool
    transparency_signal_type: bool
    occlusion_mode: str
    occlusion_fraction: float
    rep: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def balanced_latin_square(n: int) -> List[List[int]]:
    """Return ``n`` orderings of ``0..n-1`` forming a balanced Latin square.

    For even ``n`` this is the standard balanced square (each condition appears in each
    position once, and every ordered pair is balanced). For odd ``n`` we append the reverse
    of the even construction to retain first-order balance.
    """

    def row(i: int) -> List[int]:
        seq = [0]
        k = 1
        forward = True
        while len(seq) < n:
            seq.append((seq[-1] + k) % n if forward else (seq[-1] - k) % n)
            forward = not forward
            k += 1
        return [(x + i) % n for x in seq]

    rows = [row(i) for i in range(n)]
    if n % 2 == 1:
        rows += [list(reversed(r)) for r in rows]
    return rows


@dataclass
class StudyProtocol:
    conditions: List[Condition] = field(default_factory=default_conditions)
    observability: List[ObservabilityLevel] = field(default_factory=default_observability)
    design: str = "within"  # "within" | "between"
    reps_per_cell: int = 1
    n_participants: int = 24
    instruments: List[str] = field(default_factory=lambda: ["nasa_tlx", "jian_trust"])
    task: str = "reach_grasp_lift_color"
    seed: int = 1337
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyProtocol":
        conds = d.get("conditions")
        observ = d.get("observability")
        conditions = (
            [Condition(**{k: c[k] for k in ("idx", "name", "controller_kind", "transparency_enabled", "transparency_signal_type")}) for c in conds]
            if conds
            else default_conditions()
        )
        observability = (
            [ObservabilityLevel(mode=str(o["mode"]), fraction=float(o.get("fraction", 0.0))) for o in observ]
            if observ
            else default_observability()
        )
        return cls(
            conditions=conditions,
            observability=observability,
            design=str(d.get("design", "within")),
            reps_per_cell=int(d.get("reps_per_cell", 1)),
            n_participants=int(d.get("n_participants", 24)),
            instruments=list(d.get("instruments", ["nasa_tlx", "jian_trust"])),
            task=str(d.get("task", "reach_grasp_lift_color")),
            seed=int(d.get("seed", 1337)),
            notes=str(d.get("notes", "")),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "StudyProtocol":
        import yaml

        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(d.get("study", d))


@dataclass
class SessionPlan:
    participant_id: str
    design: str
    seed: int
    trials: List[Trial]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "design": self.design,
            "seed": self.seed,
            "n_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials],
        }

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SessionPlan":
        d = json.loads(Path(path).read_text())
        trials = [Trial(**{k: t[k] for k in Trial.__dataclass_fields__ if k in t}) for t in d.get("trials", [])]  # type: ignore[attr-defined]
        return cls(participant_id=d["participant_id"], design=d.get("design", "within"), seed=int(d.get("seed", 0)), trials=trials)


def generate_session_plan(
    participant_idx: int,
    protocol: StudyProtocol,
    *,
    participant_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> SessionPlan:
    """Build a reproducible, counterbalanced trial order for one participant."""

    pid = participant_id or f"P{participant_idx:03d}"
    base_seed = int(seed if seed is not None else protocol.seed) + int(participant_idx)
    rng = random.Random(base_seed)
    conds = protocol.conditions
    obs = protocol.observability
    trials: List[Trial] = []
    tidx = 0

    if protocol.design == "between":
        cond = conds[participant_idx % len(conds)]
        order = list(range(len(obs)))
        rng.shuffle(order)
        for rep in range(max(1, protocol.reps_per_cell)):
            for oi in order:
                o = obs[oi]
                trials.append(_mk_trial(tidx, 0, cond, o, rep))
                tidx += 1
        return SessionPlan(pid, protocol.design, base_seed, trials)

    # within-subjects: counterbalance condition order via Latin square
    square = balanced_latin_square(len(conds))
    cond_order = square[participant_idx % len(square)]
    for block_idx, ci in enumerate(cond_order):
        cond = conds[ci]
        order = list(range(len(obs)))
        rng.shuffle(order)
        for rep in range(max(1, protocol.reps_per_cell)):
            for oi in order:
                o = obs[oi]
                trials.append(_mk_trial(tidx, block_idx, cond, o, rep))
                tidx += 1
    return SessionPlan(pid, protocol.design, base_seed, trials)


def _mk_trial(tidx: int, block_idx: int, cond: Condition, o: ObservabilityLevel, rep: int) -> Trial:
    return Trial(
        trial_idx=tidx,
        block_idx=block_idx,
        condition_idx=cond.idx,
        condition_name=cond.name,
        controller_kind=cond.controller_kind,
        transparency_enabled=cond.transparency_enabled,
        transparency_signal_type=cond.transparency_signal_type,
        occlusion_mode=o.mode,
        occlusion_fraction=float(o.fraction),
        rep=rep,
    )

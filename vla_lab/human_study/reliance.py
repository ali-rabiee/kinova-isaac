"""Appropriate reliance, the over-trust trap, and trust calibration (memo §5.3–§5.4).

The headline HRI construct. For each trial we observe whether the robot was ultimately
**correct** and whether the human **intervened** (took manual control — distinct from merely
answering a robot's query, which is collaboration, not intervention). The classic 2×2
(Lee & See; Wang & Yin) gives:

    robot_correct & not intervened   → appropriate acceptance  (trusted good autonomy)
    not correct  & intervened        → appropriate rejection   (caught the error)
    not correct  & not intervened    → **over-trust**          (the dangerous miss)
    robot_correct & intervened       → under-trust             (needless takeover)

The over-trust trap is the robot being most dangerous to its partner exactly when it is
**confidently wrong** — low self-uncertainty, so the human has no internal-signal reason to
intervene. The target finding (§5.4) is that a controller which distinguishes "I should
think" from "I must ask you" — and *communicates which* — reduces over-trust without
inflating under-trust.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

CORRECT_RELIANCE = "correct_reliance"
CORRECT_INTERVENTION = "correct_intervention"
OVER_TRUST = "over_trust"
UNDER_TRUST = "under_trust"


@dataclass
class RelianceTrial:
    """One trial's reliance-relevant outcome."""

    robot_correct: bool
    human_intervened: bool
    success: Optional[bool] = None
    robot_queried: bool = False
    robot_signaled_uncertainty: bool = False
    confidently_wrong: Optional[bool] = None  # robot wrong AND did not signal/seek help
    condition_name: str = ""
    participant_id: str = ""
    occlusion_fraction: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidently_wrong is None:
            self.confidently_wrong = (not self.robot_correct) and (not self.robot_signaled_uncertainty) and (not self.robot_queried)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelianceTrial":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        base = {k: d[k] for k in d if k in known}
        return cls(**base)


def classify_reliance(trial: RelianceTrial) -> str:
    if trial.robot_correct and not trial.human_intervened:
        return CORRECT_RELIANCE
    if (not trial.robot_correct) and trial.human_intervened:
        return CORRECT_INTERVENTION
    if (not trial.robot_correct) and not trial.human_intervened:
        return OVER_TRUST
    return UNDER_TRUST


def _phi(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def reliance_summary(trials: Sequence[RelianceTrial]) -> Dict[str, float]:
    """Rates over a set of trials (one condition, or pooled)."""

    n = len(trials)
    if n == 0:
        return {"n": 0}
    labels = [classify_reliance(t) for t in trials]
    n_correct_robot = sum(1 for t in trials if t.robot_correct)
    n_wrong_robot = n - n_correct_robot
    over = labels.count(OVER_TRUST)
    under = labels.count(UNDER_TRUST)
    appropriate = labels.count(CORRECT_RELIANCE) + labels.count(CORRECT_INTERVENTION)
    interventions = sum(1 for t in trials if t.human_intervened)
    queries = sum(1 for t in trials if t.robot_queried)
    successes = sum(1 for t in trials if t.success)
    conf_wrong = sum(1 for t in trials if t.confidently_wrong)

    return {
        "n": int(n),
        "appropriate_reliance_rate": appropriate / n,
        "over_trust_rate": over / n,
        # Conditional miss rate: among robot errors, how often did the human fail to catch it.
        "over_trust_miss_rate": (over / n_wrong_robot) if n_wrong_robot else float("nan"),
        "under_trust_rate": under / n,
        # Among correct robot actions, how often the human needlessly intervened.
        "under_trust_overintervention_rate": (under / n_correct_robot) if n_correct_robot else float("nan"),
        "intervention_rate": interventions / n,
        "query_rate": queries / n,
        "success_rate": successes / n,
        "confidently_wrong_rate": conf_wrong / n,
        "robot_accuracy": n_correct_robot / n,
    }


def trust_calibration(trials: Sequence[RelianceTrial]) -> Dict[str, float]:
    """How well human intervention tracks robot errors.

    ``discrimination`` = P(intervene | robot wrong) − P(intervene | robot correct); higher is
    better-calibrated reliance. ``phi`` is the correlation between intervening and the robot
    being wrong.
    """

    if not trials:
        return {"n": 0}
    wrong = [0.0 if t.robot_correct else 1.0 for t in trials]
    interv = [1.0 if t.human_intervened else 0.0 for t in trials]
    n_wrong = sum(wrong)
    n_correct = len(trials) - n_wrong
    p_int_wrong = (sum(i for i, w in zip(interv, wrong) if w > 0.5) / n_wrong) if n_wrong else float("nan")
    p_int_correct = (sum(i for i, w in zip(interv, wrong) if w < 0.5) / n_correct) if n_correct else float("nan")
    disc = (p_int_wrong - p_int_correct) if (n_wrong and n_correct) else float("nan")
    return {
        "n": int(len(trials)),
        "p_intervene_given_wrong": float(p_int_wrong),
        "p_intervene_given_correct": float(p_int_correct),
        "discrimination": float(disc),
        "phi_intervene_wrong": _phi(interv, wrong),
    }


def group_by(trials: Sequence[RelianceTrial], key: Callable[[RelianceTrial], Any]) -> Dict[Any, List[RelianceTrial]]:
    out: Dict[Any, List[RelianceTrial]] = defaultdict(list)
    for t in trials:
        out[key(t)].append(t)
    return dict(out)


def reliance_by_condition(trials: Sequence[RelianceTrial]) -> Dict[str, Dict[str, float]]:
    groups = group_by(trials, lambda t: t.condition_name)
    return {str(k): {**reliance_summary(v), **{f"cal_{kk}": vv for kk, vv in trust_calibration(v).items()}} for k, v in sorted(groups.items())}


def reliance_by_condition_occlusion(trials: Sequence[RelianceTrial]) -> Dict[str, Dict[str, float]]:
    groups = group_by(trials, lambda t: f"{t.condition_name}@{t.occlusion_fraction:g}")
    return {str(k): reliance_summary(v) for k, v in sorted(groups.items())}

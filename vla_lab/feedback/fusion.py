"""Quality-aware fusion: decide what to DO with an interjection.

This is the "cover for the shortcomings of human input" piece: instead of
trusting every correction (YAY-Robot-style) or ignoring the human, each parsed
refinement is gated by an online estimate of THIS human's reliability and by
how strongly it conflicts with the robot's own intent posterior.

- :class:`ReliabilityTracker` — Beta posterior over "this human's inputs are
  correct", updated from verifiable events (sim oracle grading, confirmation
  answers, or episode outcomes on the real robot).
- :func:`fuse` — refinement + reliability + intent posterior ->
  apply | verify (ask a confirmation question first) | ignore.

Safety-critical refinements (stop) always apply. Nudges apply with their
magnitude scaled by reliability (cheap, reversible). Target overrides are the
high-stakes case and get the full trust logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .types import Refinement


@dataclass
class ReliabilityTracker:
    """Beta posterior over P(this human's input is correct)."""

    alpha0: float = 2.0  # mildly optimistic prior: mean 2/3
    beta0: float = 1.0
    n_correct: float = 0.0
    n_wrong: float = 0.0

    def update(self, correct: bool, weight: float = 1.0) -> None:
        if correct:
            self.n_correct += float(weight)
        else:
            self.n_wrong += float(weight)

    @property
    def mean(self) -> float:
        a = self.alpha0 + self.n_correct
        b = self.beta0 + self.n_wrong
        return float(a / max(1e-9, a + b))

    @property
    def n_observations(self) -> float:
        return float(self.n_correct + self.n_wrong)

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "n_correct": self.n_correct,
            "n_wrong": self.n_wrong,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
        }


@dataclass
class FusionConfig:
    trust_threshold: float = 0.55  # reliability >= this -> apply directly
    ignore_threshold: float = 0.30  # reliability < this -> ignore (with log)
    # A target override that contradicts a CONFIDENT visual-intent posterior is
    # suspicious: verify when the named target's posterior is below this and
    # the robot's top-1 is above trust in itself.
    conflict_posterior_floor: float = 0.10
    verify_enabled: bool = True  # False -> "verify" downgrades to "apply" (trust-all ablation)
    nudge_min_reliability: float = 0.20

    def to_dict(self) -> Dict[str, float]:
        return {
            "trust_threshold": float(self.trust_threshold),
            "ignore_threshold": float(self.ignore_threshold),
            "conflict_posterior_floor": float(self.conflict_posterior_floor),
            "verify_enabled": bool(self.verify_enabled),
            "nudge_min_reliability": float(self.nudge_min_reliability),
        }


@dataclass
class FusionDecision:
    action: str  # apply | verify | ignore
    reason: str
    scale: float = 1.0  # for nudges: reliability-scaled gain
    extra: Dict[str, float] = field(default_factory=dict)


def fuse(
    refinement: Refinement,
    *,
    reliability: float,
    intent_posterior: Optional[Dict[str, float]] = None,
    cfg: Optional[FusionConfig] = None,
) -> FusionDecision:
    """Route one parsed refinement given human reliability + visual evidence."""

    cfg = cfg or FusionConfig()
    r = float(reliability)
    kind = refinement.kind

    if kind == "stop":
        return FusionDecision(action="apply", reason="safety: stop always applies")
    if kind in ("resume", "confirm_yes", "confirm_no"):
        return FusionDecision(action="apply", reason="control-flow refinement")
    if kind == "unknown":
        return FusionDecision(action="ignore", reason="unparseable utterance")

    if kind == "nudge":
        if r < cfg.nudge_min_reliability:
            return FusionDecision(action="ignore", reason=f"reliability {r:.2f} below nudge floor")
        return FusionDecision(
            action="apply", reason="nudge scaled by reliability", scale=max(0.0, min(1.0, r))
        )

    if kind in ("target", "avoid"):
        if r < cfg.ignore_threshold:
            return FusionDecision(action="ignore", reason=f"reliability {r:.2f} < ignore threshold")
        conflicted = False
        p_named = None
        if intent_posterior and refinement.target_label is not None and kind == "target":
            p_named = float(intent_posterior.get(refinement.target_label, 0.0))
            top1 = max(intent_posterior.values()) if intent_posterior else 0.0
            conflicted = p_named < cfg.conflict_posterior_floor and top1 > 0.5
        if r >= cfg.trust_threshold and not conflicted:
            return FusionDecision(
                action="apply", reason=f"reliability {r:.2f} >= trust threshold",
                extra={"p_named": p_named if p_named is not None else -1.0},
            )
        if cfg.verify_enabled:
            return FusionDecision(
                action="verify",
                reason=(
                    "correction conflicts with confident visual intent"
                    if conflicted
                    else f"reliability {r:.2f} in verify band"
                ),
                extra={"p_named": p_named if p_named is not None else -1.0},
            )
        return FusionDecision(action="apply", reason="verify disabled -> trust")

    return FusionDecision(action="ignore", reason=f"unhandled kind {kind!r}")

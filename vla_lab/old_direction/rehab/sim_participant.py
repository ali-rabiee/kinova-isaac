"""W6 — the generative participant.

This is what makes W1–W5 and the whole analysis pipeline testable end-to-end **before any
hardware or IRB approval exists**, and it is the substrate for the power analysis (W16). It is
the Phase 0 analogue of :mod:`vla_lab.feedback.sim_human` — same idea (a seeded synthetic human
with explicit quality knobs), written fresh because the semantics share nothing.

The generative model, per participant ``p``:

.. code-block:: text

    logit pi*_p(l) = a_p * (s(l) - c_p) + d_p * (x(l) - x_bar)          # the estimand
    logit P(nonpref at t) = logit pi*_p(l_t) + beta_p * kappa_t + phi_p * (t / T)
    kappa_{t+1} = lambda_p^{Delta_t} * (kappa_t + g_p * strength_t * 1[a_t == COACH])

with a lapse rate (uniform choice with probability ``lapse``) on top, and observer
**misdetection** applied at the observation layer.

- ``s`` is the nonpreferred-signed lateral coordinate, so ``pi*`` is monotone increasing in
  ``s`` for every drawn participant regardless of handedness.
- ``phi_p`` is the **fatigue** drift: the participant reverts toward the preferred arm as the
  session wears on. It is included because fatigue is not a nuisance here — it drifts arm
  choice in a way that would masquerade as carryover if unmodelled, so the pipeline must be
  stress-tested against it (§11).
- The misdetection rate is configurable because online arm-choice detection is the study's
  highest-risk sensing component (W8): if the pipeline only works at perfect detection, that
  is a finding to discover here rather than at the pilot.

The choice model is deliberately **identical in form** to the inference model in
:mod:`vla_lab.rehab.carryover`, including the immediate-effect convention on COACH trials.
Recovery tests are therefore about *whether the estimator works*, not about whether two
different equations happen to agree — model misspecification is studied separately by
perturbing these parameters, not by baking a mismatch in.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import ARM_NONPREFERRED, ARM_PREFERRED, COACH
from .workspace import SIDE_LEFT, SIDE_RIGHT, TargetGrid, TargetSpec, nonpreferred_lateral, reach_distances


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


# ---------------------------------------------------------------------------
# Population prior
# ---------------------------------------------------------------------------


@dataclass
class PopulationPrior:
    """The population the synthetic participants are drawn from.

    Defaults are **priors**, not measurements. Every number here is a modelling assumption
    that the lab pilot (M4) is supposed to replace; the power memo (W16) must trace its
    assumptions back to those measurements before it means anything (§13/M5).
    """

    # --- the estimand ------------------------------------------------------
    crossover_mean_m: float = 0.02     # crossover offset from the midline, toward nonpref (+)
    crossover_sd_m: float = 0.06       # between-person spread: the headline heterogeneity
    steepness_mean: float = 12.0       # d logit(pi*) / d s, per metre
    steepness_sd_log: float = 0.35     # lognormal spread of the steepness
    depth_coef_sd: float = 1.5         # per-metre depth effect, centred at 0

    # --- carryover ---------------------------------------------------------
    lambda_a: float = 6.0              # Beta(a, b) prior on the per-person decay
    lambda_b: float = 2.5
    beta_mean: float = 1.0             # sensitivity to the carryover state
    beta_sd: float = 0.3
    g_mean: float = 0.8                # prompt gain
    g_sd: float = 0.25

    # --- nuisance / confounds ---------------------------------------------
    fatigue_mean: float = -0.4         # logit drift over a full session (negative = to preferred)
    fatigue_sd: float = 0.3
    lapse_rate: float = 0.02
    misdetect_rate: float = 0.03       # observer flips the label

    # --- kinematics --------------------------------------------------------
    reach_time_base_ms: float = 420.0
    reach_time_per_m_ms: float = 900.0
    nonpreferred_penalty_ms: float = 120.0
    reach_time_cv: float = 0.18
    success_base: float = 0.97
    success_effort_penalty: float = 0.06

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PopulationPrior":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ParticipantParams:
    """One drawn participant. ``pi*`` is a deterministic function of these."""

    participant_idx: int
    nonpreferred_side: str
    crossover_m: float
    steepness: float
    depth_coef: float
    lam: float
    beta: float
    g: float
    fatigue: float
    lapse: float
    misdetect: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 5)
        return d


def draw_participant(
    participant_idx: int,
    prior: Optional[PopulationPrior] = None,
    *,
    nonpreferred_side: Optional[str] = None,
    seed: int = 0,
) -> ParticipantParams:
    """Draw one participant. Deterministic in ``(participant_idx, seed)``."""

    pr = prior or PopulationPrior()
    rng = random.Random(int(seed) * 100003 + int(participant_idx))
    # ~90% right-handed, matching the population, unless the caller pins it.
    side = nonpreferred_side or (SIDE_LEFT if rng.random() < 0.9 else SIDE_RIGHT)
    lam = _beta_sample(rng, pr.lambda_a, pr.lambda_b)
    return ParticipantParams(
        participant_idx=int(participant_idx),
        nonpreferred_side=str(side),
        crossover_m=float(rng.gauss(pr.crossover_mean_m, pr.crossover_sd_m)),
        steepness=float(math.exp(math.log(max(1e-6, pr.steepness_mean)) + rng.gauss(0.0, pr.steepness_sd_log))),
        depth_coef=float(rng.gauss(0.0, pr.depth_coef_sd)),
        lam=float(min(0.99, max(0.01, lam))),
        beta=float(max(0.0, rng.gauss(pr.beta_mean, pr.beta_sd))),
        g=float(max(0.0, rng.gauss(pr.g_mean, pr.g_sd))),
        fatigue=float(rng.gauss(pr.fatigue_mean, pr.fatigue_sd)),
        lapse=float(pr.lapse_rate),
        misdetect=float(pr.misdetect_rate),
        seed=int(seed) * 100003 + int(participant_idx),
    )


def _beta_sample(rng: random.Random, a: float, b: float) -> float:
    """Beta draw without numpy's Generator, so the whole simulator is stdlib-seeded."""

    x = rng.gammavariate(max(1e-6, float(a)), 1.0)
    y = rng.gammavariate(max(1e-6, float(b)), 1.0)
    return float(x / max(1e-12, x + y))


# ---------------------------------------------------------------------------
# The participant
# ---------------------------------------------------------------------------


@dataclass
class ParticipantResponse:
    """One trial's ground truth, before the observer gets a look at it."""

    arm: str
    physical_side: str
    reach_time_ms: int
    success: bool
    p_nonpreferred: float
    kappa: float
    lapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["p_nonpreferred"] = round(float(self.p_nonpreferred), 5)
        d["kappa"] = round(float(self.kappa), 5)
        return d


class SimulatedParticipant:
    """A seeded generative participant. One instance per simulated session."""

    def __init__(
        self,
        params: ParticipantParams,
        grid: TargetGrid,
        *,
        prior: Optional[PopulationPrior] = None,
        total_trials: int = 100,
        seed_offset: int = 0,
    ) -> None:
        self.p = params
        self.grid = grid
        self.prior = prior or PopulationPrior()
        self.total_trials = max(1, int(total_trials))
        self.rng = random.Random(int(params.seed) + int(seed_offset))
        self.kappa = 0.0
        self.n_trials = 0
        self._depth_ref = float(sum(t.x_m for t in grid) / max(1, len(grid)))

    # -- the estimand ------------------------------------------------------
    def pi_star(self, target: TargetSpec) -> float:
        s = nonpreferred_lateral(target.y_m, self.p.nonpreferred_side)
        z = self.p.steepness * (s - self.p.crossover_m) + self.p.depth_coef * (target.x_m - self._depth_ref)
        return float(_sigmoid(z))

    def pi_star_map(self) -> Dict[int, float]:
        """The **true** ``pi*`` over every target. Ground truth for tests and power only."""

        return {t.target_id: self.pi_star(t) for t in self.grid}

    # -- one trial ---------------------------------------------------------
    def select(
        self,
        target: Optional[TargetSpec],
        *,
        action: str,
        strength: float = 1.0,
        delta: float = 1.0,
        effort_index: Optional[float] = None,
    ) -> Optional[ParticipantResponse]:
        """Simulate one slot. Returns ``None`` for WAIT (nothing is observed, kappa decays)."""

        eff_kappa = self.kappa + (self.p.g * float(strength) if str(action) == COACH else 0.0)
        if target is None:
            self.kappa = (self.p.lam ** float(delta)) * eff_kappa
            self.n_trials += 1
            return None

        fatigue = self.p.fatigue * (self.n_trials / float(self.total_trials))
        z = math.log(max(1e-12, self.pi_star(target)) / max(1e-12, 1.0 - self.pi_star(target)))
        p_np = _sigmoid(z + self.p.beta * eff_kappa + fatigue)
        lapsed = self.rng.random() < float(self.p.lapse)
        chose_np = (self.rng.random() < 0.5) if lapsed else (self.rng.random() < p_np)

        arm = ARM_NONPREFERRED if chose_np else ARM_PREFERRED
        side = self._physical_side(arm)
        d = reach_distances(target, self.grid.cfg)[side]
        pr = self.prior
        mean_ms = (
            pr.reach_time_base_ms
            + pr.reach_time_per_m_ms * d
            + (pr.nonpreferred_penalty_ms if chose_np else 0.0)
        )
        reach_ms = max(120.0, self.rng.gauss(mean_ms, mean_ms * pr.reach_time_cv))
        eidx = float(effort_index if effort_index is not None else target.effort_index)
        p_success = max(0.5, pr.success_base - pr.success_effort_penalty * eidx)

        self.kappa = (self.p.lam ** float(delta)) * eff_kappa
        self.n_trials += 1
        return ParticipantResponse(
            arm=arm,
            physical_side=side,
            reach_time_ms=int(round(reach_ms)),
            success=bool(self.rng.random() < p_success),
            p_nonpreferred=float(p_np),
            kappa=float(eff_kappa),
            lapsed=bool(lapsed),
        )

    def _physical_side(self, arm: str) -> str:
        np_side = str(self.p.nonpreferred_side)
        pref_side = SIDE_RIGHT if np_side == SIDE_LEFT else SIDE_LEFT
        return np_side if arm == ARM_NONPREFERRED else pref_side

    # -- observation -------------------------------------------------------
    def observe(self, response: ParticipantResponse) -> Tuple[str, str, float]:
        """Apply observer misdetection. Returns ``(arm, physical_side, confidence)``.

        The scheduler consumes *this*, not the ground truth — which is the point: a
        misdetection propagates into ``kappa_hat`` and into every subsequent decision, and the
        pipeline has to survive that (W8, §14 risks).
        """

        if self.rng.random() < float(self.p.misdetect):
            flipped = ARM_PREFERRED if response.arm == ARM_NONPREFERRED else ARM_NONPREFERRED
            return (flipped, self._physical_side(flipped), float(0.55 + 0.2 * self.rng.random()))
        return (response.arm, response.physical_side, float(0.90 + 0.09 * self.rng.random()))

    def describe(self) -> Dict[str, Any]:
        return {"params": self.p.to_dict(), "total_trials": int(self.total_trials)}


# ---------------------------------------------------------------------------
# The observer that watches a simulated participant
# ---------------------------------------------------------------------------


class SimulatedObserver:
    """An :class:`~vla_lab.rehab.observation.base.ArmChoiceObserver` over a simulated participant.

    Lets the *same* session runner drive a synthetic study: the observer protocol is the seam,
    so ``rehab_pilot.sh`` exercises the real scheduling, logging, and safety code paths rather
    than a parallel simulation loop. Misdetection is applied here (via
    :meth:`SimulatedParticipant.observe`), so the scheduler consumes an imperfect label exactly
    as it would online.
    """

    name = "simulated"

    def __init__(self, participant: SimulatedParticipant) -> None:
        self.participant = participant
        self.trial_idx: Optional[int] = None
        self.t_go_ms: Optional[int] = None
        self._pending: Optional[ParticipantResponse] = None
        self._truth: Dict[int, ParticipantResponse] = {}
        self._latched: Optional[Any] = None

    def prepare(
        self,
        target: Optional[Any],
        *,
        action: str,
        strength: float = 1.0,
        delta: float = 1.0,
        effort_index: Optional[float] = None,
    ) -> Optional[ParticipantResponse]:
        """Draw the trial's ground truth before the GO signal. Returns ``None`` for WAIT."""

        self._pending = self.participant.select(
            target, action=action, strength=strength, delta=delta, effort_index=effort_index
        )
        return self._pending

    # -- observer protocol -------------------------------------------------
    def begin_trial(self, trial_idx: int, t_ms: int) -> None:
        self.trial_idx = int(trial_idx)
        self.t_go_ms = int(t_ms)
        self._latched = None
        if self._pending is not None:
            self._truth[int(trial_idx)] = self._pending

    def poll(self, t_ms: int):
        from .observation.base import SOURCE_ONLINE, ArmSelection

        if self._pending is None or self.t_go_ms is None or self._latched is not None:
            return self._latched
        if int(t_ms) < int(self.t_go_ms) + int(self._pending.reach_time_ms):
            return None
        arm, side, conf = self.participant.observe(self._pending)
        self._latched = ArmSelection(
            arm=arm,
            physical_side=side,
            t_ms=int(self.t_go_ms) + int(self._pending.reach_time_ms),
            confidence=float(conf),
            observer=self.name,
            source=SOURCE_ONLINE,
            extra={
                "p_nonpreferred": round(float(self._pending.p_nonpreferred), 4),
                "clean": bool(self._pending.success),
            },
        )
        return self._latched

    def end_trial(self, t_ms: int):
        from .observation.base import SOURCE_ONLINE, ArmSelection
        from . import ARM_NONE

        if self._latched is None:
            self.poll(int(t_ms))
        if self._latched is not None:
            return self._latched
        return ArmSelection(
            arm=ARM_NONE, physical_side=ARM_NONE, t_ms=int(t_ms),
            confidence=0.0, observer=self.name, source=SOURCE_ONLINE,
            extra={"reason": "no reach within the GO window"},
        )

    def truth_labels(self) -> Dict[int, str]:
        """Ground-truth arm per trial. For scoring the observer, never for the estimand."""

        return {k: v.arm for k, v in self._truth.items()}


__all__ = [
    "PopulationPrior",
    "ParticipantParams",
    "ParticipantResponse",
    "SimulatedParticipant",
    "SimulatedObserver",
    "draw_participant",
]

"""W5 — B4, the carryover-aware personalized policy, and its two ablations.

B4's advantage is expected to come from two distinguishable sources, and §1.4 requires the
analysis to separate them:

(i) **better probe placement in time** — knowing *this person's* ``lambda`` instead of a
    population constant, and
(ii) **using mildly-contaminated observations by de-biasing them** — the corrected estimator
     of :mod:`vla_lab.rehab.estimand`.

So this module exposes a single policy with two independent switches, giving the four
schedulers the design calls for::

    adaptive_schedule  estimator_correction   condition
    -----------------  --------------------   ---------------------------
    True               True                   B4 (proposed)
    True               False                  ablation: schedule-only
    False              True                   ablation: estimator-only
    False              False                  == B2 (fixed washout)

**The objective.** Every action is scored by what it does to the *final* estimate of
``logit pi*``, not by a hand-tuned utility. Treating the block's observations as
independent Bernoulli draws with per-observation Fisher information ``I_t = q_t (1 - q_t)``
and per-observation logit offset ``b_t = beta * kappa_t``, the final estimate has

.. math::
    \\operatorname{Var} \\approx \\frac{1}{\\sum_t I_t}, \\qquad
    \\operatorname{Bias} \\approx \\frac{\\sum_t I_t b_t}{\\sum_t I_t}

and the policy minimizes ``Var + Bias^2``. The bias term is present only when the estimator
**cannot** de-bias; with ``estimator_correction`` on it drops out, which is exactly the
mechanism (ii) claim expressed as an objective rather than asserted. Waiting therefore buys
something only when contamination would otherwise bias the estimate (or when the correction
itself is too uncertain to trust) — and costs a probe every time.

**The policy family.** Rather than a free per-slot search, the policy optimizes over the
one-parameter family "wait ``w`` free slots after each COACH, then probe", re-deriving ``w``
at every decision from the current posterior. This is:

- **deterministic** given the history (a real-time scheduler whose decisions are not
  reproducible cannot be audited),
- **personalized** — ``w`` follows this participant's ``(lambda, beta, g)`` posterior, and
- **exactly reducible to B2** when the posterior is a point mass at population values, which
  is what :func:`population_washout_slots` computes and what the ablation test checks.

The objective is evaluated under the top-``top_k`` cells of the carryover posterior and
averaged, so a policy is never chosen on the strength of a parameter value the data has not
supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import ASSESS, COACH, WAIT
from ..carryover import CarryoverConfig, CarryoverPosterior, logit
from ..estimand import SpatialLogisticEstimator, TrialObservation
from ..trial import History
from ..workspace import TargetGrid, nonpreferred_lateral
from .base import BlockBudget, DeltaModel, Scheduler, SchedulerDecision, Slot

_EPS = 1e-12


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    """One carryover-posterior grid cell used for the lookahead."""

    lam: float
    beta: float
    g: float
    weight: float
    kappa: float


def _policy_actions(
    *,
    slots_total: int,
    coach_slots: Sequence[int],
    start_idx: int,
    since_coach: Optional[int],
    w: int,
) -> List[Tuple[int, str]]:
    """The remaining ``(slot_idx, action)`` sequence under "wait ``w`` after each COACH"."""

    coach = set(int(i) for i in coach_slots)
    out: List[Tuple[int, str]] = []
    since = since_coach
    for i in range(int(start_idx), int(slots_total)):
        if i in coach:
            out.append((i, COACH))
            since = 0
            continue
        if since is not None and since < int(w):
            out.append((i, WAIT))
        else:
            out.append((i, ASSESS))
        if since is not None:
            since += 1
    return out


def _objective(
    *,
    cells: Sequence[_Cell],
    actions: Sequence[Tuple[int, str]],
    pi_hat: Dict[int, float],
    target_of_slot: Dict[int, int],
    strength_of_slot: Dict[int, float],
    delta: DeltaModel,
    past_info: Dict[int, float],
    past_info_bias: Dict[int, float],
    corrected: bool,
) -> float:
    """Posterior-averaged ``Var + Bias^2`` of the final logit estimate. Lower is better."""

    total = 0.0
    wsum = 0.0
    for ci, cell in enumerate(cells):
        kappa = float(cell.kappa)
        info = float(past_info.get(ci, 0.0))
        info_bias = float(past_info_bias.get(ci, 0.0))
        for slot_idx, action in actions:
            strength = float(strength_of_slot.get(slot_idx, 1.0))
            eff = kappa + (cell.g * strength if action == COACH else 0.0)
            if action != WAIT:
                tid = target_of_slot.get(slot_idx)
                p = float(pi_hat.get(int(tid), 0.5)) if tid is not None else 0.5
                b = cell.beta * eff
                q = 1.0 / (1.0 + math.exp(-(logit(p) + b)))
                inf = q * (1.0 - q)
                info += inf
                info_bias += inf * b
            kappa = (cell.lam ** delta.for_action(action)) * eff
        var = 1.0 / max(_EPS, info)
        bias = 0.0 if corrected else (info_bias / max(_EPS, info))
        total += float(cell.weight) * (var + bias * bias)
        wsum += float(cell.weight)
    return float(total / wsum) if wsum > 0 else float("inf")


def population_washout_slots(
    *,
    lam: float,
    beta: float,
    g: float,
    slots_total: int,
    coach_slots: Sequence[int],
    delta: Optional[DeltaModel] = None,
    pi_hat: float = 0.5,
    max_w: int = 12,
    corrected: bool = False,
) -> int:
    """B2's ``w``: the carryover-aware objective run once against population parameters.

    Deriving B2 this way rather than picking a round number is what makes B2 and B4
    commensurable — they optimize the same thing, one with a population constant and one with
    a per-person posterior — and it is what the schedule-only ablation collapses onto when
    the posterior is forced to a point mass (see :meth:`CarryoverAwareScheduler.decide_free_slot`).
    """

    dm = delta or DeltaModel()
    cells = [_Cell(lam=float(lam), beta=float(beta), g=float(g), weight=1.0, kappa=0.0)]
    target_of_slot = {i: 0 for i in range(int(slots_total))}
    strength_of_slot = {i: 1.0 for i in range(int(slots_total))}
    best_w, best_obj = 0, float("inf")
    for w in range(0, int(max_w) + 1):
        actions = _policy_actions(
            slots_total=int(slots_total), coach_slots=coach_slots, start_idx=0, since_coach=None, w=w
        )
        obj = _objective(
            cells=cells,
            actions=actions,
            pi_hat={0: float(pi_hat)},
            target_of_slot=target_of_slot,
            strength_of_slot=strength_of_slot,
            delta=dm,
            past_info={},
            past_info_bias={},
            corrected=bool(corrected),
        )
        if obj < best_obj - 1e-15:
            best_obj, best_w = obj, w
    return int(best_w)


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------


@dataclass
class CarryoverAwareConfig:
    max_w: int = 8
    top_k: int = 24                # carryover-posterior cells resampled for the lookahead
    refit_every: int = 1           # refit the internal pi-hat every N observations
    fixed_w: int = 2               # used when adaptive_schedule is off (the estimator-only ablation)
    effort_strength: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_w": int(self.max_w),
            "top_k": int(self.top_k),
            "refit_every": int(self.refit_every),
            "fixed_w": int(self.fixed_w),
            "effort_strength": dict(self.effort_strength),
        }


class CarryoverAwareScheduler(Scheduler):
    """B4 and its ablations (see the module docstring for the switch table)."""

    def __init__(
        self,
        grid: TargetGrid,
        nonpreferred_side: str,
        *,
        adaptive_schedule: bool = True,
        estimator_correction: bool = True,
        carryover_cfg: Optional[CarryoverConfig] = None,
        delta: Optional[DeltaModel] = None,
        cfg: Optional[CarryoverAwareConfig] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.grid = grid
        self.nonpreferred_side = str(nonpreferred_side)
        self.adaptive_schedule = bool(adaptive_schedule)
        self.estimator_correction = bool(estimator_correction)
        self.uses_estimator_correction = bool(estimator_correction)
        self.carryover_cfg = carryover_cfg or CarryoverConfig()
        self.delta = delta or DeltaModel(mode=self.carryover_cfg.decay_mode)
        self.cfg = cfg or CarryoverAwareConfig()
        self.posterior = CarryoverPosterior(self.carryover_cfg)
        self._estimator = SpatialLogisticEstimator()
        self._seq: List[TrialObservation] = []
        self._pi_hat: Dict[int, float] = {t.target_id: 0.5 for t in grid}
        self._since_last_fit = 0
        self.last_w: int = 0

        self.name = {
            (True, True): "b4_carryover_aware",
            (True, False): "b4_ablation_schedule_only",
            (False, True): "b4_ablation_estimator_only",
            (False, False): "b2_equivalent",
        }[(self.adaptive_schedule, self.estimator_correction)]
        self.condition = {
            (True, True): "carryover_aware",
            (True, False): "ablation_schedule_only",
            (False, True): "ablation_estimator_only",
            (False, False): "fixed_washout",
        }[(self.adaptive_schedule, self.estimator_correction)]

    # -- lifecycle ---------------------------------------------------------
    def reset(self, budget: BlockBudget) -> None:
        super().reset(budget)
        # The carryover posterior deliberately persists across blocks within a session: it is
        # a property of the *person*, and re-learning lambda from scratch every block would
        # spend the participant's budget on something already known.

    def observe(self, record: Any) -> None:
        """Fold one completed trial into the carryover posterior and the internal pi-hat."""

        tr, res = record.trial, record.result
        s_m = depth = 0.0
        if tr.target_id is not None:
            t = self.grid.get(int(tr.target_id))
            s_m = nonpreferred_lateral(t.y_m, self.nonpreferred_side)
            depth = float(t.x_m)
        strength = float(self.cfg.effort_strength.get(str(tr.effort_level), 1.0))
        y = res.chose_nonpreferred
        pi = self._pi_hat.get(int(tr.target_id)) if tr.target_id is not None else None
        self.posterior.step(
            action=tr.action,
            delta=self.delta.for_action(tr.action),
            pi_star=(pi if y is not None else None),
            chose_nonpreferred=(y if pi is not None else None),
            strength=strength,
        )
        self._seq.append(
            TrialObservation(
                trial_idx=int(tr.trial_idx),
                action=str(tr.action),
                target_id=(int(tr.target_id) if tr.target_id is not None else None),
                s_m=float(s_m),
                depth_m=float(depth),
                y=y,
                delta=self.delta.for_action(tr.action),
                strength=strength,
                block_idx=int(tr.block_idx),
                condition=str(tr.condition),
            )
        )
        if y is not None:
            self._since_last_fit += 1
            if self._since_last_fit >= max(1, int(self.cfg.refit_every)):
                self._since_last_fit = 0
                self._refit_pi_hat()

    def _refit_pi_hat(self) -> None:
        post = self._estimator.fit(self._seq, self.grid, self.nonpreferred_side)
        self._pi_hat = post.mean()

    # -- lookahead ---------------------------------------------------------
    def _cells(self) -> List[_Cell]:
        """Cells representing the carryover posterior (see
        :meth:`~vla_lab.rehab.carryover.CarryoverPosterior.resample_cells` for why this is a
        resample and not the top-``k`` by weight)."""

        return [
            _Cell(lam=c["lam"], beta=c["beta"], g=c["g"], weight=c["weight"], kappa=c["kappa"])
            for c in self.posterior.resample_cells(int(self.cfg.top_k))
        ]

    def _past_info(self, cells: Sequence[_Cell]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Information and bias-weighted information already banked, per cell.

        Recomputed from the block history rather than accumulated incrementally, because the
        offsets depend on the cell and the history is at most a few dozen trials.
        """

        info: Dict[int, float] = {}
        info_bias: Dict[int, float] = {}
        for ci, cell in enumerate(cells):
            kappa = 0.0
            acc_i = acc_ib = 0.0
            for o in self._seq:
                eff = kappa + (cell.g * float(o.strength) if o.action == COACH else 0.0)
                if o.observed:
                    p = float(self._pi_hat.get(int(o.target_id), 0.5))  # type: ignore[arg-type]
                    b = cell.beta * eff
                    q = 1.0 / (1.0 + math.exp(-(logit(p) + b)))
                    inf = q * (1.0 - q)
                    acc_i += inf
                    acc_ib += inf * b
                kappa = (cell.lam ** float(o.delta)) * eff
            info[ci] = acc_i
            info_bias[ci] = acc_ib
        return info, info_bias

    def best_w(self, history: History, slot: Slot) -> Tuple[int, Dict[str, float]]:
        """Argmin of the objective over the policy family, at the current position."""

        budget = self.budget
        if budget is None:
            return (0, {})
        cells = self._cells()
        # Two separate quantities: ``cell.kappa`` is the live state entering *this* slot (the
        # forward sim starts there), while ``past_info`` is what the block's completed
        # observations already banked (replayed from kappa=0, which is where the block began).
        past_info, past_info_bias = self._past_info(cells)
        target_of_slot = {i: int(tid) for i, tid in enumerate(budget.target_sequence)}
        strength_of_slot = {
            i: float(self.cfg.effort_strength.get(str(lvl), 1.0))
            for i, lvl in enumerate(budget.effort_levels)
        }
        since = history.trials_since_last_coach()
        values: Dict[str, float] = {}
        best_w, best_obj = 0, float("inf")
        for w in range(0, int(self.cfg.max_w) + 1):
            actions = _policy_actions(
                slots_total=int(budget.slots_total),
                coach_slots=budget.coach_slots,
                start_idx=int(slot.slot_idx),
                since_coach=since,
                w=w,
            )
            obj = _objective(
                cells=cells,
                actions=actions,
                pi_hat=self._pi_hat,
                target_of_slot=target_of_slot,
                strength_of_slot=strength_of_slot,
                delta=self.delta,
                past_info=past_info,
                past_info_bias=past_info_bias,
                corrected=self.estimator_correction,
            )
            values[f"obj_w{w}"] = float(obj)
            if obj < best_obj - 1e-15:
                best_obj, best_w = obj, w
        values["objective"] = float(best_obj)
        return int(best_w), values

    # -- decision ----------------------------------------------------------
    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        cont = self.posterior.contamination()
        if self.adaptive_schedule:
            w, values = self.best_w(history, slot)
        else:
            w, values = int(self.cfg.fixed_w), {}
        self.last_w = int(w)
        since = history.trials_since_last_coach()
        m = self.posterior.mean()
        values.update({
            "w": float(w),
            "lambda_mean": float(m["lambda"]),
            "beta_mean": float(m["beta"]),
            "g_mean": float(m["g"]),
            "contamination_mean": float(cont["mean"]),
            "contamination_sd": float(cont["sd"]),
            "slots_since_coach": float(-1 if since is None else since),
        })
        if since is not None and since < w:
            return SchedulerDecision(
                action=WAIT,
                target_id=None,
                rationale=(
                    f"wait {since + 1}/{w}: expected contamination {cont['mean']:.2f}+-{cont['sd']:.2f} logits "
                    f"(lambda~{m['lambda']:.2f})"
                ),
                kappa_prior_mean=float(m["kappa"]),
                kappa_prior_sd=float(self.posterior.sd()["kappa"]),
                values=values,
            )
        return SchedulerDecision(
            action=ASSESS,
            target_id=int(slot.target_id),
            rationale=(
                f"probe: contamination {cont['mean']:.2f}+-{cont['sd']:.2f} logits, "
                f"{'correctable' if self.estimator_correction else 'uncorrected'}"
            ),
            kappa_prior_mean=float(m["kappa"]),
            kappa_prior_sd=float(self.posterior.sd()["kappa"]),
            values=values,
        )

    # -- reporting ---------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            **super().describe(),
            "adaptive_schedule": bool(self.adaptive_schedule),
            "estimator_correction": bool(self.estimator_correction),
            "carryover": self.carryover_cfg.to_dict(),
            "delta": self.delta.to_dict(),
            "cfg": self.cfg.to_dict(),
        }


__all__ = [
    "CarryoverAwareConfig",
    "CarryoverAwareScheduler",
    "population_washout_slots",
]

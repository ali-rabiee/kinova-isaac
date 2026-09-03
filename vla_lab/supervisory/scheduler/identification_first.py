r"""**B6 -- identification-first.** Spend the opening demonstration gaps identifying the decay
rate, then exploit it.

The scheduling half of the proposed policy (B5) fails for a reason the Fisher information
makes explicit. Under Eqs. (3)--(4) the decay rate ``lambda`` enters the likelihood only
through the elapsed-time exponent of the residue,

.. math::
    \frac{\partial \eta_t}{\partial \lambda} = \rho_t\,\beta_p \sum_j g_p s_j d_j\,
        \Delta_{j\to t}\,\lambda^{\Delta_{j\to t} - 1},

so the information one observation carries about it is
``sigma(eta_t)(1 - sigma(eta_t)) * (d eta_t / d lambda)^2``: large only when the residue is
observed at an elapsed time near ``-1 / ln(lambda)`` (where ``Delta * lambda^Delta`` peaks),
with leverage on the response (``eta_t`` near zero, i.e. a scene near the supervisor's own
crossover), and with enough compliance for the residue to register. The natural schedule --
wait a fixed number of slots and then probe whatever scene comes next -- observes the residue
at nearly the same elapsed time every gap, which is why one session identifies ``lambda`` for
fewer than half the supervisors (:mod:`vla_lab.supervisory.analyze`). Spreading the first probe
after each demonstration over **log-spaced** delays, and putting the most leveraged scenes at
those probes, was the design the brief expected to identify it. Measured, no schedule does: the
information is bounded by the residue's magnitude, and a 60-slot session does not carry enough
of it for anyone. The point of running B6 was to measure the schedule's worth, and its measured
worth is the answer.

Two variants ship, because they answer different questions:

``identification_first``
    May **permute** the free-slot scene sequence (the same multiset of scenes, in a different
    order), so the most crossover-leveraged scenes land at the informative delays and the least
    leveraged ones are spent on waits. The budget is matched: the same scenes are presented the
    same number of times as in every other condition; only their order differs.
``ablation_b6_fixed_scenes``
    Identical timing, no permutation. The difference between the two is what scene choice is
    worth on top of timing.

After identification -- the posterior over ``lambda`` has moved off its prior by a
total-variation threshold, or the identification budget of gaps is spent -- the policy hands
over to the B5 machinery, which now re-derives its ``(w, k)`` from an identified decay rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import COACH, COUNTER, PROBE, WAIT
from ..carryover import CarryoverConfig
from ..estimand import PsychometricConfig
from ..scenes import SceneGrid
from .base import BlockBudget, DeltaModel, History, HistoryRecord, SchedulerDecision, Slot
from .carryover_aware import CarryoverAwareConfig, CarryoverAwareScheduler


@dataclass
class IdentificationFirstConfig:
    """The identification phase's knobs. Deliberately few."""

    #: Demonstration gaps the identification phase may consume before exploitation begins.
    identify_gaps: int = 4
    #: First-probe delay (free slots) after each demonstration, cycled across gaps. The brief
    #: proposed a log-spaced ladder (0, 1, 3). **Measured, that is worse for identification than
    #: not waiting at all** (N = 48: 23% of supervisors against 38% for the plain adaptive
    #: policy), and the Fisher information says why: under time decay every slot -- probe or
    #: wait -- advances the residue clock by about 1.5 units, so consecutive probes already
    #: observe the residue at distinct elapsed times, and a wait forfeits an observation to buy a
    #: delay that probing would have bought anyway. The default is therefore no wait; the ladder
    #: is kept as the ``ablation_b6_ladder`` variant and reported.
    wait_ladder: Tuple[int, ...] = (0,)
    #: Permute the free-slot scene multiset so leveraged scenes fall at the informative probes.
    reorder_scenes: bool = True
    #: Stop identifying as soon as the posterior over ``lambda`` has moved off its prior.
    stop_when_identified: bool = True
    tv_threshold: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IdentificationFirstScheduler(CarryoverAwareScheduler):
    """B6. See the module docstring."""

    name = "identification_first"

    def __init__(
        self,
        grid: SceneGrid,
        *,
        ident_cfg: Optional[IdentificationFirstConfig] = None,
        cfg: Optional[CarryoverAwareConfig] = None,
        carryover_cfg: Optional[CarryoverConfig] = None,
        delta_model: Optional[DeltaModel] = None,
        psych_cfg: Optional[PsychometricConfig] = None,
        seed: int = 0,
        name: Optional[str] = None,
        log_prior: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(grid, cfg=cfg or CarryoverAwareConfig.full(), carryover_cfg=carryover_cfg,
                         delta_model=delta_model, psych_cfg=psych_cfg, seed=seed, name=name, log_prior=log_prior)
        self.ident = ident_cfg or IdentificationFirstConfig()
        self._pool: List[int] = []
        self._identified_at: Optional[int] = None
        self._phase = "identify"

    # -- bookkeeping ----------------------------------------------------------
    def reset(self, budget: BlockBudget) -> None:
        super().reset(budget)
        coach = set(budget.coach_slots)
        self._pool = [int(budget.scene_sequence[i]) for i in range(budget.n_slots) if i not in coach]
        self._identified_at = None
        self._phase = "identify"

    def _leverage(self, scene_id: int) -> float:
        q = float(self._pi_hat.get(int(scene_id), 0.5))
        return float(self._band_w.get(int(scene_id), 0.0)) * q * (1.0 - q)

    def _take_scene(self, slot: Slot, *, best: bool) -> int:
        """Pop the most (or least) leveraged remaining scene; without reordering, the slot's own."""
        if not self.ident.reorder_scenes or not self._pool:
            if self._pool:
                try:
                    self._pool.remove(int(slot.scene_id))
                except ValueError:
                    self._pool.pop(0)
            return int(slot.scene_id)
        ranked = sorted(self._pool, key=self._leverage, reverse=bool(best))
        sid = ranked[0]
        self._pool.remove(sid)
        return int(sid)

    def _remaining_plan(self, slot: Slot, *, consumed: int, w: int, k: int) -> List[Tuple[str, int, int]]:
        """With reordering, the lookahead sees the pool in leverage order rather than the
        protocol's order -- that is what the policy will actually present."""
        plan = super()._remaining_plan(slot, consumed=consumed, w=w, k=k)
        if not self.ident.reorder_scenes:
            return plan
        ranked = sorted(self._pool, key=self._leverage, reverse=True)
        out: List[Tuple[str, int, int]] = []
        j = 0
        for action, sid, direction in plan:
            if action == COACH or j >= len(ranked):
                out.append((action, sid, direction))
            else:
                out.append((action, ranked[j], direction))
                j += 1
        return out

    # -- the decision --------------------------------------------------------
    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        diag = self.posterior.lambda_diagnostic(tv_threshold=self.ident.tv_threshold)
        gap_index = sum(1 for r in history if r.action == COACH)
        if self._phase == "identify":
            done = (self.ident.stop_when_identified and diag["lambda_identified"]) or \
                   gap_index >= int(self.ident.identify_gaps)
            if done:
                self._phase = "exploit"
                self._identified_at = int(slot.index)
        if self._phase == "exploit":
            dec = super().decide_free_slot(history, slot)
            # Exploitation keeps the pool consistent with what is presented.
            sid = self._take_scene(slot, best=(dec.action != WAIT))
            dec.scene_id = sid
            dec.rationale.update({"phase": "exploit", "identified_at_slot": self._identified_at,
                                  "reordered": bool(self.ident.reorder_scenes)})
            return dec

        consumed = history.n_since_last_coach()
        ladder = self.ident.wait_ladder
        w_target = int(ladder[gap_index % len(ladder)]) if gap_index > 0 or history else int(ladder[0])
        base = {
            "phase": "identify", "gap_index": gap_index, "w_target": w_target, "waited": int(consumed),
            "free_remaining": int(slot.free_remaining), "reordered": bool(self.ident.reorder_scenes),
            "lambda_info_gain_if_probe_now": self.posterior.lambda_information(
                pi_star=float(self._pi_hat.get(int(slot.scene_id), 0.5)))["gain_if_observe"],
            **diag,
        }
        if consumed < w_target and slot.free_remaining > 1:
            sid = self._take_scene(slot, best=False)
            base["reason"] = "identification: log-spaced delay before the first probe of this gap"
            return SchedulerDecision(WAIT, sid, base)
        sid = self._take_scene(slot, best=True)
        base["reason"] = "identification: probe the most leveraged remaining scene at this delay"
        base["scene_leverage"] = self._leverage(sid)
        return SchedulerDecision(PROBE, sid, base)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["identification"] = self.ident.to_dict()
        return d


__all__ = ["IdentificationFirstConfig", "IdentificationFirstScheduler"]

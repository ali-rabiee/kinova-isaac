r"""**B5 -- the carryover-aware policy**, and the three ablations that decompose it.

The policy's advantage can come from three distinguishable places, and a result that does not
separate them does not support a mechanism claim:

(i) **when to probe** -- knowing *this* supervisor's ``lambda`` instead of a population
    constant, so the wait is as long as it needs to be and no longer;
(ii) **how to read a contaminated probe** -- the corrected estimator of
     :mod:`vla_lab.supervisory.estimand`, which extracts signal from an answer given while the
     residue was still non-zero instead of discarding it;
(iii) **re-opening the closed option** -- spending a counter-proposal, which attenuates the
      residue's grip on the answer at the cost of interaction burden.

So this module exposes one policy with three independent switches::

    adaptive_schedule  estimator_correction  counter_enabled   condition
    -----------------  --------------------  ---------------   -----------------------------
    True               True                  True              B5 (proposed)
    True               False                 False             ablation: schedule-only
    False              True                  False             ablation: estimator-only
    False              False                 True              ablation: counter-only
    False              False                 False             == B2 (fixed washout)

**The objective.** Every action is scored by what it does to the *final* estimate of
``logit pi*`` in the crossover band, not by a hand-tuned utility. Treating the block's
observations as independent Bernoulli draws with per-observation Fisher information
``I_t = q_t(1 - q_t)`` and per-observation logit offset ``b_t = rho_t * beta * kappa_t``, the
final estimate has

.. math::
    \operatorname{Var} \approx \frac{1}{\sum_t I_t}, \qquad
    \operatorname{Bias} \approx \frac{\sum_t I_t b_t}{\sum_t I_t}

and the policy minimises ``Var + Bias^2 + burden``. The bias term is present only when the
estimator **cannot** de-bias; with ``estimator_correction`` on it drops out -- which is
mechanism (ii) expressed as an objective rather than asserted. Waiting therefore buys
something only when contamination would otherwise bias the estimate, and it costs a probe
every time. A counter-proposal shrinks ``b_t`` by ``rho`` without costing a probe, and is
charged ``counter_burden`` for the interruption.

**The policy family.** Rather than a free per-slot search, the policy optimises over the
two-parameter family "wait ``w`` free slots after each demonstration, then spend ``k``
counter-proposals, then plain probes", re-deriving ``(w, k)`` at every decision from the
current posterior. This is

- **deterministic** given the history -- a real-time policy whose decisions are not
  reproducible from the log cannot be audited;
- **personalised** -- ``(w, k)`` follow this supervisor's ``(lambda, beta, g)`` posterior; and
- **exactly reducible to B2** when the posterior is a point mass at the population values and
  both other switches are off, which :func:`population_washout_slots` computes and the
  reduction test checks.

The objective is evaluated under the top-``top_k`` posterior cells and averaged, so a policy is
never chosen on the strength of a parameter value the data has not supported.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import COACH, COUNTER, PROBE, STRATEGY_A, WAIT
from .._numerics import logit, sigmoid
from ..carryover import CarryoverConfig, CarryoverPosterior
from ..estimand import (
    METHOD_CORRECTED,
    METHOD_PSYCHOMETRIC,
    Observation,
    PsychometricConfig,
    PsychometricEstimator,
)
from ..scenes import SceneGrid
from .base import BlockBudget, DeltaModel, History, HistoryRecord, Scheduler, SchedulerDecision, Slot


@dataclass
class CarryoverAwareConfig:
    """Switches and knobs. The three switches are the ablation axes."""

    adaptive_schedule: bool = True
    estimator_correction: bool = True
    counter_enabled: bool = True
    #: Extra time a counter-proposal costs as a fraction of a probe, from the contract's
    #: timing. The counter is then charged the **opportunity cost of the slot time it
    #: consumes**, in the same currency as everything else -- the marginal variance a probe
    #: would have bought -- rather than an arbitrary penalty. That matters: a tuned constant
    #: would make "how often should the robot re-open the option" a hyper-parameter instead of
    #: a result, and the whole point is that the answer should follow from the supervisor.
    counter_time_ratio: float = 0.38
    #: Optional flat surcharge per counter-proposal, on top of the time cost, for deployments
    #: where being asked twice is itself costly. Zero by default; the analysis sweeps it.
    counter_surcharge: float = 0.0
    max_wait: int = 6
    max_counter: int = 8
    top_k: int = 16
    refit_every: int = 4
    #: Coordinate-ascent sweeps between the plug-in map and the carryover posterior at each
    #: refit. One sweep is the naive plug-in; two or three largely remove the plug-in bias.
    belief_sweeps: int = 3
    #: Contamination below this (in logits) is treated as clean; used only for the reported
    #: personalised washout, never inside the objective.
    clean_tau: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def full(cls, **kw: Any) -> "CarryoverAwareConfig":
        """B5 as proposed: all three switches on. Reported as proposed-and-rejected on the
        scheduling half (see :meth:`recommended`)."""
        return cls(adaptive_schedule=True, estimator_correction=True, counter_enabled=True, **kw)

    @classmethod
    def recommended(cls, **kw: Any) -> "CarryoverAwareConfig":
        """**The configuration the evidence supports: corrected estimator + counter-proposals,
        adaptive scheduling OFF.**

        The adaptive schedule is kept behind its flag and reported, not shipped. Two results
        argue against it, and an identifiability argument predicted both before they were
        measured: the decay rate it personalises on is identified in a minority of supervisors
        (so the schedule it computes is the population schedule wearing a different name), and
        in the high-compliance tercile -- exactly where it should help -- the full policy is
        worse than the same corrected estimator without it. A test asserts that this
        configuration never turns the schedule on.
        """
        return cls(adaptive_schedule=False, estimator_correction=True, counter_enabled=True, **kw)


def population_washout_slots(
    *,
    cfg: CarryoverConfig,
    delta_model: DeltaModel,
    tau: float = 0.10,
    beta: Optional[float] = None,
    g: Optional[float] = None,
    lam: Optional[float] = None,
    max_wait: int = 12,
) -> int:
    """Free slots a *population*-parameter supervisor needs before a probe is clean.

    This is what sets B2's ``w``, and it is deliberately computed from the same priors the
    adaptive policy starts from -- so B2 is the best fixed rule those priors support, not a
    weak one chosen to be beaten.
    """
    prior = CarryoverPosterior(cfg)
    m = prior.mean()
    beta = float(m["beta"]) if beta is None else float(beta)
    g = float(m["g"]) if g is None else float(g)
    lam = float(m["lambda"]) if lam is None else float(lam)
    kappa = g
    d_wait = delta_model.for_action(WAIT)
    for w in range(int(max_wait) + 1):
        if abs(beta * kappa) <= float(tau):
            return w
        kappa *= lam ** d_wait
    return int(max_wait)


@dataclass
class _Cell:
    lam: float
    beta: float
    g: float
    kappa: float
    w: float


class CarryoverAwareScheduler(Scheduler):
    """The proposed policy. See the module docstring for the objective and the switches."""

    name = "carryover_aware"

    def __init__(
        self,
        grid: SceneGrid,
        *,
        cfg: Optional[CarryoverAwareConfig] = None,
        carryover_cfg: Optional[CarryoverConfig] = None,
        delta_model: Optional[DeltaModel] = None,
        psych_cfg: Optional[PsychometricConfig] = None,
        seed: int = 0,
        name: Optional[str] = None,
        log_prior: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(seed=seed)
        #: Population prior over ``(lambda, beta, g)``, learned from other supervisors. A
        #: deployed robot has met people before; this is what that is worth.
        self.log_prior = None if log_prior is None else np.asarray(log_prior, dtype=float)
        self.grid = grid
        self.cfg = cfg or CarryoverAwareConfig()
        self.carryover_cfg = carryover_cfg or CarryoverConfig()
        self.delta = delta_model or DeltaModel.from_config(self.carryover_cfg)
        self.psych = PsychometricEstimator(psych_cfg)
        self.posterior = CarryoverPosterior(self.carryover_cfg, log_prior=log_prior)
        self.observations: List[Observation] = []
        self._pi_hat: Dict[int, float] = {s.scene_id: 0.5 for s in grid.probe_scenes()}
        # Relative importance on a mean-1 scale, not a probability distribution. The
        # distribution form would make every per-observation Fisher term ~1/n_scenes, which
        # inflates the variance term by two orders of magnitude relative to the bias term and
        # would decide the policy by unit choice rather than by evidence.
        raw = grid.band_weights(crossover_weighted=True)
        scale = (sum(raw.values()) / max(len(raw), 1)) or 1.0
        self._band_w = {k: v / scale for k, v in raw.items()}
        self._n_since_refit = 0
        self._x_cache: Dict[int, np.ndarray] = {}
        self._fallback_w = population_washout_slots(
            cfg=self.carryover_cfg, delta_model=self.delta, tau=self.cfg.clean_tau
        )
        if name:
            self.name = str(name)

    @property
    def estimator(self) -> str:  # type: ignore[override]
        return METHOD_CORRECTED if self.cfg.estimator_correction else METHOD_PSYCHOMETRIC

    # -- belief -------------------------------------------------------------
    def reset(self, budget: BlockBudget) -> None:
        super().reset(budget)
        self.posterior = CarryoverPosterior(self.carryover_cfg, log_prior=self.log_prior)
        self.observations = []
        self._pi_hat = {s.scene_id: 0.5 for s in self.grid.probe_scenes()}
        self._n_since_refit = 0

    def observe(self, record: HistoryRecord) -> None:
        has_obs = record.action in (PROBE, COUNTER) and record.instructed in ("A", "B")
        self.posterior.step(
            action=record.action,
            delta=float(record.delta),
            pi_star=float(self._pi_hat.get(int(record.scene_id), 0.5)) if has_obs else None,
            chose_a=(record.instructed == STRATEGY_A) if has_obs else None,
            direction=int(record.coach_direction),
            strength=float(record.coach_strength),
        )
        self.observations.append(
            Observation(
                slot=int(record.slot),
                action=str(record.action),
                scene_id=int(record.scene_id),
                c=float(record.c),
                clutter=int(record.clutter),
                instructed=record.instructed,
                delta=float(record.delta),
                coach_direction=int(record.coach_direction),
                coach_strength=float(record.coach_strength),
                duration_s=float(record.duration_s),
            )
        )
        if has_obs:
            self._n_since_refit += 1
            if self._n_since_refit >= int(self.cfg.refit_every):
                self._refit_pi_hat()
                self._n_since_refit = 0
        elif record.action == COACH:
            # A demonstration changes every cell's kappa but adds no evidence; the incremental
            # step above already advanced it, so nothing further is needed here. Left explicit
            # because "COACH does not update the weights" is easy to break by accident.
            pass

    def _refit_pi_hat(self) -> None:
        """Refit the plug-in map and the carryover posterior by coordinate ascent.

        The naive version of this -- fit ``pi*`` on the raw answers, then infer the residue as
        whatever is left over -- is **biased toward finding no carryover**, and badly so: a map
        fitted on contaminated observations has already absorbed part of the residue into its
        own shape, so the residue the posterior is then asked to explain is smaller than the
        one that is really there. A policy running on that belief under-corrects and
        under-waits, and it does so most at exactly the moment the contamination is worst.

        So each refit alternates: form the posterior-mean offsets, refit the map *with* those
        offsets, replay the posterior against the refitted map, repeat. Two or three sweeps
        bring the online belief close to the offline joint fit at a cost of a few milliseconds,
        which is affordable between interactions -- and unlike the offline fit, it is available
        to the policy while the session is still running.
        """
        obs = [o for o in self.observations if o.observed]
        if not obs:
            return
        try:
            for _ in range(max(1, int(self.cfg.belief_sweeps))):
                offsets = self._posterior_mean_offsets()
                w, cov, _ = self.psych.fit_weights(self.observations, offsets=offsets)
                self._pi_hat = self.psych.posterior_from_weights(
                    w, cov, self.grid, method=METHOD_PSYCHOMETRIC
                ).mean()
                self._replay_posterior()
        except Exception:  # pragma: no cover - a degenerate fit must not stop a session
            pass

    def _posterior_mean_offsets(self) -> np.ndarray:
        """Posterior-mean logit offset ``E[rho * beta * kappa]`` at each observed slot."""
        w = self.posterior.weights()
        lam, beta, gg = self.posterior.lam, self.posterior.beta, self.posterior.g
        kappa = np.zeros_like(lam)
        strength = float(self.budget.coach_strength) if self.budget is not None else 1.0
        out: List[float] = []
        for o in self.observations:
            inc = gg * float(o.coach_strength) * float(o.coach_direction) if o.action == COACH else 0.0
            eff = kappa + inc
            if o.observed:
                out.append(float(np.dot(w, self.carryover_cfg.rho_for(o.action) * beta * eff)))
            kappa = np.power(lam, float(o.delta)) * eff
        return np.array(out, dtype=float)

    def _replay_posterior(self) -> None:
        """Rebuild the carryover posterior from the prior against the current plug-in map."""
        post = CarryoverPosterior(self.carryover_cfg, log_prior=self.log_prior)
        for o in self.observations:
            has = o.observed
            post.step(
                action=o.action,
                delta=float(o.delta),
                pi_star=float(self._pi_hat.get(int(o.scene_id), 0.5)) if has else None,
                chose_a=(o.instructed == STRATEGY_A) if has else None,
                direction=int(o.coach_direction),
                strength=float(o.coach_strength),
            )
        self.posterior = post

    def _cells(self) -> List[_Cell]:
        return [
            _Cell(c["lambda"], c["beta"], c["g"], c["kappa"], c["w"])
            for c in self.posterior.resample_cells(self.cfg.top_k)
        ]

    # -- the objective, in the estimator's own coordinates -------------------
    #
    # The reported estimate is a three-parameter psychometric fit, not fifteen independent
    # per-scene proportions, so the objective is written in *its* coordinates. For a planned
    # sequence of observations with Fisher weights ``I_t`` and design rows ``x_t``:
    #
    #     A     = sum_t I_t x_t x_t^T + P            (P = the estimator's own prior precision)
    #     Sigma = A^-1                               (Laplace covariance of the fitted weights)
    #     u     = sum_t I_t b_t x_t                  (b_t = the residual logit offset)
    #     dw    = Sigma u                            (first-order shift of the fitted weights)
    #     s2    = sum_t I_t b_t^2 / sum_t I_t        (dispersion of the offsets)
    #
    # and then, per scene, ``Var_s = x_s Sigma x_s^T`` and
    #
    #     Bias_s = x_s dw  +  (a - 1) * eta_s,       a = 1 / sqrt(1 + s2 / 2.9)
    #
    # The objective is the band-weighted mean of ``Var_s + Bias_s^2``.
    #
    # **The attenuation term is not a refinement.** An offset that varies from observation to
    # observation -- which is exactly what a decaying residue produces, large just after a
    # demonstration and near zero later -- acts on a logistic fit the way measurement error in a
    # covariate acts on a regression: it flattens the fitted slope toward chance. That pulls the
    # whole estimated map toward 0.5, hardest in the crossover band where the map matters most,
    # and it is invisible to a first-order shift term because its *mean* can be zero. Without
    # it the objective sees contamination as a small nuisance and the policy declines to spend
    # anything on it; with it, waiting and counter-proposing are priced against the damage they
    # actually prevent. ``2.9 = 15*pi^2/16`` is the standard logistic-probit scaling.
    #
    # Two things this gets right that a per-scene ``1/sum I`` cannot. It knows the estimator
    # **pools across scenes**, so an observation at one scene shrinks the variance everywhere,
    # which is why probing is worth so much and waiting so expensive. And it keeps the bias
    # **directional in weight space**: because demonstrations alternate direction, the average
    # offset over a block is near zero and any aggregate-bias objective would conclude there is
    # nothing to correct -- while the actual damage is that residue tilts the fitted *slope*,
    # since the scenes probed just after a demonstration are not the ones probed later. That
    # tilt is exactly ``x_s dw``.

    def _accumulate(
        self,
        steps: Sequence[Tuple[str, int, int]],
        *,
        cells: Sequence[_Cell],
        kappa0: np.ndarray,
        A0: Optional[np.ndarray] = None,
        U0: Optional[np.ndarray] = None,
        S20: Optional[np.ndarray] = None,
        I0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Fold ``steps`` into ``(A, U, kappa, n_counter)``.

        ``A`` is (3, 3) and cell-independent -- the Fisher weight of a slot depends on the scene
        and the current map estimate, not on the carryover parameters. Only the offset vector
        ``U`` (n_cells, 3) is cell-specific, which is what makes the whole search cheap.
        """
        lam = np.array([c.lam for c in cells])
        beta = np.array([c.beta for c in cells])
        gg = np.array([c.g for c in cells])
        pw = np.array([c.w for c in cells])
        A = np.zeros((3, 3)) if A0 is None else A0.copy()
        U = np.zeros((len(cells), 3)) if U0 is None else U0.copy()
        S2 = np.zeros(len(cells)) if S20 is None else S20.copy()
        Itot = 0.0 if I0 is None else float(I0)
        kappa = np.asarray(kappa0, dtype=float).copy()
        strength = float(self.budget.coach_strength) if self.budget is not None else 1.0
        n_counter = 0

        for action, sid, direction in steps:
            inc = gg * strength * float(direction) if action == COACH else 0.0
            eff = kappa + inc
            if action in (PROBE, COUNTER):
                rho = float(self.carryover_cfg.rho_counter) if action == COUNTER else 1.0
                q = float(self._pi_hat.get(int(sid), 0.5))
                bw = float(self._band_w.get(int(sid), 0.0))
                x = self._x_row(int(sid))
                i_t = bw * q * (1.0 - q)
                A += i_t * np.outer(x, x)
                offs = rho * beta * eff
                b_t = offs - float(np.dot(pw, offs)) if self.cfg.estimator_correction else offs
                U += (i_t * b_t)[:, None] * x[None, :]
                S2 += i_t * b_t**2
                Itot += i_t
                n_counter += int(action == COUNTER)
            kappa = np.power(lam, self.delta.for_action(action)) * eff
        return A, U, S2, kappa, Itot, n_counter

    def _x_row(self, scene_id: int) -> np.ndarray:
        if scene_id not in self._x_cache:
            sc = self.grid.by_id(int(scene_id))
            self._x_cache[scene_id] = np.array(
                [1.0, float(sc.c), float(sc.clutter) - float(self.psych.cfg.clutter_ref)]
            )
        return self._x_cache[scene_id]

    def _past_steps(self) -> List[Tuple[str, int, int]]:
        return [(o.action, int(o.scene_id), int(o.coach_direction)) for o in self.observations]

    def _objective(
        self,
        A: np.ndarray,
        U: np.ndarray,
        S2: np.ndarray,
        Itot: float,
        pw: np.ndarray,
        n_counter: int,
        n_obs: int,
    ) -> float:
        prior_prec = np.diag(
            [self.psych.cfg.intercept_precision, self.psych.cfg.slope_precision, self.psych.cfg.clutter_precision]
        )
        try:
            Sigma = np.linalg.inv(A + prior_prec)
        except np.linalg.LinAlgError:  # pragma: no cover
            Sigma = np.linalg.pinv(A + prior_prec)
        dW = U @ Sigma.T  # (n_cells, 3)
        # Offset dispersion -> slope attenuation, per cell.
        s2 = S2 / max(float(Itot), 1e-9)
        atten = 1.0 / np.sqrt(1.0 + s2 / 2.9) - 1.0  # (n_cells,), <= 0
        total = 0.0
        var_total = 0.0
        wsum = 0.0
        for sid, bw in self._band_w.items():
            x = self._x_row(int(sid))
            var_s = float(x @ Sigma @ x)
            eta_s = logit(float(np.clip(self._pi_hat.get(int(sid), 0.5), 1e-4, 1 - 1e-4)))
            bias_s = dW @ x + atten * eta_s  # (n_cells,)
            total += bw * float(np.dot(pw, var_s + bias_s**2))
            var_total += bw * var_s
            wsum += bw
        wsum = max(wsum, 1e-9)
        # Opportunity cost of the extra time a counter-proposal takes, priced at the marginal
        # variance one probe's worth of slot time would have bought.
        marginal = (var_total / wsum) / max(int(n_obs), 1)
        burden = int(n_counter) * (float(self.cfg.counter_time_ratio) * marginal + float(self.cfg.counter_surcharge))
        return float(total / wsum + burden)

    def _remaining_plan(self, slot: Slot, *, consumed: int, w: int, k: int) -> List[Tuple[str, int, int]]:
        """``(action, scene_id, coach_direction)`` for every slot from ``slot`` to block end.

        The lookahead covers the **whole remaining block**, not just the current gap. That is
        what "the final estimate" means, and it changes the answer: judged against the six
        slots of one gap, giving one up to wait looks like a 17% loss of information and no
        plausible bias is worth it; judged against the thirty that actually enter the reported
        map, the same wait costs 3%. Scoring the gap alone does not make the policy
        conservative, it makes it wrong.

        ``consumed`` is how many free slots have already been spent in the current gap, so the
        (w, k) rule resumes mid-gap rather than restarting.
        """
        assert self.budget is not None
        b = self.budget
        coach = dict(zip(b.coach_slots, b.coach_directions))
        coach_order = {sl: i for i, sl in enumerate(b.coach_slots)}
        out: List[Tuple[str, int, int]] = []
        used = int(consumed)
        for i in range(int(slot.index), int(b.n_slots)):
            if i in coach:
                idx = coach_order[i]
                sid = b.coach_scene_sequence[idx] if idx < len(b.coach_scene_sequence) else b.scene_sequence[i]
                out.append((COACH, int(sid), int(coach[i])))
                used = 0
                continue
            action = WAIT if used < w else (COUNTER if (used - w) < k else PROBE)
            out.append((action, int(b.scene_sequence[i]), 0))
            used += 1
        return out

    def best_policy(self, slot: Slot, *, consumed: int) -> Tuple[int, int, Dict[str, Any]]:
        """Re-derive ``(w, k)`` for the rest of the block, from the current posterior."""
        free = max(1, int(slot.free_remaining))
        cells = self._cells()
        pw = np.array([c.w for c in cells])
        A0, U0, S20, kappa0, I0, _n_past = self._accumulate(
            self._past_steps(), cells=cells, kappa0=np.zeros(len(cells))
        )

        max_w = min(int(self.cfg.max_wait), max(0, free - 1))
        max_k = min(int(self.cfg.max_counter), free) if self.cfg.counter_enabled else 0
        w_candidates: Sequence[int] = (
            list(range(0, max_w + 1))
            if self.cfg.adaptive_schedule
            else [min(self._fallback_w, max(0, free - 1))]
        )

        best: Optional[Tuple[float, int, int]] = None
        scored: List[Dict[str, float]] = []
        for w in w_candidates:
            for k in range(0, max_k + 1):
                plan = self._remaining_plan(slot, consumed=consumed, w=w, k=k)
                A, U, S2, _, Itot, nc = self._accumulate(
                    plan, cells=cells, kappa0=kappa0, A0=A0, U0=U0, S20=S20, I0=I0
                )
                n_obs = sum(1 for a, _, _ in plan if a in (PROBE, COUNTER)) + len(
                    [o for o in self.observations if o.observed]
                )
                sc = self._objective(A, U, S2, Itot, pw, nc, n_obs)
                scored.append({"w": int(w), "k": int(k), "score": float(sc)})
                if best is None or sc < best[0]:
                    best = (sc, int(w), int(k))
        assert best is not None
        _, w, k = best
        return w, k, {
            "w": int(w),
            "k": int(k),
            "free_remaining": free,
            "consumed_in_gap": int(consumed),
            "scored": scored[:32],
            "adaptive": bool(self.cfg.adaptive_schedule),
            "correction": bool(self.cfg.estimator_correction),
            "counter": bool(self.cfg.counter_enabled),
        }

    # -- the decision -------------------------------------------------------
    def decide_free_slot(self, history: History, slot: Slot) -> SchedulerDecision:
        consumed = history.n_since_last_coach()
        w, k, rationale = self.best_policy(slot, consumed=consumed)
        cont = self.posterior.contamination()
        m = self.posterior.mean()
        rationale.update(
            {
                "kappa_mean": float(m["kappa"]),
                "contamination_mean": float(cont["mean"]),
                "contamination_sd": float(cont["sd"]),
                "washout_delta": float(self.posterior.washout_delta(self.cfg.clean_tau)),
                "waited": int(consumed),
                "lambda_mean": float(m["lambda"]),
                "beta_g_mean": float(m["beta_g"]),
                # The prospective identification readout: is the schedule about to act on a
                # decay rate the data have not identified? Logged every slot, for every policy
                # that personalises, so the answer is in the record rather than in hindsight.
                **self.posterior.lambda_diagnostic(),
            }
        )
        if consumed < w and slot.free_remaining > 1:
            rationale["reason"] = "personalised washout not yet satisfied"
            return SchedulerDecision(WAIT, slot.scene_id, rationale)
        if k > 0 and self.cfg.counter_enabled and (consumed - w) < k:
            rationale["reason"] = "re-open the demonstrated option with a counter-proposal"
            return SchedulerDecision(COUNTER, slot.scene_id, rationale)
        rationale["reason"] = "probe"
        return SchedulerDecision(PROBE, slot.scene_id, rationale)

    def describe(self) -> Dict[str, Any]:
        d = {"name": self.name, "estimator": self.estimator, "seed": self.seed}
        d["config"] = self.cfg.to_dict()
        d["carryover"] = self.carryover_cfg.to_dict()
        d["delta_model"] = self.delta.to_dict()
        d["fallback_w"] = int(self._fallback_w)
        return d

    def belief_summary(self) -> Dict[str, Any]:
        return self.posterior.summary()


__all__ = ["CarryoverAwareConfig", "CarryoverAwareScheduler", "population_washout_slots"]

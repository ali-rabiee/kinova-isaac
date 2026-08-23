r"""The latent carryover state: the mechanism claim, as code.

A robot that demonstrates and narrates a strategy leaves a residue in the person watching it.
This module is the model of that residue and the sequential posterior over its parameters.

.. math::
    \kappa^{\mathrm{eff}}_t &= \kappa_t + g_p\,s_t\,d_t\,\mathbb{1}[a_t = \mathrm{COACH}] \\
    \Pr[y_t = A \mid c_t, \kappa_t] &= \sigma\!\big(\operatorname{logit}\pi^*_p(c_t)
        + \rho_t\,\beta_p\,\kappa^{\mathrm{eff}}_t\big) \\
    \kappa_{t+1} &= \lambda_p^{\,\Delta_t}\,\kappa^{\mathrm{eff}}_t

with

``lambda_p`` in (0,1)
    the person's decay rate -- how fast a demonstration stops mattering.
``g_p > 0``
    the demonstration gain -- how much residue one COACH deposits.
``beta_p >= 0``
    the person's **compliance sensitivity** -- how much residue moves what they say. This is
    the parameter with the interesting between-person story: ``beta_p = 0`` is a supervisor who
    says what they think regardless of what the robot just did.
``d_t`` in {+1, -1}
    the **direction** of the demonstration: +1 when the robot demonstrated the cautious
    strategy A, -1 for the efficient strategy B.
``s_t``
    demonstration strength (narration only, narration + repetition, ...), from the contract.
``rho_t`` in [0,1]
    **counter-proposal attenuation.** 1 for a plain PROBE. For a COUNTER -- where the robot
    explicitly names the alternative it did *not* demonstrate -- the residue's grip on the
    answer is weakened, because the option compliance had closed is re-opened out loud. This
    is the model of the study's active de-biasing action, and ``rho`` is itself an estimable
    quantity (:func:`fit_rho`), not a knob tuned to make the method win.

**Three deliberate departures from the arm-choice ancestor** of this model
(``vla_lab/old_direction/rehab/carryover.py``), each of which buys something:

1. ``kappa`` is **signed**. The robot can coach either strategy, so the design can be
   counterbalanced: if compliance is real, priming A must move commands toward A and priming
   B toward B, with the same decay. A one-sided design cannot separate "the prompt worked"
   from "the session drifted", and drift is the confound this study is most exposed to.
2. The COACH increment takes effect **immediately**, then decays with everything else. The
   ancestor applied it at ``t+1``, leaving the demonstration's own slot unmodelled. For every
   non-COACH slot the two are identical, and COACH slots carry no observation here anyway
   (the robot acts, the supervisor does not speak), so this is a modelling tidiness rather
   than a change of substance -- but it makes the simulator and the estimator share one
   equation exactly.
3. ``rho`` exists at all, because the action set has a fourth member.

**What this module owns.** It infers ``(lambda, beta, g)`` **given a plugged-in ``pi*``**.
That is all a real-time scheduler can afford, and it is *biased*: a ``pi*`` fitted on
contaminated observations has already absorbed some of the carryover, leaving less residue for
this model to find, so the online posterior understates the effect. The unbiased quantity --
the joint posterior over ``(pi*, theta)`` -- is fitted offline by
:func:`vla_lab.supervisory.estimand.joint_carryover_posterior` and is what the paper's H1/H2
report. The online posterior is what the *policy acted on* and is logged for audit; the two
are never conflated.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B
from ._numerics import logit, logsumexp, sigmoid_np

DECAY_TIME = "time"
DECAY_TRIALS = "trials"

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CarryoverConfig:
    """Grid and priors for ``(lambda, beta, g)``, plus the counter-proposal attenuation.

    ``0.0`` is a grid point for both ``beta`` and ``g`` on purpose: "this person shows no
    compliance carryover at all" must be a *representable* hypothesis, or the posterior cannot
    report it and the study cannot answer its own go/no-go question.
    """

    #: ``time`` decay (memory) is the default. Under pure *interference* decay a neutral WAIT
    #: interferes less than a probe, so waiting buys less decay than probing *and* costs a
    #: slot -- WAIT is then strictly dominated and the scheduling question is vacuous. That is
    #: a real prediction of the interference account and worth reporting if the data support
    #: it, but it cannot be the account the study is designed around. Both are fitted.
    decay_mode: str = DECAY_TIME
    time_unit_s: float = 30.0
    n_lambda: int = 12
    n_beta: int = 11
    n_g: int = 9
    lambda_range: Tuple[float, float] = (0.05, 0.97)
    beta_range: Tuple[float, float] = (0.0, 2.5)
    g_range: Tuple[float, float] = (0.0, 2.0)
    lambda_prior_ab: Tuple[float, float] = (2.0, 2.0)
    beta_prior_scale: float = 1.0
    g_prior_scale: float = 0.8
    #: Prior mean of the counter-proposal attenuation. 1.0 would mean counter-proposals do
    #: nothing; 0.0 would mean they fully restore unprompted judgement. The default is a
    #: deliberately unflattering prior -- a counter-proposal removes rather less than half the
    #: residue's grip -- so that the policy is not handed its own advantage by assumption.
    rho_counter: float = 0.6
    rho_source: str = "prior"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("lambda_range", "beta_range", "g_range", "lambda_prior_ab"):
            d[k] = [float(x) for x in getattr(self, k)]
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CarryoverConfig":
        d = dict(d or {})
        for k in ("lambda_range", "beta_range", "g_range", "lambda_prior_ab"):
            if k in d and d[k] is not None:
                d[k] = tuple(float(x) for x in d[k])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def rho_for(self, action: str) -> float:
        """Attenuation applied to the residue for an observation taken under ``action``."""
        return float(self.rho_counter) if str(action) == COUNTER else 1.0


# ---------------------------------------------------------------------------
# Pure kappa dynamics -- used by the simulator, the tests, and the scheduler lookahead
# ---------------------------------------------------------------------------
def kappa_trace(
    actions: Sequence[str],
    *,
    lam: float,
    g: float,
    directions: Optional[Sequence[int]] = None,
    deltas: Optional[Sequence[float]] = None,
    strengths: Optional[Sequence[float]] = None,
    kappa0: float = 0.0,
) -> List[float]:
    """Effective ``kappa`` at each slot -- the value the answer on that slot sees.

    ``deltas[t]`` is the decay exponent applied *after* slot ``t``: 1.0 per slot in ``trials``
    mode, elapsed time in ``time_unit_s`` units in ``time`` mode.
    """
    n = len(actions)
    dl = list(deltas) if deltas is not None else [1.0] * n
    st = list(strengths) if strengths is not None else [1.0] * n
    dr = list(directions) if directions is not None else [1] * n
    out: List[float] = []
    k = float(kappa0)
    for t in range(n):
        inc = float(g) * float(st[t]) * float(dr[t]) if str(actions[t]) == COACH else 0.0
        eff = k + inc
        out.append(float(eff))
        k = float(lam) ** float(dl[t]) * eff
    return out


def delta_for(
    action: str,
    *,
    cfg: CarryoverConfig,
    duration_s: float,
    slot_interference: float = 1.0,
) -> float:
    """Decay exponent charged by one slot, under whichever decay mode is configured.

    In ``time`` mode a slot decays the residue in proportion to how long it took, which is why
    a WAIT filler is worth anything at all. In ``trials`` mode it decays in proportion to how
    much *interference* it caused, and a neutral filler causes less than a real interaction.
    """
    if str(cfg.decay_mode) == DECAY_TRIALS:
        return float(slot_interference)
    return float(max(0.0, float(duration_s)) / max(_EPS, float(cfg.time_unit_s)))


# ---------------------------------------------------------------------------
# The posterior
# ---------------------------------------------------------------------------
class CarryoverPosterior:
    """Sequential grid posterior over ``(lambda, beta, g)`` plus the live ``kappa`` state.

    ``kappa`` is tracked **per grid cell**, because it is a deterministic function of
    ``(lambda, g)`` and the action history: cell ``i`` believes ``kappa_i``, and the
    posterior-averaged contamination is ``E_i[beta_i * kappa_i]``.

    The update is a deterministic grid sweep rather than a sampler. That is a requirement, not
    a preference: the scheduler runs between interactions and its decisions must be exactly
    reproducible from the log, or an adaptive policy's behaviour cannot be audited after the
    fact.

    Typical use, once per slot::

        post.step(action=PROBE,   pi_star=0.42, chose_a=True, delta=1.0)
        post.step(action=COUNTER, pi_star=0.51, chose_a=False, delta=1.0)
        post.step(action=COACH,   direction=+1, delta=1.0)   # robot acts; nobody answers
        post.step(action=WAIT,    delta=1.0)                 # no observation; kappa decays
    """

    def __init__(
        self,
        cfg: Optional[CarryoverConfig] = None,
        *,
        log_prior: Optional[np.ndarray] = None,
    ) -> None:
        """``log_prior``, when given, replaces the weakly-informative default with a prior
        learned from other supervisors (see :func:`fit_population_prior`). It must be defined on
        the same grid, which :class:`CarryoverConfig` fixes."""
        self.cfg = cfg or CarryoverConfig()
        lam = np.linspace(self.cfg.lambda_range[0], self.cfg.lambda_range[1], int(self.cfg.n_lambda))
        beta = np.linspace(self.cfg.beta_range[0], self.cfg.beta_range[1], int(self.cfg.n_beta))
        g = np.linspace(self.cfg.g_range[0], self.cfg.g_range[1], int(self.cfg.n_g))
        L, B, G = np.meshgrid(lam, beta, g, indexing="ij")
        self.lam = L.ravel().astype(np.float64)
        self.beta = B.ravel().astype(np.float64)
        self.g = G.ravel().astype(np.float64)
        self._axes = {"lambda": lam, "beta": beta, "g": g}
        if log_prior is not None:
            lp = np.asarray(log_prior, dtype=float).ravel()
            if lp.size != self.lam.size:
                raise ValueError(f"log_prior has {lp.size} cells but the grid has {self.lam.size}")
            self.log_prior = lp - logsumexp(lp)
        else:
            self.log_prior = self._log_prior()
        self.log_w = self.log_prior.copy()
        self.kappa = np.zeros_like(self.lam)
        self.n_observations = 0
        self.n_coach = 0
        self.n_counter = 0

    # -- priors ------------------------------------------------------------
    def _log_prior(self) -> np.ndarray:
        a, b = self.cfg.lambda_prior_ab
        lam = np.clip(self.lam, 1e-6, 1.0 - 1e-6)
        lp = (a - 1.0) * np.log(lam) + (b - 1.0) * np.log1p(-lam)
        lp = lp - 0.5 * (self.beta / max(_EPS, self.cfg.beta_prior_scale)) ** 2
        lp = lp - 0.5 * (self.g / max(_EPS, self.cfg.g_prior_scale)) ** 2
        return lp - logsumexp(lp)

    # -- sequential update -------------------------------------------------
    def step(
        self,
        *,
        action: str,
        delta: float = 1.0,
        pi_star: Optional[float] = None,
        chose_a: Optional[bool] = None,
        direction: int = 1,
        strength: float = 1.0,
    ) -> None:
        """Fold in one slot: its observation if it has one, then advance ``kappa``.

        ``pi_star`` is the plug-in estimate of ``pi*(c_t)`` for the presented scene. When it or
        the outcome is missing -- WAIT slots, COACH slots, timeouts, ungrounded utterances --
        only the state advances, which is exactly right: an unobserved slot still decays the
        residue.
        """
        act = str(action)
        inc = self.g * float(strength) * float(direction) if act == COACH else 0.0
        eff = self.kappa + inc
        if pi_star is not None and chose_a is not None:
            rho = self.cfg.rho_for(act)
            lo = logit(float(pi_star))
            p = np.clip(sigmoid_np(lo + rho * self.beta * eff), 1e-12, 1.0 - 1e-12)
            y = 1.0 if bool(chose_a) else 0.0
            self.log_w = self.log_w + (y * np.log(p) + (1.0 - y) * np.log1p(-p))
            self.log_w -= logsumexp(self.log_w)
            self.n_observations += 1
        if act == COACH:
            self.n_coach += 1
        if act == COUNTER:
            self.n_counter += 1
        self.kappa = np.power(self.lam, float(delta)) * eff

    # -- summaries ---------------------------------------------------------
    def weights(self) -> np.ndarray:
        w = np.exp(self.log_w - logsumexp(self.log_w))
        s = float(w.sum())
        return w / s if s > 0 else np.full_like(w, 1.0 / w.size)

    @property
    def beta_g(self) -> np.ndarray:
        r"""``beta * g`` -- the immediate logit shift one COACH produces.

        ``beta`` and ``g`` enter the likelihood almost exclusively as a product, so their
        *marginals* stay wide even when the data are informative. The product is what is
        actually identified, and it is also the quantity the go/no-go asks about ("is there a
        detectable compliance effect at all?"). Report this, not ``beta`` alone.
        """
        return self.beta * self.g

    def _array(self, name: str) -> np.ndarray:
        return {"lambda": self.lam, "beta": self.beta, "g": self.g, "beta_g": self.beta_g, "kappa": self.kappa}[
            str(name)
        ]

    def mean(self) -> Dict[str, float]:
        w = self.weights()
        return {k: float(np.dot(w, self._array(k))) for k in ("lambda", "beta", "g", "beta_g", "kappa")}

    def sd(self) -> Dict[str, float]:
        w = self.weights()
        out: Dict[str, float] = {}
        for name in ("lambda", "beta", "g", "beta_g", "kappa"):
            arr = self._array(name)
            m = float(np.dot(w, arr))
            out[name] = float(math.sqrt(max(0.0, float(np.dot(w, (arr - m) ** 2)))))
        return out

    def credible_interval(self, name: str, level: float = 0.9) -> Tuple[float, float]:
        """Equal-tailed credible interval of a marginal, from the weighted grid."""
        arr = self._array(name)
        w = self.weights()
        order = np.argsort(arr)
        a, cw = arr[order], np.cumsum(w[order])
        lo_q, hi_q = (1.0 - float(level)) / 2.0, 1.0 - (1.0 - float(level)) / 2.0
        return (
            float(a[min(len(a) - 1, int(np.searchsorted(cw, lo_q)))]),
            float(a[min(len(a) - 1, int(np.searchsorted(cw, hi_q)))]),
        )

    # -- contamination -----------------------------------------------------
    def contamination(
        self,
        *,
        after_delta: float = 0.0,
        if_coach: bool = False,
        direction: int = 1,
        strength: float = 1.0,
        action: str = PROBE,
    ) -> Dict[str, float]:
        """Expected logit bias ``E[rho * beta * kappa]`` and its sd -- now, or after waiting.

        ``after_delta`` decays ``kappa`` first, which is exactly the quantity a fixed washout
        rule is implicitly betting on. ``if_coach`` adds a prospective COACH increment, which
        is what the scheduler needs to price a demonstration slot. ``action`` selects the
        attenuation, so the caller can ask "and what if I counter-propose instead?".
        """
        rho = self.cfg.rho_for(action)
        k = self.kappa + (self.g * float(strength) * float(direction) if if_coach else 0.0)
        k = np.power(self.lam, float(after_delta)) * k
        vals = rho * self.beta * k
        w = self.weights()
        m = float(np.dot(w, vals))
        sd = float(math.sqrt(max(0.0, float(np.dot(w, (vals - m) ** 2)))))
        return {"mean": m, "sd": sd, "abs_mean": float(np.dot(w, np.abs(vals)))}

    def predict_contamination(self, delta: float = 0.0, *, action: str = PROBE) -> Tuple[float, float]:
        c = self.contamination(after_delta=delta, action=action)
        return c["mean"], c["sd"]

    def washout_delta(self, tau: float = 0.1, max_delta: float = 400.0) -> float:
        """Smallest wait (in decay units) after which ``|E[beta*kappa]| <= tau``.

        This is the personalised analogue of the fixed washout constant: the number the
        proposed policy computes and a stopwatch cannot.
        """
        if abs(self.contamination()["mean"]) <= float(tau):
            return 0.0
        lo, hi = 0.0, float(max_delta)
        if abs(self.contamination(after_delta=hi)["mean"]) > float(tau):
            return float(max_delta)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if abs(self.contamination(after_delta=mid)["mean"]) > float(tau):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -- diagnostics -------------------------------------------------------
    def identifiability(self, *, tv_threshold: float = 0.05) -> Dict[str, Any]:
        """Has the data moved the posterior away from the prior, and on which axis?

        Reported per axis as total variation between the prior and posterior marginals. An
        axis that has not moved is one the session did not identify, and a scheduler that
        personalises on an unidentified ``lambda`` is personalising on its prior.
        """
        prior = np.exp(self.log_prior - logsumexp(self.log_prior))
        post = self.weights()
        out: Dict[str, Any] = {}
        for name in ("lambda", "beta", "g", "beta_g"):
            arr = self._array(name)
            edges = np.unique(np.round(arr, 9))
            pri = np.array([prior[np.isclose(arr, e)].sum() for e in edges])
            pos = np.array([post[np.isclose(arr, e)].sum() for e in edges])
            tv = 0.5 * float(np.abs(pri - pos).sum())
            out[name] = {"tv": tv, "identified": bool(tv >= float(tv_threshold))}
        out["n_observations"] = int(self.n_observations)
        out["n_coach"] = int(self.n_coach)
        return out

    def effect(self, *, level: float = 0.9, threshold: float = 0.15) -> Dict[str, Any]:
        """Is there a compliance effect at all? The go/no-go, as a number.

        ``threshold`` is the smallest ``beta*g`` (an immediate logit shift) the study regards
        as an effect worth scheduling around. ``p_above`` is the posterior mass above it.
        """
        w = self.weights()
        bg = self.beta_g
        lo, hi = self.credible_interval("beta_g", level=level)
        return {
            "beta_g_mean": float(np.dot(w, bg)),
            "beta_g_ci": [lo, hi],
            "p_above_threshold": float(w[bg >= float(threshold)].sum()),
            "threshold": float(threshold),
            "level": float(level),
        }

    def summary(self) -> Dict[str, Any]:
        m, s = self.mean(), self.sd()
        return {
            "mean": m,
            "sd": s,
            "ci90": {k: list(self.credible_interval(k, 0.9)) for k in ("lambda", "beta", "g", "beta_g")},
            "contamination": self.contamination(),
            "washout_delta": self.washout_delta(),
            "identifiability": self.identifiability(),
            "effect": self.effect(),
            "n_observations": int(self.n_observations),
            "n_coach": int(self.n_coach),
            "n_counter": int(self.n_counter),
        }

    # -- scheduler support -------------------------------------------------
    def resample_cells(self, k: int, *, mode: str = "systematic") -> List[Dict[str, float]]:
        """``k`` particles representing the posterior, each carrying weight ``1/k``.

        **Systematic resampling, not top-k.** Taking the ``k`` highest-weight grid cells is
        wrong here in a way that fails silently: the priors on ``beta`` and ``g`` are half-
        normals peaked at zero, so the modal cells are precisely the *no-carryover* ones. A
        top-k sample of an uninformed posterior therefore contains almost no contamination at
        all, every downstream bias term evaluates to zero, and a scheduler using it concludes
        it has nothing to correct for -- at exactly the moment in the session when it knows
        least and should be most cautious. Systematic resampling represents the whole
        posterior, including the tail cells that carry the risk.

        ``mode="top"`` keeps the old behaviour for diagnostics that genuinely want the mode.
        """
        w = self.weights()
        k = max(1, int(k))
        if str(mode) == "top":
            idx = np.argsort(w)[::-1][:k]
            z = float(w[idx].sum()) or 1.0
            weights = w[idx] / z
        else:
            positions = (np.arange(k) + 0.5) / float(k)
            idx = np.clip(np.searchsorted(np.cumsum(w), positions), 0, w.size - 1)
            weights = np.full(k, 1.0 / k)
        return [
            {
                "lambda": float(self.lam[i]),
                "beta": float(self.beta[i]),
                "g": float(self.g[i]),
                "kappa": float(self.kappa[i]),
                "w": float(wt),
            }
            for i, wt in zip(idx, weights)
        ]

    def copy(self) -> "CarryoverPosterior":
        out = CarryoverPosterior.__new__(CarryoverPosterior)
        out.cfg = self.cfg
        for attr in ("lam", "beta", "g", "log_prior", "log_w", "kappa"):
            setattr(out, attr, getattr(self, attr).copy())
        out._axes = self._axes
        out.n_observations = self.n_observations
        out.n_coach = self.n_coach
        out.n_counter = self.n_counter
        return out

    def force_point_mass(self, *, lam: float, beta: Optional[float] = None, g: Optional[float] = None) -> None:
        """Collapse onto the grid cell nearest the given values. For tests and for the
        population-washout reduction check, never for a live session."""

        def nearest(axis: np.ndarray, value: float) -> float:
            return float(axis[int(np.argmin(np.abs(axis - float(value))))])

        keep = np.isclose(self.lam, nearest(self._axes["lambda"], lam))
        if beta is not None:
            keep &= np.isclose(self.beta, nearest(self._axes["beta"], beta))
        if g is not None:
            keep &= np.isclose(self.g, nearest(self._axes["g"], g))
        lw = np.where(keep, 0.0, -np.inf)
        self.log_w = lw - logsumexp(lw)


# ---------------------------------------------------------------------------
# Offline helpers
# ---------------------------------------------------------------------------
def fit_history(
    records: Sequence[Dict[str, Any]],
    *,
    pi_star: Dict[int, float],
    cfg: Optional[CarryoverConfig] = None,
) -> CarryoverPosterior:
    """Replay a session's records into a fresh posterior, given a plug-in ``pi*`` per scene."""
    post = CarryoverPosterior(cfg)
    for r in records:
        act = str(r.get("action"))
        obs = r.get("instructed_strategy")
        sid = r.get("scene_id")
        has_obs = act in (PROBE, COUNTER) and obs in (STRATEGY_A, "B") and sid is not None
        post.step(
            action=act,
            delta=float(r.get("delta", 1.0)),
            pi_star=float(pi_star[int(sid)]) if has_obs and int(sid) in pi_star else None,
            chose_a=(str(obs) == STRATEGY_A) if has_obs else None,
            direction=int(r.get("coach_direction", 1) or 1),
            strength=float(r.get("coach_strength", 1.0) or 1.0),
        )
    return post


def fit_rho(
    records: Sequence[Dict[str, Any]],
    *,
    pi_star: Dict[int, float],
    lam: float,
    beta: float,
    g: float,
    cfg: Optional[CarryoverConfig] = None,
    grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21)),
) -> Dict[str, Any]:
    """Profile-likelihood estimate of the counter-proposal attenuation ``rho``.

    Fitted **pooled across supervisors** at their per-person ``(lambda, beta, g)``, and only
    from sessions that actually contain COUNTER slots. Estimating ``rho`` is what turns "the
    counter-proposal helps" from an assumption into a measurement -- and a fitted ``rho`` near
    1 would say the counter-proposal does nothing, which the analysis must be able to report.
    """
    cfg = cfg or CarryoverConfig()
    rows: List[Tuple[float, float, bool, float]] = []  # (logit pi*, kappa_eff, is_counter, y)
    k = 0.0
    for r in records:
        act = str(r.get("action"))
        inc = float(g) * float(r.get("coach_strength", 1.0) or 1.0) * float(r.get("coach_direction", 1) or 1)
        eff = k + (inc if act == COACH else 0.0)
        sid = r.get("scene_id")
        y = r.get("instructed_strategy")
        if act in (PROBE, COUNTER) and sid is not None and int(sid) in pi_star and y in (STRATEGY_A, STRATEGY_B):
            rows.append((logit(float(pi_star[int(sid)])), eff, act == COUNTER, 1.0 if str(y) == STRATEGY_A else 0.0))
        k = float(lam) ** float(r.get("delta", 1.0)) * eff

    n_counter = sum(1 for r in rows if r[2])
    if not rows or n_counter == 0:
        return {"rho": float(cfg.rho_counter), "source": "prior", "n_counter": 0, "loglik": None, "n_obs": len(rows)}

    lo = np.array([r[0] for r in rows])
    eff = np.array([r[1] for r in rows])
    is_c = np.array([1.0 if r[2] else 0.0 for r in rows])
    y = np.array([r[3] for r in rows])
    best, best_ll = float(cfg.rho_counter), -np.inf
    for rho in grid:
        r_eff = np.where(is_c > 0.5, float(rho), 1.0)
        p = np.clip(sigmoid_np(lo + r_eff * float(beta) * eff), 1e-12, 1.0 - 1e-12)
        ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
        if ll > best_ll:
            best, best_ll = float(rho), ll
    return {"rho": best, "source": "measured", "n_counter": int(n_counter), "loglik": best_ll, "n_obs": len(rows)}


def fit_population_prior(
    posteriors: Sequence["CarryoverPosterior"],
    *,
    exclude: Optional[int] = None,
    smoothing: float = 0.15,
) -> Optional[np.ndarray]:
    """Empirical-Bayes prior over ``(lambda, beta, g)``, pooled from other supervisors.

    **Why this is not a shortcut.** A deployed robot does not meet its first supervisor with a
    flat prior over how people respond to being coached; it has met other people. Fitting the
    prior from the cohort is what that looks like, and it is the reason personalisation is
    affordable at a session length a person will actually sit through: with a realistic budget
    a single session identifies ``beta*g`` well and ``lambda`` only loosely, and the population
    prior supplies the rest. The honest headline is therefore not "personalisation works" but
    "personalisation works *on top of* a population prior", and the study reports the
    flat-prior variant alongside so the reader can see how much of the benefit is which.

    **Leakage.** ``exclude`` drops one supervisor's own posterior, so a leave-one-out prior can
    be built for each of them. Fitting the prior on the full cohort and then evaluating on it
    would let each supervisor's own data inform their prior, which inflates every downstream
    number. The study runner always passes ``exclude``.

    ``smoothing`` mixes back toward the uniform grid so that a cell no cohort member visited is
    improbable rather than impossible -- an empirical prior with exact zeros can make a new
    supervisor's true parameters unreachable.
    """
    picks = [p for i, p in enumerate(posteriors) if exclude is None or i != int(exclude)]
    if not picks:
        return None
    w = np.mean([p.weights() for p in picks], axis=0)
    w = (1.0 - float(smoothing)) * w + float(smoothing) / float(w.size)
    return np.log(np.clip(w, 1e-300, None))


__all__ = [
    "DECAY_TIME",
    "DECAY_TRIALS",
    "CarryoverConfig",
    "CarryoverPosterior",
    "kappa_trace",
    "delta_for",
    "fit_history",
    "fit_rho",
    "fit_population_prior",
]

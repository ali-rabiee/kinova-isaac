"""W3 — the latent carryover model: the mechanism claim, as code.

The robot's own prior COACH actions bias the very quantity it is trying to measure (deck
slide 12). ``rehab.md`` §1.2 models this with a latent carryover state ``kappa_t >= 0``:

.. math::
    \\Pr[a_t=\\text{nonpref}\\mid \\ell_t,\\kappa_t]
        = \\sigma\\big(\\operatorname{logit}\\pi^*(\\ell_t) + \\beta_p\\,\\kappa_t\\big)

    \\kappa_{t+1} = \\lambda_p^{\\Delta_t}\\,\\kappa_t + g_p\\cdot\\mathbb{1}[a_t=\\text{COACH}]

``(lambda_p, beta_p, g_p)`` are person-specific and unknown — that is the entire point. A
fixed washout is a bet on a population-level ``lambda``; the proposed policy estimates
``lambda_p`` online.

**One deliberate refinement of §1.2.** The spec applies the COACH increment at ``t+1``, which
leaves the COACH trial's *own* choice unmodelled even though the prompt has obviously landed
by then. Here the increment takes effect immediately and then decays with everything else::

    kappa_eff(t) = kappa_t + g * strength_t * 1[a_t == COACH]
    kappa_{t+1}  = lambda^{Delta_t} * kappa_eff(t)

For every non-COACH trial this is *identical* to §1.2 — including every ASSESS trial, which
is where the estimand's information comes from. On COACH trials it models the prompt's
immediate effect instead of ignoring it, and the difference from §1.2 for later trials is a
factor ``lambda^{Delta}`` on the injected residue, which is absorbed by ``g``.

**What this module owns, and what it does not.** ``pi*`` is also unknown, and inferring both
jointly is the hard part. There are two paths, and they are not interchangeable:

*Online* (this module's :meth:`CarryoverPosterior.step`)
    Infer ``(lambda, beta, g)`` **given a plugged-in ``pi*``**. Cheap, sequential, and the
    only thing a real-time scheduler can afford. It is **biased**: a ``pi*`` fitted on
    contaminated data absorbs the carryover, so the residual left for this model to find is
    too small. On synthetic data with a true ``beta*g = 1.2`` the plug-in recovers ~0.45.
    Good enough to decide "probe now or wait", not good enough to report.
*Offline* (:func:`vla_lab.rehab.estimand.joint_carryover_posterior`)
    **Marginalize** ``pi*`` out by Laplace approximation instead of plugging it in, giving a
    proper profile marginal likelihood per grid cell. Recovers ~1.14 on the same data. This
    is what the analysis and the paper use.

The scheduler's own posterior is therefore an *operational* belief, logged for auditability;
the reported posterior is refit offline. Anywhere that distinction matters, the code says
which one it is holding.

The posterior is a **grid**, not a particle filter: the parameter space is 3-D, so a grid is
sufficient, exhaustive, and — the property that actually matters for a real-time scheduler —
deterministic. Two identical histories always produce identical decisions.

Both decay parameterizations of §12.3 are supported behind ``decay_mode``: ``"time"`` (memory
decay, ``Delta`` in units of ``time_unit_s``) and ``"trials"`` (interference, ``Delta`` = 1
per intervening trial). They are different mechanisms with different WAIT semantics — waiting
idle is cheap in trials and expensive in wall-clock — so the pilot fits both and the
comparison is a reportable secondary result, not a modelling nuisance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import COACH

DECAY_TIME = "time"
DECAY_TRIALS = "trials"

_EPS = 1e-9


def logit(p: float) -> float:
    p = float(min(1.0 - 1e-12, max(1e-12, p)))
    return math.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CarryoverConfig:
    """Grid + priors for ``(lambda, beta, g)``.

    ``0.0`` is a grid point for both ``beta`` and ``g`` on purpose: the degenerate
    "no carryover at all" hypothesis must be *representable*, or the posterior cannot report
    it and the study cannot answer its own go/no-go question (§12.7).
    """

    #: ``time`` is the default (§12.3). Under a pure *interference* parameterization a neutral
    #: WAIT filler causes less interference than a reach trial, so waiting buys less decay than
    #: probing *and* costs a probe — waiting is strictly dominated and the scheduling question
    #: is vacuous. That is a real prediction of the interference account and worth reporting if
    #: the pilot supports it, but it cannot be the default the study is designed around.
    decay_mode: str = DECAY_TIME
    time_unit_s: float = 10.0
    n_lambda: int = 12
    n_beta: int = 11
    n_g: int = 9
    lambda_range: Tuple[float, float] = (0.05, 0.97)
    beta_range: Tuple[float, float] = (0.0, 2.5)
    g_range: Tuple[float, float] = (0.0, 2.0)
    # Weakly-informative priors. Beta(a,b) on lambda; half-normal scales on beta and g.
    lambda_prior_ab: Tuple[float, float] = (2.0, 2.0)
    beta_prior_scale: float = 1.0
    g_prior_scale: float = 0.8

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


# ---------------------------------------------------------------------------
# Pure kappa dynamics (used by the simulator, the tests, and the scheduler's lookahead)
# ---------------------------------------------------------------------------


def kappa_trace(
    actions: Sequence[str],
    *,
    lam: float,
    g: float,
    deltas: Optional[Sequence[float]] = None,
    strengths: Optional[Sequence[float]] = None,
    kappa0: float = 0.0,
) -> List[float]:
    """Effective ``kappa`` at each trial (the value the choice on that trial sees).

    ``deltas[t]`` is the decay exponent applied *after* trial ``t`` (1.0 per trial in
    ``trials`` mode; elapsed time in units of ``time_unit_s`` in ``time`` mode).
    """

    n = len(actions)
    dl = list(deltas) if deltas is not None else [1.0] * n
    st = list(strengths) if strengths is not None else [1.0] * n
    out: List[float] = []
    k = float(kappa0)
    for t in range(n):
        eff = k + (float(g) * float(st[t]) if str(actions[t]) == COACH else 0.0)
        out.append(float(eff))
        k = float(lam) ** float(dl[t]) * eff
    return out


# ---------------------------------------------------------------------------
# The posterior
# ---------------------------------------------------------------------------


class CarryoverPosterior:
    """Sequential grid posterior over ``(lambda, beta, g)`` plus the live ``kappa`` state.

    ``kappa`` is tracked **per grid cell**, because it is a deterministic function of
    ``(lambda, g)`` and the action history: cell ``i`` believes ``kappa_i``, and the
    posterior-averaged contamination is ``E_i[beta_i * kappa_i]``.

    Typical use, once per trial::

        post.step(action=ASSESS, pi_star=0.42, chose_nonpreferred=True, delta=1.0)
        post.step(action=COACH,  pi_star=0.55, chose_nonpreferred=True, delta=1.0)
        post.step(action=WAIT,   delta=1.0)          # no observation; kappa still decays
    """

    def __init__(self, cfg: Optional[CarryoverConfig] = None) -> None:
        self.cfg = cfg or CarryoverConfig()
        lam = np.linspace(self.cfg.lambda_range[0], self.cfg.lambda_range[1], int(self.cfg.n_lambda))
        beta = np.linspace(self.cfg.beta_range[0], self.cfg.beta_range[1], int(self.cfg.n_beta))
        g = np.linspace(self.cfg.g_range[0], self.cfg.g_range[1], int(self.cfg.n_g))
        L, B, G = np.meshgrid(lam, beta, g, indexing="ij")
        self.lam = L.ravel().astype(np.float64)
        self.beta = B.ravel().astype(np.float64)
        self.g = G.ravel().astype(np.float64)
        self._lam_axis, self._beta_axis, self._g_axis = lam, beta, g
        self.log_prior = self._log_prior()
        self.log_w = self.log_prior.copy()
        self.kappa = np.zeros_like(self.lam)
        self.n_observations = 0
        self.n_coach = 0

    # -- priors ------------------------------------------------------------
    def _log_prior(self) -> np.ndarray:
        a, b = self.cfg.lambda_prior_ab
        lam = np.clip(self.lam, 1e-6, 1.0 - 1e-6)
        lp = (a - 1.0) * np.log(lam) + (b - 1.0) * np.log1p(-lam)
        lp = lp - 0.5 * (self.beta / max(_EPS, self.cfg.beta_prior_scale)) ** 2
        lp = lp - 0.5 * (self.g / max(_EPS, self.cfg.g_prior_scale)) ** 2
        return lp - _logsumexp(lp)

    # -- sequential update -------------------------------------------------
    def step(
        self,
        *,
        action: str,
        delta: float = 1.0,
        pi_star: Optional[float] = None,
        chose_nonpreferred: Optional[bool] = None,
        strength: float = 1.0,
    ) -> None:
        """Fold in one trial: (optionally) its observation, then advance ``kappa``.

        ``pi_star`` is the plug-in estimate of ``pi*(l_t)`` at this trial's target; when it
        or the outcome is missing (WAIT slots, timeouts, halts) only the state advances.
        """

        eff = self.kappa + (self.g * float(strength) if str(action) == COACH else 0.0)
        if pi_star is not None and chose_nonpreferred is not None:
            lo = logit(float(pi_star))
            p = np.clip(sigmoid(lo + self.beta * eff), 1e-12, 1.0 - 1e-12)
            y = 1.0 if bool(chose_nonpreferred) else 0.0
            self.log_w = self.log_w + (y * np.log(p) + (1.0 - y) * np.log1p(-p))
            self.log_w -= _logsumexp(self.log_w)
            self.n_observations += 1
        if str(action) == COACH:
            self.n_coach += 1
        self.kappa = np.power(self.lam, float(delta)) * eff

    # -- summaries ---------------------------------------------------------
    def weights(self) -> np.ndarray:
        w = np.exp(self.log_w - _logsumexp(self.log_w))
        s = float(w.sum())
        return w / s if s > 0 else np.full_like(w, 1.0 / w.size)

    @property
    def beta_g(self) -> np.ndarray:
        """``beta * g`` — the immediate logit shift one COACH produces.

        ``beta`` and ``g`` are only weakly separable (they enter the likelihood almost
        exclusively as a product), so their *marginals* stay wide even when the data are
        informative. Their product is the quantity that is actually identified, and it is also
        the one §12.7's go/no-go asks about: "is there a detectable carryover effect at all?"
        Report this, not ``beta`` alone.
        """

        return self.beta * self.g

    def mean(self) -> Dict[str, float]:
        w = self.weights()
        return {
            "lambda": float(np.dot(w, self.lam)),
            "beta": float(np.dot(w, self.beta)),
            "g": float(np.dot(w, self.g)),
            "beta_g": float(np.dot(w, self.beta_g)),
            "kappa": float(np.dot(w, self.kappa)),
        }

    def sd(self) -> Dict[str, float]:
        w = self.weights()
        out: Dict[str, float] = {}
        for name in ("lambda", "beta", "g", "beta_g", "kappa"):
            arr = self._array(name)
            m = float(np.dot(w, arr))
            out[name] = float(math.sqrt(max(0.0, float(np.dot(w, (arr - m) ** 2)))))
        return out

    def _array(self, name: str) -> np.ndarray:
        return {
            "lambda": self.lam,
            "beta": self.beta,
            "g": self.g,
            "beta_g": self.beta_g,
            "kappa": self.kappa,
        }[str(name)]

    def credible_interval(self, name: str, level: float = 0.9) -> Tuple[float, float]:
        """Equal-tailed credible interval of a marginal, from the weighted grid."""

        arr = self._array(name)
        w = self.weights()
        order = np.argsort(arr)
        a, cw = arr[order], np.cumsum(w[order])
        lo_q, hi_q = (1.0 - float(level)) / 2.0, 1.0 - (1.0 - float(level)) / 2.0
        return (float(a[int(np.searchsorted(cw, lo_q))]), float(a[min(len(a) - 1, int(np.searchsorted(cw, hi_q)))]))

    # -- contamination -----------------------------------------------------
    def contamination(self, *, after_delta: float = 0.0, if_coach: bool = False, strength: float = 1.0) -> Dict[str, float]:
        """Expected logit bias ``E[beta * kappa]`` (and its sd) now, or after waiting.

        ``after_delta`` decays ``kappa`` by that many decay units first — that is exactly the
        quantity a washout rule is implicitly betting on. ``if_coach`` adds a prospective
        COACH increment, which is what the scheduler needs to price a COACH slot.
        """

        k = self.kappa + (self.g * float(strength) if if_coach else 0.0)
        if after_delta:
            k = np.power(self.lam, float(after_delta)) * k
        bk = self.beta * k
        w = self.weights()
        m = float(np.dot(w, bk))
        v = float(np.dot(w, (bk - m) ** 2))
        return {"mean": m, "sd": float(math.sqrt(max(0.0, v))), "kappa_mean": float(np.dot(w, k))}

    def predict_contamination(self, delta: float = 0.0) -> Tuple[float, float]:
        """``(expected bias, sd)`` in logit units after ``delta`` decay units. §6/W3."""

        c = self.contamination(after_delta=delta)
        return (c["mean"], c["sd"])

    def washout_delta(self, tau: float = 0.1, max_delta: float = 200.0) -> float:
        """Smallest ``delta`` for which expected contamination falls below ``tau``.

        This is the quantity a **fixed washout** (B2) hard-codes from a population
        ``lambda``, and the quantity B4 estimates per person. Returned in decay units
        (trials, or ``time_unit_s`` seconds).
        """

        if self.predict_contamination(0.0)[0] <= float(tau):
            return 0.0
        lo, hi = 0.0, 1.0
        while hi < float(max_delta) and self.predict_contamination(hi)[0] > float(tau):
            lo, hi = hi, hi * 2.0
        if self.predict_contamination(hi)[0] > float(tau):
            return float(max_delta)
        for _ in range(40):  # bisection to ~1e-12 of the bracket
            mid = 0.5 * (lo + hi)
            if self.predict_contamination(mid)[0] > float(tau):
                lo = mid
            else:
                hi = mid
        return float(hi)

    # -- identifiability ---------------------------------------------------
    def identifiability(self, *, tv_threshold: float = 0.05) -> Dict[str, Any]:
        """Which parameters the data actually moved, per marginal.

        A posterior that equals its prior is *not* a confident answer, and reporting its mean
        as an estimate is the failure mode this guards against. ``lambda`` is unidentified
        whenever no COACH has occurred (nothing has decayed yet), and ``beta`` and ``g`` trade
        off when either is near zero — all three show up here as small total variation from
        the prior.
        """

        wp = np.exp(self.log_prior - _logsumexp(self.log_prior))
        w = self.weights()
        out: Dict[str, Any] = {"n_observations": int(self.n_observations), "n_coach": int(self.n_coach)}
        for name, axis, arr in (
            ("lambda", self._lam_axis, self.lam),
            ("beta", self._beta_axis, self.beta),
            ("g", self._g_axis, self.g),
        ):
            mp = np.array([wp[np.isclose(arr, v)].sum() for v in axis])
            mq = np.array([w[np.isclose(arr, v)].sum() for v in axis])
            tv = float(0.5 * np.abs(mp - mq).sum())
            out[name] = {"tv_from_prior": tv, "identified": bool(tv >= float(tv_threshold))}
        if self.n_coach == 0:
            out["lambda"]["identified"] = False
            out["lambda"]["note"] = "no COACH events: nothing has decayed, lambda cannot be identified"
        eff = self.effect()
        out["beta_g"] = eff
        if not eff["detected"]:
            out["note"] = (
                "beta*g is not credibly above zero: the model says there is little or no "
                "carryover to schedule around (§12.7 go/no-go), and lambda is then unidentified"
            )
            out["lambda"]["identified"] = False
        return out

    def effect(self, *, level: float = 0.9, threshold: float = 0.1) -> Dict[str, Any]:
        """The §12.7 go/no-go quantity: is there a detectable, decaying carryover effect?

        ``detected`` is true when the lower credible bound on ``beta * g`` — the immediate
        logit shift one COACH produces — exceeds ``threshold``. "Decaying" is the companion
        question: it is only answerable when ``detected`` holds, because with no effect there
        is nothing whose decay could be observed.
        """

        lo, hi = self.credible_interval("beta_g", level)
        m, s = self.mean(), self.sd()
        return {
            "beta_g_mean": round(float(m["beta_g"]), 5),
            "beta_g_sd": round(float(s["beta_g"]), 5),
            "beta_g_ci": [round(float(lo), 5), round(float(hi), 5)],
            "level": float(level),
            "threshold": float(threshold),
            "detected": bool(lo > float(threshold)),
            "lambda_mean": round(float(m["lambda"]), 5),
            "lambda_ci": [round(float(x), 5) for x in self.credible_interval("lambda", level)],
        }

    # -- serialization -----------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        m, s = self.mean(), self.sd()
        return {
            "config": self.cfg.to_dict(),
            "n_observations": int(self.n_observations),
            "n_coach": int(self.n_coach),
            "mean": {k: round(v, 5) for k, v in m.items()},
            "sd": {k: round(v, 5) for k, v in s.items()},
            "ci90": {
                k: [round(x, 5) for x in self.credible_interval(k, 0.9)]
                for k in ("lambda", "beta", "g", "beta_g")
            },
            "effect": self.effect(),
            "contamination": {k: round(v, 5) for k, v in self.contamination().items()},
            "identifiability": self.identifiability(),
        }

    def resample_cells(self, k: int) -> List[Dict[str, float]]:
        """``k`` grid cells that *represent* the posterior, each with weight ``1/k``.

        Systematic (low-variance) resampling at the deterministic quantiles ``(i + 0.5)/k`` —
        no RNG, so two identical histories always produce identical decisions.

        **Not the top-``k`` cells by weight.** Under a diffuse prior the highest-density cells
        all sit at ``beta ~ 0, g ~ 0`` (both priors are half-normals peaking at zero), so a
        top-``k`` selection reports "no carryover" no matter what the posterior's mass is doing.
        That collapse silently turned the adaptive scheduler into "never wait", which is the
        kind of bug that would have looked like a *finding*.
        """

        k = max(1, int(k))
        w = self.weights()
        cw = np.cumsum(w)
        out: List[Dict[str, float]] = []
        for i in range(k):
            u = (i + 0.5) / k
            idx = int(np.searchsorted(cw, u))
            idx = min(idx, w.size - 1)
            out.append(
                {
                    "lam": float(self.lam[idx]),
                    "beta": float(self.beta[idx]),
                    "g": float(self.g[idx]),
                    "kappa": float(self.kappa[idx]),
                    "weight": 1.0 / k,
                    "index": float(idx),
                }
            )
        return out

    def copy(self) -> "CarryoverPosterior":
        """Deep copy — the scheduler's lookahead rolls forward on a copy, never in place."""

        other = CarryoverPosterior.__new__(CarryoverPosterior)
        other.cfg = self.cfg
        for attr in ("lam", "beta", "g", "log_prior", "log_w", "kappa"):
            setattr(other, attr, getattr(self, attr).copy())
        other._lam_axis, other._beta_axis, other._g_axis = self._lam_axis, self._beta_axis, self._g_axis
        other.n_observations = self.n_observations
        other.n_coach = self.n_coach
        return other

    def force_point_mass(self, *, lam: float, beta: Optional[float] = None, g: Optional[float] = None) -> None:
        """Collapse the posterior onto the grid cell(s) nearest the given values.

        Used to express "a population constant instead of a per-person estimate" — which is
        precisely what B2's fixed washout assumes. With ``lambda`` forced this way, B4's
        schedule-only ablation reduces to B2 (see
        :mod:`vla_lab.rehab.scheduler.carryover_aware`).
        """

        mask = np.isclose(self.lam, self._nearest(self._lam_axis, lam))
        if beta is not None:
            mask &= np.isclose(self.beta, self._nearest(self._beta_axis, beta))
        if g is not None:
            mask &= np.isclose(self.g, self._nearest(self._g_axis, g))
        if not mask.any():
            raise ValueError("force_point_mass selected no grid cells")
        self.log_w = np.where(mask, 0.0, -np.inf)
        self.log_w -= _logsumexp(self.log_w)

    @staticmethod
    def _nearest(axis: np.ndarray, value: float) -> float:
        return float(axis[int(np.argmin(np.abs(axis - float(value))))])


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x)) if x.size else 0.0
    if not np.isfinite(m):
        return m
    return float(m + math.log(float(np.exp(x - m).sum())))


# ---------------------------------------------------------------------------
# Replaying a recorded history
# ---------------------------------------------------------------------------


def delta_for(
    cfg: CarryoverConfig,
    *,
    dt_ms: Optional[float],
    n_trials: float = 1.0,
) -> float:
    """The decay exponent between two consecutive trials, per ``cfg.decay_mode``."""

    if str(cfg.decay_mode) == DECAY_TIME:
        if dt_ms is None:
            return 0.0
        return float(max(0.0, float(dt_ms) / 1000.0 / max(_EPS, float(cfg.time_unit_s))))
    return float(n_trials)


def fit_history(
    records: Sequence[Any],
    pi_star_of_target: Any,
    *,
    cfg: Optional[CarryoverConfig] = None,
    strength_of: Optional[Any] = None,
) -> CarryoverPosterior:
    """Replay a session's :class:`~vla_lab.rehab.trial.TrialRecord`s into a posterior.

    ``pi_star_of_target(target_id) -> float`` is the plug-in ``pi*`` (see the module
    docstring on why this is conditional). ``strength_of(record) -> float`` supplies the
    COACH effort strength; it defaults to 1.0.
    """

    post = CarryoverPosterior(cfg or CarryoverConfig())
    recs = list(records)

    # ``delta`` is the decay applied AFTER a trial, so it is the gap to the NEXT trial.
    # Collect the anchors first rather than lagging by one and getting the direction wrong.
    anchors: List[Optional[int]] = []
    for rec in recs:
        tr = rec.trial
        anchors.append(tr.t_go_ms if tr.t_go_ms is not None else tr.t_present_ms)

    for i, rec in enumerate(recs):
        tr, res = rec.trial, rec.result
        dt = None
        if i + 1 < len(recs) and anchors[i] is not None and anchors[i + 1] is not None:
            dt = float(int(anchors[i + 1]) - int(anchors[i]))  # type: ignore[arg-type]
        delta = delta_for(post.cfg, dt_ms=dt) if i + 1 < len(recs) else 0.0
        y = res.chose_nonpreferred
        pi = None
        if y is not None and tr.target_id is not None:
            try:
                pi = float(pi_star_of_target(tr.target_id))
            except (KeyError, TypeError, ValueError):
                pi = None
        post.step(
            action=tr.action,
            delta=delta,
            pi_star=pi,
            chose_nonpreferred=(y if pi is not None else None),
            strength=(float(strength_of(rec)) if strength_of is not None else 1.0),
        )
    return post


__all__ = [
    "DECAY_TIME",
    "DECAY_TRIALS",
    "CarryoverConfig",
    "CarryoverPosterior",
    "kappa_trace",
    "fit_history",
    "delta_for",
    "logit",
    "sigmoid",
]

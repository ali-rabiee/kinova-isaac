r"""The estimand ``pi*(c)`` and its estimators. The primary outcome is defined here.

.. math::
    \pi^*_p(c) = \Pr[\,p \text{ instructs the cautious strategy A}
                     \mid \text{scene at } c,\ \text{no recent robot coaching}\,]

Two properties of ``pi*`` drive every downstream decision:

- **It is not uniform.** Far on the tight side ``pi* -> 1``; far on the roomy side
  ``pi* -> 0``. Nearly all the information -- and nearly all the room a demonstration has to
  move an answer -- lives in the crossover band near ``c = 0``, so error metrics are
  crossover-weighted (:meth:`~vla_lab.supervisory.scenes.SceneGrid.band_weights`).
- **It is a probability, not a label.** Each probe is one Bernoulli draw, so the budget per
  scene is tiny and estimators that share strength across nearby scenes are a requirement
  rather than a refinement.

Three estimators, all returning a :class:`PiStarPosterior` over every scene:

``pooled``
    Per-scene Beta-Bernoulli, ignoring carryover. The honest naive baseline, and exactly what
    the **Memoryless VLA** condition uses.
``psychometric``
    Bayesian logistic over the scene coordinate ``c`` with a clutter term, fitted by Laplace
    approximation. The prior mean on the ``c`` slope is positive because the direction is not
    in doubt -- a tighter gap makes the cautious strategy more attractive -- but the prior is
    weak enough that a supervisor who genuinely runs the other way is reported as such rather
    than fitted away.
``carryover_corrected``
    ``psychometric`` with each observation's likelihood offset by ``rho_t * beta * kappa_t``,
    **marginalised over the carryover posterior** rather than plugged in at its mean. Plugging
    in the mean would let the corrected estimator claim a precision it has not earned; the
    whole point of the calibration outcome is that it must not.

Beyond error and calibration this module also scores **regret** (:func:`decision_regret`).
That is possible here and was not in the arm-choice ancestor of this design: because the scene
coordinate is defined through a task-value model, executing the wrong strategy has a price in
task value, so compliance bias is a measurable *performance* loss and not only a measurement
artifact. It is the outcome a practitioner cares about.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..calibration.metrics import expected_calibration_error, reliability_bins
from . import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, WAIT
from ._numerics import logit, logsumexp, sigmoid, sigmoid_np
from .carryover import CarryoverConfig, CarryoverPosterior
from .scenes import SceneGrid, SceneSpec

METHOD_POOLED = "pooled"
METHOD_PSYCHOMETRIC = "psychometric"
METHOD_CORRECTED = "carryover_corrected"
METHODS = (METHOD_POOLED, METHOD_PSYCHOMETRIC, METHOD_CORRECTED)

_GRID_N = 201


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """One interaction slot, as the estimators see it.

    ``instructed`` is the strategy the supervisor's utterance grounded to, or ``None`` when the
    slot carried no answer (COACH, WAIT) or the utterance could not be grounded. An ungrounded
    utterance is **not** silently coded: it advances the carryover state and contributes
    nothing to the likelihood, and the count of them is a reported quantity.
    """

    slot: int
    action: str
    scene_id: Optional[int] = None
    c: float = 0.0
    clutter: int = 0
    instructed: Optional[str] = None
    delta: float = 1.0
    coach_direction: int = 1
    coach_strength: float = 1.0
    #: What the robot actually executed (may differ from ``instructed`` only in conditions
    #: where the policy overrides, which none of the shipped conditions do).
    executed: Optional[str] = None
    success: Optional[bool] = None
    duration_s: float = 0.0

    @property
    def observed(self) -> bool:
        return self.action in (PROBE, COUNTER) and self.instructed in (STRATEGY_A, STRATEGY_B)

    @property
    def y(self) -> float:
        return 1.0 if self.instructed == STRATEGY_A else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def sequence_from_records(records: Sequence[Dict[str, Any]], grid: SceneGrid) -> List[Observation]:
    """Build the estimator input from a session's trial records."""
    out: List[Observation] = []
    for i, r in enumerate(records):
        sid = r.get("scene_id")
        scene = grid.by_id(int(sid)) if sid is not None else None
        out.append(
            Observation(
                slot=int(r.get("slot", i)),
                action=str(r.get("action")),
                scene_id=int(sid) if sid is not None else None,
                c=float(scene.c) if scene is not None else 0.0,
                clutter=int(scene.clutter) if scene is not None else 0,
                instructed=r.get("instructed_strategy"),
                delta=float(r.get("delta", 1.0)),
                coach_direction=int(r.get("coach_direction", 1) or 1),
                coach_strength=float(r.get("coach_strength", 1.0) or 1.0),
                executed=r.get("executed_strategy"),
                success=r.get("success"),
                duration_s=float(r.get("duration_s", 0.0) or 0.0),
            )
        )
    return out


# ---------------------------------------------------------------------------
# The posterior representation
# ---------------------------------------------------------------------------
def _prob_grid(n: int = _GRID_N) -> np.ndarray:
    return np.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n)


@dataclass
class PiStarPosterior:
    """A posterior over ``pi*(c)`` at every scene, as a discrete density on a shared grid.

    One representation for all three estimators keeps means, sds, intervals, calibration, and
    the corrected estimator's marginalisation uniform -- and avoids needing an incomplete-beta
    inverse, which would drag in SciPy for no scientific gain.
    """

    scene_ids: Tuple[int, ...]
    density: np.ndarray  # (n_scenes, n_grid), each row normalised
    grid: np.ndarray = field(default_factory=_prob_grid)
    method: str = METHOD_POOLED
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def _row(self, scene_id: int) -> np.ndarray:
        return self.density[self.scene_ids.index(int(scene_id))]

    def mean(self) -> Dict[int, float]:
        return {sid: float(np.dot(self.density[i], self.grid)) for i, sid in enumerate(self.scene_ids)}

    def sd(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for i, sid in enumerate(self.scene_ids):
            m = float(np.dot(self.density[i], self.grid))
            out[sid] = float(math.sqrt(max(0.0, float(np.dot(self.density[i], (self.grid - m) ** 2)))))
        return out

    def at(self, scene_id: int) -> float:
        return float(np.dot(self._row(scene_id), self.grid))

    def interval(self, level: float = 0.9) -> Dict[int, Tuple[float, float]]:
        lo_q, hi_q = (1.0 - float(level)) / 2.0, 1.0 - (1.0 - float(level)) / 2.0
        out: Dict[int, Tuple[float, float]] = {}
        for i, sid in enumerate(self.scene_ids):
            cw = np.cumsum(self.density[i])
            lo = float(self.grid[min(len(self.grid) - 1, int(np.searchsorted(cw, lo_q)))])
            hi = float(self.grid[min(len(self.grid) - 1, int(np.searchsorted(cw, hi_q)))])
            out[sid] = (lo, hi)
        return out

    def argmax_strategy(self) -> Dict[int, str]:
        """The strategy the map says the supervisor prefers, per scene."""
        return {sid: (STRATEGY_A if v >= 0.5 else STRATEGY_B) for sid, v in self.mean().items()}

    def to_dict(self, *, level: float = 0.9) -> Dict[str, Any]:
        mean, sd, itv = self.mean(), self.sd(), self.interval(level)
        return {
            "method": self.method,
            "level": float(level),
            "scenes": {
                str(sid): {"mean": mean[sid], "sd": sd[sid], "lo": itv[sid][0], "hi": itv[sid][1]}
                for sid in self.scene_ids
            },
            "diagnostics": self.diagnostics,
        }


def _beta_density(grid: np.ndarray, a: float, b: float) -> np.ndarray:
    lg = (a - 1.0) * np.log(grid) + (b - 1.0) * np.log1p(-grid)
    d = np.exp(lg - lg.max())
    return d / max(float(d.sum()), 1e-300)


def _logitnormal_density(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-4))
    z = (np.log(grid / (1.0 - grid)) - float(mu)) / sigma
    d = np.exp(-0.5 * z * z) / (grid * (1.0 - grid))
    s = float(d.sum())
    return d / s if s > 0 else np.full_like(grid, 1.0 / grid.size)


# ---------------------------------------------------------------------------
# Estimator 1: pooled Beta-Bernoulli
# ---------------------------------------------------------------------------
@dataclass
class PooledBetaEstimator:
    """Independent Beta-Bernoulli per scene. The naive estimator, and the Memoryless VLA's."""

    prior_a: float = 1.0
    prior_b: float = 1.0

    def fit(self, seq: Sequence[Observation], grid: SceneGrid) -> PiStarPosterior:
        ids = tuple(s.scene_id for s in grid.probe_scenes())
        counts = {sid: [float(self.prior_a), float(self.prior_b)] for sid in ids}
        for o in seq:
            if o.observed and o.scene_id in counts:
                counts[o.scene_id][0 if o.instructed == STRATEGY_A else 1] += 1.0
        g = _prob_grid()
        dens = np.stack([_beta_density(g, *counts[sid]) for sid in ids])
        n_obs = sum(1 for o in seq if o.observed)
        return PiStarPosterior(ids, dens, g, METHOD_POOLED, {"n_observations": n_obs})


# ---------------------------------------------------------------------------
# Estimator 2: the psychometric function over c
# ---------------------------------------------------------------------------
@dataclass
class PsychometricConfig:
    """Prior for the logistic ``logit pi* = w0 + w1 * c + w2 * (clutter - clutter_ref)``.

    ``w1`` is given a positive prior mean: the direction of the effect is not in question --
    a tighter gap makes the cautious strategy more attractive -- only its magnitude and where
    it crosses. The precision is low enough (``slope_precision``) that roughly a dozen
    observations dominate the prior, so a supervisor who genuinely runs the other way is
    *reported*, not fitted away.
    """

    slope_prior_mean: float = 1.2
    slope_precision: float = 0.35
    intercept_precision: float = 0.25
    clutter_precision: float = 1.0
    clutter_ref: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PsychometricEstimator:
    """Bayesian logistic in ``(c, clutter)``, fitted by Laplace approximation."""

    def __init__(self, cfg: Optional[PsychometricConfig] = None) -> None:
        self.cfg = cfg or PsychometricConfig()

    def _design(self, cs: np.ndarray, clutter: np.ndarray) -> np.ndarray:
        return np.stack([np.ones_like(cs), cs, clutter - float(self.cfg.clutter_ref)], axis=1)

    def _prior(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.array([0.0, float(self.cfg.slope_prior_mean), 0.0])
        prec = np.diag([self.cfg.intercept_precision, self.cfg.slope_precision, self.cfg.clutter_precision])
        return mean, prec

    def fit_weights(
        self,
        seq: Sequence[Observation],
        *,
        offsets: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        obs = [o for o in seq if o.observed]
        mean, prec = self._prior()
        if not obs:
            return mean, np.linalg.inv(prec), 0
        cs = np.array([o.c for o in obs], dtype=float)
        cl = np.array([float(o.clutter) for o in obs], dtype=float)
        y = np.array([o.y for o in obs], dtype=float)
        X = self._design(cs, cl)
        off = np.zeros(len(obs)) if offsets is None else np.asarray(offsets, dtype=float)
        # The three weights want different prior precisions (the slope's is the loose one),
        # so this uses the diagonal-prior IRLS below rather than the isotropic helper.
        w, cov = _irls_diag_prior(X, y, offset=off, prior_mean=mean, prior_precision_diag=np.diag(prec))
        return w, cov, len(obs)

    def posterior_from_weights(self, w: np.ndarray, cov: np.ndarray, grid: SceneGrid, *, method: str,
                               diagnostics: Optional[Dict[str, Any]] = None) -> PiStarPosterior:
        scenes = grid.probe_scenes()
        ids = tuple(s.scene_id for s in scenes)
        cs = np.array([s.c for s in scenes], dtype=float)
        cl = np.array([float(s.clutter) for s in scenes], dtype=float)
        X = self._design(cs, cl)
        mu = X @ w
        var = np.einsum("ij,jk,ik->i", X, cov, X)
        g = _prob_grid()
        dens = np.stack([_logitnormal_density(g, float(m), float(math.sqrt(max(v, 1e-8)))) for m, v in zip(mu, var)])
        diag = dict(diagnostics or {})
        diag.update({"weights": [float(x) for x in w], "slope_c": float(w[1]),
                     "crossover_c": float(-w[0] / w[1]) if abs(w[1]) > 1e-6 else None,
                     "monotone": bool(w[1] > 0.0)})
        return PiStarPosterior(ids, dens, g, method, diag)

    def fit(self, seq: Sequence[Observation], grid: SceneGrid) -> PiStarPosterior:
        w, cov, n = self.fit_weights(seq)
        return self.posterior_from_weights(w, cov, grid, method=METHOD_PSYCHOMETRIC, diagnostics={"n_observations": n})


def _irls_diag_prior(
    X: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision_diag: np.ndarray,
    max_iter: int = 80,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """IRLS MAP fit with a diagonal Gaussian prior. Returns ``(w, Laplace covariance)``."""
    P = np.diag(np.asarray(prior_precision_diag, dtype=float))
    mu0 = np.asarray(prior_mean, dtype=float)
    w = mu0.copy()
    H = P
    for _ in range(int(max_iter)):
        eta = X @ w + offset
        p = sigmoid_np(eta)
        s = np.clip(p * (1.0 - p), 1e-9, None)
        grad = X.T @ (y - p) - P @ (w - mu0)
        H = X.T @ (X * s[:, None]) + P
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:  # pragma: no cover
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w + step
        if float(np.max(np.abs(step))) < tol:
            break
    return w, np.linalg.inv(H)


# ---------------------------------------------------------------------------
# Estimator 3: carryover-corrected
# ---------------------------------------------------------------------------
def _offsets_for_cell(seq: Sequence[Observation], *, lam: float, beta: float, g: float,
                      cfg: CarryoverConfig) -> np.ndarray:
    """Per-observation logit offset ``rho_t * beta * kappa_t`` under one carryover cell."""
    out: List[float] = []
    k = 0.0
    for o in seq:
        inc = float(g) * float(o.coach_strength) * float(o.coach_direction) if o.action == COACH else 0.0
        eff = k + inc
        if o.observed:
            out.append(cfg.rho_for(o.action) * float(beta) * eff)
        k = float(lam) ** float(o.delta) * eff
    return np.array(out, dtype=float)


class CarryoverCorrectedEstimator:
    """The psychometric estimator with contaminated observations de-biased.

    Rather than plugging in the carryover posterior's mean, the fit is repeated under
    ``top_k`` particles systematically resampled from the carryover posterior, and the
    resulting ``pi*`` densities are **mixed**. The mixture is wider than any single fit, which is the honest representation: the
    correction is uncertain, and an estimator that hid that uncertainty would win on point
    error and lose on the calibration outcome -- which is exactly the failure mode the study is
    set up to catch.
    """

    def __init__(self, cfg: Optional[PsychometricConfig] = None, *, top_k: int = 24) -> None:
        self.psych = PsychometricEstimator(cfg)
        self.top_k = int(top_k)

    def fit(
        self,
        seq: Sequence[Observation],
        grid: SceneGrid,
        posterior: CarryoverPosterior,
    ) -> PiStarPosterior:
        cells = posterior.resample_cells(self.top_k)
        mixed: Optional[np.ndarray] = None
        ids: Tuple[int, ...] = ()
        slopes: List[float] = []
        gprob = _prob_grid()
        for cell in cells:
            off = _offsets_for_cell(seq, lam=cell["lambda"], beta=cell["beta"], g=cell["g"], cfg=posterior.cfg)
            w, cov, n = self.psych.fit_weights(seq, offsets=off)
            post = self.psych.posterior_from_weights(w, cov, grid, method=METHOD_CORRECTED)
            ids = post.scene_ids
            mixed = post.density * cell["w"] if mixed is None else mixed + post.density * cell["w"]
            slopes.append(float(w[1]))
        if mixed is None:
            return self.psych.fit(seq, grid)
        mixed = mixed / np.clip(mixed.sum(axis=1, keepdims=True), 1e-300, None)
        n_obs = sum(1 for o in seq if o.observed)
        diag = {
            "n_observations": n_obs,
            "n_cells": len(cells),
            "slope_c": float(np.mean(slopes)) if slopes else None,
            "carryover_mean": posterior.mean(),
            "rho_counter": float(posterior.cfg.rho_counter),
            "rho_source": str(posterior.cfg.rho_source),
        }
        return PiStarPosterior(ids, mixed, gprob, METHOD_CORRECTED, diag)


# ---------------------------------------------------------------------------
# The offline joint fit -- what H1/H2 actually report
# ---------------------------------------------------------------------------
def joint_carryover_posterior(
    seq: Sequence[Observation],
    grid: SceneGrid,
    *,
    cfg: Optional[CarryoverConfig] = None,
    psych_cfg: Optional[PsychometricConfig] = None,
    log_prior: Optional[np.ndarray] = None,
) -> Tuple[CarryoverPosterior, PiStarPosterior]:
    """Joint posterior over ``(pi*, lambda, beta, g)``, fitted offline over the whole session.

    For every carryover cell the psychometric map is refitted with that cell's offsets, and the
    cell is weighted by its profile marginal likelihood. This is the *unbiased* view: the
    online posterior of :mod:`vla_lab.supervisory.carryover` conditions on a ``pi*`` that has
    already absorbed part of the carryover and therefore understates it. Sessions report both,
    labelled, and never conflate them: one is what the policy acted on, the other is what the
    mechanism claim rests on.
    """
    cfg = cfg or CarryoverConfig()
    post = CarryoverPosterior(cfg, log_prior=log_prior)
    psych = PsychometricEstimator(psych_cfg)
    obs = [o for o in seq if o.observed]
    if not obs:
        return post, psych.fit(seq, grid)

    y = np.array([o.y for o in obs], dtype=float)
    cs = np.array([o.c for o in obs], dtype=float)
    cl = np.array([float(o.clutter) for o in obs], dtype=float)
    X = psych._design(cs, cl)
    mean0, prec0 = psych._prior()

    log_lik = np.full(post.lam.shape, -np.inf)
    for i in range(post.lam.size):
        off = _offsets_for_cell(seq, lam=float(post.lam[i]), beta=float(post.beta[i]), g=float(post.g[i]), cfg=cfg)
        w, cov = _irls_diag_prior(X, y, offset=off, prior_mean=mean0, prior_precision_diag=np.diag(prec0))
        p = np.clip(sigmoid_np(X @ w + off), 1e-12, 1.0 - 1e-12)
        ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
        # Laplace evidence: + 1/2 log det(cov) penalises cells that need a sharply-tuned map.
        sign, logdet = np.linalg.slogdet(cov)
        log_lik[i] = ll + (0.5 * float(logdet) if sign > 0 else 0.0)

    post.log_w = post.log_prior + log_lik
    post.log_w -= logsumexp(post.log_w)
    post.n_observations = len(obs)
    post.n_coach = sum(1 for o in seq if o.action == COACH)
    post.n_counter = sum(1 for o in seq if o.action == COUNTER)
    # Advance kappa per cell to the end of the session so downstream summaries are meaningful.
    k = np.zeros_like(post.lam)
    for o in seq:
        inc = post.g * float(o.coach_strength) * float(o.coach_direction) if o.action == COACH else 0.0
        k = np.power(post.lam, float(o.delta)) * (k + inc)
    post.kappa = k

    pi_post = CarryoverCorrectedEstimator(psych_cfg).fit(seq, grid, post)
    return post, pi_post


def fit_all(
    seq: Sequence[Observation],
    grid: SceneGrid,
    *,
    posterior: Optional[CarryoverPosterior] = None,
    psych_cfg: Optional[PsychometricConfig] = None,
    log_prior: Optional[np.ndarray] = None,
) -> Dict[str, PiStarPosterior]:
    """All three estimators on one observation sequence."""
    out = {
        METHOD_POOLED: PooledBetaEstimator().fit(seq, grid),
        METHOD_PSYCHOMETRIC: PsychometricEstimator(psych_cfg).fit(seq, grid),
    }
    post = posterior
    if post is None:
        post, _ = joint_carryover_posterior(seq, grid, psych_cfg=psych_cfg, log_prior=log_prior)
    out[METHOD_CORRECTED] = CarryoverCorrectedEstimator(psych_cfg).fit(seq, grid, post)
    return out


# ---------------------------------------------------------------------------
# Reference map
# ---------------------------------------------------------------------------
def reference_map_from_observations(
    seq: Sequence[Observation],
    grid: SceneGrid,
    *,
    psych_cfg: Optional[PsychometricConfig] = None,
) -> Dict[int, float]:
    """The operational reference ``tilde-pi``, fitted from an uncontaminated reference block.

    Uses the psychometric estimator, not the pooled one: with a realistic reference budget the
    per-scene counts are single digits, and a pooled reference would carry so much sampling
    noise that every condition's "estimation error" would be dominated by the error in the
    thing it is measured against.
    """
    return PsychometricEstimator(psych_cfg).fit(seq, grid).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mae(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    grid: SceneGrid,
    *,
    crossover_weighted: bool = True,
) -> float:
    """Weighted mean absolute error against the reference map."""
    w = grid.band_weights(crossover_weighted)
    mean = estimate.mean()
    num = sum(w[sid] * abs(mean[sid] - float(reference[sid])) for sid in mean if sid in w and sid in reference)
    den = sum(w[sid] for sid in mean if sid in w and sid in reference) or 1.0
    return float(num / den)


def brier_vs_reference(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    grid: SceneGrid,
    *,
    crossover_weighted: bool = True,
) -> float:
    w = grid.band_weights(crossover_weighted)
    mean = estimate.mean()
    num = sum(w[sid] * (mean[sid] - float(reference[sid])) ** 2 for sid in mean if sid in w and sid in reference)
    den = sum(w[sid] for sid in mean if sid in w and sid in reference) or 1.0
    return float(num / den)


def interval_coverage(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    grid: SceneGrid,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> Dict[str, float]:
    """Fraction of scenes whose reference value falls inside the credible interval.

    A condition that wins on point error while under-covering has not won: an over-confident
    estimator is worse than useless to a robot that has to decide whether to act on its belief.
    """
    out: Dict[str, float] = {}
    ids = [s.scene_id for s in grid.probe_scenes() if s.scene_id in reference]
    for lv in levels:
        itv = estimate.interval(lv)
        hit = sum(1 for sid in ids if itv[sid][0] <= float(reference[sid]) <= itv[sid][1])
        out[f"coverage@{int(round(lv * 100))}"] = float(hit / max(len(ids), 1))
    return out


def trial_level_calibration(
    estimate: PiStarPosterior,
    seq: Sequence[Observation],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """ECE and reliability bins of the map's predictions against the actual answers given."""
    preds, outs = [], []
    mean = estimate.mean()
    for o in seq:
        if o.observed and o.scene_id in mean:
            preds.append(float(mean[o.scene_id]))
            outs.append(bool(o.instructed == STRATEGY_A))
    if not preds:
        return {"ece": None, "n": 0, "bins": []}
    return {
        "ece": float(expected_calibration_error(preds, outs, n_bins=n_bins)),
        "n": len(preds),
        "bins": reliability_bins(preds, outs, n_bins=n_bins),
    }


# ---------------------------------------------------------------------------
# Regret -- the outcome a practitioner cares about
# ---------------------------------------------------------------------------
def _preferred(p: float) -> str:
    return STRATEGY_A if float(p) >= 0.5 else STRATEGY_B


def decision_regret(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    grid: SceneGrid,
    *,
    crossover_weighted: bool = True,
) -> Dict[str, float]:
    """**Deployment regret**: what acting on the estimated map would cost, per scene.

    If, after the session, the robot were to execute what it believes each supervisor prefers,
    it would sometimes execute the other strategy. Where the map is wrong, that costs real task
    value under the scene physics. This converts an estimation error into the currency a
    practitioner can price, and it is the number that answers "so what if the map is a bit
    off?".

    Also returns ``alignment``: the fraction of scenes where the estimated preference matches
    the reference preference. Regret is the better outcome (it weights mistakes by how much
    they cost), alignment the more legible one.
    """
    w = grid.band_weights(crossover_weighted)
    mean = estimate.mean()
    phys = grid.physics
    num = 0.0
    den = 0.0
    aligned = 0
    n = 0
    for s in grid.probe_scenes():
        sid = s.scene_id
        if sid not in reference or sid not in mean or sid not in w:
            continue
        want = _preferred(reference[sid])
        got = _preferred(mean[sid])
        loss = phys.value(want, s.margin_m) - phys.value(got, s.margin_m)
        num += w[sid] * max(0.0, loss)
        den += w[sid]
        aligned += int(want == got)
        n += 1
    return {
        "deployment_regret": float(num / (den or 1.0)),
        "alignment": float(aligned / max(n, 1)),
        "n_scenes": n,
    }


def executed_regret(seq: Sequence[Observation], reference: Dict[int, float], grid: SceneGrid) -> Dict[str, float]:
    """**Executed regret**: what the supervisor lost *during* the session to compliance.

    For each answered slot, the value of the strategy the supervisor's unprompted preference
    would have chosen, minus the value of the one they actually instructed. Positive means the
    session's coaching pushed them into instructing something worth less than what they would
    have picked cold. This is the loss the study is trying to prevent, measured directly rather
    than inferred from a map.
    """
    phys = grid.physics
    total = 0.0
    n = 0
    flips = 0
    for o in seq:
        if not o.observed or o.scene_id is None or o.scene_id not in reference:
            continue
        scene = grid.by_id(o.scene_id)
        want = _preferred(reference[o.scene_id])
        got = str(o.instructed)
        total += max(0.0, phys.value(want, scene.margin_m) - phys.value(got, scene.margin_m))
        flips += int(want != got)
        n += 1
    return {
        "executed_regret_per_slot": float(total / max(n, 1)),
        "executed_regret_total": float(total),
        "flip_rate": float(flips / max(n, 1)),
        "n_slots": n,
    }


def evaluate(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    grid: SceneGrid,
    seq: Sequence[Observation],
) -> Dict[str, Any]:
    """Every outcome for one (condition, participant) cell, in the order the paper reports."""
    out: Dict[str, Any] = {
        "method": estimate.method,
        "mae": mae(estimate, reference, grid, crossover_weighted=False),
        "mae_crossover": mae(estimate, reference, grid, crossover_weighted=True),
        "brier": brier_vs_reference(estimate, reference, grid, crossover_weighted=False),
        "brier_crossover": brier_vs_reference(estimate, reference, grid, crossover_weighted=True),
    }
    out.update(interval_coverage(estimate, reference, grid))
    out["calibration"] = trial_level_calibration(estimate, seq)
    out.update(decision_regret(estimate, reference, grid))
    out.update(executed_regret(seq, reference, grid))
    out["n_probe"] = sum(1 for o in seq if o.action == PROBE)
    out["n_counter"] = sum(1 for o in seq if o.action == COUNTER)
    out["n_wait"] = sum(1 for o in seq if o.action == WAIT)
    out["n_coach"] = sum(1 for o in seq if o.action == COACH)
    out["n_ungrounded"] = sum(1 for o in seq if o.action in (PROBE, COUNTER) and not o.observed)
    out["wall_clock_s"] = float(sum(o.duration_s for o in seq))
    return out


__all__ = [
    "METHOD_POOLED",
    "METHOD_PSYCHOMETRIC",
    "METHOD_CORRECTED",
    "METHODS",
    "Observation",
    "sequence_from_records",
    "PiStarPosterior",
    "PooledBetaEstimator",
    "PsychometricConfig",
    "PsychometricEstimator",
    "CarryoverCorrectedEstimator",
    "joint_carryover_posterior",
    "fit_all",
    "reference_map_from_observations",
    "mae",
    "brier_vs_reference",
    "interval_coverage",
    "trial_level_calibration",
    "decision_regret",
    "executed_regret",
    "evaluate",
]

"""W4 — the estimand ``pi*(l)`` and its three estimators. The primary outcome is defined here.

``pi*_p(l) = P[participant selects the nonpreferred arm | target at l, no recent robot prompt]``
(``rehab.md`` §1.1). Two properties of ``pi*`` drive everything:

- **It is not uniform.** Far to the nonpreferred side ``pi* -> 1``; far to the preferred side
  ``pi* -> 0``. Nearly all the information — and nearly all the between-person variance —
  lives in the crossover band near the midline, so error metrics are *crossover-weighted*.
- **It is a probability, not a label.** Each presentation is one Bernoulli draw, so the budget
  per target is tiny and estimators that share strength across nearby targets are not a
  refinement but a requirement.

Three estimators, all returning a :class:`PiStarPosterior` over every target:

``pooled``
    Per-target Beta-Bernoulli, ignoring carryover. The honest naive baseline.
``spatial``
    Bayesian logistic over workspace coordinates — a two-parameter psychometric function in
    the nonpreferred-signed lateral coordinate ``s`` plus a depth term — fit by Laplace
    approximation. Monotone in ``s`` by construction, which is the one strong prior the
    science actually supports.
``carryover_corrected``
    ``spatial`` with each observation's likelihood conditioned on ``beta * kappa_t`` from
    :mod:`vla_lab.rehab.carryover`, **marginalized over the carryover posterior** rather than
    plugged in at its mean — otherwise the corrected estimator would claim a precision it
    does not have.

Every posterior is represented the same way: a discrete distribution on a shared probability
grid. That keeps means, sds, credible intervals, and the marginalization in (c) uniform
across estimators (and avoids needing an incomplete-beta inverse, which would drag in scipy —
the repo deliberately has no scipy dependency).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..calibration.metrics import expected_calibration_error, reliability_bins
from . import ASSESS, COACH
from .carryover import CarryoverConfig, CarryoverPosterior, delta_for, logit, sigmoid
from .workspace import TargetGrid, TargetSpec, nonpreferred_lateral

METHOD_POOLED = "pooled"
METHOD_SPATIAL = "spatial"
METHOD_CORRECTED = "carryover_corrected"

_GRID_N = 201


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass
class TrialObservation:
    """One trial as the estimators see it. ``y is None`` = no usable selection.

    The Phase 0 analogue of the VLA track's ``calibration.records.CalibrationRecord``
    (which is dispersion/occlusion-shaped and does not apply here).
    """

    trial_idx: int
    action: str
    target_id: Optional[int]
    s_m: float           # nonpreferred-signed lateral coordinate
    depth_m: float
    y: Optional[bool]    # chose nonpreferred
    delta: float = 1.0   # decay units to the NEXT trial
    strength: float = 1.0
    block_idx: int = 0
    condition: str = ""

    @property
    def observed(self) -> bool:
        return self.y is not None and self.target_id is not None


def sequence_from_records(
    records: Sequence[Any],
    grid: TargetGrid,
    nonpreferred_side: str,
    *,
    cfg: Optional[CarryoverConfig] = None,
    effort_strength: Optional[Dict[str, float]] = None,
) -> List[TrialObservation]:
    """Convert :class:`~vla_lab.rehab.trial.TrialRecord`s into estimator inputs.

    The **full** sequence is returned, WAIT slots and timeouts included, because ``kappa``
    depends on every slot even though only some slots yield an observation.
    """

    cfg = cfg or CarryoverConfig()
    recs = list(records)
    anchors: List[Optional[int]] = [
        (r.trial.t_go_ms if r.trial.t_go_ms is not None else r.trial.t_present_ms) for r in recs
    ]
    out: List[TrialObservation] = []
    for i, rec in enumerate(recs):
        tr, res = rec.trial, rec.result
        dt = None
        if i + 1 < len(recs) and anchors[i] is not None and anchors[i + 1] is not None:
            dt = float(int(anchors[i + 1]) - int(anchors[i]))  # type: ignore[arg-type]
        delta = delta_for(cfg, dt_ms=dt) if i + 1 < len(recs) else 0.0
        s_m, depth = 0.0, 0.0
        if tr.target_id is not None:
            t = grid.get(int(tr.target_id))
            s_m = nonpreferred_lateral(t.y_m, nonpreferred_side)
            depth = float(t.x_m)
        strength = 1.0
        if effort_strength is not None:
            strength = float(effort_strength.get(str(tr.effort_level), 1.0))
        out.append(
            TrialObservation(
                trial_idx=int(tr.trial_idx),
                action=str(tr.action),
                target_id=(int(tr.target_id) if tr.target_id is not None else None),
                s_m=float(s_m),
                depth_m=float(depth),
                y=res.chose_nonpreferred,
                delta=float(delta),
                strength=float(strength),
                block_idx=int(tr.block_idx),
                condition=str(tr.condition),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Posterior container
# ---------------------------------------------------------------------------


def _prob_grid(n: int = _GRID_N) -> np.ndarray:
    # Interior points only: the endpoints have zero density under every family used here
    # and would only create 0/0 in the normalizations.
    return np.linspace(0.5 / n, 1.0 - 0.5 / n, n)


@dataclass
class PiStarPosterior:
    """A discrete posterior over ``pi*(l)`` for every target in ``L``."""

    target_ids: List[int]
    grid: np.ndarray                 # shape [G], the shared probability grid
    density: Dict[int, np.ndarray]   # target_id -> normalized weights on the grid
    method: str = ""
    n_observations: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- summaries ---------------------------------------------------------
    def mean(self) -> Dict[int, float]:
        return {tid: float(np.dot(self.density[tid], self.grid)) for tid in self.target_ids}

    def sd(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for tid in self.target_ids:
            w = self.density[tid]
            m = float(np.dot(w, self.grid))
            out[tid] = float(math.sqrt(max(0.0, float(np.dot(w, (self.grid - m) ** 2)))))
        return out

    def interval(self, level: float = 0.9) -> Dict[int, Tuple[float, float]]:
        lo_q, hi_q = (1.0 - float(level)) / 2.0, 1.0 - (1.0 - float(level)) / 2.0
        out: Dict[int, Tuple[float, float]] = {}
        for tid in self.target_ids:
            cw = np.cumsum(self.density[tid])
            lo = float(self.grid[int(np.searchsorted(cw, lo_q))])
            hi = float(self.grid[min(len(self.grid) - 1, int(np.searchsorted(cw, hi_q)))])
            out[tid] = (lo, hi)
        return out

    def at(self, target_id: int) -> float:
        return float(np.dot(self.density[int(target_id)], self.grid))

    def to_dict(self, *, level: float = 0.9) -> Dict[str, Any]:
        m, s, ci = self.mean(), self.sd(), self.interval(level)
        return {
            "method": self.method,
            "n_observations": int(self.n_observations),
            "level": float(level),
            "targets": {
                str(tid): {
                    "mean": round(m[tid], 5),
                    "sd": round(s[tid], 5),
                    "ci": [round(ci[tid][0], 5), round(ci[tid][1], 5)],
                }
                for tid in self.target_ids
            },
            **({"extra": self.extra} if self.extra else {}),
        }


def _logitnormal_density(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(1e-6, float(sigma))
    z = (np.log(grid / (1.0 - grid)) - float(mu)) / sigma
    # 1/(p(1-p)) Jacobian folded in; normalization is by sum, so constants drop out.
    d = np.exp(-0.5 * z * z) / (grid * (1.0 - grid))
    s = float(d.sum())
    return d / s if s > 0 else np.full_like(d, 1.0 / d.size)


def _beta_density(grid: np.ndarray, a: float, b: float) -> np.ndarray:
    la = (max(1e-6, float(a)) - 1.0) * np.log(grid) + (max(1e-6, float(b)) - 1.0) * np.log1p(-grid)
    la -= la.max()
    d = np.exp(la)
    s = float(d.sum())
    return d / s if s > 0 else np.full_like(d, 1.0 / d.size)


# ---------------------------------------------------------------------------
# (a) pooled Beta-Bernoulli
# ---------------------------------------------------------------------------


@dataclass
class PooledBetaEstimator:
    """Per-target Beta-Bernoulli. Ignores carryover *and* ignores spatial structure.

    With a realistic budget most targets get a handful of draws, so this is deliberately the
    weak baseline — it is in the paper to show what "just count the choices" buys you.
    """

    prior_a: float = 1.0
    prior_b: float = 1.0

    def fit(self, seq: Sequence[TrialObservation], grid: TargetGrid) -> PiStarPosterior:
        g = _prob_grid()
        counts: Dict[int, Tuple[float, float]] = {t.target_id: (self.prior_a, self.prior_b) for t in grid}
        n = 0
        for o in seq:
            if not o.observed:
                continue
            a, b = counts.get(int(o.target_id), (self.prior_a, self.prior_b))  # type: ignore[arg-type]
            counts[int(o.target_id)] = (a + (1.0 if o.y else 0.0), b + (0.0 if o.y else 1.0))  # type: ignore[arg-type]
            n += 1
        dens = {tid: _beta_density(g, *counts[tid]) for tid in counts}
        return PiStarPosterior(
            target_ids=sorted(counts),
            grid=g,
            density=dens,
            method=METHOD_POOLED,
            n_observations=n,
        )


# ---------------------------------------------------------------------------
# (b) spatial Bayesian logistic
# ---------------------------------------------------------------------------


@dataclass
class SpatialLogisticConfig:
    """Gaussian priors on the psychometric weights (units: 1 / metre for the slopes)."""

    intercept_sd: float = 2.0
    slope_sd: float = 20.0     # d logit(pi*) / d s ; ~20 covers a 0.2 m-wide crossover
    depth_sd: float = 6.0
    max_iter: int = 60
    tol: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _features(seq_s: np.ndarray, seq_depth: np.ndarray, depth_ref: float) -> np.ndarray:
    return np.column_stack([np.ones_like(seq_s), seq_s, seq_depth - float(depth_ref)])


def _stable_sigmoid(eta: np.ndarray) -> np.ndarray:
    """Overflow-free logistic. Separable data drives ``eta`` to +-1e3 during Newton steps."""

    e = np.asarray(eta, dtype=np.float64)
    out = np.empty_like(e)
    pos = e >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-e[pos]))
    ex = np.exp(e[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _laplace_logistic(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    prior_sd: np.ndarray,
    *,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Newton MAP + Laplace covariance for logistic regression with a fixed ``offset``.

    The posterior is Gaussian in weight space; ``offset`` carries the carryover correction
    ``beta * kappa_t``, which is known per observation and therefore not a parameter.
    """

    d = X.shape[1]
    prior_prec = np.diag(1.0 / np.maximum(1e-12, np.asarray(prior_sd, dtype=np.float64) ** 2))
    w = np.zeros(d, dtype=np.float64)
    if X.shape[0] == 0:
        return w, np.linalg.inv(prior_prec)
    for _ in range(int(max_iter)):
        eta = X @ w + offset
        p = _stable_sigmoid(eta)
        grad = X.T @ (y - p) - prior_prec @ w
        W = np.clip(p * (1.0 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + prior_prec
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # Step-halving keeps Newton honest when a separable dataset sends it to infinity.
        scale = 1.0
        for _ in range(20):
            cand = w + scale * step
            if np.all(np.isfinite(cand)):
                break
            scale *= 0.5
        w_new = w + scale * step
        if float(np.max(np.abs(w_new - w))) < float(tol):
            w = w_new
            break
        w = w_new
    eta = X @ w + offset
    p = _stable_sigmoid(eta)
    W = np.clip(p * (1.0 - p), 1e-9, None)
    H = X.T @ (X * W[:, None]) + prior_prec
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    return w, cov


@dataclass
class SpatialLogisticEstimator:
    """Bayesian logistic over ``(s, depth)``; nearby targets share strength.

    ``nonpreferred_side`` is baked into ``s`` upstream (:func:`sequence_from_records`), so the
    fitted slope is positive for every participant and the crossover location is directly
    interpretable as "how far past the midline the switch happens".
    """

    cfg: SpatialLogisticConfig = field(default_factory=SpatialLogisticConfig)

    def _fit_weights(
        self, seq: Sequence[TrialObservation], grid: TargetGrid, offsets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        obs = [o for o in seq if o.observed]
        depth_ref = float(np.mean([t.x_m for t in grid])) if len(grid) else 0.0
        if not obs:
            X = np.zeros((0, 3))
            y = np.zeros(0)
            off = np.zeros(0)
        else:
            X = _features(
                np.array([o.s_m for o in obs], dtype=np.float64),
                np.array([o.depth_m for o in obs], dtype=np.float64),
                depth_ref,
            )
            y = np.array([1.0 if o.y else 0.0 for o in obs], dtype=np.float64)
            off = np.zeros(len(obs)) if offsets is None else np.asarray(offsets, dtype=np.float64)
        prior_sd = np.array([self.cfg.intercept_sd, self.cfg.slope_sd, self.cfg.depth_sd], dtype=np.float64)
        w, cov = _laplace_logistic(X, y, off, prior_sd, max_iter=self.cfg.max_iter, tol=self.cfg.tol)
        return w, cov, depth_ref

    def _posterior_from_weights(
        self,
        w: np.ndarray,
        cov: np.ndarray,
        depth_ref: float,
        grid: TargetGrid,
        nonpreferred_side: str,
        *,
        method: str,
        n_obs: int,
        weight: float = 1.0,
        accumulate: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, np.ndarray]:
        pg = _prob_grid()
        out = accumulate if accumulate is not None else {}
        for t in grid:
            phi = np.array([1.0, nonpreferred_lateral(t.y_m, nonpreferred_side), t.x_m - depth_ref])
            mu = float(phi @ w)
            var = float(phi @ cov @ phi)
            d = _logitnormal_density(pg, mu, math.sqrt(max(1e-12, var)))
            out[t.target_id] = out.get(t.target_id, 0.0) + float(weight) * d  # type: ignore[assignment]
        return out

    def fit(self, seq: Sequence[TrialObservation], grid: TargetGrid, nonpreferred_side: str) -> PiStarPosterior:
        w, cov, depth_ref = self._fit_weights(seq, grid)
        dens = self._posterior_from_weights(
            w, cov, depth_ref, grid, nonpreferred_side, method=METHOD_SPATIAL, n_obs=0
        )
        dens = {k: v / max(1e-300, float(v.sum())) for k, v in dens.items()}
        return PiStarPosterior(
            target_ids=sorted(dens),
            grid=_prob_grid(),
            density=dens,
            method=METHOD_SPATIAL,
            n_observations=sum(1 for o in seq if o.observed),
            extra={
                "weights": [round(float(x), 5) for x in w],
                "depth_ref_m": round(float(depth_ref), 5),
                "crossover_s_m": _crossover_from_weights(w),
            },
        )


def _crossover_from_weights(w: np.ndarray) -> Optional[float]:
    """Where the fitted psychometric function crosses 0.5, in ``s`` (metres)."""

    if abs(float(w[1])) < 1e-9:
        return None
    return round(float(-w[0] / w[1]), 5)


# ---------------------------------------------------------------------------
# (c) carryover-corrected
# ---------------------------------------------------------------------------


@dataclass
class CarryoverCorrectedEstimator:
    """``spatial`` with a per-observation ``beta * kappa_t`` offset, marginalized over
    the carryover posterior.

    Marginalization is over ``n_cells`` cells **resampled** from the carryover posterior (see
    :meth:`~vla_lab.rehab.carryover.CarryoverPosterior.resample_cells`), not the top-``k`` by
    weight — a diffuse posterior's highest-density cells all sit at ``beta ~ 0``, which would
    quietly turn the corrected estimator back into the uncorrected one. Plugging in the
    posterior *mean* instead would understate the uncertainty exactly where the paper's
    calibration claim lives, so that is not offered as an option either.
    """

    spatial: SpatialLogisticEstimator = field(default_factory=SpatialLogisticEstimator)
    n_cells: int = 32

    def fit(
        self,
        seq: Sequence[TrialObservation],
        grid: TargetGrid,
        nonpreferred_side: str,
        posterior: CarryoverPosterior,
    ) -> PiStarPosterior:
        cells = posterior.resample_cells(int(self.n_cells))
        k = len(cells)

        pg = _prob_grid()
        acc: Dict[int, np.ndarray] = {t.target_id: np.zeros_like(pg) for t in grid}
        used_weights: List[List[float]] = []
        for cell in cells:
            lam, beta, gg, cw = cell["lam"], cell["beta"], cell["g"], cell["weight"]
            offsets = _offsets_for_cell(seq, lam=lam, beta=beta, g=gg)
            w, cov, depth_ref = self.spatial._fit_weights(seq, grid, offsets=offsets)
            used_weights.append([round(float(x), 5) for x in w])
            self.spatial._posterior_from_weights(
                w, cov, depth_ref, grid, nonpreferred_side,
                method=METHOD_CORRECTED, n_obs=0, weight=float(cw), accumulate=acc,
            )
        dens = {kk: v / max(1e-300, float(v.sum())) for kk, v in acc.items()}
        return PiStarPosterior(
            target_ids=sorted(dens),
            grid=pg,
            density=dens,
            method=METHOD_CORRECTED,
            n_observations=sum(1 for o in seq if o.observed),
            extra={
                "n_cells_marginalized": int(k),
                "carryover_mean": {kk: round(v, 5) for kk, v in posterior.mean().items()},
                "map_cell_weights": used_weights[0] if used_weights else [],
            },
        )


def _offsets_for_cell(seq: Sequence[TrialObservation], *, lam: float, beta: float, g: float) -> np.ndarray:
    """``beta * kappa_t`` for the observed trials, under one ``(lambda, beta, g)`` cell."""

    kappa = 0.0
    out: List[float] = []
    for o in seq:
        eff = kappa + (g * float(o.strength) if o.action == COACH else 0.0)
        if o.observed:
            out.append(beta * eff)
        kappa = (lam ** float(o.delta)) * eff
    return np.asarray(out, dtype=np.float64)


def joint_carryover_posterior(
    seq: Sequence[TrialObservation],
    grid: TargetGrid,
    *,
    cfg: Optional[CarryoverConfig] = None,
    spatial_cfg: Optional[SpatialLogisticConfig] = None,
) -> CarryoverPosterior:
    """Posterior over ``(lambda, beta, g)`` with ``pi*`` **integrated out**, not plugged in.

    :meth:`vla_lab.rehab.carryover.CarryoverPosterior.step` conditions on a supplied ``pi*``,
    which is what a real-time scheduler can afford. Offline that plug-in is badly biased: a
    ``pi*`` fitted on contaminated data absorbs the carryover, leaving little residual for the
    carryover model to find, and the two then keep each other wrong. Measured on synthetic
    data with a true ``beta*g = 1.2``, the plug-in recovers ~0.45.

    Here the nuisance weights of the spatial model are **marginalized** instead, by Laplace
    approximation, giving a proper profile marginal likelihood per grid cell::

        log p(data | lambda, beta, g)
            ~ log p(data | w_MAP, offsets) + log p(w_MAP) + 0.5 * log det Sigma

    which recovers ~1.14 on the same data.

    **Include the reference block in ``seq``.** Its observations carry ``kappa = 0`` by
    construction and are what pins the intercept of ``pi*``: without them, a slowly-decaying
    carryover is nearly constant and therefore nearly confounded with the intercept. That is
    not a limitation of this estimator, it is why reference-first/retest-last exists (§12.2) —
    and it is visible in the numbers (with ``lambda = 0.9``, recovery of ``beta*g`` goes from
    1.08 to 1.59 against a truth of 1.80 when the reference block is included).
    """

    post = CarryoverPosterior(cfg or CarryoverConfig())
    scfg = spatial_cfg or SpatialLogisticConfig()
    obs = [o for o in seq if o.observed]
    if not obs:
        return post

    depth_ref = float(np.mean([t.x_m for t in grid])) if len(grid) else 0.0
    X = _features(
        np.array([o.s_m for o in obs], dtype=np.float64),
        np.array([o.depth_m for o in obs], dtype=np.float64),
        depth_ref,
    )
    y = np.array([1.0 if o.y else 0.0 for o in obs], dtype=np.float64)
    prior_sd = np.array([scfg.intercept_sd, scfg.slope_sd, scfg.depth_sd], dtype=np.float64)

    log_ml = np.zeros(post.lam.size, dtype=np.float64)
    for i in range(post.lam.size):
        off = _offsets_for_cell(seq, lam=float(post.lam[i]), beta=float(post.beta[i]), g=float(post.g[i]))
        w, cov = _laplace_logistic(X, y, off, prior_sd, max_iter=scfg.max_iter, tol=scfg.tol)
        p = np.clip(_stable_sigmoid(X @ w + off), 1e-12, 1.0 - 1e-12)
        ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
        lp = float(-0.5 * np.sum((w / prior_sd) ** 2))
        _, logdet = np.linalg.slogdet(cov)
        log_ml[i] = ll + lp + 0.5 * float(logdet)

    post.log_w = post.log_prior + log_ml
    post.log_w -= float(np.max(post.log_w))
    # Advance kappa to the end of the sequence so the posterior's live state is usable.
    for o in seq:
        eff = post.kappa + (post.g * float(o.strength) if o.action == COACH else 0.0)
        post.kappa = np.power(post.lam, float(o.delta)) * eff
    post.n_observations = len(obs)
    post.n_coach = sum(1 for o in seq if o.action == COACH)
    return post


def fit_all(
    seq: Sequence[TrialObservation],
    grid: TargetGrid,
    nonpreferred_side: str,
    *,
    cfg: Optional[CarryoverConfig] = None,
    spatial_cfg: Optional[SpatialLogisticConfig] = None,
) -> Dict[str, Any]:
    """Fit all three estimators plus the joint carryover posterior on one sequence."""

    spatial = SpatialLogisticEstimator(spatial_cfg or SpatialLogisticConfig())
    carry = joint_carryover_posterior(seq, grid, cfg=cfg, spatial_cfg=spatial_cfg)
    return {
        "carryover": carry,
        METHOD_POOLED: PooledBetaEstimator().fit(seq, grid),
        METHOD_SPATIAL: spatial.fit(seq, grid, nonpreferred_side),
        METHOD_CORRECTED: CarryoverCorrectedEstimator(spatial=spatial).fit(
            seq, grid, nonpreferred_side, carry
        ),
    }


# ---------------------------------------------------------------------------
# Error + calibration metrics (the primary outcome)
# ---------------------------------------------------------------------------


def _weights_for(grid: TargetGrid, crossover_weighted: bool) -> Dict[int, float]:
    return grid.crossover_weights() if crossover_weighted else {t.target_id: 1.0 for t in grid}


def mae(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    *,
    grid: TargetGrid,
    crossover_weighted: bool = True,
) -> float:
    """Weighted mean absolute error of ``pi_hat`` against the reference map."""

    m = estimate.mean()
    w = _weights_for(grid, crossover_weighted)
    num = den = 0.0
    for tid, ref in reference.items():
        if tid not in m:
            continue
        wi = float(w.get(tid, 1.0))
        num += wi * abs(float(m[tid]) - float(ref))
        den += wi
    return float(num / den) if den > 0 else float("nan")


def brier_vs_reference(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    *,
    grid: TargetGrid,
    crossover_weighted: bool = True,
) -> float:
    """Weighted mean **squared** error against the reference *map*.

    This is the "Brier score against ``tilde-pi*``" of §1.5. It is not the classical Brier
    score, which scores probabilities against binary outcomes — that one is
    :func:`brier_vs_outcomes`, and the paper must say which it reports.
    """

    m = estimate.mean()
    w = _weights_for(grid, crossover_weighted)
    num = den = 0.0
    for tid, ref in reference.items():
        if tid not in m:
            continue
        wi = float(w.get(tid, 1.0))
        num += wi * (float(m[tid]) - float(ref)) ** 2
        den += wi
    return float(num / den) if den > 0 else float("nan")


def brier_vs_outcomes(predicted: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Classical Brier score of per-trial predictions against realized binary choices."""

    p = np.asarray(predicted, dtype=np.float64)
    y = np.asarray([1.0 if bool(v) else 0.0 for v in outcomes], dtype=np.float64)
    if p.size == 0 or p.size != y.size:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def interval_coverage(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    *,
    level: float = 0.9,
    grid: Optional[TargetGrid] = None,
    crossover_only: bool = False,
) -> Dict[str, float]:
    """Fraction of targets whose reference value lies inside the credible interval.

    The second primary outcome (§1.5): an estimator whose intervals do not cover at their
    nominal rate is not usable for the scheduling decision it is meant to support, however
    small its point error.
    """

    ci = estimate.interval(level)
    ids = list(reference)
    if crossover_only and grid is not None:
        band = {t.target_id for t in grid.crossover_targets()}
        ids = [t for t in ids if t in band]
    hit = tot = 0
    for tid in ids:
        if tid not in ci:
            continue
        lo, hi = ci[tid]
        tot += 1
        hit += int(lo <= float(reference[tid]) <= hi)
    return {
        "level": float(level),
        "n_targets": int(tot),
        "coverage": float(hit / tot) if tot else float("nan"),
    }


def trial_level_calibration(
    estimate: PiStarPosterior,
    seq: Sequence[TrialObservation],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """ECE + reliability bins of ``pi_hat`` against the realized choices.

    Delegates to :mod:`vla_lab.calibration.metrics` — the same reliability machinery the VLA
    track uses (``rehab.md`` §5: SHARE), applied to a different pair of arrays.
    """

    m = estimate.mean()
    preds: List[float] = []
    outs: List[float] = []
    for o in seq:
        if not o.observed or int(o.target_id) not in m:  # type: ignore[arg-type]
            continue
        preds.append(float(m[int(o.target_id)]))  # type: ignore[arg-type]
        outs.append(1.0 if o.y else 0.0)
    if not preds:
        return {"n": 0, "ece": float("nan"), "brier": float("nan"), "bins": {}}
    return {
        "n": len(preds),
        "ece": expected_calibration_error(preds, outs, n_bins),
        "brier": brier_vs_outcomes(preds, [bool(v) for v in outs]),
        "bins": reliability_bins(preds, outs, n_bins),
    }


def reference_map_from_observations(
    seq: Sequence[TrialObservation],
    grid: TargetGrid,
    nonpreferred_side: str,
    *,
    estimator: Optional[SpatialLogisticEstimator] = None,
) -> Dict[int, float]:
    """The operational reference map ``tilde-pi*`` from a no-prompt block (§12.2).

    Fitted with the *spatial* estimator: the reference block is uncontaminated by
    construction (zero COACH), so no correction applies, but its per-target budget is just as
    thin as everywhere else and pooling per target would make the reference itself noisy.
    """

    est = estimator or SpatialLogisticEstimator()
    return est.fit(seq, grid, nonpreferred_side).mean()


def evaluate(
    estimate: PiStarPosterior,
    reference: Dict[int, float],
    seq: Sequence[TrialObservation],
    *,
    grid: TargetGrid,
    level: float = 0.9,
) -> Dict[str, Any]:
    """The full primary/secondary outcome bundle for one estimator on one block."""

    return {
        "method": estimate.method,
        "n_observations": int(estimate.n_observations),
        "mae_crossover_weighted": mae(estimate, reference, grid=grid, crossover_weighted=True),
        "mae_unweighted": mae(estimate, reference, grid=grid, crossover_weighted=False),
        "brier_vs_reference": brier_vs_reference(estimate, reference, grid=grid, crossover_weighted=True),
        "coverage": interval_coverage(estimate, reference, level=level, grid=grid),
        "coverage_crossover": interval_coverage(
            estimate, reference, level=level, grid=grid, crossover_only=True
        ),
        "calibration": trial_level_calibration(estimate, seq),
    }


__all__ = [
    "METHOD_POOLED",
    "METHOD_SPATIAL",
    "METHOD_CORRECTED",
    "TrialObservation",
    "PiStarPosterior",
    "PooledBetaEstimator",
    "SpatialLogisticConfig",
    "SpatialLogisticEstimator",
    "CarryoverCorrectedEstimator",
    "joint_carryover_posterior",
    "fit_all",
    "sequence_from_records",
    "reference_map_from_observations",
    "mae",
    "brier_vs_reference",
    "brier_vs_outcomes",
    "interval_coverage",
    "trial_level_calibration",
    "evaluate",
]

"""Shared numerics. NumPy only -- the package deliberately carries no SciPy dependency.

Everything here is either (a) a numerically-careful version of a one-liner that bites when the
argument is large, or (b) a small solver used in more than one place. Nothing here knows
anything about the study.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "logit",
    "sigmoid",
    "sigmoid_np",
    "logsumexp",
    "irls_logistic",
    "normal_ppf",
    "trapz_normalize",
]

_EPS = 1e-9


def logit(p: float) -> float:
    """Log-odds, clamped away from the asymptotes."""
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Scalar logistic, overflow-safe in both tails."""
    x = float(x)
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Vectorised logistic, overflow-safe in both tails."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x)) if x.size else 0.0
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


def irls_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    prior_precision: float = 1e-2,
    prior_mean: Optional[np.ndarray] = None,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """MAP fit of a logistic regression by IRLS, with a Gaussian prior on the weights.

    Returns ``(w, cov)`` where ``cov`` is the inverse observed Fisher information at the mode
    -- i.e. the Laplace posterior covariance. The Gaussian prior is what keeps the fit finite
    under perfect separation, which happens constantly here: a supervisor whose every observed
    answer at one end of the scene range is the same strategy is the *normal* case, not a
    pathology, and an unregularised fit would return an infinite slope for them.

    ``offset`` is a per-observation additive term in the linear predictor that is **not**
    fitted. That is how a contaminated observation is de-biased: pass ``beta * kappa_t``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {y.shape[0]}")
    w_obs = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    mu0 = np.zeros(d) if prior_mean is None else np.asarray(prior_mean, dtype=float).ravel()
    P = float(prior_precision) * np.eye(d)

    w = mu0.copy()
    cov = np.linalg.inv(P)
    for _ in range(int(max_iter)):
        eta = X @ w + off
        p = sigmoid_np(eta)
        s = np.clip(p * (1.0 - p), 1e-9, None) * w_obs
        grad = X.T @ (w_obs * (y - p)) - P @ (w - mu0)
        H = X.T @ (X * s[:, None]) + P
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:  # pragma: no cover - singular only on degenerate input
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w + step
        cov = np.linalg.inv(H)
        if float(np.max(np.abs(step))) < tol:
            break
    return w, cov


# Abramowitz & Stegun 26.2.23 refined by one Newton step against the erf-based CDF.
def normal_ppf(q: float) -> float:
    """Standard-normal quantile. Accurate to ~1e-9 over (1e-12, 1-1e-12)."""
    q = min(max(float(q), 1e-12), 1.0 - 1e-12)
    if q < 0.5:
        return -normal_ppf(1.0 - q)
    t = math.sqrt(-2.0 * math.log(1.0 - q))
    x = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )
    for _ in range(3):
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        x -= (cdf - q) / max(pdf, 1e-300)
    return x


def trapz_normalize(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Normalise a density sampled on ``grid`` so it integrates to 1 (trapezoid rule)."""
    density = np.asarray(density, dtype=float)
    z = float(np.trapz(density, grid))
    if not np.isfinite(z) or z <= 0.0:
        return np.full_like(density, 1.0 / max(len(density), 1))
    return density / z

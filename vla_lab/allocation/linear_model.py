"""A tiny, dependency-free logistic-regression model (NumPy only).

Used for (a) the allocator's **irreducibility detector** — ``P(compute is useless |
uncertainty features)`` — and (b) the **INSIGHT** learned-help-trigger baseline. We ship a
hand-rolled fit so the package has no sklearn dependency (it is not installed in the Isaac
env). Features are standardized internally; the model serializes to/from a plain dict so a
fitted detector can live inside ``allocator_fit.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass
class LogisticModel:
    """Standardized logistic regression with an explicit, ordered feature schema."""

    feature_keys: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    fitted: bool = False

    # ---- inference -------------------------------------------------------
    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        return np.asarray([float(features.get(str(k), 0.0)) for k in self.feature_keys], dtype=np.float64)

    def predict_proba_features(self, features: Dict[str, float]) -> float:
        if not self.fitted or not self.feature_keys:
            return float("nan")
        x = self._vectorize(features)
        mu = np.asarray(self.mean, dtype=np.float64)
        sd = np.asarray(self.std, dtype=np.float64)
        sd = np.where(sd <= 1e-9, 1.0, sd)
        z = ((x - mu) / sd) @ np.asarray(self.weights, dtype=np.float64) + float(self.bias)
        return float(_sigmoid(np.asarray(z)))

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        mu = np.asarray(self.mean, dtype=np.float64)
        sd = np.asarray(self.std, dtype=np.float64)
        sd = np.where(sd <= 1e-9, 1.0, sd)
        z = ((X - mu) / sd) @ np.asarray(self.weights, dtype=np.float64) + float(self.bias)
        return _sigmoid(z)

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_keys": list(self.feature_keys),
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "mean": [float(m) for m in self.mean],
            "std": [float(s) for s in self.std],
            "fitted": bool(self.fitted),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "LogisticModel":
        return cls(
            feature_keys=[str(k) for k in d.get("feature_keys", [])],
            weights=[float(w) for w in d.get("weights", [])],
            bias=float(d.get("bias", 0.0)),
            mean=[float(m) for m in d.get("mean", [])],
            std=[float(s) for s in d.get("std", [])],
            fitted=bool(d.get("fitted", False)),
        )


def fit_logistic(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_keys: Sequence[str],
    *,
    l2: float = 1e-3,
    lr: float = 0.5,
    epochs: int = 2000,
    class_balance: bool = True,
    seed: int = 0,
) -> LogisticModel:
    """Fit a :class:`LogisticModel` by full-batch gradient descent on standardized X.

    ``class_balance`` reweights the (typically rare) positive class so a detector trained
    on mostly-clean data still learns the masked/irreducible minority. Returns a model with
    ``fitted=False`` when the data is degenerate (single class / empty).
    """

    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    keys = [str(k) for k in feature_keys]
    if Xa.ndim != 2 or Xa.shape[0] == 0 or Xa.shape[1] != len(keys):
        return LogisticModel(feature_keys=keys, fitted=False)
    if len(set(ya.tolist())) < 2:
        # Degenerate: return a constant-bias model reflecting the base rate.
        p = float(np.clip(ya.mean(), 1e-4, 1 - 1e-4))
        bias = float(np.log(p / (1.0 - p)))
        return LogisticModel(
            feature_keys=keys,
            weights=[0.0] * len(keys),
            bias=bias,
            mean=[0.0] * len(keys),
            std=[1.0] * len(keys),
            fitted=False,
        )

    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    sd = np.where(sd <= 1e-9, 1.0, sd)
    Z = (Xa - mu) / sd

    n, d = Z.shape
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, 0.01, size=d)
    b = 0.0

    if class_balance:
        pos = float(ya.sum())
        neg = float(n - pos)
        w_pos = n / (2.0 * max(1.0, pos))
        w_neg = n / (2.0 * max(1.0, neg))
        sample_w = np.where(ya > 0.5, w_pos, w_neg)
    else:
        sample_w = np.ones(n)
    sample_w = sample_w / sample_w.mean()

    for _ in range(int(epochs)):
        p = _sigmoid(Z @ w + b)
        grad_core = sample_w * (p - ya)
        grad_w = Z.T @ grad_core / n + l2 * w
        grad_b = float(np.sum(grad_core) / n)
        w -= lr * grad_w
        b -= lr * grad_b

    return LogisticModel(
        feature_keys=keys,
        weights=[float(x) for x in w],
        bias=float(b),
        mean=[float(x) for x in mu],
        std=[float(x) for x in sd],
        fitted=True,
    )

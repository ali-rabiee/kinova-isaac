"""Distribution-free conformal calibration for the "useless-compute" detector.

The principled answer to *"am I in the regime where extra compute cannot help?"* is a
conformal threshold on a nonconformity score (memo §4.3, following KnowNo). Split
conformal gives finite-sample coverage **iff the calibration set is exchangeable with
deployment**. Occlusion is a deliberate covariate shift, so vanilla split-conformal
coverage degrades under exactly our manipulation — we *measure* that degradation
(``coverage_vs_bucket``) rather than claim a guarantee we do not have, and offer
mask-stratified (:class:`MondrianConformal`) and importance-weighted
(:func:`weighted_quantile`) variants that are approximately valid under shift.

Conventions
-----------
A nonconformity score ``s`` is "higher = more anomalous / less typical". For a target
coverage ``1 - alpha`` we take the finite-sample quantile

    q_hat = the ceil((n+1)(1-alpha)) / n  empirical quantile of the calibration scores

and flag ``s > q_hat`` as anomalous (escalate). Everything here is pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample split-conformal quantile of ``scores`` at level ``1 - alpha``.

    Uses the standard ``ceil((n+1)(1-alpha))/n`` correction. Returns ``+inf`` when the
    correction exceeds 1 (too few calibration points for the requested coverage), which
    makes the detector never fire — a safe, explicit degenerate case.
    """

    s = np.asarray(list(scores), dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    a = float(alpha)
    a = min(max(a, 0.0), 1.0)
    level = np.ceil((n + 1) * (1.0 - a)) / n
    if level >= 1.0:
        return float("inf")
    return float(np.quantile(s, level, method="higher"))


def weighted_quantile(scores: Sequence[float], weights: Sequence[float], level: float) -> float:
    """Weighted empirical quantile (for importance-weighted conformal under shift)."""

    s = np.asarray(list(scores), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    if s.size == 0:
        return float("inf")
    if s.shape != w.shape:
        raise ValueError("scores and weights must be the same length")
    order = np.argsort(s)
    s = s[order]
    w = np.clip(w[order], 0.0, None)
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(s[-1])
    target = float(min(max(level, 0.0), 1.0)) * total
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(idx, s.size - 1)
    return float(s[idx])


@dataclass
class SplitConformal:
    """Split-conformal anomaly detector over a 1-D nonconformity score."""

    alpha: float = 0.1
    q_hat: float = float("inf")
    n_calib: int = 0

    def calibrate(self, scores: Sequence[float], alpha: Optional[float] = None) -> "SplitConformal":
        if alpha is not None:
            self.alpha = float(alpha)
        s = np.asarray(list(scores), dtype=np.float64)
        s = s[np.isfinite(s)]
        self.q_hat = conformal_quantile(s, self.alpha)
        self.n_calib = int(s.size)
        return self

    def is_anomalous(self, score: float) -> bool:
        """True when ``score`` exceeds the calibrated quantile (escalate)."""

        return bool(float(score) > self.q_hat)

    def to_dict(self) -> Dict[str, float]:
        return {"alpha": float(self.alpha), "q_hat": float(self.q_hat), "n_calib": int(self.n_calib)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "SplitConformal":
        return cls(
            alpha=float(d.get("alpha", 0.1)),
            q_hat=float(d.get("q_hat", float("inf"))),
            n_calib=int(d.get("n_calib", 0)),
        )


@dataclass
class MondrianConformal:
    """Mask-stratified ("Mondrian") conformal: one quantile per discrete bucket.

    Buckets are typically occlusion-fraction bins. Calibrating per bucket keeps coverage
    approximately valid under the occlusion covariate shift, at the cost of fewer points
    per bucket. Unknown buckets fall back to a pooled quantile.
    """

    alpha: float = 0.1
    q_by_bucket: Dict[str, float] = None  # type: ignore[assignment]
    q_pooled: float = float("inf")

    def __post_init__(self) -> None:
        if self.q_by_bucket is None:
            self.q_by_bucket = {}

    def calibrate(
        self, scores: Sequence[float], buckets: Sequence[str], alpha: Optional[float] = None
    ) -> "MondrianConformal":
        if alpha is not None:
            self.alpha = float(alpha)
        s = np.asarray(list(scores), dtype=np.float64)
        b = np.asarray([str(x) for x in buckets], dtype=object)
        if s.shape[0] != b.shape[0]:
            raise ValueError("scores and buckets must be the same length")
        self.q_pooled = conformal_quantile(s, self.alpha)
        self.q_by_bucket = {}
        for bucket in sorted(set(b.tolist())):
            mask = b == bucket
            self.q_by_bucket[str(bucket)] = conformal_quantile(s[mask], self.alpha)
        return self

    def quantile_for(self, bucket: str) -> float:
        return float(self.q_by_bucket.get(str(bucket), self.q_pooled))

    def is_anomalous(self, score: float, bucket: str) -> bool:
        return bool(float(score) > self.quantile_for(bucket))

    def to_dict(self) -> Dict[str, object]:
        return {
            "alpha": float(self.alpha),
            "q_pooled": float(self.q_pooled),
            "q_by_bucket": {k: float(v) for k, v in self.q_by_bucket.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "MondrianConformal":
        obj = cls(alpha=float(d.get("alpha", 0.1)), q_pooled=float(d.get("q_pooled", float("inf"))))
        obj.q_by_bucket = {str(k): float(v) for k, v in dict(d.get("q_by_bucket", {})).items()}
        return obj


def empirical_coverage(scores: Sequence[float], q_hat: float) -> float:
    """Fraction of ``scores`` at or below ``q_hat`` (the realized coverage)."""

    s = np.asarray(list(scores), dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s <= float(q_hat)))


def coverage_vs_bucket(
    scores: Sequence[float], buckets: Sequence[str], q_hat: float
) -> Dict[str, float]:
    """Realized coverage of a *single* pooled ``q_hat`` within each bucket.

    This is the §4.3 figure: with one calibration quantile fitted on (mostly clean)
    data, coverage **falls** as the occlusion bucket gets harder — the measured
    consequence of split-conformal's exchangeability assumption breaking under shift.
    """

    s = np.asarray(list(scores), dtype=np.float64)
    b = np.asarray([str(x) for x in buckets], dtype=object)
    out: Dict[str, float] = {}
    for bucket in sorted(set(b.tolist())):
        mask = b == bucket
        out[str(bucket)] = empirical_coverage(s[mask].tolist(), q_hat)
    return out


def prediction_set(option_scores: Sequence[float], q_hat: float) -> List[int]:
    """KnowNo-style prediction set: indices of options with score <= q_hat.

    Here ``option_scores`` are nonconformity scores per candidate option (lower = more
    plausible). A set with more than one option means residual ambiguity → KnowNo asks.
    """

    s = np.asarray(list(option_scores), dtype=np.float64)
    keep = [int(i) for i in range(s.size) if s[i] <= float(q_hat)]
    if not keep and s.size:
        keep = [int(np.argmin(s))]  # never return an empty set
    return keep


def bucketize(value: float, edges: Iterable[float]) -> str:
    """Label a continuous value (e.g. occlusion fraction) by half-open bins.

    ``edges=[0.1, 0.2, 0.3]`` -> labels ``"[0.00,0.10)"``, ``"[0.10,0.20)"``, ... .
    """

    e = sorted(float(x) for x in edges)
    lo = 0.0
    v = float(value)
    for hi in e:
        if v < hi:
            return f"[{lo:.2f},{hi:.2f})"
        lo = hi
    return f"[{lo:.2f},inf)"

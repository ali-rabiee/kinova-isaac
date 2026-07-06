"""Result-2 metrics: calibration, agreement→correctness decoupling, coverage under shift.

Implements the three falsifiable predictions from memo §4.2 as pure-NumPy functions:

1. **ECE degrades monotonically with occlusion** — :func:`ece_vs_bucket`.
2. **The agreement→correctness mapping decouples or inverts under heavy occlusion** —
   :func:`decoupling_table` (point-biserial correlation and AUROC per occlusion bucket).
3. **A query rule built only on self-agreement under-queries the high-error masked cases** —
   the AUROC of "agreement predicts correctness" collapsing toward / below 0.5 in
   :func:`decoupling_table`, and :func:`coverage_vs_occlusion` (§4.3).

"Confidence" here is a monotone decreasing transform of dispersion (high self-agreement =
high confidence): ``confidence = exp(-dispersion / scale)``. The mapping's ``scale`` is a
free knob owned by the analysis, not baked into the logs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..allocation.conformal import bucketize, conformal_quantile
from .records import CalibrationRecord


def confidence_from_dispersion(dispersion, scale: float = 0.1):
    """Map dispersion to a confidence proxy in (0, 1]: ``exp(-d/scale)``."""

    d = np.asarray(dispersion, dtype=np.float64)
    s = max(1e-9, float(scale))
    return np.exp(-np.clip(d, 0.0, None) / s)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def _label(rec: CalibrationRecord) -> Optional[float]:
    if rec.correct is not None:
        return 1.0 if rec.correct else 0.0
    if rec.success is not None:
        return 1.0 if rec.success else 0.0
    return None


def extract(
    records: Sequence[CalibrationRecord], scale: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned arrays ``(confidence, correct, dispersion, occlusion_fraction)``.

    Records without any correctness label are dropped.
    """

    corr: List[float] = []
    disp: List[float] = []
    occ: List[float] = []
    for r in records:
        y = _label(r)
        if y is None:
            continue
        disp.append(float(r.dispersion))
        corr.append(float(y))
        occ.append(float(r.occlusion_fraction))
    d = np.asarray(disp, dtype=np.float64)
    return confidence_from_dispersion(d, scale), np.asarray(corr, dtype=np.float64), d, np.asarray(occ, dtype=np.float64)


# ---------------------------------------------------------------------------
# Calibration / reliability
# ---------------------------------------------------------------------------


def reliability_bins(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> Dict[str, list]:
    """Equal-width reliability bins on [0,1]. Returns parallel lists for plotting."""

    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    lo, hi, bconf, bacc, bcount = [], [], [], [], []
    for i in range(int(n_bins)):
        a, b = edges[i], edges[i + 1]
        mask = (c >= a) & (c < b) if i < n_bins - 1 else (c >= a) & (c <= b)
        n = int(mask.sum())
        lo.append(float(a))
        hi.append(float(b))
        bcount.append(n)
        bconf.append(float(c[mask].mean()) if n else float("nan"))
        bacc.append(float(y[mask].mean()) if n else float("nan"))
    return {"bin_lo": lo, "bin_hi": hi, "bin_conf": bconf, "bin_acc": bacc, "bin_count": bcount}


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> float:
    """Weighted |accuracy - confidence| across equal-width bins (lower = better calibrated)."""

    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    n = c.size
    if n == 0:
        return float("nan")
    b = reliability_bins(c, y, n_bins)
    ece = 0.0
    for conf, acc, cnt in zip(b["bin_conf"], b["bin_acc"], b["bin_count"]):
        if cnt == 0 or np.isnan(conf) or np.isnan(acc):
            continue
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def ece_vs_bucket(
    records: Sequence[CalibrationRecord], *, scale: float = 0.1, edges: Optional[Sequence[float]] = None, n_bins: int = 10
) -> Dict[str, Dict[str, float]]:
    """ECE within each occlusion bucket (Prediction 1)."""

    edges = list(edges) if edges is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    conf, corr, _, occ = extract(records, scale)
    buckets = np.asarray([bucketize(o, edges) for o in occ], dtype=object)
    out: Dict[str, Dict[str, float]] = {}
    for bucket in sorted(set(buckets.tolist())):
        m = buckets == bucket
        out[str(bucket)] = {
            "n": int(m.sum()),
            "ece": expected_calibration_error(conf[m], corr[m], n_bins),
            "accuracy": float(corr[m].mean()) if m.sum() else float("nan"),
            "mean_confidence": float(conf[m].mean()) if m.sum() else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Agreement -> correctness coupling
# ---------------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1..n), like scipy.stats.rankdata (no scipy dependency)."""

    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """AUROC of ``scores`` ranking positives above negatives (Mann–Whitney). NaN if degenerate."""

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def point_biserial(x: Sequence[float], binary: Sequence[float]) -> float:
    """Pearson correlation between continuous ``x`` and a {0,1} label. NaN if no variance."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(binary, dtype=np.float64)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def decoupling_table(
    records: Sequence[CalibrationRecord], *, scale: float = 0.1, edges: Optional[Sequence[float]] = None
) -> Dict[str, Dict[str, float]]:
    """Per occlusion bucket: accuracy, agreement→correctness corr, and AUROC (Predictions 2 & 3).

    A high AUROC / positive correlation means self-agreement tracks correctness (querying on
    low agreement would be well-targeted). As occlusion grows, AUROC collapsing toward 0.5
    (or below) is the decoupling/inversion finding — the regime where an agreement-only query
    rule under-queries the confidently-wrong masked cases.
    """

    edges = list(edges) if edges is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    conf, corr, disp, occ = extract(records, scale)
    buckets = np.asarray([bucketize(o, edges) for o in occ], dtype=object)
    out: Dict[str, Dict[str, float]] = {}
    for bucket in sorted(set(buckets.tolist())):
        m = buckets == bucket
        out[str(bucket)] = {
            "n": int(m.sum()),
            "accuracy": float(corr[m].mean()) if m.sum() else float("nan"),
            "mean_confidence": float(conf[m].mean()) if m.sum() else float("nan"),
            "mean_dispersion": float(disp[m].mean()) if m.sum() else float("nan"),
            "corr_agreement_correct": point_biserial(conf[m], corr[m]),
            "auroc_agreement_correct": auroc(conf[m], corr[m]),
        }
    return out


# ---------------------------------------------------------------------------
# Conformal coverage under occlusion shift (§4.3)
# ---------------------------------------------------------------------------


def coverage_vs_occlusion(
    records: Sequence[CalibrationRecord],
    *,
    alpha: float = 0.1,
    edges: Optional[Sequence[float]] = None,
    score: str = "dispersion",
) -> Dict[str, object]:
    """Calibrate a split-conformal "act-when-confident" threshold on the cleanest bucket, then
    show its guarantee breaking under occlusion.

    The threshold ``q_hat`` is set so that on clean data we accept (act on) ``1-alpha`` of
    steps. We then report, per occlusion bucket:

    - ``acceptance_rate``    — ``P(score <= q_hat)``: how often the rule still deems the step
      confident (stays high even when the policy is **confidently wrong**), and
    - ``selective_accuracy`` — ``P(correct | score <= q_hat)``: the realized reliability of
      those confident decisions, which **falls below the ``target = 1-alpha``** as occlusion
      grows. That gap is the §4.3 finding: split-conformal's exchangeability assumption is
      violated by our deliberate covariate shift, so the calibration guarantee degrades.

    ``coverage`` is kept as an alias of ``selective_accuracy`` for back-compatible plotting.
    """

    edges = list(edges) if edges is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    vals: List[float] = []
    labs: List[float] = []
    occ: List[float] = []
    for r in records:
        if score == "irreducibility":
            s = float(r.irreducible_score)
        elif score == "nonconformity" and float(r.nonconformity_score) != 0.0:
            # 0.0 is the dataclass default = "never computed" (controllers without
            # a conformal stage); fall back to dispersion so mixed logs stay usable.
            s = float(r.nonconformity_score)
        else:
            s = float(r.dispersion)
        y = _label(r)
        vals.append(s)
        labs.append(float("nan") if y is None else float(y))
        occ.append(float(r.occlusion_fraction))
    sv = np.asarray(vals, dtype=np.float64)
    ly = np.asarray(labs, dtype=np.float64)
    ov = np.asarray(occ, dtype=np.float64)
    buckets = np.asarray([bucketize(o, edges) for o in ov], dtype=object)

    uniq = sorted(set(buckets.tolist()))
    target = 1.0 - float(alpha)
    if not uniq:
        return {"q_hat": float("inf"), "calib_bucket": None, "coverage": {}, "selective_accuracy": {}, "acceptance_rate": {}, "target": target}
    # Cleanest bucket = the lexicographically-first "[0.00,..)" bin (lowest occlusion).
    calib_bucket = uniq[0]
    q_hat = conformal_quantile(sv[buckets == calib_bucket].tolist(), alpha)

    selective_accuracy: Dict[str, float] = {}
    acceptance_rate: Dict[str, float] = {}
    for bucket in uniq:
        m = buckets == bucket
        accepted = m & (sv <= q_hat)
        acceptance_rate[str(bucket)] = float(np.mean(sv[m] <= q_hat)) if m.sum() else float("nan")
        lab_acc = ly[accepted]
        lab_acc = lab_acc[~np.isnan(lab_acc)]
        selective_accuracy[str(bucket)] = float(lab_acc.mean()) if lab_acc.size else float("nan")
    return {
        "q_hat": float(q_hat),
        "calib_bucket": str(calib_bucket),
        "target": target,
        "acceptance_rate": acceptance_rate,
        "selective_accuracy": selective_accuracy,
        "coverage": selective_accuracy,
    }


def summary(
    records: Sequence[CalibrationRecord], *, scale: float = 0.1, alpha: float = 0.1, edges: Optional[Sequence[float]] = None, n_bins: int = 10
) -> Dict[str, object]:
    """Bundle every Result-2 metric into one JSON-able dict (used by the CLI and tests)."""

    edges = list(edges) if edges is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    conf, corr, _, _ = extract(records, scale)
    return {
        "n_records": int(len(records)),
        "n_labeled": int(conf.size),
        "overall_ece": expected_calibration_error(conf, corr, n_bins),
        "overall_accuracy": float(corr.mean()) if conf.size else float("nan"),
        "overall_auroc_agreement_correct": auroc(conf, corr),
        "ece_vs_occlusion": ece_vs_bucket(records, scale=scale, edges=edges, n_bins=n_bins),
        "decoupling": decoupling_table(records, scale=scale, edges=edges),
        "coverage_vs_occlusion": coverage_vs_occlusion(records, alpha=alpha, edges=edges),
        "params": {"scale": float(scale), "alpha": float(alpha), "edges": [float(x) for x in edges], "n_bins": int(n_bins)},
    }

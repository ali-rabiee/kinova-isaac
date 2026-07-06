"""Fit the act/compute/query allocator from a calibration log (the "training" step).

This is what unlocks the **query** branch (memo C3). From a calibration JSONL produced by the
robot-only runs, it fits and serializes an ``allocator_fit.json``:

1. the **irreducibility detector** — a logistic model ``P(compute is useless | features)``
   trained against a ground-truth ``identifiable`` oracle when available (else a documented
   confidently-wrong proxy);
2. the **conformal escalation rule** — a mask-stratified (Mondrian) or split-conformal
   quantile on the chosen nonconformity score (memo §4.3);
3. a dedicated **split-conformal detector on dispersion** for the KnowNo baseline; and
4. carries the **VoC/VoI** value + cost parameters (defaults, or CLI overrides; ``act_tau``
   can be estimated from the dispersion distribution).

Usage:
    python -m vla_lab.fit_allocator \
        --calib vla_lab/calibration_runs/*/records.jsonl \
        --out vla_lab/fits/allocator_fit.json \
        --conformal mondrian --alpha 0.1 --label auto

It deliberately depends only on NumPy + the in-repo allocation package (no sklearn), so it
runs in the Isaac env or any plain Python env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .allocation.allocator import AllocatorFit
from .allocation.conformal import MondrianConformal, SplitConformal, bucketize
from .allocation.linear_model import fit_logistic
from .allocation.value_of_information import CostModel, ValueParams
from .calibration.metrics import auroc, coverage_vs_occlusion
from .calibration.records import CalibrationRecord, read_many

DEFAULT_FEATURES = ["dispersion", "occlusion_fraction"]


def _record_features(r: CalibrationRecord) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "dispersion": float(r.dispersion),
        "occlusion_fraction": float(r.occlusion_fraction),
        "k_used": float(r.k_used),
        "nonconformity_score": float(r.nonconformity_score),
    }
    for k, v in (r.extra or {}).items():
        try:
            feats[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return feats


def _derive_labels(records: Sequence[CalibrationRecord], mode: str) -> Tuple[np.ndarray, str]:
    """Return (y_irreducible in {0,1}, label_source). 1 = optimal action unidentifiable."""

    have_ident = any(r.identifiable is not None for r in records)
    have_correct = any(r.correct is not None for r in records)
    src = mode
    if mode == "auto":
        src = "identifiable" if have_ident else ("confidently_wrong" if have_correct else "occlusion")

    if src == "identifiable":
        return np.asarray([1.0 if (r.identifiable is False) else 0.0 for r in records], dtype=np.float64), src
    if src == "confidently_wrong":
        disp = np.asarray([float(r.dispersion) for r in records], dtype=np.float64)
        med = float(np.median(disp)) if disp.size else 0.0
        y = np.asarray(
            [1.0 if (r.correct is False and float(r.dispersion) <= med) else 0.0 for r in records], dtype=np.float64
        )
        return y, src
    # occlusion proxy
    return np.asarray([1.0 if float(r.occlusion_fraction) > 0.3 else 0.0 for r in records], dtype=np.float64), src


def fit_from_records(
    records: Sequence[CalibrationRecord],
    *,
    features: Sequence[str] = DEFAULT_FEATURES,
    label_mode: str = "auto",
    alpha: float = 0.1,
    knowno_alpha: float = 0.2,
    conformal_kind: str = "mondrian",
    score_kind: str = "irreducibility",
    bucket_edges: Optional[Sequence[float]] = None,
    k_compute: int = 4,
    value_params: Optional[ValueParams] = None,
    cost: Optional[CostModel] = None,
    estimate_tau: bool = True,
) -> Tuple[AllocatorFit, Dict[str, object]]:
    """Core fitting routine (importable + unit-tested). Returns ``(fit, report)``."""

    keys = list(features)
    edges = list(bucket_edges) if bucket_edges is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    feat_dicts = [_record_features(r) for r in records]
    X = np.asarray([[fd.get(k, 0.0) for k in keys] for fd in feat_dicts], dtype=np.float64)
    y_irr, label_src = _derive_labels(records, label_mode)

    model = fit_logistic(X, y_irr, keys, epochs=2000)

    # Nonconformity score per record for the allocator's conformal rule.
    if score_kind == "irreducibility" and model.fitted:
        scores = model.predict_proba_matrix(X)
    else:
        score_kind = "dispersion"
        scores = np.asarray([float(r.dispersion) for r in records], dtype=np.float64)

    occ = [float(r.occlusion_fraction) for r in records]
    buckets = [bucketize(o, edges) for o in occ]

    conformal = None
    if conformal_kind == "mondrian":
        conformal = MondrianConformal(alpha=alpha).calibrate(scores.tolist(), buckets, alpha=alpha)
    elif conformal_kind == "split":
        conformal = SplitConformal(alpha=alpha).calibrate(scores.tolist(), alpha=alpha)

    dispersions = [float(r.dispersion) for r in records]
    knowno_conf = SplitConformal(alpha=knowno_alpha).calibrate(dispersions, alpha=knowno_alpha)

    vp = value_params or ValueParams()
    if estimate_tau and dispersions:
        # Set the act threshold near a low percentile of dispersion (cheap-act on the easy mass).
        vp.act_dispersion_tau = float(np.quantile(np.asarray(dispersions), 0.25))

    fit = AllocatorFit(
        value_params=vp,
        cost=cost or CostModel(),
        irreducibility=model,
        conformal=conformal,
        conformal_kind=conformal_kind if conformal is not None else "none",
        knowno_conformal=knowno_conf,
        score_kind=score_kind,
        occlusion_bucket_edges=edges,
        k_compute=int(k_compute),
        default_irreducible=0.0,
        allow_voi_query=True,
    )

    report: Dict[str, object] = {
        "n_records": int(len(records)),
        "features": keys,
        "label_source": label_src,
        "irreducible_base_rate": float(np.mean(y_irr)) if y_irr.size else float("nan"),
        "model_fitted": bool(model.fitted),
        "train_auroc_irreducible": auroc(scores.tolist(), y_irr.tolist()) if model.fitted else float("nan"),
        "score_kind": score_kind,
        "conformal_kind": fit.conformal_kind,
        "act_dispersion_tau": float(vp.act_dispersion_tau),
        "knowno_q_hat": float(knowno_conf.q_hat),
        "coverage_vs_occlusion": coverage_vs_occlusion(records, alpha=alpha, edges=edges),
    }
    if isinstance(conformal, MondrianConformal):
        report["conformal_q_by_bucket"] = {k: float(v) for k, v in conformal.q_by_bucket.items()}
    elif isinstance(conformal, SplitConformal):
        report["conformal_q_hat"] = float(conformal.q_hat)
    return fit, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the act/compute/query allocator from calibration logs")
    ap.add_argument("--calib", action="append", default=[], help="calibration JSONL path or glob (repeatable)")
    ap.add_argument("--out", type=str, default="vla_lab/fits/allocator_fit.json")
    ap.add_argument("--report", type=str, default=None, help="optional path for the fit report JSON")
    ap.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--label", type=str, default="auto", choices=["auto", "identifiable", "confidently_wrong", "occlusion"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--knowno-alpha", type=float, default=0.2)
    ap.add_argument("--conformal", type=str, default="mondrian", choices=["mondrian", "split", "none"])
    ap.add_argument("--score-kind", type=str, default="irreducibility", choices=["irreducibility", "dispersion"])
    ap.add_argument("--bucket-edges", type=str, default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--k-compute", type=int, default=4)
    ap.add_argument("--no-estimate-tau", action="store_true")
    # value / cost overrides
    ap.add_argument("--voc-gain", type=float, default=None)
    ap.add_argument("--voi-gain", type=float, default=None)
    ap.add_argument("--query-cost-s", type=float, default=None)
    args = ap.parse_args()

    if not args.calib:
        print("[fit_allocator] no --calib paths given.")
        return 2
    records = read_many(args.calib)
    if not records:
        print("[fit_allocator] no calibration records loaded.")
        return 2

    vp = ValueParams()
    if args.voc_gain is not None:
        vp.voc_gain = float(args.voc_gain)
    if args.voi_gain is not None:
        vp.voi_gain = float(args.voi_gain)
    cost = CostModel()
    if args.query_cost_s is not None:
        cost.query_cost_s = float(args.query_cost_s)

    fit, report = fit_from_records(
        records,
        features=[s for s in args.features.split(",") if s.strip()],
        label_mode=args.label,
        alpha=args.alpha,
        knowno_alpha=args.knowno_alpha,
        conformal_kind=args.conformal,
        score_kind=args.score_kind,
        bucket_edges=[float(x) for x in args.bucket_edges.split(",") if x.strip()],
        k_compute=args.k_compute,
        value_params=vp,
        cost=cost,
        estimate_tau=not args.no_estimate_tau,
    )

    out = Path(args.out)
    fit.save(out)
    print(f"[fit_allocator] {report['n_records']} records | label={report['label_source']} "
          f"base_rate={report['irreducible_base_rate']:.3f} | model_fitted={report['model_fitted']} "
          f"train_AUROC={report['train_auroc_irreducible']:.3f}")
    print(f"[fit_allocator] act_tau={report['act_dispersion_tau']:.4f}  knowno_q={report['knowno_q_hat']:.4f}  "
          f"conformal={report['conformal_kind']}")
    print(f"[fit_allocator] wrote {out}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[fit_allocator] wrote report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

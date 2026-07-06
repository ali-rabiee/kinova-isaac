"""CLI: turn calibration logs into the Result-2 figures + a summary JSON.

    python -m vla_lab.calibration.analyze \
        --calib vla_lab/calibration_runs/*/records.jsonl \
        --out-dir vla_lab/results/calibration --format pdf

Produces (under ``--out-dir``):
  - ``calibration_summary.json``      — every metric in :func:`metrics.summary`
  - ``reliability_diagram.<fmt>``     — confidence vs. accuracy (overall + worst bucket)
  - ``ece_vs_occlusion.<fmt>``        — Prediction 1
  - ``agreement_vs_correctness.<fmt>``— Prediction 2/3 (AUROC + correlation per bucket)
  - ``coverage_vs_occlusion.<fmt>``   — §4.3 conformal-coverage degradation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import metrics
from .records import read_many


def _bucket_midpoint(label: str) -> float:
    # label like "[0.10,0.20)" or "[0.50,inf)"
    try:
        inner = label.strip()[1:-1]
        lo, hi = inner.split(",")
        lo = float(lo)
        hi = 0.6 if hi.startswith("inf") else float(hi)
        return (lo + hi) / 2.0
    except Exception:
        return 0.0


def _plot_reliability(records, scale, n_bins, out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    conf, corr, _, _ = metrics.extract(records, scale)
    if conf.size == 0:
        return
    b = metrics.reliability_bins(conf, corr, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    xs = [c for c in b["bin_conf"]]
    ys = [a for a in b["bin_acc"]]
    ax.plot(xs, ys, "o-", color="#1f77b4", label="empirical")
    ece = metrics.expected_calibration_error(conf, corr, n_bins)
    ax.set_xlabel("Confidence (self-agreement)")
    ax.set_ylabel("Accuracy (correctness)")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability_diagram.{fmt}", dpi=150)
    plt.close(fig)


def _plot_ece_vs_occlusion(table: dict, out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    items = sorted(table.items(), key=lambda kv: _bucket_midpoint(kv[0]))
    if not items:
        return
    x = [_bucket_midpoint(k) for k, _ in items]
    y = [v.get("ece", float("nan")) for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o-", color="#d62728")
    ax.set_xlabel("Occlusion fraction (bucket midpoint)")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Prediction 1: calibration degrades with occlusion")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"ece_vs_occlusion.{fmt}", dpi=150)
    plt.close(fig)


def _plot_agreement_vs_correctness(table: dict, out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    items = sorted(table.items(), key=lambda kv: _bucket_midpoint(kv[0]))
    if not items:
        return
    x = [_bucket_midpoint(k) for k, _ in items]
    auc = [v.get("auroc_agreement_correct", float("nan")) for _, v in items]
    cor = [v.get("corr_agreement_correct", float("nan")) for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.5, ls="--", color="#888", label="AUROC = 0.5 (no signal)")
    ax.plot(x, auc, "o-", color="#2ca02c", label="AUROC(agreement→correct)")
    ax.plot(x, cor, "s-", color="#9467bd", label="corr(agreement, correct)")
    ax.set_xlabel("Occlusion fraction (bucket midpoint)")
    ax.set_ylabel("Agreement↔correctness coupling")
    ax.set_title("Prediction 2/3: self-agreement decouples from correctness")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"agreement_vs_correctness.{fmt}", dpi=150)
    plt.close(fig)


def _plot_coverage(cov: dict, out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    coverage = cov.get("coverage", {})
    items = sorted(coverage.items(), key=lambda kv: _bucket_midpoint(kv[0]))
    if not items:
        return
    x = [_bucket_midpoint(k) for k, _ in items]
    y = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(cov.get("target", 0.9), ls="--", color="#888", label=f"target = {cov.get('target', 0.9):.2f}")
    ax.plot(x, y, "o-", color="#ff7f0e", label="realized coverage")
    ax.set_xlabel("Occlusion fraction (bucket midpoint)")
    ax.set_ylabel("Conformal coverage")
    ax.set_title("§4.3: split-conformal coverage degrades under occlusion shift")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"coverage_vs_occlusion.{fmt}", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration / Result-2 analysis")
    ap.add_argument("--calib", action="append", default=[], help="Calibration JSONL path or glob (repeatable)")
    ap.add_argument("--out-dir", type=str, default="vla_lab/results/calibration")
    ap.add_argument("--scale", type=float, default=0.1, help="dispersion->confidence scale (exp(-d/scale))")
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage level")
    ap.add_argument("--edges", type=str, default="0.1,0.2,0.3,0.4,0.5", help="occlusion bucket edges (comma-sep)")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.calib:
        print("[calibration.analyze] no --calib paths given.")
        return 2
    records = read_many(args.calib)
    if not records:
        print("[calibration.analyze] no records loaded.")
        return 2
    edges = [float(x) for x in str(args.edges).split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = metrics.summary(records, scale=args.scale, alpha=args.alpha, edges=edges, n_bins=args.n_bins)
    (out_dir / "calibration_summary.json").write_text(json.dumps(s, indent=2))
    print(f"[calibration.analyze] {s['n_labeled']}/{s['n_records']} labeled records  "
          f"overall ECE={s['overall_ece']:.3f}  AUROC(agree→correct)={s['overall_auroc_agreement_correct']:.3f}")
    print(f"[calibration.analyze] wrote {out_dir / 'calibration_summary.json'}")

    if not args.no_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("[calibration.analyze] matplotlib not installed; skipping figures (summary JSON still written).")
            return 0
        _plot_reliability(records, args.scale, args.n_bins, out_dir, args.format)
        _plot_ece_vs_occlusion(s["ece_vs_occlusion"], out_dir, args.format)
        _plot_agreement_vs_correctness(s["decoupling"], out_dir, args.format)
        _plot_coverage(s["coverage_vs_occlusion"], out_dir, args.format)
        print(f"[calibration.analyze] wrote figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate paper-style figures from TinyVLA training logs and eval JSON outputs.

Reads:
  - `<run_dir>/metrics.jsonl` — written by `vla_lab.train`
  - Optional eval summaries: `results_*.json` from `vla_lab.eval_isaaclab`

Usage:
    python -m vla_lab.plot_metrics --run-dir vla_lab/checkpoints/tiny_v0
    python -m vla_lab.plot_metrics --run-dir vla_lab/checkpoints/run1 \\
        --eval-json vla_lab/eval_results/run1/results_1730000000.json \\
        --eval-json vla_lab/eval_results/run1/results_1730000100.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .stats_utils import wilson_ci


def _load_metrics_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_logs: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    if not path.exists():
        return train_logs, epoch_rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "train_log":
            train_logs.append(obj)
        elif t == "epoch_end":
            epoch_rows.append(obj)
    train_logs.sort(key=lambda x: int(x.get("step", 0)))
    epoch_rows.sort(key=lambda x: int(x.get("epoch", 0)))
    return train_logs, epoch_rows


def _plot_learning_curves(
    train_logs: List[Dict[str, Any]],
    epoch_rows: List[Dict[str, Any]],
    out_dir: Path,
    fmt: str,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    if train_logs:
        steps = [int(x["step"]) for x in train_logs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, [float(x["loss"]) for x in train_logs], label="train loss", color="#1f77b4", linewidth=1.2)
        ax.plot(steps, [float(x["act_loss"]) for x in train_logs], label="action MSE", color="#ff7f0e", alpha=0.85, linewidth=1.0)
        if any(float(x.get("fa_loss", 0.0)) > 0 for x in train_logs):
            ax.plot(steps, [float(x["fa_loss"]) for x in train_logs], label="align loss", color="#2ca02c", alpha=0.85, linewidth=1.0)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss (logged batches)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"train_loss_vs_step.{fmt}", dpi=150)
        plt.close(fig)

        lrs = [float(x["lr"]) for x in train_logs]
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(steps, lrs, color="#9467bd", linewidth=1.2)
        ax2.set_xlabel("Optimization step")
        ax2.set_ylabel("Learning rate")
        ax2.set_title("Learning-rate schedule")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"lr_vs_step.{fmt}", dpi=150)
        plt.close(fig2)

    if epoch_rows:
        epochs = [int(x["epoch"]) for x in epoch_rows]
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(epochs, [float(x["train_loss"]) for x in epoch_rows], "o-", label="train (epoch mean)", color="#1f77b4")
        has_val = any(x.get("val_loss") is not None for x in epoch_rows)
        if has_val:
            ve = [float(x["val_loss"]) if x.get("val_loss") is not None else float("nan") for x in epoch_rows]
            ax3.plot(epochs, ve, "s-", label="val action MSE", color="#d62728")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Train / validation loss per epoch")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(out_dir / f"train_val_per_epoch.{fmt}", dpi=150)
        plt.close(fig3)


def _load_eval_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _plot_eval_comparison(summaries: List[Tuple[str, Dict[str, Any]]], out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    if not summaries:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    rates: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    for label, s in summaries:
        n_ep = int(s.get("num_episodes", 0))
        ns = int(s.get("num_success", 0))
        rate = float(s.get("success_rate", ns / max(1, n_ep)))
        lo, hi = wilson_ci(ns, n_ep)
        labels.append(label)
        rates.append(rate)
        lows.append(lo)
        highs.append(hi)

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4))
    colors = plt.cm.tab10(range(len(labels)))
    ax.bar(x, rates, yerr=([rates[i] - lows[i] for i in x], [highs[i] - rates[i] for i in x]), capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Evaluation success rate (95% Wilson CI)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"eval_success_rates.{fmt}", dpi=150)
    plt.close(fig)

    if len(summaries) == 1:
        s0 = summaries[0][1]
        elapsed = [float(r.get("elapsed_s", 0.0)) for r in s0.get("results", []) or [] if r.get("elapsed_s") is not None]
        if len(elapsed) > 2:
            figt, axt = plt.subplots(figsize=(6, 3.5))
            axt.hist(elapsed, bins=min(20, max(5, len(elapsed) // 3)), color="#8c564b", edgecolor="black", alpha=0.85)
            axt.set_xlabel("Episode wall time (s)")
            axt.set_ylabel("Count")
            axt.set_title("Eval episode duration")
            axt.grid(True, axis="y", alpha=0.3)
            figt.tight_layout()
            figt.savefig(out_dir / f"eval_episode_duration_hist.{fmt}", dpi=150)
            plt.close(figt)

    # Per-target breakdown when present
    all_targets: List[str] = []
    for _, s in summaries:
        for r in s.get("results", []) or []:
            tl = r.get("target_leaf") or r.get("instruction", "")
            if tl and str(tl) not in all_targets:
                all_targets.append(str(tl))
    if len(summaries) == 1 and all_targets:
        s0 = summaries[0][1]
        by_t: Dict[str, List[bool]] = {}
        for r in s0.get("results", []) or []:
            key = str(r.get("target_leaf") or r.get("instruction", "unknown"))
            by_t.setdefault(key, []).append(bool(r.get("success", False)))
        if len(by_t) > 1:
            fig2, ax2 = plt.subplots(figsize=(max(5, 0.5 * len(by_t)), 4))
            tlabels = sorted(by_t.keys())
            tr = []
            err_lo = []
            err_hi = []
            for k in tlabels:
                succ = sum(1 for v in by_t[k] if v)
                n = len(by_t[k])
                p = succ / max(1, n)
                lo, hi = wilson_ci(succ, n)
                tr.append(p)
                err_lo.append(p - lo)
                err_hi.append(hi - p)
            ax2.bar(range(len(tlabels)), tr, yerr=(err_lo, err_hi), capsize=4, color="#17becf", edgecolor="black", linewidth=0.5)
            ax2.set_xticks(range(len(tlabels)))
            ax2.set_xticklabels(tlabels, rotation=20, ha="right")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Success rate")
            ax2.set_title("Success by target (single eval run)")
            ax2.grid(True, axis="y", alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(out_dir / f"eval_success_by_target.{fmt}", dpi=150)
            plt.close(fig2)

    _plot_success_vs_k_mean_frontier(summaries, out_dir, fmt)


def _plot_success_vs_k_mean_frontier(summaries: List[Tuple[str, Dict[str, Any]]], out_dir: Path, fmt: str) -> None:
    """Success vs mean effective K (gated TTC / τ sweeps). Requires ``ttc_aggregate.k_mean`` in JSON."""

    import matplotlib.pyplot as plt

    pts: List[Tuple[float, float, float, float, str]] = []
    for label, s in summaries:
        agg = s.get("ttc_aggregate") or {}
        k_bar = agg.get("k_mean")
        if k_bar is None:
            continue
        n_ep = int(s.get("num_episodes", 0))
        ns = int(s.get("num_success", 0))
        rate = float(s.get("success_rate", ns / max(1, n_ep)))
        lo, hi = wilson_ci(ns, n_ep)
        pts.append((float(k_bar), rate, lo, hi, label))
    if len(pts) < 2:
        return
    pts.sort(key=lambda x: x[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    kx = [p[0] for p in pts]
    ry = [p[1] for p in pts]
    yerr_lo = [p[1] - p[2] for p in pts]
    yerr_hi = [p[3] - p[1] for p in pts]
    ax.errorbar(kx, ry, yerr=(yerr_lo, yerr_hi), fmt="o-", capsize=4, color="#8c564b", ecolor="#333333")
    for p in pts:
        ax.annotate(p[4], (p[0], p[1]), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.85)
    ax.set_xlabel(r"Mean effective $K$ per policy query ($\bar K$)")
    ax.set_ylabel("Success rate")
    ax.set_title("Success vs. mean effective K (95% Wilson CI)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"eval_success_vs_k_mean.{fmt}", dpi=150)
    plt.close(fig)


def _export_metrics_csv(train_logs: List[Dict[str, Any]], epoch_rows: List[Dict[str, Any]], out_csv: Path) -> None:
    lines = ["kind,step,epoch,train_loss,val_loss,act_loss,fa_loss,lr,wall_time_s"]
    for x in train_logs:
        lines.append(
            "train_log,{step},{epoch},{loss},,{act},{fa},{lr},{wt}".format(
                step=x.get("step", ""),
                epoch=x.get("epoch", ""),
                loss=x.get("loss", ""),
                act=x.get("act_loss", ""),
                fa=x.get("fa_loss", ""),
                lr=x.get("lr", ""),
                wt=x.get("wall_time_s", ""),
            )
        )
    for x in epoch_rows:
        vl = x.get("val_loss", "")
        lines.append(
            "epoch_end,{step},{epoch},{tl},{vl},,,,{wt}".format(
                step=x.get("step", ""),
                epoch=x.get("epoch", ""),
                tl=x.get("train_loss", ""),
                vl=vl if vl is not None else "",
                wt=x.get("wall_time_s", ""),
            )
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n")


def plot_run_dir(
    run_dir: Path,
    eval_json_paths: Optional[List[Path]] = None,
    fig_format: str = "pdf",
) -> None:
    """Write figures under ``run_dir/figures``. Shared by CLI and ``train.post_plot``."""

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[plot_metrics] matplotlib is required: pip install matplotlib")
        return

    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    metrics_path = run_dir / "metrics.jsonl"
    train_logs, epoch_rows = _load_metrics_jsonl(metrics_path)
    if not train_logs and not epoch_rows:
        print(f"[plot_metrics] WARNING: no metrics at {metrics_path}.")

    _plot_learning_curves(train_logs, epoch_rows, fig_dir, fig_format)
    _export_metrics_csv(train_logs, epoch_rows, fig_dir / "metrics_training.csv")

    eval_specs: List[Tuple[str, Dict[str, Any]]] = []
    for p in eval_json_paths or []:
        p = Path(p)
        if p.exists():
            eval_specs.append((p.stem, _load_eval_summary(p)))
        else:
            print(f"[plot_metrics] WARNING: missing eval file {p}")

    if eval_specs:
        _plot_eval_comparison(eval_specs, fig_dir, fig_format)

    print(f"[plot_metrics] wrote figures and metrics_training.csv under {fig_dir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot TinyVLA training / eval metrics")
    parser.add_argument("--run-dir", type=str, required=True, help="Checkpoint dir containing metrics.jsonl")
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: <run-dir>/figures)",
    )
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"], help="Figure format")
    parser.add_argument(
        "--eval-json",
        action="append",
        default=[],
        help="Eval summary JSON (repeatable). Label defaults to filename stem.",
    )
    parser.add_argument(
        "--eval-label",
        action="append",
        default=[],
        help="Label for each --eval-json (same order)",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[plot_metrics] matplotlib is required: pip install matplotlib")
        return 2

    run_dir = Path(args.run_dir)
    labels = list(args.eval_label)

    fig_dir = Path(args.figures_dir) if args.figures_dir else run_dir / "figures"
    metrics_path = run_dir / "metrics.jsonl"
    train_logs, epoch_rows = _load_metrics_jsonl(metrics_path)
    if not train_logs and not epoch_rows:
        print(f"[plot_metrics] WARNING: no metrics at {metrics_path}. Train with vla_lab.train first.")

    _plot_learning_curves(train_logs, epoch_rows, fig_dir, args.format)
    _export_metrics_csv(train_logs, epoch_rows, fig_dir / "metrics_training.csv")

    eval_specs: List[Tuple[str, Dict[str, Any]]] = []
    for i, p in enumerate(args.eval_json):
        path = Path(p)
        label = labels[i] if i < len(labels) else path.stem
        if path.exists():
            eval_specs.append((label, _load_eval_summary(path)))
        else:
            print(f"[plot_metrics] WARNING: missing eval file {path}")

    if eval_specs:
        _plot_eval_comparison(eval_specs, fig_dir, args.format)

    print(f"[plot_metrics] wrote figures and metrics_training.csv under {fig_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

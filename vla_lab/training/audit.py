"""Audit a training run, or a whole sweep, from what it wrote to disk.

Every claim about a model in this project should be checkable without re-running it. That means
each run has to be reconstructable from its own directory, and that its *provenance* -- not only
its metrics -- has to be visible. The three provenance facts that have each already caused a
wrong conclusion here are checked explicitly and shown at the top of every card:

``image source``
    ``schematic`` runs validate the pipeline. They must never be read as perception results.
``adaptation applied``
    "LoRA" and "frozen backbone" are different experiments. When ``peft`` is missing the trainer
    silently falls back, and the manifest is the only place that difference survives.
``prompt truncation``
    A verbalised context that overflows the backbone's instruction budget is cut from the LEFT
    on SmolVLM, removing exactly the informative half. That produced a run that read as
    "verbalised context does not work" and was a tokenizer budget, not an architecture.

    python -m vla_lab.training.audit vla_lab/results/models
    python -m vla_lab.training.audit vla_lab/results/models --figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def read_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """One run directory -> manifest + metric history, or ``None`` if it never finished."""
    run_dir = Path(run_dir)
    man_p, met_p = run_dir / "manifest.json", run_dir / "metrics.jsonl"
    if not man_p.exists():
        return None
    manifest = json.loads(man_p.read_text())
    history = []
    if met_p.exists():
        history = [json.loads(l) for l in met_p.read_text().splitlines() if l.strip()]
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None
    return {"dir": str(run_dir), "manifest": manifest, "history": history, "summary": summary,
            "finished": summary is not None}


def find_runs(root: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(Path(root).rglob("manifest.json")):
        r = read_run(p.parent)
        if r:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def audit_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Provenance and sanity checks. ``flags`` are things a reader must be told."""
    m, hist = run["manifest"], run["history"]
    data, model = m.get("data", {}), m.get("model", {})
    flags: List[str] = []
    checks: Dict[str, Any] = {}

    src = data.get("image_source")
    checks["image_source"] = src
    if src != "isaac":
        flags.append(f"images are {src}: pipeline validation only, not a perception result")

    applied = (model.get("adapt") or {}).get("applied")
    requested = (model.get("adapt") or {}).get("requested")
    checks["adapt"] = f"{requested} -> {applied}"
    if requested != applied:
        flags.append(f"adaptation fell back from {requested} to {applied}: "
                     f"{(model.get('adapt') or {}).get('reason', 'no reason recorded')}")

    pa = m.get("prompt_audit") or {}
    checks["prompt_audit"] = pa
    if pa.get("checked") and not pa.get("ok", True):
        flags.append(f"{pa['frac_truncated']*100:.0f}% of prompts truncated at "
                     f"{pa['budget']} tokens: the context was cut, not the instruction")

    tr, va = set(data.get("train_supervisors", [])), set(data.get("val_supervisors", []))
    checks["split"] = {"train": len(tr), "val": len(va), "disjoint": not (tr & va)}
    if tr & va:
        flags.append(f"train/val share {len(tr & va)} supervisors: the split leaks")
    if not va:
        flags.append("empty validation split")

    checks["finished"] = run["finished"]
    if not run["finished"]:
        flags.append("no summary.json: this run did not finish")

    if hist:
        last = hist[-1]
        checks["epochs_run"] = len(hist)
        checks["final"] = {k: last.get(k) for k in
                           ("train_loss", "val_loss", "acc_said", "debias_gain_brier",
                            "mean_abs_debias_gap", "debias_kappa_corr", "ask_rank_corr")}
        val = [h.get("val_loss") for h in hist if h.get("val_loss") is not None]
        if len(val) > 3 and val[-1] > min(val) * 1.15:
            flags.append(f"validation loss ended {100*(val[-1]/min(val)-1):.0f}% above its best: overfitting")
        gains = [h.get("debias_gain_brier") for h in hist if h.get("debias_gain_brier") is not None]
        if gains and max(gains) < 0.01:
            flags.append("de-biasing gain never exceeded 0.01: this model did not learn to de-bias")
        if last.get("acc_said", 1.0) < 0.8:
            flags.append(f"grounding accuracy only {last.get('acc_said', float('nan')):.2f}: "
                         "the model is not reading the instruction, so any de-biasing number is moot")
    pre = m.get("preflight") or {}
    checks["peak_vram_gb"] = pre.get("peak_gb")

    return {"dir": run["dir"], "model": m.get("card", {}).get("display", model.get("model_key")),
            "context": m.get("context_mode"), "context_style": m.get("context_style"),
            "checks": checks, "flags": flags}


def render(audits: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for a in audits:
        c = a["checks"]
        f = c.get("final") or {}
        lines.append("=" * 96)
        lines.append(f"{a['model']}  |  context={a['context']} ({a['context_style']})  |  {a['dir']}")
        lines.append(f"  provenance : images={c.get('image_source')}  adapt={c.get('adapt')}  "
                     f"split={c.get('split', {}).get('train')}/{c.get('split', {}).get('val')} "
                     f"(disjoint={c.get('split', {}).get('disjoint')})  peak={c.get('peak_vram_gb')}GB")
        pa = c.get("prompt_audit") or {}
        if pa.get("checked"):
            lines.append(f"  prompts    : max {pa.get('max_tokens')} tok vs budget {pa.get('budget')}  "
                         f"truncated {pa.get('frac_truncated', 0)*100:.0f}%")
        if f:
            lines.append(f"  final      : train {f.get('train_loss'):.4f}  val {f.get('val_loss'):.4f}  "
                         f"acc_said {f.get('acc_said'):.3f}  gain(Brier) {f.get('debias_gain_brier'):+.4f}  "
                         f"gap {f.get('mean_abs_debias_gap'):.2f}  gap~k {f.get('debias_kappa_corr'):+.3f}  "
                         f"ask rho {f.get('ask_rank_corr'):+.3f}")
        if a["flags"]:
            for fl in a["flags"]:
                lines.append(f"  [!] {fl}")
        else:
            lines.append("  [ok] no provenance flags")
    lines.append("=" * 96)
    n_flag = sum(1 for a in audits if a["flags"])
    lines.append(f"{len(audits)} run(s); {n_flag} carry at least one flag a reader must be told about.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_curves(runs: Sequence[Dict[str, Any]], out: Path) -> Optional[Path]:
    """Learning curves for every run, on the four axes that matter."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 160, "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

    panels = [
        ("val_loss", "validation loss", False),
        ("debias_gain_brier", "de-biasing gain (Brier)", True),
        ("debias_kappa_corr", r"gap $\sim\kappa$ (context is used)", True),
        ("acc_said", "grounding accuracy", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(runs), 1)))
    for (key, label, zero_line), ax in zip(panels, axes):
        for r, col in zip(runs, colors):
            h = r["history"]
            ys = [x.get(key) for x in h if x.get(key) is not None]
            if not ys:
                continue
            name = f"{r['manifest'].get('context_mode')}"
            ax.plot(range(len(ys)), ys, color=col, lw=1.3, label=name)
        if zero_line:
            ax.axhline(0.0, color="#444", lw=0.8, ls="--")
        ax.set_xlabel("epoch")
        ax.set_title(label, loc="left", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, title="context", title_fontsize=6)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_training_curves.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def figure_context_effect(runs: Sequence[Dict[str, Any]], out: Path) -> Optional[Path]:
    """The architectural claim in one panel: does the model's correction track the residue?"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 160, "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    rows = []
    for r in runs:
        h = r["history"]
        if not h:
            continue
        gains = [x.get("debias_gain_brier") for x in h if x.get("debias_gain_brier") is not None]
        corrs = [x.get("debias_kappa_corr") for x in h if x.get("debias_kappa_corr") is not None]
        if not gains:
            continue
        rows.append({
            "label": f"{r['manifest'].get('card', {}).get('display', '?').split('-')[0]}\n{r['manifest'].get('context_mode')}",
            "mode": r["manifest"].get("context_mode"),
            "gain": max(gains),
            "corr": float(np.mean(corrs[-3:])) if corrs else float("nan"),
        })
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    palette = {"none": "#c0392b", "text": "#8e44ad", "token": "#2c6fbb", "film": "#16a085"}
    for r in rows:
        ax.scatter(r["corr"], r["gain"], s=70, color=palette.get(r["mode"], "#7f8c8d"),
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(r["label"], (r["corr"], r["gain"]), fontsize=6.5, xytext=(5, 3),
                    textcoords="offset points")
    ax.axvline(0.0, color="#444", lw=0.9, ls="--")
    ax.set_xlabel(r"rank correlation of the de-biasing gap with the residue $\kappa$")
    ax.set_ylabel("best de-biasing gain (Brier)")
    ax.set_title("Does the model modulate its correction, or apply a constant one?",
                 loc="left", fontsize=8.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_context_effect.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a run directory or a directory of runs")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    runs = find_runs(args.root)
    if not runs:
        print(f"no runs under {args.root}")
        return 1
    audits = [audit_run(r) for r in runs]
    text = render(audits)
    print(text)
    out = args.out or (args.root / "audit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.txt").write_text(text + "\n")
    (args.json or (out / "audit.json")).write_text(json.dumps(audits, indent=2, default=str) + "\n")
    if args.figures:
        for fn in (figure_curves, figure_context_effect):
            p = fn(runs, out)
            if p:
                print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

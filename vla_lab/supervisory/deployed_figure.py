"""Picture of the closed-loop reversal: offline skill against deployed behaviour.

The result this draws is the section's whole point and is hard to see in a table, because it is a
statement about *two numbers failing to line up*. Offline, the ranking of the
context-injection mechanisms is one thing; in the loop it is another; and the variable that
explains the difference is neither of them but a third column, how often the model declines to
answer inside the region the estimand is determined in.

    python -m vla_lab.supervisory.deployed_figure

Left panel: crossover-weighted MAE by grounding channel, against the lexical reference. Right
panel: each context mode's offline residue-tracking on the x-axis against its closed-loop error on
the y-axis, sized by in-band abstention. If offline skill predicted deployment the points would
fall on a downward line; the figure exists because they do not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PAPER = Path("vla_lab/paper/hri2027_carryover_vla")
COLORS = {"none": "#6d8299", "text": "#8e44ad", "token": "#2e7d32", "film": "#c62828"}


def _rows(deployed: Dict[str, Any], condition: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, e in deployed.items():
        cell = e.get("conditions", {}).get(condition, {})
        g = e.get("grounder", {})
        ctx = key.split("__")[-1] if "__" in key else None
        out.append({
            "key": key,
            "read": ("lexical" if key == "lexical" else ("said" if key.startswith("policy_said") else "unprompted")),
            "context": ctx,
            "mae": (cell.get("mae_crossover") or {}).get("mean"),
            "align": (cell.get("alignment") or {}).get("mean"),
            "ungrounded": (cell.get("n_ungrounded") or {}).get("mean"),
            "abstain": g.get("abstain_rate"),
            "abstain_band": g.get("abstain_rate_band"),
        })
    return out


def figure(runs: Sequence[Tuple[str, Dict[str, Any], Sequence[Dict[str, Any]]]], out: Path,
           *, condition: str = "carryover_aware") -> Dict[str, Any]:
    """``runs`` is [(backbone label, deployed summary, model table rows), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.35, 1.0]})

    order = ["none", "text", "token", "film"]
    bars: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []
    for label, deployed, models in runs:
        rows = _rows(deployed, condition)
        ref = next((r["mae"] for r in rows if r["read"] == "lexical"), None)
        offline = {str(m.get("context")): m for m in models if m.get("debias_kappa_corr") is not None}
        for r in sorted((r for r in rows if r["read"] == "unprompted"),
                        key=lambda r: order.index(r["context"]) if r["context"] in order else 9):
            rel = (r["mae"] - ref) if (ref is not None and r["mae"] is not None) else None
            bars.append({**r, "backbone": label, "rel": rel})
            m = offline.get(r["context"])
            if m is not None and rel is not None:
                points.append({**r, "backbone": label, "rel": rel,
                               "gain": float(m["debias_gain_brier"])})

    xs = np.arange(len(bars))
    ax.bar(xs, [b["rel"] for b in bars], color=[COLORS.get(b["context"], "#999") for b in bars])
    for patch, b in zip(ax.patches, bars):
        patch.set_alpha(0.55 if b["backbone"].startswith("Tiny") else 1.0)
    ax.axhline(0.0, color="#333", lw=1.0)
    short = {"TinyVLA-2M": "2M", "SmolVLA-450M": "450M"}
    ax.set_xticks(xs, [f"{b['context']}\n{short.get(b['backbone'], b['backbone'])}" for b in bars],
                  fontsize=7)
    ax.set_ylabel("deployed MAE $-$ lexical reference")
    ax.set_title("Below zero beats the keyword grounder it replaces", loc="left", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.4, axis="y")

    marks = {"Tiny": "o", "Smol": "s"}

    def short_b(name: str) -> str:
        return {"TinyVLA-2M": "2M", "SmolVLA-450M": "450M"}.get(name, name)

    for p in points:
        bx.scatter([p["gain"]], [p["rel"]],
                   s=1500.0 * float(p.get("abstain_band") or 0.0) + 45.0,
                   c=COLORS.get(p["context"], "#999"),
                   marker=marks.get(p["backbone"][:4], "o"),
                   alpha=0.85, edgecolors="white", linewidths=1.2, zorder=3)
        bx.annotate(f"{p['context']} ({short_b(p['backbone'])})", (p["gain"], p["rel"]),
                    textcoords="offset points", xytext=(10, -3), fontsize=6.5)
    bx.axhline(0.0, color="#333", lw=1.0)
    bx.set_xlabel("offline de-biasing gain (Brier)")
    bx.set_ylabel("deployed MAE $-$ reference")
    bx.set_title("Offline skill barely orders deployment;\ncircles from-scratch, squares pretrained;\n"
                 "marker area is in-band abstention", loc="left", fontsize=8.0)
    bx.grid(alpha=0.25, lw=0.4)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"n_cells": len(bars), "condition": condition, "figure": str(out)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/deployed/fig_deployed.pdf"))
    ap.add_argument("--condition", default="carryover_aware")
    args = ap.parse_args(argv)

    spec = [("TinyVLA-2M", Path("vla_lab/results/deployed/summary.json"),
             Path("vla_lab/results/models_isaac/table.json")),
            ("SmolVLA-450M", Path("vla_lab/results/deployed_smolvla/summary.json"),
             Path("vla_lab/results/models_isaac_smolvla/table.json"))]
    runs = [(label, json.loads(dp.read_text()),
             json.loads(mp.read_text()) if mp.exists() else [])
            for label, dp, mp in spec if dp.exists()]
    if not runs:
        print("no deployed summaries found")
        return 1
    print(json.dumps(figure(runs, Path(args.out), condition=args.condition), indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

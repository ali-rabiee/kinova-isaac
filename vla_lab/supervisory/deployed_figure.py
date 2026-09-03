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


def policy_condition(deployed: Dict[str, Any], default: str = "carryover_aware") -> str:
    """The policy the checkpoints drove, from the summary's own metadata."""
    return str((deployed.get("_meta") or {}).get("policy") or default)


def _rows(deployed: Dict[str, Any], condition: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, e in deployed.items():
        if key.startswith("_"):
            continue
        cell = e.get("conditions", {}).get(condition, {})
        g = e.get("grounder", {})
        # Keys are ``policy_<read>@<model>__<context>[__s<seed>]``: take the context segment,
        # never the last one -- a seed suffix silently emptied the whole facts file once.
        parts = key.split("@")[-1].split("__")
        ctx = parts[1] if len(parts) > 1 else None
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


def deployed_facts(runs: Sequence[Tuple[str, Dict[str, Any], Sequence[Dict[str, Any]]]],
                   *, condition: str = "carryover_aware") -> Dict[str, Any]:
    """What the closed-loop cells support, stated without a correlation coefficient.

    The cell count is far below the minimum the analysis allows a coefficient on
    (:data:`vla_lab.stats_utils.MIN_N_FOR_CORRELATION`), so the guard withholds it and the facts
    reported are the two the evidence actually carries: whether the best offline cell on each
    backbone is also the best deployed one, and how in-band abstention lines up with failure.
    """
    from ..stats_utils import guarded_correlation

    cells: List[Dict[str, Any]] = []
    for label, deployed, models in runs:
        rows = _rows(deployed, policy_condition(deployed, condition))
        ref = next((r["mae"] for r in rows if r["read"] == "lexical"), None)
        offline = {str(m.get("context")): m for m in models if m.get("debias_gain_brier") is not None}
        for r in rows:
            if r["read"] != "unprompted" or r["mae"] is None or ref is None:
                continue
            m = offline.get(r["context"])
            cells.append({"backbone": label, "context": r["context"], "deployed_rel": r["mae"] - ref,
                          "deployed_mae": r["mae"], "reference_mae": ref,
                          "offline_gain": (float(m["debias_gain_brier"]) if m else None),
                          "offline_kappa_corr": (float(m["debias_kappa_corr"]) if m else None),
                          "abstain_band": float(r.get("abstain_band") or 0.0)})
    scored = [c for c in cells if c["offline_gain"] is not None]
    gain_vs_dep = guarded_correlation([c["offline_gain"] for c in scored], [c["deployed_rel"] for c in scored])
    abst_vs_dep = guarded_correlation([c["abstain_band"] for c in scored], [c["deployed_rel"] for c in scored])
    per_backbone: Dict[str, Any] = {}
    for label in {c["backbone"] for c in scored}:
        mine = [c for c in scored if c["backbone"] == label]
        best_off = max(mine, key=lambda c: c["offline_gain"])
        by_dep = sorted(mine, key=lambda c: c["deployed_rel"])
        per_backbone[label] = {
            "best_offline_context": best_off["context"],
            "best_offline_deployed_rank": 1 + by_dep.index(best_off),
            "n_contexts": len(mine),
            "best_deployed_context": by_dep[0]["context"],
            "worst_deployed_context": by_dep[-1]["context"],
            "worst_deployed_rel": by_dep[-1]["deployed_rel"],
        }
    worst = max(scored, key=lambda c: c["deployed_rel"]) if scored else None
    most_abst = max(scored, key=lambda c: c["abstain_band"]) if scored else None
    return {
        "n_cells": len(scored),
        "condition": condition,
        "cells": cells,
        "offline_gain_vs_deployed": gain_vs_dep,
        "abstention_vs_deployed": abst_vs_dep,
        "per_backbone": per_backbone,
        "worst_cell": worst,
        "most_abstaining_cell": most_abst,
        "worst_is_most_abstaining": bool(worst and most_abst and worst is most_abst),
        "cells_above_11pct_band_abstention": [f"{c['backbone']}/{c['context']}" for c in scored if c["abstain_band"] > 0.11],
        "cells_failing": [f"{c['backbone']}/{c['context']}" for c in scored
                          if c["deployed_rel"] > 0.005],
    }


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
        rows = _rows(deployed, policy_condition(deployed, condition))
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
    facts = deployed_facts(runs, condition=condition)
    facts["figure"] = str(out)
    out.with_name("facts.json").write_text(json.dumps(facts, indent=2, default=float) + "\n")
    return facts


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

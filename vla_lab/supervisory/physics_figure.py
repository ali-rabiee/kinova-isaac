"""Picture of the measured scene physics: what the two strategies actually do, and where they cross.

The task-value model is the least visible and most load-bearing object in the paper --- every
scene coordinate, every regret number and the whole notion of an ``ambiguous band'' are defined
in terms of it. A table of fitted parameters does not let a reader check it. Two panels do:

*Left* --- the measured success fraction of each scripted expert at each clearance gap, with
binomial intervals, against the fitted curves. This is the raw evidence, and it is where a
degenerate fit (a step, or two curves lying on top of each other) is immediately obvious.

*Right* --- the resulting task values and their difference, with the crossover and the ambiguous
band marked. If the two value curves do not cross inside the bracket, the study has no ambiguous
region and the figure says so rather than leaving it to be inferred.

    python -m vla_lab.supervisory.physics_figure vla_lab/results/physics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import STRATEGY_A, STRATEGY_B


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson interval -- honest at 0/n and n/n, where the normal interval collapses to a point."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def figure(rollouts: Sequence[Dict[str, Any]], physics, out: Path, *, lo: float = 0.0,
           hi: float = 0.20) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from collections import defaultdict

    tally: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
    for r in rollouts:
        key = (round(float(r["margin_m"]), 4), str(r["strategy"]))
        tally[key][0] += int(bool(r["success"]))
        tally[key][1] += 1
    margins = sorted({k[0] for k in tally})

    grid = np.linspace(lo, hi, 241)
    cross = float(physics.crossover_margin(lo, hi))
    width = float(physics.transition_width_m())
    degenerate = bool(physics.is_degenerate(lo, hi))

    style = {STRATEGY_A: ("#1565c0", "o", "CLEAR-FIRST (A)"), STRATEGY_B: ("#c62828", "s", "DIRECT (B)")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    for st, (col, mk, lab) in style.items():
        ks = np.array([tally[(m, st)][0] for m in margins], dtype=float)
        ns = np.array([tally[(m, st)][1] for m in margins], dtype=float)
        ps = np.divide(ks, np.maximum(ns, 1))
        los, his = zip(*[_wilson(int(k), int(n)) for k, n in zip(ks, ns)])
        xs = np.array(margins) * 100
        ax.errorbar(xs, ps, yerr=[ps - np.array(los), np.array(his) - ps], fmt=mk, ms=4.5,
                    color=col, capsize=2.5, lw=1.0, label=f"{lab}, measured")
        ax.plot(grid * 100, [physics.p_success(st, m) for m in grid], color=col, lw=1.6, alpha=0.85)
    ax.set_xlabel("clearance gap (cm)")
    ax.set_ylabel("P(success)")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(f"Measured, {len(rollouts)} rollouts", fontsize=9.5)

    for st, (col, _mk, lab) in style.items():
        bx.plot(grid * 100, [physics.value(st, m) for m in grid], color=col, lw=1.8, label=lab)
    gapv = np.array([physics.value_gap(m) for m in grid])
    bx.plot(grid * 100, gapv, color="#455a64", lw=1.1, ls="--", label="$V_A - V_B$")
    bx.axhline(0.0, color="#90a4ae", lw=0.7)
    if not degenerate:
        bx.axvline(cross * 100, color="#6a1b9a", lw=1.2)
        band = [physics.margin_for_coordinate(c, lo=lo, hi=hi) for c in (-1.2, 1.2)]
        bx.axvspan(min(band) * 100, max(band) * 100, color="#ce93d8", alpha=0.22, lw=0)
        bx.annotate(f"crossover {cross * 100:.1f} cm\nwidth {width * 100:.2f} cm",
                    xy=(cross * 100, 0.0), xytext=(6, 10), textcoords="offset points", fontsize=7.5)
    else:
        bx.text(0.5, 0.5, "no crossover inside the bracket:\nthe scene has no ambiguous region",
                transform=bx.transAxes, ha="center", va="center", fontsize=9, color="#b71c1c")
    bx.set_xlabel("clearance gap (cm)")
    bx.set_ylabel("task value")
    bx.legend(fontsize=7.5, loc="best", framealpha=0.92)
    bx.grid(alpha=0.25, lw=0.4)
    bx.set_title("Fitted value model and the ambiguous band", fontsize=9.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"crossover_margin_m": cross, "transition_width_m": width, "degenerate": degenerate,
            "n_rollouts": len(rollouts), "margins": margins, "figure": str(out)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, nargs="?", default=Path("vla_lab/results/physics"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from .apparatus.measure import ROLLOUTS_FILE, read_rollouts
    from .scenes import load_physics

    root = Path(args.root)
    rollouts = read_rollouts(root / ROLLOUTS_FILE)
    physics = load_physics(root / "physics.json")
    info = figure(rollouts, physics, Path(args.out) if args.out else root / "fig_physics.pdf")
    print(json.dumps(info, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

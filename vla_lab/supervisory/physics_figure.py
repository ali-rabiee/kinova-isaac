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
           hi: float = 0.20, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Two panels. Left: the data, the primary fit, the isotonic step, and the legacy fit it
    replaced. Right: the value curves, the crossover with its bootstrap interval, the band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from collections import defaultdict

    report = report or {}
    tally: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
    for r in rollouts:
        key = (round(float(r["margin_m"]), 4), str(r["strategy"]))
        tally[key][0] += int(bool(r["success"]))
        tally[key][1] += 1
    margins = sorted({k[0] for k in tally})
    x_hi = min(hi, max(margins) + 0.01) if margins else hi

    grid = np.linspace(lo, x_hi, 401)
    cross = float(physics.crossover_margin(lo, hi))
    width = float(physics.transition_width_m())
    degenerate = bool(physics.is_degenerate(lo, hi))

    legacy = None
    if report.get("legacy_fit", {}).get("params"):
        from .scenes import ScenePhysics
        legacy = ScenePhysics.from_dict(report["legacy_fit"]["params"])
    iso = report.get("isotonic") or {}
    boot = report.get("bootstrap") or {}

    style = {STRATEGY_A: ("#1565c0", "o", "CLEAR-FIRST (A)"), STRATEGY_B: ("#c62828", "s", "DIRECT (B)")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    for st, (col, mk, lab) in style.items():
        ks = np.array([tally[(m, st)][0] for m in margins], dtype=float)
        ns = np.array([tally[(m, st)][1] for m in margins], dtype=float)
        ps = np.divide(ks, np.maximum(ns, 1))
        los, his = zip(*[_wilson(int(k), int(n)) for k, n in zip(ks, ns)])
        xs = np.array(margins) * 100
        ax.errorbar(xs, ps, yerr=[ps - np.array(los), np.array(his) - ps], fmt=mk, ms=4.5,
                    color=col, capsize=2.5, lw=1.0, label=f"{lab}, measured")
        ax.plot(grid * 100, [physics.p_success(st, m) for m in grid], color=col, lw=1.7, alpha=0.9,
                label=f"{lab}, lapse fit" if st == STRATEGY_A else None)
        curve = iso.get("curve_a" if st == STRATEGY_A else "curve_b")
        if curve:
            ax.step(np.array(curve["margins_m"]) * 100, curve["p"], where="mid", color=col, lw=0.9,
                    alpha=0.55, ls=":", label="isotonic" if st == STRATEGY_A else None)
        if legacy is not None:
            ax.plot(grid * 100, [legacy.p_success(st, m) for m in grid], color=col, lw=1.0, ls="--",
                    alpha=0.45, label="legacy fit (per-metre prior)" if st == STRATEGY_A else None)
    ax.set_xlabel("clearance gap (cm)")
    ax.set_ylabel("P(success)")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(lo * 100 - 0.4, x_hi * 100)
    ax.legend(fontsize=6.6, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(f"Measured, {len(rollouts)} rollouts", fontsize=9.5)

    for st, (col, _mk, lab) in style.items():
        bx.plot(grid * 100, [physics.value(st, m) for m in grid], color=col, lw=1.8, label=lab)
    gapv = np.array([physics.value_gap(m) for m in grid])
    bx.plot(grid * 100, gapv, color="#455a64", lw=1.1, ls="--", label="$V_A - V_B$")
    bx.axhline(0.0, color="#90a4ae", lw=0.7)
    if not degenerate:
        ci = boot.get("crossover_margin_m") or {}
        if "p2.5" in ci:
            bx.axvspan(ci["p2.5"] * 100, ci["p97.5"] * 100, color="#6a1b9a", alpha=0.12, lw=0,
                       label="95% bootstrap CI on $m^*$")
        bx.axvline(cross * 100, color="#6a1b9a", lw=1.2)
        band = [physics.margin_for_coordinate(c, lo=lo, hi=hi) for c in (-1.2, 1.2)]
        bx.axvspan(min(band) * 100, max(band) * 100, color="#ce93d8", alpha=0.22, lw=0, label="ambiguous band")
        wci = boot.get("transition_width_m") or {}
        wtxt = (f"width {width * 100:.2f} cm [{wci['p2.5'] * 100:.2f}, {wci['p97.5'] * 100:.2f}]"
                if "p2.5" in wci else f"width {width * 100:.2f} cm")
        ctxt = (f"crossover {cross * 100:.1f} cm [{ci['p2.5'] * 100:.1f}, {ci['p97.5'] * 100:.1f}]"
                if "p2.5" in ci else f"crossover {cross * 100:.1f} cm")
        bx.annotate(f"{ctxt}\n{wtxt}", xy=(cross * 100, 0.0), xytext=(0.98, 0.35),
                    textcoords="axes fraction", ha="right", fontsize=7.2)
    else:
        bx.text(0.5, 0.5, "no crossover inside the bracket:\nthe scene has no ambiguous region",
                transform=bx.transAxes, ha="center", va="center", fontsize=9, color="#b71c1c")
    bx.set_xlabel("clearance gap (cm)")
    bx.set_ylabel("task value")
    bx.set_xlim(lo * 100 - 0.4, x_hi * 100)
    bx.legend(fontsize=6.6, loc="lower right", framealpha=0.92)
    bx.grid(alpha=0.25, lw=0.4)
    bx.set_title("Fitted value model, crossover interval, ambiguous band", fontsize=9.5)

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
    rep_path = root / "physics_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    files = report.get("rollout_files") or [root / ROLLOUTS_FILE]
    rollouts: List[Dict[str, Any]] = []
    for f in files:
        rollouts.extend(read_rollouts(Path(f)))
    physics = load_physics(root / "physics.json")
    info = figure(rollouts, physics, Path(args.out) if args.out else root / "fig_physics.pdf", report=report)
    print(json.dumps(info, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

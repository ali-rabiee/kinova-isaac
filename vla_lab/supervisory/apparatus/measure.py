"""Measuring the scene physics: the margin sweep that turns priors into numbers.

Every quantity the Tier-1 study depends on -- how often each strategy works at each clearance,
how long each takes, and therefore where the value crossover sits and what the ambiguity
coordinate means -- is a **measurement**, and this is where it is made. Until this has run, the
whole pipeline is running on the defaults in
:class:`~vla_lab.supervisory.scenes.ScenePhysics`, ``ScenePhysics.source`` says ``"prior"``, and
every figure carries that label.

Two entry points:

``sweep``
    Run the scripted experts across a grid of clearance gaps in Isaac and write one row per
    rollout. This is the expensive part -- a few hundred episodes -- and it is run once.
``fit_from_rollouts``
    Read those rows, fit the success and duration curves, rebuild the scene grid on the fitted
    coordinate, and write the physics json the study then loads.

The sweep is deliberately **strategy-balanced and margin-stratified**: equal rollouts of each
strategy at each gap. An unbalanced sweep would fit one strategy's curve well and the other's
badly, and since the coordinate is a *difference* of the two, the error would land squarely on
the thing every downstream number is defined in terms of.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .. import STRATEGY_A, STRATEGY_B
from ..scenes import SOURCE_MEASURED, ScenePhysics, build_scene_grid, save_physics

ROLLOUTS_FILE = "rollouts.jsonl"
PHYSICS_FILE = "physics.json"


def default_margin_grid(lo: float = 0.0, hi: float = 0.16, n: int = 9) -> List[float]:
    return [float(x) for x in np.linspace(lo, hi, int(n))]


def read_rollouts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarise(rows: Sequence[Dict[str, Any]]) -> str:
    """Per-(strategy, gap) success and duration. Printed before any fit, because a fit to a
    sweep with an empty cell is worse than no fit at all -- it silently extrapolates."""
    by: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((str(r["strategy"]), round(float(r["margin_m"]), 4)), []).append(r)
    lines = [f"{'gap(cm)':>8}{'A n':>6}{'A succ':>8}{'A s':>7}{'B n':>6}{'B succ':>8}{'B s':>7}"]
    lines.append("-" * len(lines[0]))
    gaps = sorted({g for _, g in by})
    for g in gaps:
        cells = []
        for s in (STRATEGY_A, STRATEGY_B):
            rs = by.get((s, g), [])
            n = len(rs)
            sr = (sum(1 for r in rs if r.get("success")) / n) if n else float("nan")
            du = statistics.mean([float(r["duration_s"]) for r in rs if r.get("duration_s") is not None]) if n else float("nan")
            cells.append(f"{n:>6}{sr:>8.2f}{du:>7.1f}")
        lines.append(f"{g * 100:>8.1f}" + "".join(cells))
    empty = [(s, g) for g in gaps for s in (STRATEGY_A, STRATEGY_B) if not by.get((s, g))]
    if empty:
        lines.append(f"WARNING: {len(empty)} empty (strategy, gap) cells -- the fit would extrapolate over them.")
    return "\n".join(lines)


def fit_residuals(rows: Sequence[Dict[str, Any]], phys: ScenePhysics,
                  band: Tuple[float, float] = (0.048, 0.123)) -> Dict[str, Any]:
    """How far the fitted curves sit from the measurement, per cell and worst-case.

    A two-parameter logistic cannot follow a near-step, and this scene's success curves are
    close to one at the tight end. That is not a reason to hide the fit -- the study only uses
    it inside the ambiguous band -- but it is a reason to report *where* it holds rather than
    quoting a single R-squared. Reported per cell, and summarised separately inside and outside
    the band the estimand is defined over.
    """
    from collections import defaultdict

    tally: Dict[Tuple[float, str], List[Any]] = defaultdict(lambda: [0, 0, []])
    for r in rows:
        k = (round(float(r["margin_m"]), 4), str(r["strategy"]))
        tally[k][0] += int(bool(r["success"]))
        tally[k][1] += 1
        if r.get("duration_s") is not None:
            tally[k][2].append(float(r["duration_s"]))

    cells: List[Dict[str, Any]] = []
    for (m, st), (k, n, durs) in sorted(tally.items()):
        obs_p, pred_p = k / max(n, 1), phys.p_success(st, m)
        obs_d = (sum(durs) / len(durs)) if durs else float("nan")
        cells.append({
            "margin_m": m, "strategy": st, "n": n,
            "p_observed": obs_p, "p_fitted": pred_p, "dp": obs_p - pred_p,
            "duration_observed_s": obs_d, "duration_fitted_s": phys.duration_s(st, m),
            "dd_s": obs_d - phys.duration_s(st, m),
            "in_band": bool(band[0] <= m <= band[1]),
        })

    def worst(pred) -> float:
        vals = [abs(c["dp"]) for c in cells if pred(c)]
        return max(vals) if vals else float("nan")

    return {
        "cells": cells,
        "worst_abs_dp": worst(lambda c: True),
        "worst_abs_dp_in_band": worst(lambda c: c["in_band"]),
        "worst_abs_dd_s": max((abs(c["dd_s"]) for c in cells if c["dd_s"] == c["dd_s"]), default=float("nan")),
        "band_m": list(band),
    }


def fit_from_rollouts(
    rows: Sequence[Dict[str, Any]],
    *,
    base: Optional[ScenePhysics] = None,
) -> Tuple[ScenePhysics, Dict[str, Any]]:
    """Fit :class:`ScenePhysics` and report what the fit implies."""
    phys = ScenePhysics.fit(rows, base=base)
    report = {
        "source": phys.source,
        "n_rollouts": int(phys.n_measured),
        "crossover_margin_m": phys.crossover_margin(),
        "transition_width_m": phys.transition_width_m(),
        "degenerate": phys.is_degenerate(),
        "p_success": {
            f"{int(m * 100)}cm": {"A": phys.p_success(STRATEGY_A, m), "B": phys.p_success(STRATEGY_B, m)}
            for m in (0.0, 0.04, 0.08, 0.12, 0.16)
        },
        "coordinate": {f"{int(m * 100)}cm": phys.coordinate(m) for m in (0.0, 0.04, 0.08, 0.12, 0.16)},
        "fit_residuals": fit_residuals(rows, phys),
    }
    return phys, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit ScenePhysics from an existing rollouts.jsonl")
    f.add_argument("rollouts", type=Path)
    f.add_argument("--out", type=Path, default=Path("vla_lab/results/physics/physics.json"))
    f.add_argument("--quiet", action="store_true")

    g = sub.add_parser("grid", help="print the margin grid a sweep should cover")
    g.add_argument("--n", type=int, default=9)
    g.add_argument("--lo", type=float, default=0.0)
    g.add_argument("--hi", type=float, default=0.16)

    args = ap.parse_args(argv)

    if args.cmd == "grid":
        print(" ".join(f"{m:.4f}" for m in default_margin_grid(args.lo, args.hi, args.n)))
        return 0

    rows = read_rollouts(args.rollouts)
    if not rows:
        print(f"[FAIL] no rollouts in {args.rollouts}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(summarise(rows))
    phys, report = fit_from_rollouts(rows)
    save_physics(phys, args.out)
    Path(args.out).with_name("physics_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    if not args.quiet:
        print()
        print(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.out}")
        if phys.is_degenerate():
            print("[WARN] no value crossover in the usable margin range: no scene is ambiguous, "
                  "so the study as designed has nothing to measure. Re-tune the task before collecting.",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

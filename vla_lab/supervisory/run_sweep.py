"""The Isaac margin sweep: run the scripted experts across clearance gaps and fit the physics.

This is the one expensive, one-off measurement the whole study rests on. Until it has run,
``ScenePhysics.source == "prior"`` and every figure says so.

    ./vla_lab/scripts/sup_sweep.sh --headless --repeats 12

Writes ``rollouts.jsonl`` (one row per rollout) and, with ``--fit``, the fitted
``physics.json`` plus a report. The sweep is **strategy-balanced and margin-stratified** --
equal rollouts of each strategy at each gap -- because the ambiguity coordinate is a
*difference* of the two success curves, so an unbalanced sweep puts its error exactly where
every downstream number is defined.

Isaac is launched here rather than imported at module scope, so ``--dry-run`` can print the
plan and validate the geometry on a machine with no simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import STRATEGY_A, STRATEGY_B
from .apparatus.measure import ROLLOUTS_FILE, default_margin_grid, fit_from_rollouts, summarise
from .scenes import SceneSpec, build_scene_grid, save_physics


def plan(margins: Sequence[float], repeats: int) -> List[Dict[str, Any]]:
    """Interleaved by strategy and gap, so a sweep killed halfway is still balanced."""
    rows: List[Dict[str, Any]] = []
    for rep in range(int(repeats)):
        for m in margins:
            for strategy in (STRATEGY_A, STRATEGY_B):
                rows.append({"margin_m": float(m), "strategy": strategy, "rep": int(rep)})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/physics"))
    ap.add_argument("--repeats", type=int, default=12, help="rollouts per (strategy, gap)")
    ap.add_argument("--margins", type=float, nargs="+", default=None)
    ap.add_argument("--n-margins", type=int, default=9)
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=0.16)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", action="store_true", help="fit ScenePhysics when the sweep finishes")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the geometry; no simulator")
    args = ap.parse_args(argv)

    margins = list(args.margins) if args.margins else default_margin_grid(args.lo, args.hi, args.n_margins)
    rows = plan(margins, args.repeats)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        from environments.supervisory_fetch import layout_for_margin
        from environments.supervisory_fetch.experts import approach_clearance_m, waypoints_for
        import random as _random

        print(f"{len(rows)} rollouts = {len(margins)} gaps x 2 strategies x {args.repeats} repeats")
        print(f"{'gap(cm)':>8}{'finger clearance(cm)':>22}{'A waypoints':>13}{'B waypoints':>13}")
        for m in margins:
            L = layout_for_margin(m, rng=_random.Random(0))
            print(f"{m * 100:>8.1f}{approach_clearance_m(L) * 100:>22.1f}"
                  f"{len(waypoints_for(STRATEGY_A, L)):>13}{len(waypoints_for(STRATEGY_B, L)):>13}")
        return 0

    # --- launch Isaac -----------------------------------------------------
    from isaaclab.app import AppLauncher

    launcher_args = argparse.Namespace(headless=bool(args.headless), enable_cameras=True, device=args.device)
    app = AppLauncher(launcher_args)
    _sim_app = app.app

    from .apparatus.isaac import IsaacApparatus, IsaacApparatusConfig

    grid = build_scene_grid()
    appa = IsaacApparatus(grid, cfg=IsaacApparatusConfig(headless=bool(args.headless), device=args.device),
                          seed=int(args.seed))
    path = out / ROLLOUTS_FILE
    t0 = time.time()
    n_ok = 0
    with path.open("w") as fh:
        for i, row in enumerate(rows):
            scene = SceneSpec(scene_id=1000 + i, axis=grid.axis, margin_m=float(row["margin_m"]),
                              clutter=2, c=float(grid.physics.coordinate(row["margin_m"])))
            appa.reset_scene(scene)
            outc = appa.execute(scene, row["strategy"])
            rec = {**row, "success": bool(outc.success), "duration_s": float(outc.duration_s), **outc.notes}
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            n_ok += int(outc.success)
            print(f"  [{i + 1}/{len(rows)}] gap={row['margin_m'] * 100:.1f}cm {row['strategy']} "
                  f"-> {'ok' if outc.success else 'fail'}  ({time.time() - t0:.0f}s)", file=sys.stderr)
    appa.close()

    rollouts = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(summarise(rollouts))
    if args.fit:
        phys, report = fit_from_rollouts(rollouts)
        save_physics(phys, out / "physics.json")
        # The figure is written here rather than by a separate command because the value model
        # is the paper's least visible and most load-bearing object: a degenerate fit -- a step,
        # or two curves lying on top of each other -- is obvious in the picture and invisible in
        # the parameters.
        try:
            from .physics_figure import figure as _physics_figure

            report["figure"] = _physics_figure(rollouts, phys, out / "fig_physics.pdf")
        except Exception as exc:                                    # matplotlib is optional
            report["figure_error"] = f"{type(exc).__name__}: {exc}"
        (out / "physics_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
        print(json.dumps(report, indent=2, default=float))
    print(f"\n{len(rollouts)} rollouts ({n_ok} successful) in {time.time() - t0:.0f}s -> {path}")
    _exit_now(0)
    return 0



def _exit_now(code: int = 0) -> None:
    """Leave the process immediately once every output is on disk.

    Isaac Sim's shutdown reliably hangs here after a long headless run -- the last rollout is
    logged, the fit is written, and the process then sits for minutes in teardown holding the
    GPU. Every output this command produces is flushed before this is called, so there is
    nothing left to lose by not unwinding; what there is to gain is the next stage of the
    pipeline starting when the work actually finished.
    """
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))

if __name__ == "__main__":
    raise SystemExit(main())

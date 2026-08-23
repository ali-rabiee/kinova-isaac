"""Measure where this arm can actually put its gripper, and choose the scene from that.

The margin sweep kept timing out on its first descent, and every explanation we reached for was
a controller-tuning story. The trace said something simpler: the arm was being asked to place a
downward-pointing gripper 29 cm from its own base and 22 cm above the table, and it never got
below 32 cm. That is not a tuning failure. It is a scene whose objects sit inside the arm's
inner workspace, where a 6-DOF arm has to fold through itself to arrive pointing down.

So this measures the envelope instead of arguing about it. For a grid of targets in the plane
the objects live in, at the heights the grasp actually needs, it commands the end effector there
through the same controller the experts use and records whether it arrived. The output is a
reachability map: a picture of where a cube can be placed such that the scripted expert has any
chance at all, and the basis on which ``target_xy`` and the blocker bearing are set.

    ./vla_lab/scripts/sup_reach.sh --headless

Writes ``reach.jsonl``, ``reach.json`` (the summary and the recommendation) and
``fig_reach.pdf``. Cheap next to the sweep -- a few hundred short moves, no objects, no physics
settling -- and it is the thing that should have been run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REACH_FILE = "reach.jsonl"
SUMMARY_FILE = "reach.json"


def grid_points(x_lo: float, x_hi: float, nx: int, y_lo: float, y_hi: float, ny: int,
                heights: Sequence[float]) -> List[Dict[str, float]]:
    """Row-major over (z, y, x): all of one height before moving to the next.

    Ordering matters for a probe that may be interrupted: a partial run should still cover a
    whole height, because a half-covered height cannot be read as a map.
    """
    pts: List[Dict[str, float]] = []
    for z in heights:
        for y in np.linspace(y_lo, y_hi, int(ny)):
            for x in np.linspace(x_lo, x_hi, int(nx)):
                pts.append({"x": float(x), "y": float(y), "z": float(z)})
    return pts


#: A probe point is only a measurement if the arm actually started from the home pose. A hard
#: miss at the workspace edge can leave it far enough out that the re-home itself fails, and the
#: next point then measures the recovery rather than the reach.
HOME_TOL_M = 0.05


def valid_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if float(r.get("home_err_m", 0.0)) <= HOME_TOL_M]


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-height reach fractions, and the best object placement the map supports.

    Rows whose re-home failed are dropped and counted, not silently averaged in: a contaminated
    probe reads as "out of reach" and would shrink the reported envelope for the wrong reason.
    """
    n_all = len(rows)
    rows = valid_rows(rows)
    heights = sorted({round(float(r["z"]), 4) for r in rows})
    per_h: Dict[str, Any] = {}
    for z in heights:
        sub = [r for r in rows if abs(float(r["z"]) - z) < 1e-6]
        ok = [r for r in sub if r["reached"]]
        per_h[f"{z:.3f}"] = {
            "n": len(sub),
            "reached": len(ok),
            "fraction": len(ok) / max(len(sub), 1),
            "x_min_reached": min((r["x"] for r in ok), default=None),
            "x_max_reached": max((r["x"] for r in ok), default=None),
            "median_err_m": float(np.median([r["err_m"] for r in sub])) if sub else None,
        }

    # The recommendation. A scene needs *both* objects reachable at grasp height, so we look for
    # the x-interval that is reachable at the lowest height probed, and put the pair inside it.
    z_low = f"{heights[0]:.3f}" if heights else None
    lo = per_h.get(z_low, {}).get("x_min_reached") if z_low else None
    hi = per_h.get(z_low, {}).get("x_max_reached") if z_low else None
    rec: Dict[str, Any] = {"reachable_x_at_lowest_z": [lo, hi], "lowest_z_probed": heights[0] if heights else None}
    if lo is not None and hi is not None and hi > lo:
        # Objects go in the outer half of the reachable band: the inner edge is where the arm has
        # to fold, and a scene that sits on the edge of feasibility measures the arm, not the gap.
        span = hi - lo
        rec["suggested_target_x"] = float(lo + 0.80 * span)
        rec["suggested_blocker_x_at_14cm"] = float(lo + 0.80 * span - 0.19)
        rec["blocker_inside_band"] = bool(rec["suggested_blocker_x_at_14cm"] >= lo)
        rec["note"] = (
            "blocker fits on the near side" if rec["blocker_inside_band"] else
            "blocker does NOT fit on the near side at a 14 cm gap -- place it laterally "
            "(bearing +/- pi/2) instead, which the clearance definition allows"
        )
    return {"per_height": per_h, "recommendation": rec, "n": len(rows),
            "n_probed": int(n_all), "n_dropped_bad_home": int(n_all - len(rows)),
            "home_tol_m": HOME_TOL_M}


def _class_of(row: Dict[str, Any], slow_frac: float = 0.5) -> str:
    """``ok`` / ``marginal`` / ``miss`` for one probe point.

    *marginal* means the pose was reached but the arm either burned most of its budget getting
    there or could not get back to the home pose afterwards. Both are disqualifying for a scene
    that has to run hundreds of rollouts.
    """
    if not bool(row.get("reached")):
        return "miss"
    budget = float(row.get("budget") or 700)
    if float(row.get("home_err_m", 0.0)) > HOME_TOL_M or float(row.get("steps", 0)) > slow_frac * budget:
        return "marginal"
    return "ok"


def figure(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = sorted({round(float(r["z"]), 4) for r in rows})
    fig, axes = plt.subplots(1, max(len(heights), 1), figsize=(3.1 * max(len(heights), 1), 3.4), squeeze=False)
    for ax, z in zip(axes[0], heights):
        sub = [r for r in rows if abs(float(r["z"]) - z) < 1e-6]
        xs = np.array([r["x"] for r in sub])
        ys = np.array([r["y"] for r in sub])
        # Three classes, not two. "Reached" and "out of reach" hide the category that actually
        # bit this project: poses the arm gets to, but only by spending most of its step budget
        # and arriving in a configuration it cannot recover from. A scene placed there produces
        # timeouts that depend on the strategy rather than on the independent variable.
        cls = np.array([_class_of(r) for r in sub])
        ax.scatter(xs[cls == "miss"], ys[cls == "miss"], c="#c62828", marker="x", s=30,
                   linewidths=1.2, label="out of reach")
        ax.scatter(xs[cls == "marginal"], ys[cls == "marginal"], facecolors="none",
                   edgecolors="#ef6c00", marker="o", s=44, linewidths=1.4, label="marginal")
        ax.scatter(xs[cls == "ok"], ys[cls == "ok"], c="#2e7d32", marker="o", s=26, label="reached")
        ax.set_title(f"z = {z * 100:.0f} cm", fontsize=9)
        ax.set_xlabel("x (m, base frame)", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0][0].set_ylabel("y (m, base frame)", fontsize=8)
    axes[0][-1].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Measured reachability with the tool pointing down", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/reach"))
    ap.add_argument("--x-lo", type=float, default=0.22)
    ap.add_argument("--x-hi", type=float, default=0.72)
    ap.add_argument("--nx", type=int, default=11)
    ap.add_argument("--y-lo", type=float, default=-0.20)
    ap.add_argument("--y-hi", type=float, default=0.20)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.02, 0.12, 0.22])
    ap.add_argument("--budget", type=int, default=900, help="controller steps allowed per point")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keep-objects", action="store_true",
                    help="probe with the scene's cubes in place (measures occlusion, not reach)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-jsonl", type=Path, default=None,
                    help="re-summarise and re-plot an existing reach.jsonl; no simulator")
    args = ap.parse_args(argv)

    if args.from_jsonl:
        rows = [json.loads(l) for l in Path(args.from_jsonl).read_text().splitlines() if l.strip()]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        summary = summarise(rows)
        (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, default=float) + "\n")
        figure(rows, out / "fig_reach.pdf")
        print(json.dumps(summary, indent=2, default=float))
        return 0

    pts = grid_points(args.x_lo, args.x_hi, args.nx, args.y_lo, args.y_hi, args.ny, args.heights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"{len(pts)} probe points over {len(args.heights)} heights; "
              f"x in [{args.x_lo}, {args.x_hi}], y in [{args.y_lo}, {args.y_hi}]")
        return 0

    from isaaclab.app import AppLauncher

    app = AppLauncher(argparse.Namespace(headless=bool(args.headless), enable_cameras=False, device=args.device))
    _sim_app = app.app

    from environments.supervisory_fetch.experts import Waypoint
    from environments.supervisory_fetch.scene import OBJECT_NAMES, SupervisoryFetchScene

    from .apparatus.isaac import IsaacApparatusConfig
    from .scenes import SceneSpec, build_scene_grid

    grid = build_scene_grid()
    cfg = IsaacApparatusConfig(headless=bool(args.headless), device=args.device)
    scene = SupervisoryFetchScene(cfg=cfg, seed=0)
    scene.open()
    spec = SceneSpec(scene_id=0, axis=grid.axis, margin_m=0.14, clutter=0,
                     c=float(grid.physics.coordinate(0.14)))
    scene.reset_to(spec)
    if not args.keep_objects:
        # **Park the cubes before probing.** With the scene in place the probe measures where the
        # gripper can go *without hitting an object*, which is a different quantity and an easy
        # one to misread as kinematics. It cost us a wrong conclusion: at grasp height the row
        # through the objects (y ~ 0) came back unreachable for x in [0.22, 0.52] while the rows
        # 10 cm to either side were fully reachable, which looks exactly like a singularity in
        # the base's x-z plane. It was the target cube at x = 0.48 and the blocker at x = 0.29,
        # sitting precisely where the gripper was being sent.
        for k, name in enumerate(OBJECT_NAMES):
            scene._set_pose(name, (-0.60, -0.60 + 0.12 * k, 0.03))
        for _ in range(30):
            scene.sim.step(render=False)
            scene.robot.update(float(scene.sim.get_physics_dt()))
    orient = dict(getattr(scene, "_orient", {}) or {})
    print(f"[reach] wrist alignment: downwardness {orient.get('downwardness_before')} -> "
          f"{orient.get('downwardness_after')} (ok={orient.get('ok')})", file=sys.stderr)

    # Each point is measured from the same starting pose. Without this the map is
    # path-dependent: a point the arm cannot reach leaves it flung out to the workspace edge and
    # the *next* point spends its whole budget travelling back rather than testing reachability.
    # Measured: (0.27, -0.10, 0.02) was reached to 1.9 cm when approached from a neighbour and
    # missed by 38.6 cm when approached from a failure two grid rows away. A map with that in it
    # is a map of the traversal order, not of the arm.
    home = tuple(float(v) for v in getattr(cfg, "start_ee_pos_b", (0.454, 0.093, 0.210)))

    def _park() -> None:
        for k, name in enumerate(OBJECT_NAMES):
            scene._set_pose(name, (-0.60, -0.60 + 0.12 * k, 0.03))
        for _ in range(20):
            scene.sim.step(render=False)
            scene.robot.update(float(scene.sim.get_physics_dt()))

    def _rehome() -> float:
        """Return to the start pose by **resetting the episode**, not by jogging back.

        Jogging back was tried and decays: a soft re-home works for the first few dozen points
        and then stops. Measured over one pass, the residual home error grew from 1.8 cm to
        6.5 cm once the probe entered a marginal region and to 14--18 cm afterwards, at which
        point every subsequent measurement is of the arm's recovery rather than of its reach.
        A full reset costs a couple of seconds and cannot drift.
        """
        scene.reset_to(spec)
        if not args.keep_objects:
            _park()
        scene.cfg.max_steps_per_waypoint = int(args.budget) * 2
        scene._follow(Waypoint(home, 1, "rehome", 0.02))
        return float(np.linalg.norm(scene._ee() - np.array(home)))

    path = out / REACH_FILE
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    with path.open("w") as fh:
        for i, p in enumerate(pts):
            home_err = _rehome()
            wp = Waypoint((p["x"], p["y"], p["z"]), 1, "probe", float(args.tol))
            scene.cfg.max_steps_per_waypoint = int(args.budget)
            ok, steps = scene._follow(wp)
            ee = scene._ee()
            rec = {**p, "reached": bool(ok), "steps": int(steps), "home_err_m": round(home_err, 4),
                   "budget": int(args.budget),
                   "err_m": float(np.linalg.norm(ee - np.array([p["x"], p["y"], p["z"]]))),
                   "ee": [round(float(v), 4) for v in ee],
                   "downwardness": float(-scene.tool_axis_b()[2])}
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{i + 1}/{len(pts)}] ({p['x']:.2f},{p['y']:+.2f},{p['z']:.2f}) "
                  f"-> {'ok' if ok else 'MISS'} err={rec['err_m'] * 100:5.1f}cm  "
                  f"down={rec['downwardness']:+.2f}  home={home_err * 100:4.1f}cm  "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)

    summary = summarise(rows)
    summary["orient"] = orient
    summary["args"] = vars(args) | {"out": str(out)}
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, default=float) + "\n")
    figure(rows, out / "fig_reach.pdf")
    print(json.dumps(summary["per_height"], indent=2, default=float))
    print(json.dumps(summary["recommendation"], indent=2, default=float))
    _exit_now(0)
    return 0



def _exit_now(code: int = 0) -> None:
    """Leave the process immediately once every output is on disk.

    Isaac Sim's shutdown reliably hangs after a long headless run, holding the GPU for minutes
    after the work is finished. Every output this command produces is flushed before this is
    called, so there is nothing to lose by not unwinding -- and something to gain, since the next
    stage of a chained pipeline is usually waiting on this process to exit.
    """
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))

if __name__ == "__main__":
    raise SystemExit(main())

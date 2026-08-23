"""Render the scene atlas: one set of real Isaac frames per scene on the study's grid.

These frames are what the policies are trained and evaluated on, and what the paper shows when
it claims to show the task. Both uses make the same demand: **the frames must come from the
same scene the rest of the pipeline runs.** A geometry change that is not followed by a re-render
leaves the models learning one workspace and the reader looking at another, and nothing in the
training logs would say so.

Each scene is rendered several times with the per-episode domain randomisation active, so the
atlas carries the variation a policy will actually see rather than one canonical still. The
per-reset placement error -- how far the realised clearance gap is from the commanded one -- is
recorded for every render, because the gap is the study's independent variable and a scene that
fails to realise it is invisible in an image.

    ./vla_lab/scripts/sup_frames.sh --headless

Writes ``<out>/<view>/scene_NNN/<tag>.png`` and ``<out>/manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/physics/frames"))
    ap.add_argument("--variants", type=int, default=3, help="randomised renders per scene")
    ap.add_argument("--figure-views", action="store_true", default=True,
                    help="also render the oblique figure camera on the first variant")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from isaaclab.app import AppLauncher

    app = AppLauncher(argparse.Namespace(headless=bool(args.headless), enable_cameras=True, device=args.device))
    _sim_app = app.app

    from environments.supervisory_fetch.scene import SupervisoryFetchScene

    from .apparatus.isaac import IsaacApparatusConfig
    from .scenes import build_scene_grid

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = IsaacApparatusConfig(headless=bool(args.headless), device=args.device, capture_frames=True)
    scene = SupervisoryFetchScene(cfg=cfg, seed=int(args.seed), capture_dir=out)
    scene.open()
    grid = build_scene_grid()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for sc in grid.scenes:
        errs: List[float] = []
        for v in range(int(args.variants)):
            scene.reset_to(sc)
            errs.append(abs(float(scene._placement_error_m or 0.0)))
            which = ("topdown", "figure") if (v == 0 and args.figure_views) else ("topdown",)
            paths = scene.capture(sc, tag=f"v{v}", which=which)
            rows.append({"scene_id": int(sc.scene_id), "variant": v, "margin_m": float(sc.margin_m),
                         "c": float(sc.c), "placement_error_m": errs[-1],
                         "files": {k: str(p) for k, p in paths.items()}})
        print(f"[frames] scene {sc.scene_id:2d} gap={sc.margin_m * 100:5.1f}cm c={sc.c:+5.2f} "
              f"x{args.variants}  max placement err {max(errs) * 1000:.3f} mm", file=sys.stderr)

    worst = max((r["placement_error_m"] for r in rows), default=0.0)
    manifest = {
        "n_scenes": len(grid.scenes), "variants": int(args.variants), "n_renders": len(rows),
        "max_placement_error_m": worst, "elapsed_s": time.time() - t0,
        "target_xy": list(scene.scene_cfg.target_xy),
        "blocker_bearing_rad": float(scene.scene_cfg.blocker_bearing_rad),
        "cube_size_m": float(scene.scene_cfg.cube_size_m),
        "rows": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    scene.close()
    print(f"\n{len(rows)} renders over {len(grid.scenes)} scenes in {time.time() - t0:.0f}s -> {out}")
    print(f"worst realised-gap error across every reset: {worst * 1000:.3f} mm")
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

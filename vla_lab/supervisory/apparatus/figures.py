"""Select the simulator frames the paper shows, and copy them under stable names.

The scene grid is dense near the crossover, so most of its 19 scenes look nearly alike; a figure
that showed all of them would communicate less than three chosen to bracket the independent
variable. This picks the tight, crossover, and wide ends by *coordinate* rather than by index,
so it keeps working when the physics is re-measured and the grid is rebuilt.

    python -m vla_lab.supervisory.apparatus.figures \
        --frames vla_lab/results/physics/frames \
        --out vla_lab/paper/hri2027_carryover_vla/figures
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..scenes import build_scene_grid, load_physics


def pick_scenes(grid) -> Dict[str, int]:
    """The three scenes the figure brackets, chosen by ``c`` rather than by index."""
    probe = sorted(grid.probe_scenes(), key=lambda s: s.c)
    crossover = min(probe, key=lambda s: abs(s.c))
    return {"wide": probe[0].scene_id, "crossover": crossover.scene_id, "tight": probe[-1].scene_id}


def copy_figures(frames: Path, out: Path, grid, *, variant: str = "v0") -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    picks = pick_scenes(grid)
    written: List[Path] = []
    manifest: Dict[str, Dict] = {}
    for label, sid in picks.items():
        scene = grid.by_id(sid)
        for view, prefix in (("figure", "scene"), ("topdown", "top")):
            src = frames / view / f"scene_{sid:03d}" / f"{variant}.png"
            if not src.exists():
                cand = sorted((frames / view / f"scene_{sid:03d}").glob("*.png"))
                if not cand:
                    print(f"[figures] missing {src}")
                    continue
                src = cand[0]
            dst = out / f"{prefix}_{label}.png"
            shutil.copyfile(src, dst)
            written.append(dst)
        manifest[label] = {"scene_id": int(sid), "gap_cm": round(scene.margin_m * 100, 1),
                           "c": round(scene.c, 2), "clutter": int(scene.clutter),
                           "physics_source": grid.physics.source}
    (out / "scene_figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, default=Path("vla_lab/results/physics/frames"))
    ap.add_argument("--out", type=Path, default=Path("vla_lab/paper/hri2027_carryover_vla/figures"))
    ap.add_argument("--physics", type=Path, default=None)
    ap.add_argument("--variant", default="v0")
    args = ap.parse_args(argv)

    grid = build_scene_grid(physics=load_physics(args.physics) if args.physics else None)
    made = copy_figures(args.frames, args.out, grid, variant=args.variant)
    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

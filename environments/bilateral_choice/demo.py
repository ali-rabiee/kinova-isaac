"""W10 — the Phase 0 twin dry-run. The M2 gate.

    ./vla_lab/scripts/rehab_twin_dryrun.sh
    python environments/bilateral_choice/demo.py --gui --out-dir vla_lab/results/rehab_phase0/twin

Passes when, over every target in the contract:

* **100% reachable** from the study mounting pose,
* **zero** presentation trajectories intersect the seated-participant proxy, and
* a wrist-camera view is rendered per target, so the "will the wrist camera actually see the
  participant's hand arrive?" question (W11, §14) is answered with pictures rather than
  optimism.

Exits non-zero when the gate fails, so it can front a hardware bring-up checklist.

The geometric half of the check (trajectory clearance) runs **without** Isaac and is exercised
by ``vla_lab/tests/test_rehab_safety.py``; the simulator is reserved for what only it can
answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default="vla_lab/configs/rehab_twin.yaml")
    ap.add_argument("--out-dir", type=str, default="vla_lab/results/rehab_phase0/twin")
    ap.add_argument("--gui", action="store_true", help="run with the Isaac viewport (default headless)")
    ap.add_argument("--no-render", action="store_true", help="skip the per-target wrist renders")
    ap.add_argument("--geometry-only", action="store_true",
                    help="run only the Isaac-free clearance check (no simulator needed)")
    args = ap.parse_args(argv)

    from vla_lab.rehab.apparatus.isaac_apparatus import ParticipantProxy, TwinReport, check_trajectories
    from vla_lab.rehab.contract import Phase0Contract

    cfg_path = Path(args.config)
    contract = Phase0Contract.from_yaml(cfg_path) if cfg_path.exists() else Phase0Contract()
    problems = contract.validate()
    if problems:
        print("[rehab.twin] contract is invalid:")
        for p in problems:
            print(f"  - {p}")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = contract.target_grid()
    proxy = ParticipantProxy()

    if args.geometry_only:
        collisions = check_trajectories(grid, proxy=proxy)
        report = TwinReport(n_targets=len(grid), reachable=grid.ids(), trajectory_collisions=collisions)
        report.notes.append("geometry-only run: robot reachability was NOT checked in the simulator")
        _emit(report, out_dir)
        return 0 if not collisions else 1

    from vla_lab.rehab.apparatus.isaac_apparatus import IsaacApparatus

    app = IsaacApparatus(contract, proxy=proxy, headless=not args.gui)
    try:
        app.connect()
        app.home()
        report = app.dry_run(render_wrist=not args.no_render, out_dir=str(out_dir / "wrist_views"))
    finally:
        app.close()
    _emit(report, out_dir)
    return 0 if report.passed else 1


def _emit(report, out_dir: Path) -> None:
    d = report.to_dict()
    (out_dir / "twin_report.json").write_text(json.dumps(d, indent=2))
    print(f"[rehab.twin] targets: {d['n_reachable']}/{d['n_targets']} reachable")
    print(f"[rehab.twin] participant-proxy trajectory collisions: {d['n_trajectory_collisions']}")
    if d["unreachable"]:
        print(f"[rehab.twin] UNREACHABLE target ids: {d['unreachable']}")
    for c in d["trajectory_collisions"][:5]:
        print(f"[rehab.twin]   collision: target {c['target_id']} at {c['waypoint_xy_m']} — {c['reason']}")
    for n in d["notes"]:
        print(f"[rehab.twin] note: {n}")
    print(f"[rehab.twin] wrist views rendered: {len(d['wrist_views'])}")
    print(f"[rehab.twin] RESULT: {'PASS' if d['passed'] else 'FAIL'} — wrote {out_dir / 'twin_report.json'}")


if __name__ == "__main__":
    raise SystemExit(main())

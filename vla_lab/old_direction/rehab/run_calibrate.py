"""W13 — physical calibration: table homography, camera intrinsics, and the participant frame.

    ./vla_lab/scripts/rehab_calibrate.sh --participant P001 --points calib_points.json

The **participant frame is re-solved for every participant**, because the midline defines the
crossover band and therefore the informative region of the estimand (``rehab.md`` §9). A 2 cm
midline error shifts the whole crossover band by 2 cm, which at a typical psychometric slope is
a ~0.06 shift in ``pi*`` right where the primary outcome is measured — so the solve's
uncertainty is *reported*, and :meth:`CalibrationBundle.check` refuses a session whose midline
is worse than the stated tolerance.

Input JSON (produced by whatever marker-detection tooling the rig uses — camera capture is an
environment dependency, not repo code, ``rehab.md`` §8)::

    {
      "cameras": {
        "front": {"fov_deg": 78.0, "resolution": [960, 540],
                  "image_points": [[u,v], ...], "table_points": [[x,y], ...]}
      },
      "shoulders": {"left": [[x,y], ...], "right": [[x,y], ...]},
      "robot_base_in_table": {"tx": 0.0, "ty": 0.0, "yaw_rad": 0.0},
      "wrist_hand_eye": {"offset_pos": [0, -0.04, -0.06], "offset_rpy_deg": [145, 0, 0]}
    }

Re-running on a fixed rig must reproduce target positions within the stated tolerance; the
reprojection residuals in the output are that check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .observation.calibration import (
    CalibrationBundle,
    homography_residuals,
    pinhole_intrinsics,
    solve_participant_frame,
    solve_table_homography,
)
from .workspace import PlanarTransform


def calibrate(points: Dict[str, Any], *, sternum_forward_offset_m: float = 0.0) -> CalibrationBundle:
    bundle = CalibrationBundle()
    for cam, d in (points.get("cameras") or {}).items():
        if d.get("fov_deg") and d.get("resolution"):
            bundle.intrinsics[cam] = pinhole_intrinsics(float(d["fov_deg"]), tuple(d["resolution"]))
        ip, tp = d.get("image_points"), d.get("table_points")
        if ip and tp:
            h = solve_table_homography(ip, tp)
            bundle.homographies[cam] = [[float(v) for v in row] for row in h]
            bundle.residuals[cam] = homography_residuals(h, ip, tp)

    sh = points.get("shoulders") or {}
    if sh.get("left") and sh.get("right"):
        bundle.participant_frame = solve_participant_frame(
            sh["left"], sh["right"], sternum_forward_offset_m=float(sternum_forward_offset_m)
        )
    if points.get("robot_base_in_table"):
        bundle.robot_base_in_table = PlanarTransform.from_dict(points["robot_base_in_table"])
    if points.get("wrist_hand_eye"):
        bundle.wrist_hand_eye = dict(points["wrist_hand_eye"])
    return bundle


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=str, required=True, help="calibration input JSON (see the module docstring)")
    ap.add_argument("--participant", type=str, default=None, help="study ID, used for the output filename")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sternum-forward-offset-m", type=float, default=0.0)
    ap.add_argument("--max-reprojection-m", type=float, default=0.01)
    ap.add_argument("--max-midline-sd-m", type=float, default=0.02)
    args = ap.parse_args(argv)

    points = json.loads(Path(args.points).read_text())
    bundle = calibrate(points, sternum_forward_offset_m=float(args.sternum_forward_offset_m))
    problems = bundle.check(
        max_reprojection_m=float(args.max_reprojection_m),
        max_midline_sd_m=float(args.max_midline_sd_m),
    )

    pf = bundle.participant_frame
    print(f"[rehab.calibrate] participant frame: origin=({pf.participant_in_table.tx:.4f}, "
          f"{pf.participant_in_table.ty:.4f}) yaw={pf.participant_in_table.yaw_rad:.4f} rad")
    print(f"[rehab.calibrate] shoulder half-width: {100*pf.shoulder_halfwidth_m:.1f} cm  "
          f"(from {pf.n_samples} samples)")
    print(f"[rehab.calibrate] midline uncertainty: {1000*pf.midline_sd_m:.1f} mm")
    for cam, res in bundle.residuals.items():
        print(f"[rehab.calibrate] {cam}: reprojection mean {1000*res['mean_m']:.1f} mm, "
              f"max {1000*res['max_m']:.1f} mm over {res['n']} points")

    out = Path(args.out or f"logs/rehab/calibration_{args.participant or 'rig'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle.to_dict(), indent=2))
    print(f"[rehab.calibrate] wrote {out}")

    for p in problems:
        print(f"[rehab.calibrate][FAIL] {p}")
    if problems:
        print("[rehab.calibrate] RESULT: FAIL — do not run a session on this calibration.")
        return 1
    print("[rehab.calibrate] RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

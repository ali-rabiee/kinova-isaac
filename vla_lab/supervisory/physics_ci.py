"""The primary contrasts under the physics interval: how much of the headline is physics error?

``m*`` and ``w`` are estimates from a finite sweep, and every scene coordinate, band weight and
regret number inherits their sampling error. ``sup_physics_ci.sh`` re-runs the primary study
under the bootstrap draws of the physics whose transition width sits at the 2.5th and 97.5th
percentile (``run_study --physics-quantile lower|upper``); this module puts the three runs side
by side so the statement "the memoryless contrast is robust to physics estimation error; the
remedy contrasts are inside the floor under every draw" is a table rather than a sentence.

    python -m vla_lab.supervisory.physics_ci
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .scheduler import (
    CONDITION_ALWAYS_COUNTER,
    CONDITION_CARRYOVER_AWARE,
    CONDITION_IDENTIFICATION_FIRST,
    CONDITION_MEMORYLESS,
    CONDITION_RECOMMENDED,
    DISPLAY_NAMES,
    PRIMARY_COMPARATOR,
)

CONTRASTS = (CONDITION_MEMORYLESS, CONDITION_ALWAYS_COUNTER, CONDITION_CARRYOVER_AWARE,
             CONDITION_RECOMMENDED, CONDITION_IDENTIFICATION_FIRST)
DRAWS = (("lower", "vla_lab/results/tier1_physics_lower"), ("point", "vla_lab/results/tier1"),
         ("upper", "vla_lab/results/tier1_physics_upper"))


def load(draws: Sequence[tuple] = DRAWS) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tag, d in draws:
        f = Path(d) / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        from .scenes import ScenePhysics

        sp = ScenePhysics.from_dict((s.get("contract", {}).get("grid") or {}).get("physics", {}))
        row: Dict[str, Any] = {
            "draw": tag, "dir": d, "n": s.get("n_supervisors"),
            "quantile_stamp": sp.quantile,
            "mstar_cm": sp.crossover_margin() * 100.0, "w_cm": sp.transition_width_m() * 100.0,
            "floor": (s.get("test_retest", {}).get("mae_crossover") or {}).get("mean"),
            "spread_remedies": None,
            "contrasts": {},
            "absolute": {},
        }
        conds = s.get("conditions", {})
        others = [v["mae_crossover"]["mean"] for k, v in conds.items() if k != CONDITION_MEMORYLESS]
        if others:
            row["spread_remedies"] = max(others) - min(others)
        for c in CONTRASTS:
            ct = (s.get("contrasts", {}).get(c) or {}).get("mae_crossover")
            if ct:
                row["contrasts"][c] = {"delta": ct["delta"], "lo": ct["lo"], "hi": ct["hi"],
                                       "excludes_zero": bool(ct["lo"] > 0 or ct["hi"] < 0),
                                       "p_better": ct.get("p_better")}
            if c in conds:
                row["absolute"][c] = conds[c]["mae_crossover"]["mean"]
        if PRIMARY_COMPARATOR in conds:
            row["absolute"][PRIMARY_COMPARATOR] = conds[PRIMARY_COMPARATOR]["mae_crossover"]["mean"]
        rows.append(row)
    return rows


def render(rows: Sequence[Dict[str, Any]]) -> str:
    head = f"{'physics draw':14s}{'m* cm':>7}{'w cm':>6}{'floor':>8}{'spread':>8}" + \
           "".join(f"{DISPLAY_NAMES[c].split()[0]:>26}" for c in CONTRASTS)
    lines = ["primary contrasts (paired dMAE_x vs " + DISPLAY_NAMES[PRIMARY_COMPARATOR] +
             ") under the bootstrap draws of the measured physics", head, "-" * len(head)]
    for r in rows:
        cells = []
        for c in CONTRASTS:
            ct = r["contrasts"].get(c)
            if not ct:
                cells.append(f"{'--':>26}")
                continue
            cells.append(f"{ct['delta']:+.4f} [{ct['lo']:+.4f},{ct['hi']:+.4f}]{'*' if ct['excludes_zero'] else ' '}".rjust(26))
        lines.append(f"{r['draw']:14s}{r['mstar_cm']:>7.2f}{r['w_cm']:>6.2f}{(r['floor'] or float('nan')):>8.4f}"
                     f"{(r['spread_remedies'] or float('nan')):>8.4f}" + "".join(cells))
    lines.append("   * = interval excludes zero")
    mem = [r["contrasts"].get(CONDITION_MEMORYLESS, {}).get("excludes_zero") for r in rows]
    if rows and all(mem):
        lines.append("[ok] the memoryless contrast excludes zero under every physics draw")
    rem = [ct["excludes_zero"] for r in rows for c, ct in r["contrasts"].items() if c != CONDITION_MEMORYLESS]
    if rows:
        lines.append(("[ok] no remedy contrast excludes zero under any draw" if not any(rem)
                      else "[!] at least one remedy contrast excludes zero under some draw -- report which"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/physics_ci"))
    args = ap.parse_args(argv)
    rows = load()
    if not rows:
        print("no runs found")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    text = render(rows)
    print(text)
    (args.out / "table.txt").write_text(text + "\n")
    (args.out / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

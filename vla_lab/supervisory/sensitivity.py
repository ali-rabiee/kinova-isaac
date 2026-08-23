"""Aggregate the sensitivity sweep: does the conclusion survive the design choices behind it?

Every number in the main table rests on three assumptions a reader is entitled to push on: how
strong a demonstration is (the dose), whether the robot coaches one-sidedly or alternates, and
whether the belief module starts from a population prior or a flat one. Each was chosen for a
stated reason, and each could have been chosen otherwise. This runs the whole study under each
alternative and reports whether the ordering of the conditions changes.

The question is deliberately about *ordering*, not about whether the numbers move. They will
move -- a stronger dose contaminates more, and the errors rise for everybody. What would damage
the paper is a setting under which a different condition wins.

    python -m vla_lab.supervisory.sensitivity vla_lab/results/sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .scheduler import DISPLAY_NAMES

#: Reported in this order; the ordering claim is about these five.
CONDITIONS = ["memoryless", "fixed_washout", "random_static", "always_counter", "carryover_aware"]

LABELS = {
    "dose_weak": "dose: weak",
    "dose_moderate": "dose: moderate (default)",
    "dose_strong": "dose: strong",
    "regime_alternating": "coaching: alternating",
    "prior_flat": "belief prior: flat",
}


def load(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in sorted(Path(root).iterdir()):
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        cells = s.get("conditions", {})
        row: Dict[str, Any] = {
            "cell": d.name,
            "label": LABELS.get(d.name, d.name),
            "n": s.get("n_supervisors"),
            "floor": s.get("test_retest", {}).get("mae_crossover", {}).get("mean"),
            "beta_g": s.get("population", {}).get("beta_g", {}).get("mean"),
        }
        for c in CONDITIONS:
            if c in cells:
                row[c] = cells[c]["mae_crossover"]["mean"]
                row[f"{c}__ctr"] = cells[c]["n_counter"]["mean"]
        present = [c for c in CONDITIONS if c in row]
        row["best"] = min(present, key=lambda c: row[c]) if present else None
        row["worst"] = max(present, key=lambda c: row[c]) if present else None
        rows.append(row)
    return rows


def render(rows: Sequence[Dict[str, Any]]) -> str:
    head = f"{'setting':26s}" + "".join(f"{DISPLAY_NAMES[c].split()[0]:>9}" for c in CONDITIONS) + \
           f"{'floor':>9}{'B5 ctr':>8}  best / worst"
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(f"{r.get(c, float('nan')):>9.4f}" for c in CONDITIONS)
        lines.append(
            f"{r['label']:26s}{cells}{r.get('floor', float('nan')):>9.4f}"
            f"{r.get('carryover_aware__ctr', float('nan')):>8.1f}  "
            f"{DISPLAY_NAMES.get(r['best'], '?').split()[0]} / {DISPLAY_NAMES.get(r['worst'], '?').split()[0]}"
        )
    worsts = {r["worst"] for r in rows if r["worst"]}
    bests = {r["best"] for r in rows if r["best"]}
    lines.append("")
    lines.append(f"worst condition across all settings: {sorted(worsts)}")
    lines.append(f"best condition across all settings:  {sorted(bests)}")
    if worsts == {"memoryless"}:
        lines.append("[ok] the memoryless VLA is worst under EVERY setting -- the headline result is not "
                     "an artifact of the dose, the coaching regime, or the belief prior.")
    else:
        lines.append("[!] the worst condition is not stable across settings; the headline needs qualifying.")
    return "\n".join(lines)


#: Presentation order: the dose ladder ascending, then the two structural variants. The pattern
#: in this sweep is monotone in the dose, and a reader should be able to see that by reading left
#: to right rather than reconstructing it from an alphabetical listing.
CELL_ORDER = {"dose_weak": 0, "dose_moderate": 1, "dose_strong": 2, "prior_flat": 3,
              "regime_alternating": 4}


def _ordered(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: CELL_ORDER.get(str(r.get("cell", "")), 99))


def figure(rows: Sequence[Dict[str, Any]], out: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 160, "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    # Two panels. The left one is the headline -- is the ordering stable? -- and the right one is
    # the behaviour that is *supposed* to move: the carryover-aware policy is never told the dose,
    # so how often it counter-proposes is a readout of what its belief module inferred. Showing
    # only the left panel hides the one quantity in this sweep that tracks the manipulation.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.1), gridspec_kw={"width_ratios": [2.1, 1.0]})
    x = np.arange(len(rows))
    w = 0.16
    colors = {"memoryless": "#c0392b", "fixed_washout": "#7f8c8d", "random_static": "#95a5a6",
              "always_counter": "#e67e22", "carryover_aware": "#2c6fbb"}
    for i, c in enumerate(CONDITIONS):
        ys = [r.get(c, np.nan) for r in rows]
        ax.bar(x + (i - 2) * w, ys, width=w, color=colors[c],
               label=DISPLAY_NAMES[c].split(maxsplit=1)[-1])
    for i, r in enumerate(rows):
        f = r.get("floor")
        if f is not None and np.isfinite(f):
            ax.plot([i - 2.6 * w, i + 2.6 * w], [f, f], color="#333", lw=1.0, ls="--",
                    label="test–retest floor" if i == 0 else None)
    ax.set_xticks(x, [r["label"] for r in rows], rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("crossover-weighted MAE")
    ax.set_title("The ordering is stable across the design choices behind it", loc="left", fontsize=9)
    ax.legend(fontsize=6.2, frameon=False, ncol=3)

    ctr = [r.get("carryover_aware__ctr", np.nan) for r in rows]
    bx.bar(x, ctr, width=0.55, color="#2c6fbb")
    for i, v in enumerate(ctr):
        if np.isfinite(v):
            bx.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    bx.set_xticks(x, [r["label"].split(":")[-1].strip() for r in rows], rotation=18, ha="right", fontsize=7)
    bx.set_ylabel("counter-proposals / session")
    bx.set_ylim(0, max([v for v in ctr if np.isfinite(v)] + [1.0]) * 1.25)
    bx.set_title("B5 asks more when it infers more residue", loc="left", fontsize=9)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_sensitivity.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, nargs="?", default=Path("vla_lab/results/sensitivity"))
    args = ap.parse_args(argv)
    rows = _ordered(load(args.root))
    if not rows:
        print(f"no summaries under {args.root}")
        return 1
    text = render(rows)
    print(text)
    (args.root / "table.txt").write_text(text + "\n")
    (args.root / "table.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    p = figure(rows, args.root)
    if p:
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

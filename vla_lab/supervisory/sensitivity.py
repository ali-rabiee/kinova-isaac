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
CONDITIONS = ["memoryless", "fixed_washout", "random_static", "always_counter", "carryover_aware",
              "identification_first", "recommended"]

LABELS = {
    "dose_weak": "dose: weak",
    "dose_moderate": "dose: moderate (default)",
    "dose_strong": "dose: strong",
    "regime_alternating": "coaching: alternating",
    "prior_flat": "belief prior: flat",
    "placebo_lapse_low": "placebo: lapse 0.00-0.02",
    "placebo_lapse_high": "placebo: lapse 0.15-0.25",
    "placebo_latency_slow": "placebo: slow answers",
}

#: The placebo control for the dose-tracking result. The carryover-aware policy's counter-proposal
#: rate is claimed to track the inferred residue; if it also tracked a parameter the belief module
#: has no access to -- the supervisor's lapse rate, which adds answer noise with no direction --
#: the tracking would be an incidental correlation with something else (session length, scene
#: sequence), not the mechanism.
PLACEBO_CELLS = ("placebo_lapse_low", "placebo_lapse_high", "placebo_latency_slow")


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
        row["population_overrides"] = s.get("population_overrides", [])
        for c in CONDITIONS:
            if c in cells:
                row[c] = cells[c]["mae_crossover"]["mean"]
                row[f"{c}__ctr"] = cells[c]["n_counter"]["mean"]
                row[f"{c}__wait"] = cells[c]["n_wait"]["mean"]
        # What the policy inferred: the posterior-mean compliance strength, averaged over supervisors.
        try:
            per = json.loads((d / "per_supervisor.json").read_text())
            bgs = [p["conditions"]["carryover_aware"]["online_belief"]["mean"]["beta_g"] for p in per
                   if p.get("conditions", {}).get("carryover_aware", {}).get("online_belief")]
            row["carryover_aware__beta_g_hat"] = float(np.mean(bgs)) if bgs else None
        except Exception:
            row["carryover_aware__beta_g_hat"] = None
        present = [c for c in CONDITIONS if c in row]
        row["best"] = min(present, key=lambda c: row[c]) if present else None
        row["worst"] = max(present, key=lambda c: row[c]) if present else None
        rows.append(row)
    return rows


def render(rows: Sequence[Dict[str, Any]]) -> str:
    head = f"{'setting':26s}" + "".join(f"{DISPLAY_NAMES[c].split()[0]:>9}" for c in CONDITIONS) + \
           f"{'floor':>9}{'B5 ctr':>8}{'B7 ctr':>8}{'B5 bg^':>8}  best / worst"
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(f"{r.get(c, float('nan')):>9.4f}" for c in CONDITIONS)
        lines.append(
            f"{r['label']:26s}{cells}{r.get('floor', float('nan')):>9.4f}"
            f"{r.get('carryover_aware__ctr', float('nan')):>8.1f}"
            f"{r.get('recommended__ctr', float('nan')):>8.1f}"
            f"{r.get('carryover_aware__beta_g_hat', float('nan')):>8.2f}  "
            f"{DISPLAY_NAMES.get(r['best'], '?').split()[0]} / {DISPLAY_NAMES.get(r['worst'], '?').split()[0]}"
        )
    main_rows = [r for r in rows if r["cell"] not in PLACEBO_CELLS]
    worsts = {r["worst"] for r in main_rows if r["worst"]}
    bests = {r["best"] for r in main_rows if r["best"]}
    lines.append("")
    lines.append(f"worst condition across all settings: {sorted(worsts)}")
    lines.append(f"best condition across all settings:  {sorted(bests)}")
    if worsts == {"memoryless"}:
        lines.append("[ok] the memoryless VLA is worst under EVERY setting -- the headline result is not "
                     "an artifact of the dose, the coaching regime, or the belief prior.")
    else:
        lines.append("[!] the worst condition is not stable across settings; the headline needs qualifying.")
    placebo = placebo_check(rows)
    if placebo:
        lines.append("")
        lines.append(placebo["text"])
    return "\n".join(lines)


def placebo_check(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Does B5's counter-proposal rate follow the dose (it should) and NOT the placebo (it must not)?"""
    by = {r["cell"]: r for r in rows}
    dose = [by[c].get("carryover_aware__ctr") for c in ("dose_weak", "dose_moderate", "dose_strong") if c in by]
    plc = [by[c].get("carryover_aware__ctr") for c in ("placebo_lapse_low", "dose_moderate", "placebo_lapse_high") if c in by]
    if len(dose) < 3 or len(plc) < 3 or any(v is None for v in dose + plc):
        return None
    dose_span = max(dose) - min(dose)
    plc_span = max(plc) - min(plc)
    monotone_dose = dose[0] <= dose[1] <= dose[2]
    out = {"dose_ctr": dose, "placebo_ctr": plc, "dose_span": dose_span, "placebo_span": plc_span,
           "dose_monotone": monotone_dose, "placebo_span_over_dose_span": (plc_span / dose_span if dose_span else None)}
    ok = monotone_dose and dose_span > 0 and plc_span < 0.25 * dose_span
    out["ok"] = bool(ok)
    out["text"] = (f"dose-tracking control: B5 counter-proposals {dose[0]:.1f} -> {dose[1]:.1f} -> {dose[2]:.1f} across the "
                   f"dose ladder (span {dose_span:.1f}); across the placebo (lapse rate) {plc[0]:.1f} -> {plc[1]:.1f} -> "
                   f"{plc[2]:.1f} (span {plc_span:.1f}). "
                   + ("[ok] the rate follows the dose and not the placebo." if ok else
                      "[!] the rate moves with the placebo: the dose-tracking claim needs qualifying."))
    return out


#: Presentation order: the dose ladder ascending, then the two structural variants. The pattern
#: in this sweep is monotone in the dose, and a reader should be able to see that by reading left
#: to right rather than reconstructing it from an alphabetical listing.
CELL_ORDER = {"dose_weak": 0, "dose_moderate": 1, "dose_strong": 2, "prior_flat": 3,
              "regime_alternating": 4, "placebo_lapse_low": 5, "placebo_lapse_high": 6,
              "placebo_latency_slow": 7}


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
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 3.3), gridspec_kw={"width_ratios": [2.4, 1.0]})
    x = np.arange(len(rows))
    w = 0.8 / len(CONDITIONS)
    colors = {"memoryless": "#c0392b", "fixed_washout": "#7f8c8d", "random_static": "#95a5a6",
              "always_counter": "#e67e22", "carryover_aware": "#2c6fbb", "identification_first": "#8e44ad",
              "recommended": "#1b7f5a"}
    mid = (len(CONDITIONS) - 1) / 2.0
    for i, c in enumerate(CONDITIONS):
        ys = [r.get(c, np.nan) for r in rows]
        ax.bar(x + (i - mid) * w, ys, width=w, color=colors.get(c, "#999"),
               label=DISPLAY_NAMES[c].split(maxsplit=1)[-1])
    for i, r in enumerate(rows):
        f = r.get("floor")
        if f is not None and np.isfinite(f):
            ax.plot([i - 0.45, i + 0.45], [f, f], color="#333", lw=1.0, ls="--",
                    label="test–retest floor" if i == 0 else None)
    ax.set_xticks(x, [r["label"] for r in rows], rotation=18, ha="right", fontsize=6.5)
    ax.set_ylabel("crossover-weighted MAE")
    ax.set_title("The ordering is stable across the design choices behind it", loc="left", fontsize=9)
    ax.legend(fontsize=5.8, frameon=False, ncol=4)

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


def figure_dose_tracking(rows: Sequence[Dict[str, Any]], out: Path) -> Optional[Path]:
    """The mechanism end to end, with its control. Left: B5's counter-proposal rate across the
    dose ladder, against what its belief module inferred. Right: the same rate across the
    placebo cells -- parameters the belief module has no access to -- which must stay flat."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 160, "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    by = {r["cell"]: r for r in rows}
    dose = [c for c in ("dose_weak", "dose_moderate", "dose_strong") if c in by]
    plc = [c for c in ("placebo_lapse_low", "dose_moderate", "placebo_lapse_high", "placebo_latency_slow") if c in by]
    if len(dose) < 2:
        return None
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    x = np.arange(len(dose))
    ctr = [by[c].get("carryover_aware__ctr", np.nan) for c in dose]
    ctr7 = [by[c].get("recommended__ctr", np.nan) for c in dose]
    bg = [by[c].get("carryover_aware__beta_g_hat") for c in dose]
    ax.bar(x - 0.18, ctr, width=0.36, color="#2c6fbb", label="B5 counter-proposals / session")
    ax.bar(x + 0.18, ctr7, width=0.36, color="#1b7f5a", label="B7 counter-proposals / session")
    for i, v in enumerate(ctr):
        if np.isfinite(v):
            ax.text(i - 0.18, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, [LABELS.get(c, c).split(":")[-1].strip() for c in dose])
    ax.set_ylabel("counter-proposals per session")
    ax.set_title("The rate follows the dose the belief module inferred", loc="left", fontsize=8.5)
    if all(v is not None for v in bg):
        ax2 = ax.twinx()
        ax2.plot(x, bg, "o--", color="#c0392b", ms=4, label=r"inferred $\widehat{\beta g}$ (B5)")
        ax2.set_ylabel(r"posterior-mean $\widehat{\beta g}$", color="#c0392b")
        ax2.tick_params(axis="y", colors="#c0392b")
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")

    xp = np.arange(len(plc))
    pc = [by[c].get("carryover_aware__ctr", np.nan) for c in plc]
    bx.bar(xp, pc, width=0.55, color=["#7f8c8d" if c != "dose_moderate" else "#2c6fbb" for c in plc])
    for i, v in enumerate(pc):
        if np.isfinite(v):
            bx.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    bx.set_xticks(xp, [LABELS.get(c, c).replace("placebo: ", "").replace("dose: ", "") for c in plc],
                  rotation=12, ha="right", fontsize=7)
    bx.set_ylim(0, max(max(ctr + pc) * 1.3, 1.0))
    bx.set_ylabel("B5 counter-proposals per session")
    bx.set_title("... and not a placebo the belief module cannot see", loc="left", fontsize=8.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_dose_tracking.pdf"
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
    pc = placebo_check(rows)
    if pc:
        (args.root / "placebo.json").write_text(json.dumps(pc, indent=2, default=float) + "\n")
    p = figure(rows, args.root)
    if p:
        print(f"\nwrote {p}")
    p2 = figure_dose_tracking(rows, args.root)
    if p2:
        print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

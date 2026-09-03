"""Sensitivity of the headline contrasts to the task-difficulty curve: the flip diagnostic.

The study once reported a significant "asking beats waiting" effect under an *assumed* value
model and lost it under a *measured* one. That is a warning, not a tool -- nobody can act on it
without measuring their own curve. This turns it into a procedure: hold the policies, the
supervisors and the seeds fixed, sweep the assumed **transition width** ``w`` (and, secondarily,
the crossover ``m*``) of the curve the study is scored against, recompute every primary contrast
at each value, and report the range of curve parameters over which each conclusion holds. The
measured value and its bootstrap interval are marked on the same axis, so a reader can see at a
glance whether the measurement sits on the safe side of a flip point or on top of it.

A conclusion is "a paired 95% interval that excludes zero", the same standard the primary table
uses. The **flip point** of a contrast is the value of the swept parameter at which that
interval stops excluding zero; it is reported as the bracket between the two adjacent sweep
values where the status changes.

    python -m vla_lab.supervisory.flip run --param w --values 0.5 0.8 1.0 1.5 2.0 3.0 4.0 --jobs 7
    python -m vla_lab.supervisory.flip analyze vla_lab/results/flip_w

Every sweep cell is a full ``run_study`` invocation with its own manifest; the counterfactual
physics is stamped into the cell's contract (``physics.quantile = "assumed_w=..."``), so a
counterfactual run can never be pooled with a measured one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scheduler import (
    CONDITION_ALWAYS_COUNTER,
    CONDITION_CARRYOVER_AWARE,
    CONDITION_IDENTIFICATION_FIRST,
    CONDITION_MEMORYLESS,
    CONDITION_RECOMMENDED,
    DISPLAY_NAMES,
    PRIMARY_COMPARATOR,
)

#: The contrasts whose survival the diagnostic tracks, all against the primary comparator.
CONTRASTS = (CONDITION_MEMORYLESS, CONDITION_ALWAYS_COUNTER, CONDITION_CARRYOVER_AWARE,
             CONDITION_RECOMMENDED, CONDITION_IDENTIFICATION_FIRST)


def _cell_dir(root: Path, param: str, value: float) -> Path:
    return Path(root) / f"{param}_{value:.3f}cm"


def run_sweep(root: Path, *, param: str, values: Sequence[float], supervisors: int, seed: int,
              jobs: int, extra: Sequence[str] = ()) -> List[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    flag = "--assume-w-cm" if param == "w" else "--assume-mstar-cm"

    def one(v: float) -> Path:
        out = _cell_dir(root, param, v)
        cmd = [sys.executable, "-m", "vla_lab.supervisory.run_study", "--supervisors", str(supervisors),
               "--seed", str(seed), "--out", str(out), "--quiet", flag, str(v), *extra]
        log = Path(str(out) + ".log")
        out.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"  [{param}={v:g} cm] exit {rc} -> {out}", file=sys.stderr)
        return out

    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as ex:
        return list(ex.map(one, [float(v) for v in values]))


def load_sweep(root: Path, param: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in sorted(Path(root).glob(f"{param}_*cm")):
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        phys = (s.get("contract", {}).get("grid") or {}).get("physics", {})
        from .scenes import ScenePhysics

        sp = ScenePhysics.from_dict(phys)
        row: Dict[str, Any] = {
            "cell": d.name,
            "value_cm": float(d.name.split("_")[1].rstrip("cm")),
            "w_cm": sp.transition_width_m() * 100.0,
            "mstar_cm": sp.crossover_margin() * 100.0,
            "n": s.get("n_supervisors"),
            "floor": (s.get("test_retest", {}).get("mae_crossover") or {}).get("mean"),
            "contrasts": {},
        }
        for c in CONTRASTS:
            ct = (s.get("contrasts", {}).get(c) or {}).get("mae_crossover")
            if ct:
                row["contrasts"][c] = {"delta": ct["delta"], "lo": ct["lo"], "hi": ct["hi"],
                                       "excludes_zero": bool(ct["lo"] > 0 or ct["hi"] < 0),
                                       "sign": int(np.sign(ct["delta"]))}
        rows.append(row)
    return sorted(rows, key=lambda r: r["value_cm"])


def flip_points(rows: Sequence[Dict[str, Any]], contrast: str) -> Dict[str, Any]:
    """Where a contrast's interval stops (or starts) excluding zero along the sweep."""
    xs = [r["value_cm"] for r in rows if contrast in r["contrasts"]]
    st = [r["contrasts"][contrast]["excludes_zero"] for r in rows if contrast in r["contrasts"]]
    flips: List[Dict[str, Any]] = []
    for i in range(len(xs) - 1):
        if st[i] != st[i + 1]:
            flips.append({"between_cm": [xs[i], xs[i + 1]], "from": bool(st[i]), "to": bool(st[i + 1])})
    holds = [x for x, s in zip(xs, st) if s]
    return {
        "values_cm": xs,
        "excludes_zero": st,
        "flips": flips,
        "holds_at_cm": holds,
        "holds_over_cm": [min(holds), max(holds)] if holds else None,
        "holds_contiguous": bool(len(flips) <= 1),
        "holds_everywhere": bool(holds and len(holds) == len(xs)),
        "holds_nowhere": not holds,
    }


def analyze(root: Path, *, param: str, physics_report: Optional[Path] = None) -> Dict[str, Any]:
    rows = load_sweep(root, param)
    if not rows:
        raise SystemExit(f"no sweep cells under {root}")
    measured: Dict[str, Any] = {}
    rep_path = Path(physics_report) if physics_report else Path("vla_lab/results/physics/physics_report.json")
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        key = "transition_width_m" if param == "w" else "crossover_margin_m"
        ci = (rep.get("bootstrap") or {}).get(key) or {}
        measured = {"point_cm": float(rep[key]) * 100.0,
                    "ci_cm": [ci["p2.5"] * 100.0, ci["p97.5"] * 100.0] if "p2.5" in ci else None}
    out: Dict[str, Any] = {
        "param": param,
        "rows": rows,
        "measured": measured,
        "flip": {c: flip_points(rows, c) for c in CONTRASTS},
        "comparator": PRIMARY_COMPARATOR,
    }
    # The procedural summary: over which range of the curve parameter does each conclusion hold,
    # and is the measured interval inside that range?
    verdicts: Dict[str, Any] = {}
    for c, fp in out["flip"].items():
        hold = fp["holds_over_cm"]
        v: Dict[str, Any] = {"holds_over_cm": hold, "holds_at_cm": fp["holds_at_cm"],
                             "holds_contiguous": fp["holds_contiguous"], "flips": fp["flips"]}
        if measured.get("ci_cm"):
            lo, hi = measured["ci_cm"]
            inside = [x for x in fp["values_cm"] if lo - 1e-9 <= x <= hi + 1e-9]
            holding = set(fp["holds_at_cm"])
            # The conclusion is stable across the measurement's own uncertainty only if it holds
            # at EVERY swept value inside the measured interval. A flip point inside the interval
            # means the conclusion is about the measurement error, and the verdict says so.
            v["stable_inside_measured_ci"] = bool(inside) and all(x in holding for x in inside)
            v["flips_inside_measured_ci"] = [f for f in fp["flips"]
                                             if lo <= 0.5 * sum(f["between_cm"]) <= hi]
        verdicts[c] = v
    out["verdicts"] = verdicts
    return out


def render(res: Dict[str, Any]) -> str:
    param = res["param"]
    unit = "w (cm)" if param == "w" else "m* (cm)"
    lines = [f"flip diagnostic over the assumed {unit}; contrasts are paired dMAE_x vs "
             f"{DISPLAY_NAMES.get(res['comparator'], res['comparator'])} (negative = better)"]
    m = res.get("measured") or {}
    if m:
        ci = m.get("ci_cm")
        lines.append(f"measured {unit}: {m['point_cm']:.2f}" + (f"  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
    head = f"{unit:>9}{'floor':>8}" + "".join(f"{DISPLAY_NAMES[c].split()[0]:>26}" for c in CONTRASTS)
    lines += ["", head, "-" * len(head)]
    for r in res["rows"]:
        cells = []
        for c in CONTRASTS:
            ct = r["contrasts"].get(c)
            if not ct:
                cells.append(f"{'--':>26}")
                continue
            mark = "*" if ct["excludes_zero"] else " "
            cells.append(f"{ct['delta']:+.4f} [{ct['lo']:+.4f},{ct['hi']:+.4f}]{mark}".rjust(26))
        lines.append(f"{r['value_cm']:>9.2f}{(r['floor'] or float('nan')):>8.4f}" + "".join(cells))
    lines.append("   * = interval excludes zero")
    lines.append("")
    for c, v in res["verdicts"].items():
        hold = v["holds_at_cm"]
        if not hold:
            txt = "holds nowhere in the sweep"
        elif v["holds_contiguous"]:
            txt = f"holds for {unit} in [{min(hold):g}, {max(hold):g}]"
        else:
            txt = "holds at " + ", ".join(f"{x:g}" for x in hold) + f" {unit} (NOT contiguous)"
        if v["flips"]:
            txt += "; flip point(s) " + ", ".join(f"between {f['between_cm'][0]:g} and {f['between_cm'][1]:g}" for f in v["flips"])
        if "stable_inside_measured_ci" in v:
            txt += ("; STABLE across the measured interval" if v["stable_inside_measured_ci"]
                    else "; FLICKERS inside the measured interval -- a conclusion about the measurement error")
        lines.append(f"  {DISPLAY_NAMES.get(c, c).strip():36s} {txt}")
    return "\n".join(lines)


def figure(res: Dict[str, Any], out: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 160, "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    param = res["param"]
    colors = {CONDITION_MEMORYLESS: "#c0392b", CONDITION_ALWAYS_COUNTER: "#e67e22",
              CONDITION_CARRYOVER_AWARE: "#2c6fbb", CONDITION_RECOMMENDED: "#1b7f5a",
              CONDITION_IDENTIFICATION_FIRST: "#8e44ad"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1.0, 1.0]})
    for ax, conds in ((axes[0], (CONDITION_MEMORYLESS,)),
                      (axes[1], (CONDITION_ALWAYS_COUNTER, CONDITION_RECOMMENDED, CONDITION_CARRYOVER_AWARE,
                                 CONDITION_IDENTIFICATION_FIRST))):
        for c in conds:
            rows = [r for r in res["rows"] if c in r["contrasts"]]
            if not rows:
                continue
            x = np.array([r["value_cm"] for r in rows])
            d = np.array([r["contrasts"][c]["delta"] for r in rows])
            lo = np.array([r["contrasts"][c]["lo"] for r in rows])
            hi = np.array([r["contrasts"][c]["hi"] for r in rows])
            ax.fill_between(x, lo, hi, color=colors.get(c, "#999"), alpha=0.15, lw=0)
            ax.plot(x, d, "-o", ms=3.5, color=colors.get(c, "#999"), label=DISPLAY_NAMES[c].strip())
            for f in res["flip"][c]["flips"]:
                ax.axvline(0.5 * sum(f["between_cm"]), color=colors.get(c, "#999"), lw=0.8, ls=":")
        ax.axhline(0.0, color="#333", lw=0.8)
        m = res.get("measured") or {}
        if m:
            if m.get("ci_cm"):
                ax.axvspan(m["ci_cm"][0], m["ci_cm"][1], color="#000", alpha=0.06, lw=0)
            ax.axvline(m["point_cm"], color="#000", lw=1.0, ls="--")
        ax.set_xlabel("assumed transition width $w$ (cm)" if param == "w" else "assumed crossover $m^*$ (cm)")
        ax.set_ylabel(f"paired $\\Delta$MAE$_\\times$ vs. {DISPLAY_NAMES[res['comparator']].split()[0]}")
        ax.legend(fontsize=6.5, frameon=False)
    axes[0].set_title("The memoryless contrast", loc="left", fontsize=9)
    axes[1].set_title("The remedy contrasts", loc="left", fontsize=9)
    axes[1].annotate("dotted: flip points · dashed: measured value · grey band: its 95% CI",
                     xy=(0.0, 1.02), xycoords="axes fraction", fontsize=6.5, color="#555")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--param", choices=["w", "mstar"], default="w")
    r.add_argument("--values", type=float, nargs="+", required=True, help="in centimetres")
    r.add_argument("--supervisors", type=int, default=80)
    r.add_argument("--seed", type=int, default=20260822)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed through to run_study")
    a = sub.add_parser("analyze")
    a.add_argument("root", type=Path)
    a.add_argument("--param", choices=["w", "mstar"], default=None)
    a.add_argument("--physics-report", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "run":
        root = args.out or Path(f"vla_lab/results/flip_{args.param}")
        run_sweep(root, param=args.param, values=args.values, supervisors=args.supervisors,
                  seed=args.seed, jobs=args.jobs, extra=args.extra)
        res = analyze(root, param=args.param)
    else:
        root = Path(args.root)
        param = args.param or ("w" if "flip_w" in root.name else "mstar")
        res = analyze(root, param=param, physics_report=args.physics_report)
    text = render(res)
    print(text)
    (root / "table.txt").write_text(text + "\n")
    (root / "flip.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    p = figure(res, root / "fig_flip.pdf")
    if p:
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

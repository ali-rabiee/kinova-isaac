"""Outcomes and figures. Every plot in the paper is produced here, in the order it is reported.

The order is part of the commitment, not a stylistic choice:

1. **the test--retest floor** first, because it bounds the interpretability of everything below
   it -- a condition difference smaller than the floor is inside measurement noise and the
   figure says so with a shaded band rather than leaving the reader to work it out;
2. **does compliance carryover exist and does it decay** (H1, H2), because if it does not, the
   scheduling comparison is unanswerable as posed;
3. **the primary contrast** and its calibration;
4. **the error--burden frontier**, which is where the proposed policy's actual claim lives;
5. **heterogeneity**, plotted per supervisor rather than only as a mean, because "people differ"
   is the premise and a mean effect does not test it.

Matplotlib is imported lazily: the study runs and the numbers are written whether or not a
plotting backend exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scheduler import ABLATIONS, COMPARED_CONDITIONS, DISPLAY_NAMES, PRIMARY_COMPARATOR

_ORDER = list(COMPARED_CONDITIONS) + list(ABLATIONS)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 160, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    return plt


def _short(cond: str) -> str:
    return DISPLAY_NAMES.get(cond, cond).strip().split(" ", 1)[0] if cond.startswith("B") else (
        DISPLAY_NAMES.get(cond, cond).strip().replace("ablation: ", "abl. ")
    )


def fig_primary(summary: Dict[str, Any], conditions: Sequence[str], out: Path) -> Path:
    """Crossover-weighted MAE by condition, with the test--retest floor as a shaded band."""
    plt = _mpl()
    conds = [c for c in conditions if c in summary["conditions"]]
    means = [summary["conditions"][c]["mae_crossover"]["mean"] for c in conds]
    los = [summary["conditions"][c]["mae_crossover"]["lo"] for c in conds]
    his = [summary["conditions"][c]["mae_crossover"]["hi"] for c in conds]
    floor = summary.get("test_retest", {}).get("mae_crossover", {}).get("mean")

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(conds))
    err = np.array([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    colors = ["#c0392b" if c == "memoryless" else ("#2c6fbb" if c == "carryover_aware" else "#7f8c8d") for c in conds]
    ax.barh(y, means, xerr=err, color=colors, height=0.62, error_kw={"lw": 0.9, "capsize": 2})
    if floor is not None and np.isfinite(floor):
        ax.axvspan(0, floor, color="#000000", alpha=0.06, zorder=0)
        ax.axvline(floor, color="#444", lw=0.8, ls="--")
        ax.text(floor, len(conds) - 0.35, "  test–retest floor", fontsize=7, va="top", color="#444")
    ax.set_yticks(y, [DISPLAY_NAMES.get(c, c).strip() for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel("crossover-weighted MAE of $\\hat\\pi$ against the reference map (lower is better)")
    ax.set_title(f"Primary outcome  (N={summary['n_supervisors']} supervisors, paired)", loc="left", fontsize=8.5)
    fig.tight_layout()
    p = out / "fig_primary.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def fig_frontier(summary: Dict[str, Any], out: Path) -> Path:
    """The claim that actually survives: error and regret against interaction burden.

    Most conditions sit at zero counter-proposals, so labels are staggered by rank rather than
    placed at a fixed offset -- at a fixed offset they overprint each other and the panel stops
    carrying the comparison it exists to carry.
    """
    plt = _mpl()
    par = summary.get("pareto", {})
    if not par:
        return out / "fig_frontier.pdf"
    floor = summary.get("test_retest", {}).get("mae_crossover", {}).get("mean")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ykey, ylab in (
        (axes[0], "mae_crossover", "crossover-weighted MAE"),
        (axes[1], "deployment_regret", "deployment regret (task value)"),
    ):
        items = sorted(par.items(), key=lambda kv: kv[1][ykey])
        ys = [v[ykey] for _, v in items]
        span = (max(ys) - min(ys)) or 1.0
        for rank, (c, v) in enumerate(items):
            hl = c in ("carryover_aware", "always_counter", "memoryless")
            col = ("#2c6fbb" if c == "carryover_aware"
                   else "#c0392b" if c == "memoryless"
                   else "#e67e22" if c == "always_counter" else "#8d9499")
            ax.scatter(v["counters"], v[ykey], s=58 if hl else 26, color=col,
                       zorder=4 if hl else 3, edgecolor="white", linewidth=0.7)
            # Stagger vertically by rank so the cluster at zero burden stays readable. Two
            # offsets are not enough -- six of the eight conditions sit at zero burden.
            dy = (9, -13, 21, -25)[rank % 4]
            ax.annotate(_short(c), (v["counters"], v[ykey]), fontsize=6.2,
                        xytext=(7, dy), textcoords="offset points",
                        color=col if hl else "#555",
                        fontweight="bold" if hl else "normal")
        if ykey == "mae_crossover" and floor is not None and np.isfinite(floor):
            ax.axhspan(min(ys) - 0.02 * span, floor, color="#000000", alpha=0.05, zorder=0)
            ax.axhline(floor, color="#444", lw=0.8, ls="--")
            ax.text(56, floor, "test–retest floor ", fontsize=6, va="bottom",
                    ha="right", color="#444")
        ax.set_xlabel("counter-proposals per block (interaction burden)")
        ax.set_ylabel(ylab)
        ax.set_xlim(-4, 58)
        ax.margins(y=0.18)
    axes[0].set_title("Error vs. burden", loc="left", fontsize=8.5)
    axes[1].set_title("Regret vs. burden", loc="left", fontsize=8.5)
    fig.tight_layout()
    p = out / "fig_frontier.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def fig_heterogeneity(rows: Sequence[Dict[str, Any]], out: Path,
                      cond: str = "carryover_aware", ref: str = PRIMARY_COMPARATOR) -> Path:
    """Per-supervisor effect against how persuadable that supervisor actually is.

    The premise of the whole programme is that a single fixed rule cannot serve everyone. A
    mean effect does not test that; this does. Each point is one supervisor.
    """
    plt = _mpl()
    xs, ys, sizes = [], [], []
    for r in rows:
        a = r["conditions"].get(cond, {}).get("mae_crossover")
        b = r["conditions"].get(ref, {}).get("mae_crossover")
        if a is None or b is None:
            continue
        xs.append(float(r["params"]["beta"]) * float(r["params"]["g"]))
        ys.append(float(a) - float(b))
        sizes.append(8 + 40 * float(r["params"]["lam"]))
    if not xs:
        return out / "fig_heterogeneity.pdf"
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.axhline(0.0, color="#444", lw=0.8)
    ax.scatter(xs, ys, s=sizes, alpha=0.65, color="#2c6fbb", edgecolor="white", linewidth=0.5)
    if len(xs) > 3:
        z = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 50)
        ax.plot(gx, np.polyval(z, gx), color="#c0392b", lw=1.2)
    ax.set_xlabel(r"true compliance strength  $\beta_p g_p$  (logits per demonstration)")
    ax.set_ylabel(f"{_short(cond)} $-$ {_short(ref)}\ncrossover MAE (negative = better)")
    ax.set_title("Per-supervisor effect (marker size = decay $\\lambda_p$)", loc="left", fontsize=8.5)
    fig.tight_layout()
    p = out / "fig_heterogeneity.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def fig_carryover(rows: Sequence[Dict[str, Any]], out: Path) -> Path:
    """H1/H2: recovery of the compliance parameters, and their spread across people."""
    plt = _mpl()
    true_bg, est_bg, true_lam, est_lam = [], [], [], []
    for r in rows:
        jc = None
        for c in r.get("conditions", {}).values():
            if c.get("joint_carryover"):
                jc = c["joint_carryover"]
                break
        if not jc:
            continue
        true_bg.append(float(r["params"]["beta"]) * float(r["params"]["g"]))
        est_bg.append(float(jc["mean"]["beta_g"]))
        true_lam.append(float(r["params"]["lam"]))
        est_lam.append(float(jc["mean"]["lambda"]))
    if not true_bg:
        return out / "fig_carryover.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, t, e, lab in ((axes[0], true_bg, est_bg, r"$\beta_p g_p$"), (axes[1], true_lam, est_lam, r"$\lambda_p$")):
        lo, hi = min(min(t), min(e)), max(max(t), max(e))
        ax.plot([lo, hi], [lo, hi], color="#444", lw=0.8, ls="--")
        ax.scatter(t, e, s=16, alpha=0.7, color="#2c6fbb", edgecolor="white", linewidth=0.4)
        from ..stats_utils import guarded_correlation

        g = guarded_correlation(t, e, method="pearson")
        ax.set_xlabel(f"true {lab}")
        ax.set_ylabel(f"recovered {lab}")
        ax.set_title((f"r = {g['rho']:.2f} (n = {g['n']})" if g["reported"] else f"n = {g['n']}: r withheld"),
                     loc="left", fontsize=8)
    axes[2].hist(true_bg, bins=12, color="#7f8c8d", alpha=0.85)
    axes[2].set_xlabel(r"true $\beta_p g_p$ across the population")
    axes[2].set_ylabel("supervisors")
    axes[2].set_title("Heterogeneity", loc="left", fontsize=8)
    fig.tight_layout()
    p = out / "fig_carryover.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def fig_scene_model(contract_dict: Dict[str, Any], out: Path) -> Path:
    """The task-value model that defines the ambiguity coordinate. The paper's Figure 2."""
    plt = _mpl()
    from .scenes import SceneGrid

    grid = SceneGrid.from_dict(contract_dict["grid"])
    phys = grid.physics
    m = np.linspace(0.0, 0.18, 240)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))
    axes[0].plot(m * 100, [phys.p_success("A", x) for x in m], color="#2c6fbb", label="CLEAR_FIRST (A)")
    axes[0].plot(m * 100, [phys.p_success("B", x) for x in m], color="#c0392b", label="DIRECT (B)")
    axes[0].set_ylabel("P(success)")
    axes[0].legend(fontsize=6.5, frameon=False)
    axes[1].plot(m * 100, [phys.value("A", x) for x in m], color="#2c6fbb")
    axes[1].plot(m * 100, [phys.value("B", x) for x in m], color="#c0392b")
    axes[1].axvline(phys.crossover_margin() * 100, color="#444", lw=0.8, ls="--")
    axes[1].set_ylabel("expected task value")
    axes[2].plot(m * 100, [phys.coordinate(x) for x in m], color="#333")
    axes[2].axhline(0.0, color="#444", lw=0.8, ls="--")
    band = grid.crossover_halfwidth
    axes[2].axhspan(-band, band, color="#2c6fbb", alpha=0.12)
    for s in grid.probe_scenes():
        axes[2].scatter([s.margin_m * 100], [s.c], s=10, color="#2c6fbb", zorder=3)
    for s in grid.coach_scenes():
        axes[2].scatter([s.margin_m * 100], [s.c], s=14, marker="s", color="#c0392b", zorder=3)
    axes[2].set_ylabel("ambiguity coordinate $c$")
    for ax in axes:
        ax.set_xlabel("clearance gap (cm)")
    src = phys.source
    axes[0].set_title(f"physics: {src}" + (f" (n={phys.n_measured})" if src == "measured" else " — NOT YET MEASURED"),
                      loc="left", fontsize=8, color=("#2c6fbb" if src == "measured" else "#c0392b"))
    fig.tight_layout()
    p = out / "fig_scene_model.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def fig_identification(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], out: Path) -> Path:
    """B6's claim in one figure. Left: per-condition identification rate of the decay rate, with
    Wilson intervals. Right: recovered against true lambda per supervisor, under the natural
    schedule (B5) and the identification-first one (B6)."""
    plt = _mpl()
    ident = summary.get("identification") or {}
    conds = [c for c in ("fixed_washout", "carryover_aware", "recommended", "ablation_b6_fixed_scenes",
                         "identification_first") if c in ident]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = np.arange(len(conds))
    vals = [ident[c]["lambda"] for c in conds]
    err = np.array([[ident[c]["lambda"] - ident[c]["lambda_ci"][0] for c in conds],
                    [ident[c]["lambda_ci"][1] - ident[c]["lambda"] for c in conds]])
    tv = [ident[c].get("lambda_tv", np.nan) for c in conds]
    tv_nc = [ident[c].get("lambda_tv_noncomplier_rate") for c in conds]
    cols = ["#8e44ad" if c.startswith("identification") or c.startswith("ablation_b6") else "#7f8c8d" for c in conds]
    ax.barh(y - 0.18, vals, xerr=err, color=cols, height=0.34, error_kw={"lw": 0.9, "capsize": 2},
            label="posterior contraction (honest)")
    ax.barh(y + 0.18, tv, color=cols, height=0.34, alpha=0.35, label="TV criterion (first draft)")
    for yi, (t, nc) in enumerate(zip(tv, tv_nc)):
        if nc is not None and np.isfinite(t):
            ax.plot([nc], [yi + 0.18], marker="x", color="#c0392b", ms=5,
                    label=r"TV rate among $\beta{=}0$" if yi == 0 else None)
    ax.set_yticks(y, [DISPLAY_NAMES.get(c, c).strip() for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"fraction of supervisors with $\lambda$ 'identified'")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_title("Identification of the decay rate, by schedule and by criterion", loc="left", fontsize=8.5)
    for a, b, col, lab in (("carryover_aware", "B5 (natural schedule)", "#7f8c8d", "B5"),
                           ("identification_first", "B6 (identification-first)", "#8e44ad", "B6")):
        t, e = [], []
        for r in rows:
            jc = r["conditions"].get(a, {}).get("joint_carryover")
            if jc:
                t.append(float(r["params"]["lam"]))
                e.append(float(jc["mean"]["lambda"]))
        if t:
            bx.scatter(t, e, s=14, alpha=0.65, color=col, edgecolor="white", linewidth=0.4, label=b)
    bx.plot([0, 1], [0, 1], color="#444", lw=0.8, ls="--")
    bx.set_xlabel(r"true $\lambda_p$")
    bx.set_ylabel(r"recovered $\lambda_p$ (posterior mean)")
    bx.set_title("Recovery per supervisor", loc="left", fontsize=8.5)
    bx.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    p = out / "fig_identification.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def audit_study(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Study-level provenance and internal-consistency checks.

    The per-session gate (:mod:`vla_lab.supervisory.verify_session`) checks one session against
    the contract. This checks the *study*: whether the thing every number is defined in terms of
    was measured or assumed, whether the budget was matched in the realized data rather than
    only in the protocol, whether enough answers grounded to build an estimand from, and whether
    the parameters the proposed policy personalises on were actually identified. The last is the
    one that changes how a result should be read: a policy that personalises on an unidentified
    parameter is personalising on its prior, and a table that does not say so invites the reader
    to attribute a null to the method rather than to the design.
    """
    flags: List[str] = []
    checks: Dict[str, Any] = {}
    contract = summary.get("contract", {})

    phys = (contract.get("grid") or {}).get("physics", {})
    checks["physics_source"] = phys.get("source")
    checks["physics_n_measured"] = phys.get("n_measured", 0)
    if phys.get("source") != "measured":
        flags.append("scene physics is a PRIOR, not a measurement: every scene coordinate, and "
                     "therefore every regret number, is 'under the assumed physics'")
    checks["physics_fit_method"] = phys.get("fit_method")
    checks["physics_quantile"] = phys.get("quantile", "point")
    checks["crossover_margin_m"] = None
    checks["transition_width_m"] = None
    try:
        from .scenes import ScenePhysics

        sp = ScenePhysics.from_dict(phys)
        checks["crossover_margin_m"] = float(sp.crossover_margin())
        checks["transition_width_m"] = float(sp.transition_width_m())
    except Exception:                                           # pragma: no cover
        pass
    if str(phys.get("quantile", "point")) != "point":
        flags.append(f"scene physics is the '{phys.get('quantile')}' draw, NOT the point estimate: this run "
                     "exists to bound a contrast under physics uncertainty and must not be pooled with, "
                     "or quoted in place of, the point-estimate study")

    conds = summary.get("conditions_run") or []
    slots = {c: summary["conditions"][c] for c in conds if c in summary.get("conditions", {})}
    realized = {c: (v["n_probe"]["mean"] + v["n_counter"]["mean"] + v["n_wait"]["mean"])
                for c, v in slots.items()}
    checks["realized_free_slots"] = realized
    if realized and (max(realized.values()) - min(realized.values())) > 0.5:
        flags.append(f"free-slot counts differ across conditions by "
                     f"{max(realized.values()) - min(realized.values()):.1f}: the budget is not matched")

    ung = {c: v["n_ungrounded"]["mean"] for c, v in slots.items()}
    checks["ungrounded_per_block"] = ung
    obs = {c: realized.get(c, 0) - v["n_wait"]["mean"] for c, v in slots.items()}
    worst = max(((ung[c] / max(obs.get(c, 1), 1)) for c in ung), default=0.0)
    checks["worst_ungrounded_fraction"] = float(worst)
    if worst > 0.15:
        flags.append(f"up to {worst*100:.0f}% of answers could not be grounded")

    ident = {"lambda": 0, "beta_g": 0, "n": 0}
    for r in rows:
        for cell in r.get("conditions", {}).values():
            jc = cell.get("joint_carryover")
            if jc and jc.get("identifiability"):
                ident["n"] += 1
                for k in ("lambda", "beta_g"):
                    crit = jc["identifiability"].get(k, {})
                    ident[k] += int(crit.get("identified_sd", crit.get("identified", False)))
                break
    if ident["n"]:
        checks["identified_fraction"] = {k: ident[k] / ident["n"] for k in ("lambda", "beta_g")}
        if checks["identified_fraction"]["lambda"] < 0.5:
            flags.append(f"lambda identified in only {checks['identified_fraction']['lambda']*100:.0f}% "
                         "of supervisors: schedule personalisation is running on its prior, and a null "
                         "on the scheduling contrast says so rather than saying the mechanism is absent")

    floor = summary.get("test_retest", {}).get("mae_crossover", {}).get("mean")
    checks["test_retest_floor"] = floor
    if floor is not None and np.isfinite(floor):
        spread = [summary["conditions"][c]["mae_crossover"]["mean"] for c in conds
                  if c in summary.get("conditions", {}) and c != "memoryless"]
        if spread and (max(spread) - min(spread)) < floor:
            flags.append(f"the spread among non-memoryless conditions ({max(spread)-min(spread):.4f}) is "
                         f"below the test-retest floor ({floor:.4f}): those differences are inside "
                         "measurement noise and must not be interpreted")

    checks["prior"] = summary.get("prior")
    checks["n_supervisors"] = summary.get("n_supervisors")
    pop = summary.get("population", {})
    checks["n_noncompliers"] = pop.get("n_noncompliers")
    return {"checks": checks, "flags": flags}


def render_audit(audit: Dict[str, Any]) -> str:
    c = audit["checks"]
    lines = ["", "study audit", "-" * 60,
             f"  supervisors      : {c.get('n_supervisors')} "
             f"({c.get('n_noncompliers')} non-compliers)   prior: {c.get('prior')}",
             f"  scene physics    : {c.get('physics_source')} (n={c.get('physics_n_measured')}, "
             f"fit={c.get('physics_fit_method')}, draw={c.get('physics_quantile')}, "
             f"m*={(c.get('crossover_margin_m') or 0) * 100:.2f} cm, w={(c.get('transition_width_m') or 0) * 100:.2f} cm)",
             f"  test-retest floor: {c.get('test_retest_floor')}",
             f"  ungrounded (max) : {c.get('worst_ungrounded_fraction', 0)*100:.1f}%"]
    idf = c.get("identified_fraction")
    if idf:
        lines.append(f"  identified       : lambda {idf['lambda']*100:.0f}%   beta*g {idf['beta_g']*100:.0f}%")
    for f in audit["flags"]:
        lines.append(f"  [!] {f}")
    if not audit["flags"]:
        lines.append("  [ok] no study-level flags")
    return "\n".join(lines)


def write_figures(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], out: Path) -> List[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    conds = summary.get("conditions_run") or _ORDER
    made: List[Path] = []
    for fn in (
        lambda: fig_primary(summary, conds, out),
        lambda: fig_frontier(summary, out),
        lambda: fig_heterogeneity(rows, out),
        lambda: fig_carryover(rows, out),
        lambda: fig_identification(summary, rows, out) if summary.get("identification") else None,
        lambda: fig_scene_model(summary["contract"], out) if "contract" in summary else None,
    ):
        try:
            p = fn()
            if p is not None:
                made.append(p)
        except ImportError:
            print("[analyze] matplotlib not available; numbers were written, figures were not")
            break
        except Exception as exc:  # pragma: no cover - a bad figure must not lose the numbers
            print(f"[analyze] figure failed: {exc}")
    return made


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="directory written by run_study")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    summary = json.loads((args.results / "summary.json").read_text())
    rows = json.loads((args.results / "per_supervisor.json").read_text())
    out = args.out or (args.results / "figures")
    made = write_figures(summary, rows, out)
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

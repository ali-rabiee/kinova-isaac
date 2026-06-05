"""CLI: turn a study log (JSONL from :class:`vla_lab.human_study.session.StudyLogger`) into
per-condition reliance / trust / workload tables, paired McNemar tests, and figures.

    python -m vla_lab.human_study.analyze \
        --log vla_lab/study_runs/*/study.jsonl \
        --out-dir vla_lab/results/human_study --format pdf

The headline outputs are the **over-trust trap** comparison across the 5 interaction
conditions and the appropriate-reliance table (memo §5.4): does the type-aware allocator
reduce over-trust and improve appropriate reliance vs. autonomy / fixed-compute / gated /
ask-for-help baselines.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..stats_utils import mcnemar_test
from .reliance import RelianceTrial, reliance_summary, trust_calibration


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _load(paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trials: List[Dict[str, Any]] = []
    quest: List[Dict[str, Any]] = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            fp = Path(p)
            if not fp.exists():
                continue
            for line in fp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "trial":
                    trials.append(row)
                elif row.get("kind") == "questionnaire":
                    quest.append(row)
    return trials, quest


def _reliance_trial(row: Dict[str, Any]) -> RelianceTrial:
    res = row.get("result", {})
    return RelianceTrial(
        robot_correct=bool(res.get("robot_correct", False)),
        human_intervened=bool(res.get("human_intervened", False)),
        success=bool(res.get("success", False)),
        robot_queried=bool(res.get("robot_queried", False)),
        robot_signaled_uncertainty=bool(res.get("robot_signaled_uncertainty", False)),
        condition_name=str(row.get("condition_name", "")),
        participant_id=str(row.get("participant_id", "")),
        occlusion_fraction=float(row.get("occlusion_fraction", 0.0)),
    )


def _pair_key(row: Dict[str, Any]) -> Tuple:
    return (row.get("participant_id"), row.get("occlusion_mode"), row.get("occlusion_fraction"), row.get("rep", 0))


def paired_success(trials: List[Dict[str, Any]], cond_a: str, cond_b: str) -> Tuple[List[bool], List[bool]]:
    """Align success bits between two conditions on matching (participant, occlusion, rep)."""

    a = {_pair_key(r): bool(r.get("result", {}).get("success", False)) for r in trials if r.get("condition_name") == cond_a}
    b = {_pair_key(r): bool(r.get("result", {}).get("success", False)) for r in trials if r.get("condition_name") == cond_b}
    keys = sorted(set(a) & set(b), key=lambda k: tuple(str(x) for x in k))
    return [a[k] for k in keys], [b[k] for k in keys]


def _condition_questionnaires(quest: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for q in quest:
        cond = str(q.get("condition_name", ""))
        for k, v in (q.get("scores", {}) or {}).items():
            try:
                acc[cond][f"{q.get('instrument')}.{k}"].append(float(v))
            except (TypeError, ValueError):
                continue
    return {cond: {k: (sum(vs) / len(vs)) for k, vs in d.items()} for cond, d in acc.items()}


def analyze(trials: List[Dict[str, Any]], quest: List[Dict[str, Any]], *, reference: str = "allocator_type") -> Dict[str, Any]:
    by_cond: Dict[str, List[RelianceTrial]] = defaultdict(list)
    cond_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in trials:
        by_cond[str(r.get("condition_name", ""))].append(_reliance_trial(r))
        cond_rows[str(r.get("condition_name", ""))].append(r)

    per_condition: Dict[str, Any] = {}
    for cond, rts in sorted(by_cond.items()):
        summ = reliance_summary(rts)
        cal = trust_calibration(rts)
        n = summ.get("n", 0)
        ns = int(round(summ.get("success_rate", 0.0) * n))
        lo, hi = _wilson_ci(ns, n)
        per_condition[cond] = {
            "reliance": summ,
            "trust_calibration": cal,
            "success_wilson95": [lo, hi],
        }

    questionnaire_means = _condition_questionnaires(quest)
    for cond, qs in questionnaire_means.items():
        per_condition.setdefault(cond, {})["questionnaires"] = qs

    # Paired McNemar of success vs the reference condition.
    mcnemar: Dict[str, Any] = {}
    conditions = sorted(by_cond)
    if reference in conditions:
        for cond in conditions:
            if cond == reference:
                continue
            sa, sb = paired_success(trials, reference, cond)
            res = mcnemar_test(sa, sb)
            if res is not None:
                chi2, pval, ndisc = res
                mcnemar[f"{reference}_vs_{cond}"] = {
                    "n_pairs": len(sa),
                    "chi2": chi2,
                    "p_value": pval,
                    "n_discordant": ndisc,
                    "success_rate_ref": (sum(sa) / len(sa)) if sa else float("nan"),
                    "success_rate_other": (sum(sb) / len(sb)) if sb else float("nan"),
                }

    return {
        "n_trials": len(trials),
        "n_participants": len({r.get("participant_id") for r in trials}),
        "conditions": conditions,
        "reference_condition": reference,
        "per_condition": per_condition,
        "mcnemar_vs_reference": mcnemar,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_reliance_by_condition(per_condition: Dict[str, Any], out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    conds = sorted(per_condition)
    metrics = [("appropriate_reliance_rate", "#2ca02c"), ("over_trust_rate", "#d62728"), ("under_trust_rate", "#ff7f0e")]
    x = range(len(conds))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(conds)), 4))
    for i, (m, color) in enumerate(metrics):
        vals = [per_condition[c].get("reliance", {}).get(m, float("nan")) for c in conds]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=m.replace("_", " "), color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reliance by interaction condition")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"reliance_by_condition.{fmt}", dpi=150)
    plt.close(fig)


def _plot_overtrust_vs_occlusion(trials: List[Dict[str, Any]], out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    by: Dict[str, Dict[float, List[RelianceTrial]]] = defaultdict(lambda: defaultdict(list))
    for r in trials:
        by[str(r.get("condition_name", ""))][float(r.get("occlusion_fraction", 0.0))].append(_reliance_trial(r))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cond in sorted(by):
        occs = sorted(by[cond])
        ys = [reliance_summary(by[cond][o]).get("over_trust_rate", float("nan")) for o in occs]
        ax.plot(occs, ys, "o-", label=cond)
    ax.set_xlabel("Occlusion fraction")
    ax.set_ylabel("Over-trust rate")
    ax.set_title("The over-trust trap vs. observability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"overtrust_vs_occlusion.{fmt}", dpi=150)
    plt.close(fig)


def _plot_trust_workload(per_condition: Dict[str, Any], out_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    conds = sorted(per_condition)
    trust = [per_condition[c].get("questionnaires", {}).get("jian_trust.overall", float("nan")) for c in conds]
    tlx = [per_condition[c].get("questionnaires", {}).get("nasa_tlx.raw_tlx", float("nan")) for c in conds]
    if not any(t == t for t in trust) and not any(t == t for t in tlx):
        return
    fig, ax1 = plt.subplots(figsize=(max(6, 1.4 * len(conds)), 4))
    x = range(len(conds))
    ax1.bar([xi - 0.2 for xi in x], trust, 0.4, color="#1f77b4", label="Jian trust (1–7)", edgecolor="black", linewidth=0.4)
    ax1.set_ylabel("Trust (Jian overall)", color="#1f77b4")
    ax1.set_ylim(0, 7)
    ax2 = ax1.twinx()
    ax2.bar([xi + 0.2 for xi in x], tlx, 0.4, color="#d62728", label="Raw TLX (0–100)", edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("Workload (Raw TLX)", color="#d62728")
    ax2.set_ylim(0, 100)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(conds, rotation=20, ha="right")
    ax1.set_title("Trust and workload by condition")
    fig.tight_layout()
    fig.savefig(out_dir / f"trust_workload_by_condition.{fmt}", dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Human-study analysis")
    ap.add_argument("--log", action="append", default=[], help="study JSONL path or glob (repeatable)")
    ap.add_argument("--out-dir", type=str, default="vla_lab/results/human_study")
    ap.add_argument("--reference", type=str, default="allocator_type", help="condition to McNemar everything against")
    ap.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.log:
        print("[human_study.analyze] no --log paths given.")
        return 2
    trials, quest = _load(args.log)
    if not trials:
        print("[human_study.analyze] no trial records found.")
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze(trials, quest, reference=args.reference)
    (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[human_study.analyze] {summary['n_trials']} trials, {summary['n_participants']} participants, "
          f"conditions={summary['conditions']}")
    for cond in summary["conditions"]:
        rel = summary["per_condition"].get(cond, {}).get("reliance", {})
        print(f"  {cond:16s} appropriate={rel.get('appropriate_reliance_rate', float('nan')):.3f} "
              f"over_trust={rel.get('over_trust_rate', float('nan')):.3f} success={rel.get('success_rate', float('nan')):.3f}")
    for k, v in summary.get("mcnemar_vs_reference", {}).items():
        print(f"  McNemar {k}: p={v['p_value']:.4f} (n_pairs={v['n_pairs']}, discordant={v['n_discordant']})")
    print(f"[human_study.analyze] wrote {out_dir / 'study_summary.json'}")

    if not args.no_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("[human_study.analyze] matplotlib not installed; skipping figures.")
            return 0
        _plot_reliance_by_condition(summary["per_condition"], out_dir, args.format)
        _plot_overtrust_vs_occlusion(trials, out_dir, args.format)
        _plot_trust_workload(summary["per_condition"], out_dir, args.format)
        print(f"[human_study.analyze] wrote figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

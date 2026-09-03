"""Seed dispersion for the model-level tables, and the seed floor every ordering is read against.

The study half of the paper refuses to interpret any difference smaller than its test--retest
floor. The model half reported single runs. This module is the same discipline applied to the
architecture and objective sweeps: every cell is run under several seeds, every reported number
carries its seed standard deviation, and the table carries a **seed floor** -- the spread
attributable to nothing but re-running -- computed the way the test--retest floor is: as the
mean absolute difference between two runs of the *same* cell.

The seed controls everything that is random in a cell: model initialisation, data-loader
shuffling, the dialogue-generation RNG, and the supervisor draw for the training cohort (all
through ``--seed`` in :mod:`vla_lab.training.train`, which seeds ``torch``, ``random`` and
``numpy`` and passes the seed to the generator and the dataset). What it cannot control is
non-deterministic CUDA kernels; that variation is part of what the floor measures.

An ordering between two cells "clears the floor" when the absolute difference of their seed
means exceeds the pooled floor for that metric. Cells at equal seed share their training data
(the dialogues are a deterministic function of the seed), so the per-seed differences are also
reported as paired contrasts with their own seed standard deviation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

METRICS = ("debias_gain_brier", "debias_kappa_corr", "ask_rank_corr", "debias_gap", "acc_said")
HEADLINE = ("debias_gain_brier", "debias_kappa_corr")


def _cell_key(r: Dict[str, Any], group_by: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(r.get(k)) for k in group_by)


def seed_floor(rows: Sequence[Dict[str, Any]], metric: str, *, group_by: Sequence[str]) -> Dict[str, Any]:
    """Mean |difference| between two seeds of the same cell, pooled over cells.

    Also reports the 95th percentile of those pairwise differences and the pooled within-cell
    standard deviation. With ``n`` seeds per cell there are ``n(n-1)/2`` pairs per cell.
    """
    diffs: List[float] = []
    sds: List[float] = []
    cells: Dict[Tuple[str, ...], List[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v is None or r.get("skipped") or not np.isfinite(float(v)):
            continue
        cells.setdefault(_cell_key(r, group_by), []).append(float(v))
    for vals in cells.values():
        if len(vals) < 2:
            continue
        diffs.extend(abs(a - b) for a, b in itertools.combinations(vals, 2))
        sds.append(float(np.std(vals, ddof=1)))
    if not diffs:
        return {"metric": metric, "floor": None, "p95": None, "pooled_sd": None, "n_pairs": 0,
                "n_cells_with_seeds": 0}
    return {
        "metric": metric,
        "floor": float(np.mean(diffs)),
        "p95": float(np.percentile(diffs, 95)),
        "pooled_sd": float(np.sqrt(np.mean(np.square(sds)))) if sds else None,
        "n_pairs": len(diffs),
        "n_cells_with_seeds": len([v for v in cells.values() if len(v) >= 2]),
    }


def aggregate_seeds(rows: Sequence[Dict[str, Any]], *, group_by: Sequence[str] = ("model", "context"),
                    metrics: Sequence[str] = METRICS) -> Dict[str, Any]:
    """Per-cell mean +- seed SD, the seed floor per metric, and paired within-backbone contrasts."""
    rows = [r for r in rows if not r.get("skipped")]
    cells: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault(_cell_key(r, group_by), []).append(r)

    out_cells: List[Dict[str, Any]] = []
    for key, rs in cells.items():
        first = rs[0]
        cell: Dict[str, Any] = {k: first.get(k) for k in ("model", "context", "display", "pretrained", "language",
                                                            "action_head", "params", "params_trainable",
                                                            "params_total", "adapt", "image_source", "ablation")}
        cell["seeds"] = sorted(int(r.get("seed", 0)) for r in rs)
        cell["n_seeds"] = len(rs)
        for m in metrics:
            vals = [float(r[m]) for r in rs if r.get(m) is not None and np.isfinite(float(r[m]))]
            cell[m] = {
                "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "min": float(min(vals)) if vals else None,
                "max": float(max(vals)) if vals else None,
                "values": vals,
            }
        out_cells.append(cell)

    floors = {m: seed_floor(rows, m, group_by=group_by) for m in metrics}

    # Paired contrasts between cells that share every grouping key but the last (e.g. context
    # modes within a backbone), at equal seed.
    contrasts: List[Dict[str, Any]] = []
    by_parent: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = {}
    for key in cells:
        by_parent.setdefault(key[:-1], []).append(key)
    for parent, keys in by_parent.items():
        for a, b in itertools.combinations(sorted(keys), 2):
            ra = {int(r.get("seed", 0)): r for r in cells[a]}
            rb = {int(r.get("seed", 0)): r for r in cells[b]}
            common = sorted(set(ra) & set(rb))
            for m in HEADLINE:
                d = [float(rb[s][m]) - float(ra[s][m]) for s in common
                     if ra[s].get(m) is not None and rb[s].get(m) is not None]
                if not d:
                    continue
                fl = floors[m]["floor"]
                mean_d = float(np.mean(d))
                contrasts.append({
                    "metric": m, "parent": list(parent), "a": a[-1], "b": b[-1],
                    "delta_b_minus_a": mean_d,
                    "sd": float(np.std(d, ddof=1)) if len(d) > 1 else None,
                    "n_seeds": len(d),
                    "seed_floor": fl,
                    "clears_floor": bool(fl is not None and abs(mean_d) > fl),
                    "same_sign_every_seed": bool(all(x > 0 for x in d) or all(x < 0 for x in d)),
                })
    return {"group_by": list(group_by), "cells": out_cells, "seed_floor": floors, "contrasts": contrasts}


def render_seeds(agg: Dict[str, Any]) -> str:
    lines: List[str] = []
    head = (f"{'cell':44s}{'seeds':>6}{'gain(Brier)':>20}{'gap~kappa':>20}{'ask rho':>18}")
    lines += [head, "-" * len(head)]

    def pm(d: Dict[str, Any], nd: int = 3) -> str:
        if d.get("mean") is None:
            return "--"
        return f"{d['mean']:+.{nd}f}" + (f" ± {d['sd']:.{nd}f}" if d.get("sd") is not None else "   (1 seed)")

    for c in agg["cells"]:
        label = f"{c.get('display') or c.get('model')} / {c.get('context')}"
        if c.get("ablation"):
            label += f" / {c['ablation']}"
        lines.append(f"{label:44s}{c['n_seeds']:>6}{pm(c['debias_gain_brier']):>20}"
                     f"{pm(c['debias_kappa_corr']):>20}{pm(c['ask_rank_corr']):>18}")
    lines.append("")
    for m, f in agg["seed_floor"].items():
        if f.get("floor") is not None and m in HEADLINE:
            lines.append(f"seed floor {m}: mean |diff| between seeds of the same cell = {f['floor']:.4f} "
                         f"(p95 {f['p95']:.4f}, pooled sd {f['pooled_sd']:.4f}, {f['n_pairs']} pairs)")
    lines.append("")
    lines.append("paired within-backbone contrasts (b - a, at equal seed):")
    for c in agg["contrasts"]:
        if c["metric"] not in HEADLINE:
            continue
        mark = "CLEARS the seed floor" if c["clears_floor"] else "inside the seed floor"
        sign = " (same sign every seed)" if c["same_sign_every_seed"] else ""
        sd = f" ± {c['sd']:.4f}" if c.get("sd") is not None else ""
        lines.append(f"  {'/'.join(c['parent'])}: {c['b']} - {c['a']}  {c['metric']:18s} "
                     f"{c['delta_b_minus_a']:+.4f}{sd}  [{mark}{sign}]")
    return "\n".join(lines)


def write_seed_tables(rows: Sequence[Dict[str, Any]], out: Path, *, group_by: Sequence[str] = ("model", "context")) -> Dict[str, Any]:
    agg = aggregate_seeds(rows, group_by=group_by)
    out = Path(out)
    (out / "table_seeds.json").write_text(json.dumps(agg, indent=2, default=float) + "\n")
    (out / "table_seeds.txt").write_text(render_seeds(agg) + "\n")
    return agg


__all__ = ["METRICS", "HEADLINE", "seed_floor", "aggregate_seeds", "render_seeds", "write_seed_tables"]

"""Tier-1 study runner: a whole cohort, every condition, in seconds.

Runs the **real** session code path -- protocol, schedulers, grounding, event-locked records,
estimators, analysis -- with the surrogate apparatus behind the apparatus seam and a generative
supervisor behind the human seam. This is the rehearsal that makes every claim in the pipeline
verifiable before a simulator run, a checkpoint, or an IRB protocol exists.

**It is a rehearsal, not evidence about people.** Every number it produces follows from the
population prior in :mod:`vla_lab.supervisory.supervisor`. The paper labels it as such.

One thing this runner does that a human study cannot, and it is worth stating because it is why
the simulated contrasts are so much tighter than the human ones will be: each condition is run
against a **freshly instantiated, identical supervisor** -- same parameters, same protocol, same
scene sequence, residue reset to zero, drift clock reset. Conditions are therefore perfectly
paired with no order effects at all. A human study has to counterbalance instead, which is
strictly weaker, and the human protocol
(:func:`vla_lab.supervisory.protocol.build_protocol`) does exactly that.

    python -m vla_lab.supervisory.run_study --supervisors 32 --seed 20260822 --analyze
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..stats_utils import wilson_ci
from . import COACH, COUNTER, PROBE, WAIT
from .apparatus import LexicalGrounder, SimulatedSupervisorChannel, SurrogateApparatus
from .carryover import CarryoverConfig, CarryoverPosterior, fit_population_prior, fit_rho
from .contract import Contract
from .estimand import (
    METHOD_CORRECTED,
    METHOD_POOLED,
    METHOD_PSYCHOMETRIC,
    CarryoverCorrectedEstimator,
    PooledBetaEstimator,
    PsychometricEstimator,
    evaluate,
    joint_carryover_posterior,
    reference_map_from_observations,
    sequence_from_records,
)
from .narration import grounding_agreement
from .protocol import BLOCK_CONDITION, BLOCK_REFERENCE, BLOCK_RETEST, Block, build_protocol
from .scenes import SceneGrid
from .scheduler import (
    ABLATIONS,
    ALL_CONDITIONS,
    COMPARED_CONDITIONS,
    CONDITION_NO_COACH,
    DISPLAY_NAMES,
    PRIMARY_COMPARATOR,
    build_scheduler,
    estimator_for,
)
from .session import run_block
from .supervisor import SimulatedSupervisor, SupervisorParams, SupervisorPopulation, draw_supervisor


# ---------------------------------------------------------------------------
# One supervisor
# ---------------------------------------------------------------------------
def _fresh(params: SupervisorParams, contract: Contract, seed: int) -> SimulatedSupervisor:
    return SimulatedSupervisor(params, axis=contract.axis, cfg=contract.carryover, seed=seed)


def _fit_reported(condition: str, seq, grid: SceneGrid, cfg: CarryoverConfig, log_prior=None):
    """The estimate the condition actually reports, using its designated estimator."""
    method = estimator_for(condition)
    if method == METHOD_POOLED:
        return PooledBetaEstimator().fit(seq, grid), None
    if method == METHOD_PSYCHOMETRIC:
        return PsychometricEstimator().fit(seq, grid), None
    post, _ = joint_carryover_posterior(seq, grid, cfg=cfg, log_prior=log_prior)
    return CarryoverCorrectedEstimator().fit(seq, grid, post), post


def run_one_supervisor(
    *,
    index: int,
    contract: Contract,
    conditions: Sequence[str],
    seed: int,
    population: Optional[SupervisorPopulation] = None,
    log_root: Optional[Path] = None,
    log_prior=None,
    grounder: Optional[Any] = None,
) -> Dict[str, Any]:
    rng = random.Random(int(seed) * 7907 + index)
    params = draw_supervisor(rng, population, supervisor_id=f"S{index:03d}")
    protocol = build_protocol(
        supervisor_id=params.supervisor_id,
        contract=contract,
        seed=int(seed) + index,
        conditions=list(conditions),
        order_index=index,
    )
    grounder = grounder if grounder is not None else LexicalGrounder(contract.axis)
    truth_map = _fresh(params, contract, seed + index).pi_star_map(contract.grid)

    def _run(block: Block, condition: str, tag: str):
        sup = _fresh(params, contract, seed + index)
        sch = build_scheduler(
            condition,
            contract.grid,
            carryover_cfg=contract.carryover,
            delta_model=contract.delta_model(),
            seed=int(seed) + index,
            log_prior=log_prior,
        )
        root = Path(log_root) / params.supervisor_id / tag if log_root else None
        logger = None
        if root is not None:
            from .logging import SessionLogger

            root.mkdir(parents=True, exist_ok=True)
            logger = SessionLogger(root, supervisor_id=params.supervisor_id, condition=condition)
            contract.save(root / "contract.json")
            protocol.save(root / "protocol.json")
        res = run_block(
            contract=contract,
            block=block,
            apparatus=SurrogateApparatus(contract.grid, seed=int(seed) + index),
            channel=SimulatedSupervisorChannel(sup),
            grounder=grounder,
            scheduler=sch,
            logger=logger,
            seed=int(seed) + index,
        )
        if logger is not None:
            logger.write_json("truth.json", {"params": params.to_dict(), "pi_star": {str(k): v for k, v in truth_map.items()}})
            logger.close({"contract_hash": contract.hash(), "condition": condition})
        return res

    # --- reference and retest, both no-coach --------------------------------
    ref_block = protocol.reference_block()
    ref = _run(ref_block, CONDITION_NO_COACH, "reference")
    ref_seq = sequence_from_records(ref.records, contract.grid)
    reference_map = reference_map_from_observations(ref_seq, contract.grid)

    retest_map: Optional[Dict[int, float]] = None
    retest_block = protocol.retest_block()
    if retest_block is not None:
        rt = _run(retest_block, CONDITION_NO_COACH, "retest")
        retest_map = reference_map_from_observations(sequence_from_records(rt.records, contract.grid), contract.grid)

    # --- each compared condition, against an identical fresh supervisor -----
    shared_budget = protocol.condition_blocks()[0].budget
    out_conditions: Dict[str, Any] = {}
    #: Sequence used to fit this supervisor's carryover posterior for the *population* prior.
    #: Taken from whichever compared condition probes most, since that is the one whose
    #: observations carry the most information about the residue -- and it is fixed in advance
    #: rather than chosen from the results.
    ident_seq = None
    for cond in conditions:
        blk = Block(index=1, kind=BLOCK_CONDITION, condition=cond, budget=shared_budget)
        res = _run(blk, cond, cond)
        seq = sequence_from_records(res.records, contract.grid)
        est, post = _fit_reported(cond, seq, contract.grid, contract.carryover, log_prior)
        row = evaluate(est, reference_map, contract.grid, seq)
        row_truth = evaluate(est, truth_map, contract.grid, seq)
        row["vs_truth"] = {k: v for k, v in row_truth.items() if k in ("mae", "mae_crossover", "brier_crossover",
                                                                      "deployment_regret", "alignment",
                                                                      "coverage@80", "coverage@95")}
        row["condition"] = cond
        row["estimator"] = estimator_for(cond)
        row["scheduler"] = res.scheduler
        if post is not None:
            row["joint_carryover"] = {
                "mean": post.mean(),
                "effect": post.effect(),
                "identifiability": post.identifiability(),
            }
        if res.belief is not None:
            row["online_belief"] = {"mean": res.belief["mean"], "effect": res.belief["effect"],
                                    "lambda_diagnostic": res.belief.get("lambda_diagnostic"),
                                    "identifiability": res.belief.get("identifiability")}
        pairs = [(r.get("grounded"), r.get("grounded_secondary")) for r in res.records if r.get("grounded_secondary")]
        if pairs:
            row["grounder_agreement"] = grounding_agreement(pairs)
        out_conditions[cond] = row
        if ident_seq is None or row["n_probe"] + row["n_counter"] > ident_seq[1]:
            ident_seq = (seq, row["n_probe"] + row["n_counter"])

    identification, _ = joint_carryover_posterior(
        ident_seq[0] if ident_seq else [], contract.grid, cfg=contract.carryover
    )

    # --- test-retest floor ---------------------------------------------------
    tr: Dict[str, Any] = {}
    if retest_map is not None:
        w = contract.grid.band_weights(True)
        wu = contract.grid.band_weights(False)
        ids = [s.scene_id for s in contract.grid.probe_scenes()]
        tr = {
            "mae": float(sum(wu[i] * abs(reference_map[i] - retest_map[i]) for i in ids)),
            "mae_crossover": float(sum(w[i] * abs(reference_map[i] - retest_map[i]) for i in ids)),
            "reference_vs_truth_mae": float(sum(wu[i] * abs(reference_map[i] - truth_map[i]) for i in ids)),
            "retest_shift": float(sum(wu[i] * (retest_map[i] - reference_map[i]) for i in ids)),
        }

    return {
        "supervisor_id": params.supervisor_id,
        "params": params.to_dict(),
        "session_sign": int(protocol.session_sign),
        "reference_map": {str(k): float(v) for k, v in reference_map.items()},
        "truth_map": {str(k): float(v) for k, v in truth_map.items()},
        "test_retest": tr,
        "conditions": out_conditions,
        "_identification_posterior": identification,
    }


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
_NUMERIC_OUTCOMES = (
    "mae",
    "mae_crossover",
    "brier",
    "brier_crossover",
    "coverage@50",
    "coverage@80",
    "coverage@95",
    "deployment_regret",
    "alignment",
    "executed_regret_per_slot",
    "flip_rate",
    "n_probe",
    "n_counter",
    "n_wait",
    "n_ungrounded",
    "wall_clock_s",
)


def _mean_ci(xs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    se = sd / max(np.sqrt(a.size), 1.0)
    return {"mean": m, "sd": sd, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "n": int(a.size)}


def _paired_delta(rows: Sequence[Dict[str, Any]], cond: str, ref: str, key: str, n_boot: int = 4000,
                  seed: int = 0) -> Dict[str, float]:
    """Paired difference ``cond - ref`` with a bootstrap CI. Negative is better for errors."""
    d = []
    for r in rows:
        a = r["conditions"].get(cond, {}).get(key)
        b = r["conditions"].get(ref, {}).get(key)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            d.append(float(a) - float(b))
    if not d:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "p_better": float("nan")}
    arr = np.asarray(d)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(int(n_boot))])
    return {
        "delta": float(arr.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n": int(arr.size),
        "p_better": float((arr < 0).mean()),
        "win_rate_ci": list(wilson_ci(int((arr < 0).sum()), int(arr.size))),
    }


def aggregate(rows: Sequence[Dict[str, Any]], conditions: Sequence[str], *, comparator: str = PRIMARY_COMPARATOR,
              seed: int = 0) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_supervisors": len(rows), "conditions": {}, "contrasts": {}}
    for cond in conditions:
        cells = {k: _mean_ci([r["conditions"].get(cond, {}).get(k) for r in rows]) for k in _NUMERIC_OUTCOMES}
        cells["vs_truth"] = {
            k: _mean_ci([r["conditions"].get(cond, {}).get("vs_truth", {}).get(k) for r in rows])
            for k in ("mae", "mae_crossover", "deployment_regret", "alignment", "coverage@95")
        }
        summary["conditions"][cond] = cells
    for cond in conditions:
        if cond == comparator:
            continue
        summary["contrasts"][cond] = {
            k: _paired_delta(rows, cond, comparator, k, seed=seed)
            for k in ("mae_crossover", "mae", "deployment_regret", "executed_regret_per_slot", "alignment",
                      "coverage@95", "wall_clock_s", "n_counter")
        }
    summary["comparator"] = comparator

    # A second comparator, because the interesting question is not only "does the policy beat
    # the standard fixed rule" but "does it beat the obvious thing a practitioner would try" --
    # always offering the alternative. B4 is strong on error and expensive in the supervisor's
    # time, so the contrast that matters against it is error *at equal or lower burden*.
    from .scheduler import CONDITION_ALWAYS_COUNTER

    if CONDITION_ALWAYS_COUNTER in conditions:
        summary["contrasts_vs_always_counter"] = {
            cond: {
                k: _paired_delta(rows, cond, CONDITION_ALWAYS_COUNTER, k, seed=seed)
                for k in ("mae_crossover", "deployment_regret", "n_counter", "wall_clock_s")
            }
            for cond in conditions
            if cond != CONDITION_ALWAYS_COUNTER
        }

    # The error-versus-burden frontier. A policy that matches the best error while asking a
    # tenth as often is the result this study is actually able to support, so it gets its own
    # reported object rather than being left for a reader to reconstruct from two tables.
    summary["pareto"] = {
        cond: {
            "mae_crossover": summary["conditions"][cond]["mae_crossover"]["mean"],
            "deployment_regret": summary["conditions"][cond]["deployment_regret"]["mean"],
            "counters": summary["conditions"][cond]["n_counter"]["mean"],
            "waits": summary["conditions"][cond]["n_wait"]["mean"],
            "minutes": summary["conditions"][cond]["wall_clock_s"]["mean"] / 60.0,
        }
        for cond in conditions
    }

    # --- stratified contrasts -------------------------------------------------------------
    #
    # The whole premise is that people differ, so a pooled mean is the wrong headline: it
    # averages the supervisors a policy cannot help (nothing to correct) together with the ones
    # it can, and reports the dilution. H2 says a single fixed rule cannot be simultaneously
    # efficient for everyone -- testing that means splitting on how persuadable each person
    # actually is.
    #
    # Two strata are reported. ``true_beta_g`` splits on the simulator's ground truth and
    # answers "where does the mechanism help?". ``est_beta_g`` splits on the *robot's own
    # posterior* and answers the question a deployment actually faces: "can the robot tell, from
    # its own belief, which supervisors it should spend effort on?" A method that only looks
    # good under the ground-truth split cannot be operationalised.
    def _strata(key_fn, name: str) -> Dict[str, Any]:
        vals = [(key_fn(r), r) for r in rows]
        vals = [(v, r) for v, r in vals if v is not None and np.isfinite(v)]
        if len(vals) < 6:
            return {}
        xs = np.array([v for v, _ in vals])
        lo_c, hi_c = np.quantile(xs, [1 / 3, 2 / 3])
        buckets = {
            "low": [r for v, r in vals if v <= lo_c],
            "mid": [r for v, r in vals if lo_c < v <= hi_c],
            "high": [r for v, r in vals if v > hi_c],
        }
        out: Dict[str, Any] = {"split": name, "cuts": [float(lo_c), float(hi_c)]}
        for bname, brows in buckets.items():
            if not brows:
                continue
            out[bname] = {
                "n": len(brows),
                "mean_key": float(np.mean([key_fn(r) for r in brows])),
                "conditions": {
                    c: {k: _mean_ci([r["conditions"].get(c, {}).get(k) for r in brows])
                        for k in ("mae_crossover", "deployment_regret", "n_counter", "n_wait", "coverage@95")}
                    for c in conditions
                },
                "contrasts": {
                    c: {k: _paired_delta(brows, c, comparator, k, seed=seed)
                        for k in ("mae_crossover", "deployment_regret", "n_counter")}
                    for c in conditions if c != comparator
                },
            }
        return out

    def _est_bg(r: Dict[str, Any]) -> Optional[float]:
        for c in ("carryover_aware",) + tuple(conditions):
            jc = r["conditions"].get(c, {}).get("joint_carryover")
            if jc:
                return float(jc["mean"]["beta_g"])
        return None

    summary["strata"] = {
        "true_beta_g": _strata(lambda r: float(r["params"]["beta"]) * float(r["params"]["g"]), "true beta*g"),
        "true_lambda": _strata(lambda r: float(r["params"]["lam"]), "true lambda"),
        "est_beta_g": _strata(_est_bg, "robot's own posterior mean beta*g"),
    }
    summary["test_retest"] = {
        k: _mean_ci([r["test_retest"].get(k) for r in rows if r.get("test_retest")])
        for k in ("mae", "mae_crossover", "reference_vs_truth_mae", "retest_shift")
    }
    # Per-condition identification of the carryover parameters, from the OFFLINE joint fit of
    # each condition's own session. This is what B6 exists to move: the fraction of supervisors
    # whose posterior over lambda left its prior, under that condition's schedule.
    ident: Dict[str, Any] = {}
    for cond in conditions:
        lam, bg, n = 0, 0, 0
        lam_sd, bg_sd, lam_sd_nonc, n_nonc, lam_tv_nonc = 0, 0, 0, 0, 0
        lam_online, n_online = 0, 0
        for r in rows:
            jc = r["conditions"].get(cond, {}).get("joint_carryover")
            if jc and jc.get("identifiability"):
                n += 1
                il, ib = jc["identifiability"].get("lambda", {}), jc["identifiability"].get("beta_g", {})
                lam += int(il.get("identified", False))
                bg += int(ib.get("identified", False))
                lam_sd += int(il.get("identified_sd", False))
                bg_sd += int(ib.get("identified_sd", False))
                if float(r["params"]["beta"]) <= 1e-9:
                    n_nonc += 1
                    lam_sd_nonc += int(il.get("identified_sd", False))
                    lam_tv_nonc += int(il.get("identified", False))
            ob = r["conditions"].get(cond, {}).get("online_belief") or {}
            ld = ob.get("lambda_diagnostic")
            if ld:
                n_online += 1
                lam_online += int(bool(ld.get("lambda_identified_sd", ld.get("lambda_identified"))))
        if n:
            ident[cond] = {
                "n": n,
                # headline criterion: posterior contraction (a non-complier cannot satisfy it)
                "lambda": lam_sd / n, "beta_g": bg_sd / n,
                "lambda_ci": list(wilson_ci(lam_sd, n)), "beta_g_ci": list(wilson_ci(bg_sd, n)),
                # the first draft's criterion: total variation of the marginal
                "lambda_tv": lam / n, "beta_g_tv": bg / n,
                "lambda_tv_ci": list(wilson_ci(lam, n)),
                # the control: how often each criterion fires for supervisors with beta == 0
                "lambda_noncomplier_rate": (lam_sd_nonc / n_nonc) if n_nonc else None,
                "lambda_tv_noncomplier_rate": (lam_tv_nonc / n_nonc) if n_nonc else None,
                "n_noncompliers": n_nonc,
                "lambda_online": (lam_online / n_online) if n_online else None,
            }
    summary["identification"] = ident
    # Heterogeneity: the premise of the whole programme is that people differ.
    summary["population"] = {
        "beta_g": _mean_ci([r["params"]["beta"] * r["params"]["g"] for r in rows]),
        "lambda": _mean_ci([r["params"]["lam"] for r in rows]),
        "n_noncompliers": int(sum(1 for r in rows if r["params"]["beta"] <= 1e-9)),
    }
    return summary


def _fmt(v: Dict[str, float], nd: int = 4) -> str:
    if not np.isfinite(v.get("mean", float("nan"))):
        return "  --  "
    return f"{v['mean']:.{nd}f}"


def render_table(summary: Dict[str, Any], conditions: Sequence[str]) -> str:
    lines: List[str] = []
    tr = summary.get("test_retest", {})
    lines.append("")
    lines.append(f"Tier-1 synthetic study  |  N = {summary['n_supervisors']} supervisors  "
                 f"|  comparator = {DISPLAY_NAMES.get(summary['comparator'], summary['comparator'])}")
    pop = summary.get("population", {})
    lines.append(f"population: beta*g = {_fmt(pop.get('beta_g', {}), 2)}  lambda = {_fmt(pop.get('lambda', {}), 2)}  "
                 f"non-compliers = {pop.get('n_noncompliers', 0)}/{summary['n_supervisors']}")
    lines.append("")
    lines.append("test-retest floor (reference vs. terminal retest, both no-coach) -- read this first:")
    lines.append(f"    MAE {_fmt(tr.get('mae', {}))}   crossover-weighted MAE {_fmt(tr.get('mae_crossover', {}))}"
                 f"   reference-vs-true-map MAE {_fmt(tr.get('reference_vs_truth_mae', {}))}")
    lines.append("")
    head = f"{'condition':30s}{'MAE_x':>9}{'MAE':>9}{'regret':>9}{'align':>8}{'cov@95':>8}{'probe':>7}{'ctr':>6}{'wait':>6}{'min':>7}"
    lines.append(head)
    lines.append("-" * len(head))
    for cond in conditions:
        c = summary["conditions"].get(cond, {})
        lines.append(
            f"{DISPLAY_NAMES.get(cond, cond):30s}"
            f"{_fmt(c.get('mae_crossover', {})):>9}"
            f"{_fmt(c.get('mae', {})):>9}"
            f"{_fmt(c.get('deployment_regret', {})):>9}"
            f"{_fmt(c.get('alignment', {}), 3):>8}"
            f"{_fmt(c.get('coverage@95', {}), 3):>8}"
            f"{_fmt(c.get('n_probe', {}), 1):>7}"
            f"{_fmt(c.get('n_counter', {}), 1):>6}"
            f"{_fmt(c.get('n_wait', {}), 1):>6}"
            f"{(c.get('wall_clock_s', {}).get('mean', float('nan')) / 60.0):>7.1f}"
        )
    lines.append("")
    lines.append(f"paired contrasts vs. {DISPLAY_NAMES.get(summary['comparator'], summary['comparator'])} "
                 f"(negative = better for errors; bootstrap 95% CI):")
    for cond, cs in summary["contrasts"].items():
        d = cs.get("mae_crossover", {})
        r = cs.get("deployment_regret", {})
        if not np.isfinite(d.get("delta", float("nan"))):
            continue
        lines.append(
            f"    {DISPLAY_NAMES.get(cond, cond):30s} dMAE_x {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
            f"   dRegret {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]   wins {d['p_better']*100:.0f}%"
        )
    vs4 = summary.get("contrasts_vs_always_counter")
    if vs4:
        lines.append("paired contrasts vs. B4 Always counter-propose -- error at what burden:")
        for cond, cs in vs4.items():
            d = cs.get("mae_crossover", {})
            nc = cs.get("n_counter", {})
            wc = cs.get("wall_clock_s", {})
            if not np.isfinite(d.get("delta", float("nan"))):
                continue
            lines.append(
                f"    {DISPLAY_NAMES.get(cond, cond):30s} dMAE_x {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                f"   d(counter-proposals) {nc['delta']:+.1f}   d(minutes) {wc['delta'] / 60.0:+.1f}"
            )
        lines.append("")

    ident = summary.get("identification") or {}
    if ident:
        lines.append("identification of the carryover parameters (offline joint fit of each condition's own session):")
        for cond in conditions:
            d = ident.get(cond)
            if not d:
                continue
            on = f"   online: {d['lambda_online']*100:.0f}%" if d.get("lambda_online") is not None else ""
            nc = (f"   non-compliers: {d['lambda_noncomplier_rate']*100:.0f}% (TV criterion {d['lambda_tv_noncomplier_rate']*100:.0f}%)"
                  if d.get("lambda_noncomplier_rate") is not None else "")
            lines.append(f"    {DISPLAY_NAMES.get(cond, cond):30s} lambda {d['lambda']*100:3.0f}% "
                         f"[{d['lambda_ci'][0]*100:.0f}, {d['lambda_ci'][1]*100:.0f}] (TV crit. {d['lambda_tv']*100:.0f}%)"
                         f"   beta*g {d['beta_g']*100:3.0f}%{on}{nc}")
        lines.append("")
    strata = summary.get("strata", {}).get("true_beta_g") or {}
    if strata:
        lines.append(f"stratified by {strata.get('split', '')} (cuts {np.round(strata.get('cuts', []), 2).tolist()}) "
                     f"-- crossover-weighted MAE, and the paired delta vs. the comparator:")
        head2 = f"{'condition':30s}" + "".join(f"{b:>22}" for b in ("low", "mid", "high") if b in strata)
        lines.append(head2)
        lines.append("-" * len(head2))
        for cond in conditions:
            cells = []
            for b in ("low", "mid", "high"):
                if b not in strata:
                    continue
                v = strata[b]["conditions"].get(cond, {}).get("mae_crossover", {})
                d = strata[b]["contrasts"].get(cond, {}).get("mae_crossover", {})
                dv = f"{d['delta']:+.3f}" if d and np.isfinite(d.get("delta", float("nan"))) else "  ref "
                cells.append(f"{_fmt(v, 4):>10} ({dv:>7})".rjust(22))
            lines.append(f"{DISPLAY_NAMES.get(cond, cond):30s}" + "".join(cells))
        ns = "  ".join(f"{b}: n={strata[b]['n']}, mean beta*g={strata[b]['mean_key']:.2f}"
                       for b in ("low", "mid", "high") if b in strata)
        lines.append(f"    {ns}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supervisors", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--conditions", nargs="+", default=None, help="default: all compared conditions + ablations")
    ap.add_argument("--config", type=Path, default=None, help="YAML/JSON contract overrides")
    ap.add_argument("--out", type=Path, default=Path("vla_lab/results/tier1"))
    ap.add_argument("--log-root", type=Path, default=None, help="write full session records here (large)")
    ap.add_argument("--coach-regime", default=None, choices=["one_sided", "alternating", "runs"])
    ap.add_argument("--dose", default=None, choices=["weak", "moderate", "strong"])
    ap.add_argument("--physics", type=Path, default=None, help="measured ScenePhysics json from the Isaac sweep")
    ap.add_argument("--physics-quantile", default="point", choices=["lower", "point", "upper"],
                    help="rebuild the scene grid and band weights under the bootstrap draw of the physics whose "
                         "transition width sits at the 2.5th (lower) or 97.5th (upper) percentile; the files "
                         "physics_lower.json / physics_upper.json are written next to physics.json by the fit")
    ap.add_argument("--assume-w-cm", type=float, default=None,
                    help="COUNTERFACTUAL: override the transition width (cm), keeping the crossover; for the "
                         "sensitivity-to-curve sweep of vla_lab.supervisory.flip")
    ap.add_argument("--assume-mstar-cm", type=float, default=None,
                    help="COUNTERFACTUAL: override the crossover margin (cm)")
    ap.add_argument(
        "--population-prior",
        default="loo",
        choices=["loo", "none"],
        help="'loo' fits an empirical-Bayes prior over (lambda, beta, g) from the OTHER "
             "supervisors and re-runs; 'none' keeps the weakly-informative prior throughout.",
    )
    ap.add_argument("--population", action="append", default=[], metavar="FIELD=lo,hi",
                    help="override a SupervisorPopulation range, e.g. --population lapse_range=0.15,0.25. "
                         "Used by the placebo control of the dose-tracking result: vary something the belief "
                         "module has no access to and check the counter-proposal rate does NOT follow it.")
    ap.add_argument("--phrase-corpus", type=Path, default=None,
                    help="run under the EMPIRICAL phrase set rebuilt from a collected corpus "
                         "(vla_lab.human_study.phrase_corpus rebuild): the supervisor speaks from people's "
                         "phrases and hedges at their rate; the narration hash changes accordingly")
    ap.add_argument("--analyze", action="store_true", help="also write figures")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--reaggregate", type=Path, default=None, metavar="DIR",
                    help="do not run anything: rebuild summary.json, table.txt and the figures of an existing "
                         "run from its per_supervisor.json (after an aggregator change)")
    args = ap.parse_args(argv)

    if args.reaggregate is not None:
        return _reaggregate(Path(args.reaggregate), analyze=bool(args.analyze))

    corpus_info = None
    if args.phrase_corpus is not None:
        from ..human_study.phrase_corpus import install_empirical_axis

        corpus_info = install_empirical_axis(Path(args.phrase_corpus))
        print(f"[corpus] empirical axis installed: {corpus_info}", file=sys.stderr)
    contract = Contract()
    if args.config is not None:
        raw = json.loads(Path(args.config).read_text()) if args.config.suffix == ".json" else _load_yaml(args.config)
        contract = Contract.from_dict({**contract.to_dict(), **raw.get("contract", raw)})
    from .scenes import DEFAULT_PHYSICS_PATH, build_scene_grid, default_physics, load_physics

    phys = load_physics(args.physics) if args.physics is not None else None
    if args.physics_quantile != "point":
        base = Path(args.physics) if args.physics is not None else DEFAULT_PHYSICS_PATH
        qpath = base.with_name(f"physics_{args.physics_quantile}.json")
        if not qpath.exists():
            print(f"[FAIL] {qpath} does not exist; run the physics fit with --bootstrap first", file=sys.stderr)
            return 2
        phys = load_physics(qpath)
    if args.assume_w_cm is not None or args.assume_mstar_cm is not None:
        phys = phys if phys is not None else default_physics()
        if args.assume_mstar_cm is not None:
            phys = phys.with_crossover(float(args.assume_mstar_cm) / 100.0)
        if args.assume_w_cm is not None:
            phys = phys.with_transition_width(float(args.assume_w_cm) / 100.0, keep_crossover=True)
    if phys is not None:
        contract.grid = build_scene_grid(axis=contract.axis, physics=phys)
    if args.coach_regime:
        contract.budget.coach_regime = args.coach_regime
    if args.dose:
        contract.dose = args.dose

    problems = contract.check()
    if problems:
        print("[FAIL] the contract is not runnable:", file=sys.stderr)
        for p in problems:
            print(f"   - {p}", file=sys.stderr)
        return 2

    conditions = list(args.conditions) if args.conditions else list(COMPARED_CONDITIONS) + list(ABLATIONS)
    population = None
    if corpus_info is not None:
        # The hedge rate is measured, not chosen.
        r = float(corpus_info["ungrounded_rate"])
        args.population = list(args.population) + [f"ungrounded_range={r:.4f},{r:.4f}"]
    if args.population:
        overrides: Dict[str, Any] = {}
        for item in args.population:
            key, _, val = str(item).partition("=")
            parts = [float(v) for v in val.split(",")]
            overrides[key.strip()] = tuple(parts) if len(parts) > 1 else parts[0]
        population = SupervisorPopulation(**{**asdict(SupervisorPopulation()), **overrides})
    t0 = time.time()

    def _pass(priors: Optional[List[Any]], label: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(int(args.supervisors)):
            out.append(
                run_one_supervisor(
                    index=i,
                    contract=contract,
                    conditions=conditions,
                    seed=int(args.seed),
                    population=population,
                    log_root=args.log_root if priors is not None or args.population_prior == "none" else None,
                    log_prior=priors[i] if priors is not None else None,
                )
            )
            if not args.quiet:
                print(f"  [{label}] supervisor {i + 1}/{args.supervisors}   ", end="\r", file=sys.stderr)
        return out

    # Pass 1 always runs from the weakly-informative prior. It is reported in its own right --
    # it is what a robot meeting its first supervisor can do -- and it supplies the per-person
    # posteriors the population prior is built from.
    rows_flat = _pass(None, "flat prior")
    rows = rows_flat
    rows_pop: Optional[List[Dict[str, Any]]] = None
    if args.population_prior == "loo":
        posteriors = [r.pop("_identification_posterior") for r in rows_flat]
        # Leave-one-out: supervisor i's prior never sees supervisor i's own data.
        priors = [fit_population_prior(posteriors, exclude=i) for i in range(len(posteriors))]
        rows_pop = _pass(priors, "population prior")
        rows = rows_pop
    for r in rows_flat:
        r.pop("_identification_posterior", None)
    if rows_pop is not None:
        for r in rows_pop:
            r.pop("_identification_posterior", None)

    summary = aggregate(rows, conditions, seed=int(args.seed))
    summary["prior"] = args.population_prior
    if rows_pop is not None:
        summary["flat_prior"] = aggregate(rows_flat, conditions, seed=int(args.seed))
    summary["elapsed_s"] = time.time() - t0
    summary["population_overrides"] = list(args.population)
    summary["phrase_corpus"] = {"path": str(args.phrase_corpus), **corpus_info} if corpus_info else None
    summary["population_spec"] = (population or SupervisorPopulation()).to_dict()
    summary["physics_quantile"] = contract.grid.physics.quantile
    summary["seed"] = int(args.seed)
    summary["git_sha"] = _git_sha()
    summary["contract"] = contract.to_dict()
    summary["contract_hash"] = contract.hash()
    summary["conditions_run"] = conditions

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "per_supervisor.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    if rows_pop is not None:
        (out / "per_supervisor_flat_prior.json").write_text(json.dumps(rows_flat, indent=2, default=float) + "\n")
    from .analyze import audit_study, render_audit

    audit = audit_study(summary, rows)
    summary["audit"] = audit
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    table = render_table(summary, conditions) + "\n" + render_audit(audit)
    (out / "table.txt").write_text(table)
    if not args.quiet:
        print(table)
        print(f"wrote {out}/summary.json, per_supervisor.json, table.txt  ({summary['elapsed_s']:.1f}s)")

    if args.analyze:
        from .analyze import write_figures

        write_figures(summary, rows, out)
    return 0


def _reaggregate(out: Path, *, analyze: bool = False) -> int:
    from .analyze import audit_study, render_audit

    old = json.loads((out / "summary.json").read_text())
    rows = json.loads((out / "per_supervisor.json").read_text())
    conditions = list(old.get("conditions_run") or old.get("conditions", {}).keys())
    summary = aggregate(rows, conditions, seed=int(old.get("seed", 0)))
    flat = out / "per_supervisor_flat_prior.json"
    if flat.exists():
        summary["flat_prior"] = aggregate(json.loads(flat.read_text()), conditions, seed=int(old.get("seed", 0)))
    for k in ("prior", "elapsed_s", "contract", "contract_hash", "conditions_run", "population_overrides",
              "population_spec", "physics_quantile", "seed", "git_sha", "phrase_corpus"):
        if k in old:
            summary[k] = old[k]
    summary["audit"] = audit_study(summary, rows)
    summary["reaggregated"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    table = render_table(summary, conditions) + "\n" + render_audit(summary["audit"])
    (out / "table.txt").write_text(table)
    print(table)
    if analyze:
        from .analyze import write_figures

        write_figures(summary, rows, out)
    return 0


def _git_sha() -> Optional[str]:
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else None
    except Exception:                                            # pragma: no cover
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is needed for --config with a .yaml file") from exc
    return yaml.safe_load(Path(path).read_text()) or {}


if __name__ == "__main__":
    raise SystemExit(main())

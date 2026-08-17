"""W17 — Phase 0 outcomes, tables, and every paper figure, from one command.

    python -m vla_lab.rehab.analyze --session-root logs/rehab --out-dir vla_lab/results/rehab_phase0

Reports, in the order a reader should meet them (``rehab.md`` §1.5, §6/W17):

0. **Test-retest reliability of the reference map.** Reported *first*, because it bounds how
   much of the measured "estimation error" is irreducible drift. A study that cannot show
   test-retest stability of ``tilde-pi*`` cannot interpret its primary outcome (§12.2).
1. **Primary — estimation error.** Crossover-weighted MAE and Brier of ``pi-hat`` against the
   reference map, per condition, per participant.
2. **Primary — calibration.** Credible-interval coverage at nominal levels, plus ECE over the
   realized choices, delegating to :mod:`vla_lab.calibration.metrics`.
3. **The ablation decomposition.** How much of B4's advantage comes from *when it probes*
   versus *its ability to de-bias what it probed* — the two sources §1.4 requires be separated.
4. **Secondary.** Task success, the budget manipulation check, waiting cost, burden.
5. **Exploratory.** Per-person carryover posteriors; between-person heterogeneity is a headline
   figure, not a nuisance.
6. **Off-policy secondary (§12.6).** The baselines that were not run prospectively, evaluated
   on each participant's fitted carryover model — labelled model-based everywhere it appears.

**Non-circularity.** The reference block is split by trial parity: even-indexed trials define
the reference map ``tilde-pi*``, odd-indexed trials are the clean anchor every condition's
estimator may use. Without an anchor the carryover-corrected estimator cannot separate a
slowly-decaying offset from the intercept of ``pi*``; with the *whole* reference block on both
sides, the primary outcome would be scored partly against its own training data. The split
costs precision in both and is the honest trade.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..stats_utils import wilson_ci
from . import ASSESS, COACH, WAIT
from .carryover import CarryoverConfig, CarryoverPosterior
from .contract import Phase0Contract
from .estimand import (
    METHOD_CORRECTED,
    METHOD_POOLED,
    METHOD_SPATIAL,
    CarryoverCorrectedEstimator,
    PiStarPosterior,
    PooledBetaEstimator,
    SpatialLogisticEstimator,
    TrialObservation,
    brier_vs_reference,
    interval_coverage,
    joint_carryover_posterior,
    mae,
    sequence_from_records,
    trial_level_calibration,
)
from .logging import SessionReader, find_sessions
from .protocol import BLOCK_COMPARED, BLOCK_REFERENCE, BLOCK_RETEST, SessionPlan
from .scheduler import CONDITION_CARRYOVER_AWARE, CONDITION_FIXED_WASHOUT, CORRECTED_CONDITIONS
from .workspace import TargetGrid, nonpreferred_lateral


# ---------------------------------------------------------------------------
# Small statistics (kept local: the VLA track's stats_utils is left untouched)
# ---------------------------------------------------------------------------


def paired_difference(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired mean difference ``a - b`` with its sd and Cohen's dz."""

    d = np.asarray([float(x) - float(y) for x, y in zip(a, b)], dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "dz": float("nan")}
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "sd": sd,
        "dz": float(d.mean() / sd) if sd > 0 else float("inf" if d.mean() else 0.0),
    }


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 5000, level: float = 0.95, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean. Seeded, so a rerun reproduces the interval."""

    v = np.asarray([float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    means = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
    lo = float(np.quantile(means, (1.0 - level) / 2.0))
    hi = float(np.quantile(means, 1.0 - (1.0 - level) / 2.0))
    return (lo, hi)


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test. Exact for ``n <= 15``, normal approximation above.

    A rank test rather than a paired t because Phase 0's N is small and MAE across participants
    has no reason to be normal.
    """

    d = [float(x) - float(y) for x, y in zip(a, b)]
    d = [x for x in d if x == x and x != 0.0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w": float("nan"), "p_value": float("nan"), "method": "none"}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    w = min(w_plus, w_minus)

    if n <= 15:
        # Exact: enumerate every sign assignment of the ranks.
        total = 1 << n
        count = 0
        for mask in range(total):
            s = sum(ranks[k] for k in range(n) if mask & (1 << k))
            if min(s, sum(ranks) - s) <= w:
                count += 1
        p = float(min(1.0, count / total))
        method = "exact"
    else:
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mu + 0.5) / sigma if sigma > 0 else 0.0
        p = float(min(1.0, 2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
        method = "normal"
    return {"n": n, "w": float(w), "p_value": p, "method": method}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class ParticipantData:
    """One session, split into the blocks the analysis needs."""

    def __init__(self, session_dir: Path, *, carryover_cfg: Optional[CarryoverConfig] = None) -> None:
        self.dir = Path(session_dir)
        self.carryover_cfg = carryover_cfg or CarryoverConfig()
        reader = SessionReader(session_dir)
        self.contract = Phase0Contract.from_dict(reader.contract or {})
        self.participant = reader.participant or {}
        self.protocol = reader.protocol or {}
        self.plan = SessionPlan.from_dict(self.protocol) if self.protocol else None
        self.records = reader.trials()
        self.events = reader.events().rows
        self.grid: TargetGrid = self.contract.target_grid()
        self.side = str(self.participant.get("nonpreferred_side", "left"))
        self.pid = str(self.participant.get("participant_id", self.dir.parent.name))
        self.effort_strength = {
            lvl.name: float(lvl.carryover_scale) for lvl in self.contract.prompts.effort_levels
        }
        self._kinds: Dict[int, str] = {
            int(b.get("block_idx", -1)): str(b.get("kind", BLOCK_COMPARED))
            for b in self.protocol.get("blocks", [])
        }
        self._conds: Dict[int, str] = {
            int(b.get("block_idx", -1)): str(b.get("condition", ""))
            for b in self.protocol.get("blocks", [])
        }

    # -- block slicing -----------------------------------------------------
    def _seq(self, records: Sequence[Any]) -> List[TrialObservation]:
        return sequence_from_records(
            records, self.grid, self.side,
            cfg=self.carryover_cfg, effort_strength=self.effort_strength,
        )

    def block_records(self, kind: str) -> Dict[int, List[Any]]:
        out: Dict[int, List[Any]] = defaultdict(list)
        for r in self.records:
            if self._kinds.get(int(r.trial.block_idx)) == kind:
                out[int(r.trial.block_idx)].append(r)
        return dict(out)

    def reference_split(self) -> Tuple[List[TrialObservation], List[TrialObservation]]:
        """``(eval_half, anchor_half)`` of the reference block, split by trial parity."""

        recs: List[Any] = []
        for _, rs in sorted(self.block_records(BLOCK_REFERENCE).items()):
            recs += rs
        ev = [r for i, r in enumerate(recs) if i % 2 == 0]
        an = [r for i, r in enumerate(recs) if i % 2 == 1]
        return self._seq(ev), self._seq(an)

    def retest_sequence(self) -> List[TrialObservation]:
        recs: List[Any] = []
        for _, rs in sorted(self.block_records(BLOCK_RETEST).items()):
            recs += rs
        return self._seq(recs)

    def compared_blocks(self) -> List[Tuple[str, List[Any]]]:
        out: List[Tuple[str, List[Any]]] = []
        for bi, rs in sorted(self.block_records(BLOCK_COMPARED).items()):
            out.append((self._conds.get(bi, ""), rs))
        return out

    def budget_spent(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for cond, rs in self.compared_blocks():
            out[cond] = {
                "trials": len(rs),
                "coach": sum(1 for r in rs if r.trial.action == COACH),
                "assess": sum(1 for r in rs if r.trial.action == ASSESS),
                "wait": sum(1 for r in rs if r.trial.action == WAIT),
                "observations": sum(1 for r in rs if r.result.is_observation),
                "wait_ms": sum(1 for r in rs if r.trial.action == WAIT) * int(self.contract.timing.wait_dwell_ms),
            }
        return out

    def questionnaires(self) -> List[Dict[str, Any]]:
        return [dict(e.get("data") or {}) for e in self.events if e.get("type") == "questionnaire"]


# ---------------------------------------------------------------------------
# Per-participant analysis
# ---------------------------------------------------------------------------


def estimator_for(condition: str) -> str:
    """Which estimator a condition's own pipeline uses (schedule and estimator are paired)."""

    return METHOD_CORRECTED if condition in CORRECTED_CONDITIONS else METHOD_SPATIAL


def analyze_participant(
    pd: ParticipantData,
    *,
    level: float = 0.9,
    ground_truth: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Reference map, per-condition estimates and outcomes, and the carryover posterior."""

    ref_eval_seq, ref_anchor_seq = pd.reference_split()
    spatial = SpatialLogisticEstimator()
    ref_post = spatial.fit(ref_eval_seq, pd.grid, pd.side)
    reference_map = ref_post.mean()

    out: Dict[str, Any] = {
        "participant_id": pd.pid,
        "session_dir": str(pd.dir),
        "nonpreferred_side": pd.side,
        "n_reference_eval": sum(1 for o in ref_eval_seq if o.observed),
        "n_reference_anchor": sum(1 for o in ref_anchor_seq if o.observed),
        "reference_map": {str(k): round(float(v), 5) for k, v in reference_map.items()},
        "conditions": {},
        "budget": pd.budget_spent(),
    }

    # --- test-retest --------------------------------------------------------
    retest_seq = pd.retest_sequence()
    if any(o.observed for o in retest_seq):
        retest_map = spatial.fit(retest_seq, pd.grid, pd.side).mean()
        out["retest_map"] = {str(k): round(float(v), 5) for k, v in retest_map.items()}
        ids = sorted(set(reference_map) & set(retest_map))
        a = np.array([reference_map[i] for i in ids])
        b = np.array([retest_map[i] for i in ids])
        w = pd.grid.crossover_weights()
        wv = np.array([w[i] for i in ids])
        out["test_retest"] = {
            "n_targets": len(ids),
            "mae": float(np.average(np.abs(a - b), weights=wv)),
            "pearson_r": (float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else float("nan")),
        }

    # --- ground truth (synthetic pilots only) --------------------------------
    if ground_truth:
        gt = {int(k): float(v) for k, v in ground_truth.items()}
        w = pd.grid.crossover_weights()
        ids = sorted(set(gt) & set(reference_map))
        wv = np.array([w[i] for i in ids])
        out["reference_vs_truth_mae"] = float(
            np.average(np.abs([reference_map[i] - gt[i] for i in ids]), weights=wv)
        )

    # --- per condition -------------------------------------------------------
    for cond, recs in pd.compared_blocks():
        seq = pd._seq(recs)
        fit_seq = list(ref_anchor_seq) + list(seq)
        carry = joint_carryover_posterior(fit_seq, pd.grid, cfg=pd.carryover_cfg)
        method = estimator_for(cond)
        if method == METHOD_CORRECTED:
            est: PiStarPosterior = CarryoverCorrectedEstimator(spatial=spatial).fit(
                fit_seq, pd.grid, pd.side, carry
            )
        else:
            est = spatial.fit(fit_seq, pd.grid, pd.side)
        pooled = PooledBetaEstimator().fit(fit_seq, pd.grid)

        row: Dict[str, Any] = {
            "estimator": method,
            "n_observations": sum(1 for o in seq if o.observed),
            "mae": mae(est, reference_map, grid=pd.grid, crossover_weighted=True),
            "mae_unweighted": mae(est, reference_map, grid=pd.grid, crossover_weighted=False),
            "brier": brier_vs_reference(est, reference_map, grid=pd.grid, crossover_weighted=True),
            "coverage": interval_coverage(est, reference_map, level=level, grid=pd.grid),
            "coverage_crossover": interval_coverage(
                est, reference_map, level=level, grid=pd.grid, crossover_only=True
            ),
            "calibration": trial_level_calibration(est, seq),
            "mae_pooled_estimator": mae(pooled, reference_map, grid=pd.grid, crossover_weighted=True),
            "carryover": carry.summary(),
            "success_rate": _success_rate(recs),
            "pi_hat": {str(k): round(float(v), 5) for k, v in est.mean().items()},
        }
        if ground_truth:
            gt = {int(k): float(v) for k, v in ground_truth.items()}
            row["mae_vs_truth"] = mae(est, gt, grid=pd.grid, crossover_weighted=True)
        out["conditions"][cond] = row

    # --- questionnaires ------------------------------------------------------
    q = pd.questionnaires()
    if q:
        out["questionnaires"] = q
    return out


def _success_rate(records: Sequence[Any]) -> Dict[str, float]:
    presented = [r for r in records if r.trial.action != WAIT]
    n = len(presented)
    s = sum(1 for r in presented if r.result.success)
    lo, hi = wilson_ci(s, n)
    return {"n": n, "rate": (s / n) if n else float("nan"), "wilson95": [lo, hi]}


# ---------------------------------------------------------------------------
# Off-policy secondary analysis (§12.6)
# ---------------------------------------------------------------------------


def offpolicy_evaluate(
    pd: ParticipantData,
    conditions: Sequence[str],
    *,
    reference_map: Dict[int, float],
    carry: CarryoverPosterior,
    n_reps: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the un-run baselines against this participant's **fitted model**.

    Explicitly model-based: the participant here is a simulator parameterized by the posterior
    mean of ``(lambda, beta, g)`` and by the fitted ``pi*``, not the person. It answers "what
    would this baseline have done to *this* participant", which is the strongest thing a
    within-session design can say about conditions it could not afford to run (§12.6). The
    paper must say so plainly wherever these numbers appear.
    """

    from .scheduler import make_scheduler
    from .scheduler.base import DeltaModel
    from .sim_participant import ParticipantParams, SimulatedParticipant
    from .trial import History, Trial, TrialRecord, TrialResult

    if pd.plan is None:
        return {"error": "no protocol.json; cannot replay the slot layout"}
    blocks = [b for b in pd.plan.blocks if b.kind == BLOCK_COMPARED]
    if not blocks:
        return {"error": "no compared blocks in the plan"}
    layout = blocks[0]
    m = carry.mean()
    delta = DeltaModel.from_contract(pd.contract, pd.carryover_cfg)

    # Invert the fitted pi* into the simulator's parameterization by least squares in logit
    # space over the nonpreferred-signed lateral coordinate.
    ids = sorted(reference_map)
    s = np.array([nonpreferred_lateral(pd.grid.get(i).y_m, pd.side) for i in ids])
    p = np.clip(np.array([reference_map[i] for i in ids]), 1e-4, 1 - 1e-4)
    A = np.column_stack([s, np.ones_like(s)])
    coef, *_ = np.linalg.lstsq(A, np.log(p / (1 - p)), rcond=None)
    steepness = float(max(1e-3, coef[0]))
    crossover = float(-coef[1] / steepness) if steepness > 1e-6 else 0.0

    results: Dict[str, List[float]] = {c: [] for c in conditions}
    for rep in range(int(n_reps)):
        for cond in conditions:
            params = ParticipantParams(
                participant_idx=0, nonpreferred_side=pd.side, crossover_m=crossover,
                steepness=steepness, depth_coef=0.0, lam=float(m["lambda"]),
                beta=float(m["beta"]), g=float(m["g"]), fatigue=0.0, lapse=0.0,
                misdetect=0.0, seed=int(seed) * 7919 + rep * 131 + hash(cond) % 1000,
            )
            sim = SimulatedParticipant(params, pd.grid, total_trials=len(layout.slots))
            sched = make_scheduler(
                cond, grid=pd.grid, nonpreferred_side=pd.side, seed=int(seed) + rep,
                fixed_w=int(pd.plan.fixed_w), n_wait=int(pd.plan.fixed_w) * len(layout.coach_slots),
                delta=delta,
            )
            sched.reset(layout.budget())
            hist = History(
                block_idx=0, condition=cond, slots_total=len(layout.slots),
                coach_slots=layout.coach_slots,
            )
            seq: List[TrialObservation] = []
            for slot in layout.slots:
                dec = sched.decide(hist, slot)
                target = pd.grid.get(int(dec.target_id)) if dec.target_id is not None else None
                strength = float(pd.effort_strength.get(slot.effort_level, 1.0)) if dec.action == COACH else 1.0
                resp = sim.select(
                    target, action=dec.action, strength=strength,
                    delta=float(delta.for_action(dec.action)),
                )
                tr = Trial(
                    trial_idx=slot.slot_idx, block_idx=0, condition=cond, action=dec.action,
                    target_id=(int(target.target_id) if target is not None else None),
                    slot_idx=slot.slot_idx, is_coach_slot=slot.is_coach_slot,
                    effort_level=(slot.effort_level if dec.action == COACH else "none"),
                )
                res = TrialResult(arm=(resp.arm if resp is not None else "none"), success=True)
                rec = TrialRecord(trial=tr, result=res)
                hist.append(rec)
                sched.observe(rec)
                seq.append(
                    TrialObservation(
                        trial_idx=slot.slot_idx, action=dec.action,
                        target_id=(int(target.target_id) if target is not None else None),
                        s_m=(float(nonpreferred_lateral(target.y_m, pd.side)) if target is not None else 0.0),
                        depth_m=(float(target.x_m) if target is not None else 0.0),
                        y=(None if resp is None else (resp.arm == "nonpreferred")),
                        delta=float(delta.for_action(dec.action)), strength=strength,
                    )
                )
            carry_hat = joint_carryover_posterior(seq, pd.grid, cfg=pd.carryover_cfg)
            spatial = SpatialLogisticEstimator()
            est = (
                CarryoverCorrectedEstimator(spatial=spatial).fit(seq, pd.grid, pd.side, carry_hat)
                if estimator_for(cond) == METHOD_CORRECTED
                else spatial.fit(seq, pd.grid, pd.side)
            )
            results[cond].append(mae(est, sim.pi_star_map(), grid=pd.grid, crossover_weighted=True))

    return {
        "model_based": True,
        "n_reps": int(n_reps),
        "fitted": {"lambda": round(float(m["lambda"]), 4), "beta_g": round(float(m["beta_g"]), 4),
                   "steepness": round(steepness, 3), "crossover_m": round(crossover, 4)},
        "mae_by_condition": {
            c: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
            for c, v in results.items()
        },
    }


# ---------------------------------------------------------------------------
# Study-level aggregation
# ---------------------------------------------------------------------------


def aggregate(
    per_participant: Sequence[Dict[str, Any]],
    *,
    reference_condition: str = CONDITION_CARRYOVER_AWARE,
    comparator: str = CONDITION_FIXED_WASHOUT,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-condition summaries plus the primary paired contrast and the ablation decomposition."""

    conditions = sorted({c for p in per_participant for c in p.get("conditions", {})})
    by_cond: Dict[str, Dict[str, List[float]]] = {
        c: {"mae": [], "brier": [], "coverage": [], "ece": [], "success": [], "wait_ms": [], "observations": []}
        for c in conditions
    }
    for p in per_participant:
        for c, row in p.get("conditions", {}).items():
            by_cond[c]["mae"].append(float(row.get("mae", float("nan"))))
            by_cond[c]["brier"].append(float(row.get("brier", float("nan"))))
            by_cond[c]["coverage"].append(float((row.get("coverage") or {}).get("coverage", float("nan"))))
            by_cond[c]["ece"].append(float((row.get("calibration") or {}).get("ece", float("nan"))))
            by_cond[c]["success"].append(float((row.get("success_rate") or {}).get("rate", float("nan"))))
            by_cond[c]["observations"].append(float(row.get("n_observations", 0)))
            by_cond[c]["wait_ms"].append(float((p.get("budget", {}).get(c, {}) or {}).get("wait_ms", 0)))

    summary: Dict[str, Any] = {"n_participants": len(per_participant), "conditions": {}}
    for c, d in by_cond.items():
        summary["conditions"][c] = {
            k: {
                "mean": float(np.nanmean(v)) if v else float("nan"),
                "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": int(np.sum(np.isfinite(v))),
            }
            for k, v in d.items()
        }

    def _metric(row: Dict[str, Any], metric: str) -> float:
        """Pull one scalar out of a condition row. Some outcomes are nested dicts."""

        if metric == "coverage":
            return float((row.get("coverage") or {}).get("coverage", float("nan")))
        if metric == "ece":
            return float((row.get("calibration") or {}).get("ece", float("nan")))
        if metric == "success":
            return float((row.get("success_rate") or {}).get("rate", float("nan")))
        v = row.get(metric, float("nan"))
        return float(v) if isinstance(v, (int, float)) else float("nan")

    def paired(metric: str, a: str, b: str) -> Dict[str, Any]:
        pairs = [
            (_metric(p["conditions"][a], metric), _metric(p["conditions"][b], metric))
            for p in per_participant
            if a in p.get("conditions", {}) and b in p.get("conditions", {})
        ]
        pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
        if not pairs:
            return {"n": 0}
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        diff = paired_difference(xs, ys)
        lo, hi = bootstrap_ci([x - y for x, y in pairs], seed=seed)
        return {
            **diff,
            "mean_a": float(np.mean(xs)),
            "mean_b": float(np.mean(ys)),
            "ci95_of_difference": [lo, hi],
            "wilcoxon": wilcoxon_signed_rank(xs, ys),
        }

    summary["primary_contrast"] = {
        "metric": "crossover_weighted_mae",
        "a": reference_condition,
        "b": comparator,
        "note": "negative mean difference favours the proposed policy (lower error)",
        **paired("mae", reference_condition, comparator),
    }
    summary["calibration_contrast"] = {
        "metric": "credible_interval_coverage",
        "a": reference_condition,
        "b": comparator,
        **paired("coverage", reference_condition, comparator),
    }

    # --- the ablation decomposition (§1.4) ----------------------------------
    from .scheduler import CONDITION_ABLATION_ESTIMATOR, CONDITION_ABLATION_SCHEDULE

    decomposition: Dict[str, Any] = {}
    if all(c in conditions for c in (CONDITION_CARRYOVER_AWARE, CONDITION_ABLATION_SCHEDULE, CONDITION_ABLATION_ESTIMATOR, CONDITION_FIXED_WASHOUT)):
        decomposition = {
            "note": (
                "B4 vs B2 decomposed into (i) probe placement and (ii) de-biasing. Each ablation "
                "keeps one of the two mechanisms; the sum of the two need not equal the total, "
                "and any gap is the interaction."
            ),
            "total_b4_vs_b2": paired("mae", CONDITION_CARRYOVER_AWARE, CONDITION_FIXED_WASHOUT),
            "schedule_only_vs_b2": paired("mae", CONDITION_ABLATION_SCHEDULE, CONDITION_FIXED_WASHOUT),
            "estimator_only_vs_b2": paired("mae", CONDITION_ABLATION_ESTIMATOR, CONDITION_FIXED_WASHOUT),
        }
    summary["ablation_decomposition"] = decomposition

    # --- test-retest, reported before the primary outcome --------------------
    tr = [p.get("test_retest") for p in per_participant if p.get("test_retest")]
    if tr:
        summary["test_retest"] = {
            "n": len(tr),
            "mae_mean": float(np.nanmean([t["mae"] for t in tr])),
            "pearson_r_mean": float(np.nanmean([t["pearson_r"] for t in tr])),
            "note": (
                "bounds how much of the measured estimation error is irreducible drift; read "
                "this before the primary outcome (§12.2)"
            ),
        }

    # --- budget manipulation check -------------------------------------------
    budgets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for p in per_participant:
        for c, b in (p.get("budget") or {}).items():
            for k, v in b.items():
                budgets[c][k].append(float(v))
    summary["budget_manipulation_check"] = {
        c: {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in d.items()}
        for c, d in budgets.items()
    }
    matched = {
        (round(np.mean(d["trials"]), 3), round(np.mean(d["coach"]), 3)) for d in budgets.values()
    }
    summary["budget_matched"] = bool(len(matched) <= 1)

    # --- carryover heterogeneity (exploratory headline) ----------------------
    het = []
    for p in per_participant:
        for c, row in p.get("conditions", {}).items():
            car = row.get("carryover") or {}
            eff = car.get("effect") or {}
            het.append({
                "participant_id": p.get("participant_id"),
                "condition": c,
                "lambda": (car.get("mean") or {}).get("lambda"),
                "beta_g": (car.get("mean") or {}).get("beta_g"),
                "beta_g_ci": eff.get("beta_g_ci"),
                "detected": eff.get("detected"),
            })
    summary["carryover_heterogeneity"] = het
    detected = [h for h in het if h.get("detected")]
    summary["carryover_go_no_go"] = {
        "note": (
            "§12.7 go/no-go: a measurable, decaying carryover effect must exist for the "
            "scheduling question to be answerable as posed"
        ),
        "n_condition_blocks": len(het),
        "n_with_detected_effect": len(detected),
        "fraction_detected": (len(detected) / len(het)) if het else float("nan"),
    }

    # --- burden ---------------------------------------------------------------
    burden: Dict[str, List[float]] = defaultdict(list)
    for p in per_participant:
        for q in p.get("questionnaires", []) or []:
            for k, v in (q.get("scores") or {}).items():
                if isinstance(v, (int, float)):
                    burden[f"{q.get('instrument')}.{k}"].append(float(v))
    if burden:
        summary["burden"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in burden.items()}
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _fig_pi_star(per_participant: Sequence[Dict[str, Any]], data: Sequence[ParticipantData], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    if not per_participant:
        return
    p, pd0 = per_participant[0], data[0]
    ids = sorted(int(k) for k in p["reference_map"])
    # One line per depth row: two targets share each lateral position, and a single line sorted
    # by ``s`` would double back on itself and read as noise.
    depths = sorted({pd0.grid.get(i).depth_bin for i in ids})
    rows = {
        d: sorted(
            (i for i in ids if pd0.grid.get(i).depth_bin == d),
            key=lambda i: nonpreferred_lateral(pd0.grid.get(i).y_m, pd0.side),
        )
        for d in depths
    }
    styles = ["-", "--", ":", "-."]

    def xs(row_ids):
        return [nonpreferred_lateral(pd0.grid.get(i).y_m, pd0.side) for i in row_ids]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, d in enumerate(depths):
        depth_m = pd0.grid.get(rows[d][0]).x_m
        ax.plot(
            xs(rows[d]), [float(p["reference_map"][str(i)]) for i in rows[d]],
            "o" + styles[k % len(styles)], color="black", lw=2, ms=4,
            label=f"reference $\\tilde\\pi^*$ (depth {depth_m:.2f} m)",
        )
    colors = {}
    for ci, (cond, row) in enumerate(sorted(p.get("conditions", {}).items())):
        colors[cond] = f"C{ci}"
        for k, d in enumerate(depths):
            ax.plot(
                xs(rows[d]), [float(row["pi_hat"][str(i)]) for i in rows[d]],
                styles[k % len(styles)], color=colors[cond], lw=1.3, alpha=0.85,
                label=(cond if k == 0 else None),
            )
    band = pd0.grid.cfg.crossover_halfwidth_m
    ax.axvspan(-band, band, color="#cccccc", alpha=0.35, zorder=0, label="crossover band")
    ax.axhline(0.5, color="#888888", lw=0.7, ls=":")
    ax.set_xlabel("lateral position toward the nonpreferred side, $s$ (m)")
    ax.set_ylabel("$\\pi^*$ = P(nonpreferred arm)")
    ax.set_title(f"Arm-choice map, participant {p['participant_id']}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_pi_star_map.{fmt}", dpi=150)
    plt.close(fig)


def _fig_primary(summary: Dict[str, Any], per_participant: Sequence[Dict[str, Any]], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    conds = sorted(summary.get("conditions", {}))
    if not conds:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    means = [summary["conditions"][c]["mae"]["mean"] for c in conds]
    sds = [summary["conditions"][c]["mae"]["sd"] for c in conds]
    ax1.bar(range(len(conds)), means, yerr=sds, capsize=3, color="#4c72b0", edgecolor="black", linewidth=0.4)
    for pi, p in enumerate(per_participant):
        ys = [p.get("conditions", {}).get(c, {}).get("mae", float("nan")) for c in conds]
        ax1.plot(range(len(conds)), ys, "o-", color="#333333", alpha=0.35, lw=0.8, ms=3)
    ax1.set_xticks(range(len(conds)))
    ax1.set_xticklabels(conds, rotation=25, ha="right", fontsize=7)
    ax1.set_ylabel("crossover-weighted MAE of $\\hat\\pi^*$")
    ax1.set_title("Primary outcome: estimation error")
    ax1.grid(axis="y", alpha=0.3)

    cov = [summary["conditions"][c]["coverage"]["mean"] for c in conds]
    ax2.bar(range(len(conds)), cov, color="#55a868", edgecolor="black", linewidth=0.4)
    ax2.axhline(0.9, color="#c44e52", ls="--", lw=1.2, label="nominal 90%")
    ax2.set_xticks(range(len(conds)))
    ax2.set_xticklabels(conds, rotation=25, ha="right", fontsize=7)
    ax2.set_ylabel("credible-interval coverage")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Primary outcome: calibration")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_primary_outcomes.{fmt}", dpi=150)
    plt.close(fig)


def _fig_heterogeneity(summary: Dict[str, Any], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    het = [h for h in summary.get("carryover_heterogeneity", []) if h.get("lambda") is not None]
    if not het:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs = [float(h["lambda"]) for h in het]
    ys = [float(h["beta_g"]) for h in het]
    colors = ["#c44e52" if h.get("detected") else "#999999" for h in het]
    ax.scatter(xs, ys, c=colors, s=28, edgecolor="black", linewidth=0.4)
    for h in het:
        ci = h.get("beta_g_ci")
        if ci:
            ax.plot([float(h["lambda"])] * 2, [float(ci[0]), float(ci[1])], color="#666666", lw=0.8, alpha=0.6)
    ax.set_xlabel("per-person decay $\\lambda$")
    ax.set_ylabel("prompt effect $\\beta\\cdot g$ (logits)")
    ax.set_title("Carryover heterogeneity (red = effect credibly > 0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_carryover_heterogeneity.{fmt}", dpi=150)
    plt.close(fig)


def _fig_budget(summary: Dict[str, Any], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    b = summary.get("budget_manipulation_check", {})
    if not b:
        return
    conds = sorted(b)
    keys = ["coach", "assess", "wait"]
    colors = {"coach": "#c44e52", "assess": "#4c72b0", "wait": "#dd8452"}
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(conds))
    for k in keys:
        vals = np.array([float(b[c].get(k, {}).get("mean", 0.0)) for c in conds])
        ax.bar(range(len(conds)), vals, bottom=bottom, label=k, color=colors[k], edgecolor="black", linewidth=0.4)
        bottom += vals
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("slots per block")
    ax.set_title(
        "Budget manipulation check — matched"
        if summary.get("budget_matched") else "Budget manipulation check — NOT matched"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_budget_check.{fmt}", dpi=150)
    plt.close(fig)


def _fig_test_retest(per_participant: Sequence[Dict[str, Any]], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    pts = [(float(v), float(p["retest_map"][k])) for p in per_participant if p.get("retest_map")
           for k, v in p["reference_map"].items() if k in p.get("retest_map", {})]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1)
    ax.scatter([a for a, _ in pts], [b for _, b in pts], s=18, alpha=0.6, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("reference block $\\tilde\\pi^*$")
    ax.set_ylabel("retest block $\\pi^*$")
    ax.set_title("Test-retest stability of the reference map")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_test_retest.{fmt}", dpi=150)
    plt.close(fig)


def _fig_calibration(per_participant: Sequence[Dict[str, Any]], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1, label="perfect")
    agg: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for p in per_participant:
        for cond, row in (p.get("conditions") or {}).items():
            bins = ((row.get("calibration") or {}).get("bins") or {})
            for c, a, n in zip(bins.get("bin_conf", []), bins.get("bin_acc", []), bins.get("bin_count", [])):
                if n and c == c and a == a:
                    agg[cond].append((float(c), float(a), int(n)))
    for cond, rows in sorted(agg.items()):
        if not rows:
            continue
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            sel = [(c, a, n) for c, a, n in rows if edges[i] <= c < edges[i + 1] or (i == 9 and c == 1.0)]
            if not sel:
                continue
            w = sum(n for _, _, n in sel)
            xs.append(sum(c * n for c, _, n in sel) / w)
            ys.append(sum(a * n for _, a, n in sel) / w)
        if xs:
            ax.plot(xs, ys, "o-", ms=3, lw=1.1, label=cond)
    ax.set_xlabel("predicted P(nonpreferred)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration of $\\hat\\pi^*$ against realized choices")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_calibration.{fmt}", dpi=150)
    plt.close(fig)


def _fig_ablation(summary: Dict[str, Any], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    dec = summary.get("ablation_decomposition") or {}
    keys = [("total_b4_vs_b2", "B4 (both)"), ("schedule_only_vs_b2", "schedule only"), ("estimator_only_vs_b2", "estimator only")]
    rows = [(lbl, dec[k]) for k, lbl in keys if isinstance(dec.get(k), dict) and dec[k].get("n")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ys = range(len(rows))
    means = [r[1]["mean"] for r in rows]
    los = [r[1]["ci95_of_difference"][0] for r in rows]
    his = [r[1]["ci95_of_difference"][1] for r in rows]
    ax.errorbar(means, list(ys), xerr=[[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]],
                fmt="o", color="#4c72b0", capsize=4)
    ax.axvline(0, color="#c44e52", ls="--", lw=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("MAE difference vs. fixed washout (negative = better)")
    ax.set_title("Where B4's advantage comes from")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"rehab_ablation.{fmt}", dpi=150)
    plt.close(fig)


def make_figures(
    summary: Dict[str, Any],
    per_participant: Sequence[Dict[str, Any]],
    data: Sequence[ParticipantData],
    out_dir: Path,
    fmt: str = "pdf",
) -> List[str]:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[rehab.analyze] matplotlib not installed; skipping figures.")
        return []
    import matplotlib

    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    _fig_pi_star(per_participant, data, out_dir, fmt)
    _fig_primary(summary, per_participant, out_dir, fmt)
    _fig_calibration(per_participant, out_dir, fmt)
    _fig_heterogeneity(summary, out_dir, fmt)
    _fig_budget(summary, out_dir, fmt)
    _fig_test_retest(per_participant, out_dir, fmt)
    _fig_ablation(summary, out_dir, fmt)
    return sorted(str(p.name) for p in out_dir.glob(f"rehab_*.{fmt}"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append", default=[], help="session directory (repeatable)")
    ap.add_argument("--session-root", type=str, default=None, help="analyze every session under this root")
    ap.add_argument("--out-dir", type=str, default="vla_lab/results/rehab_phase0")
    ap.add_argument("--figures-dir", type=str, default=None,
                    help="where figures go (default: <out-dir>). Point at vla_lab/paper/figures/ to write straight into the paper.")
    ap.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--level", type=float, default=0.9, help="credible level for coverage")
    ap.add_argument("--reference-condition", type=str, default=CONDITION_CARRYOVER_AWARE)
    ap.add_argument("--comparator", type=str, default=CONDITION_FIXED_WASHOUT)
    ap.add_argument("--ground-truth", type=str, default=None, help="ground_truth.json from a synthetic pilot")
    ap.add_argument("--offpolicy", action="store_true", help="also run the model-based off-policy secondary analysis (§12.6)")
    ap.add_argument("--offpolicy-reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    dirs = [Path(s) for s in args.session]
    if args.session_root:
        dirs += find_sessions(args.session_root)
    if not dirs:
        ap.error("give --session and/or --session-root")

    truth: Dict[str, Any] = {}
    if args.ground_truth and Path(args.ground_truth).exists():
        truth = json.loads(Path(args.ground_truth).read_text())

    data: List[ParticipantData] = []
    per_participant: List[Dict[str, Any]] = []
    for d in dirs:
        pd = ParticipantData(d)
        gt = (truth.get(pd.pid) or {}).get("pi_star")
        res = analyze_participant(pd, level=float(args.level), ground_truth=gt)
        if args.offpolicy:
            from .scheduler import COMPARED_CONDITIONS

            run = set(res.get("conditions", {}))
            missing = [c for c in COMPARED_CONDITIONS if c not in run]
            if missing:
                ref_map = {int(k): float(v) for k, v in res["reference_map"].items()}
                any_cond = next(iter(res["conditions"].values()), None)
                carry = joint_carryover_posterior(pd._seq(list(pd.records)), pd.grid, cfg=pd.carryover_cfg)
                res["offpolicy"] = offpolicy_evaluate(
                    pd, missing, reference_map=ref_map, carry=carry,
                    n_reps=int(args.offpolicy_reps), seed=int(args.seed),
                )
        data.append(pd)
        per_participant.append(res)

    summary = aggregate(
        per_participant,
        reference_condition=str(args.reference_condition),
        comparator=str(args.comparator),
        seed=int(args.seed),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rehab_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "rehab_per_participant.json").write_text(json.dumps(per_participant, indent=2, default=float))

    # --- console report, in the order a reader should meet it -----------------
    print(f"[rehab.analyze] {summary['n_participants']} participant(s), conditions: {sorted(summary['conditions'])}")
    tr = summary.get("test_retest")
    if tr:
        print(f"[rehab.analyze] test-retest of the reference map: MAE={tr['mae_mean']:.4f}  r={tr['pearson_r_mean']:.3f}  (n={tr['n']})")
        print("[rehab.analyze]   ^ read this before the primary outcome: it bounds irreducible drift (§12.2)")
    print("[rehab.analyze] primary outcome (crossover-weighted MAE, lower is better):")
    for c in sorted(summary["conditions"]):
        m = summary["conditions"][c]
        print(
            f"    {c:26s} MAE={m['mae']['mean']:.4f}±{m['mae']['sd']:.4f}  "
            f"coverage={m['coverage']['mean']:.2f}  ECE={m['ece']['mean']:.3f}  "
            f"obs={m['observations']['mean']:.0f}"
        )
    pc = summary.get("primary_contrast", {})
    if pc.get("n"):
        lo, hi = pc["ci95_of_difference"]
        print(
            f"[rehab.analyze] PRIMARY {pc['a']} - {pc['b']}: {pc['mean']:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}] dz={pc['dz']:+.2f} "
            f"Wilcoxon p={pc['wilcoxon']['p_value']:.4f} ({pc['wilcoxon']['method']}, n={pc['n']})"
        )
    gng = summary.get("carryover_go_no_go", {})
    if gng:
        print(
            f"[rehab.analyze] §12.7 go/no-go: carryover credibly > 0 in "
            f"{gng['n_with_detected_effect']}/{gng['n_condition_blocks']} condition blocks"
        )
    print(f"[rehab.analyze] budget matched across conditions: {summary.get('budget_matched')}")

    if not args.no_figures:
        fig_dir = Path(args.figures_dir) if args.figures_dir else out_dir
        names = make_figures(summary, per_participant, data, fig_dir, args.format)
        if names:
            print(f"[rehab.analyze] wrote {len(names)} figure(s) to {fig_dir}: {', '.join(names)}")
    print(f"[rehab.analyze] wrote {out_dir / 'rehab_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

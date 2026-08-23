"""W16 — Monte-Carlo power and sample size for the Phase 0 primary contrast.

    python -m vla_lab.rehab.power --n 8,12,16,24 --sims 40

Answers the question the preregistration has to answer before the first non-pilot participant:
**how many participants, and how many trials each, to detect the effect we think is there?**

It does this by running the *whole pipeline* — session runner, schedulers, observers, gate-able
logs, estimators, and the same paired contrast :mod:`vla_lab.rehab.analyze` computes — over
synthetic populations drawn from :mod:`vla_lab.rehab.sim_participant`. Power is then the
fraction of simulated studies in which the contrast is detected, not a closed-form
approximation of a test the study does not run.

The closed-form helpers from :mod:`vla_lab.human_study.power` (``sample_size_paired``,
``power_paired_t``, ``normal_ppf``) are reused for the analytic cross-check and for turning a
measured Cohen's ``dz`` into an N — the Monte-Carlo path is the primary answer and the
analytic one is the sanity check (``rehab.md`` §5: SHARE + EXTEND).

.. warning::
   **Every number this produces inherits the population prior in ``sim_participant.py``.**
   Those are assumptions, not measurements. W16's "done when" is *a power memo whose
   assumptions are traced to pilot measurements* — so run this again after M4 with the pilot's
   fitted ``(lambda, beta, g)`` and crossover spread, and say in the memo which numbers came
   from where. Running it before the pilot tells you what the design *would* detect if the
   world matched the prior, which is useful for choosing ``T`` and useless as evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...human_study.power import normal_ppf, power_paired_t, sample_size_paired
from .analyze import analyze_participant, bootstrap_ci, paired_difference, wilcoxon_signed_rank
from .carryover import CarryoverConfig
from .contract import BudgetConfig, Phase0Contract
from .protocol import Phase0Protocol
from .scheduler import CONDITION_CARRYOVER_AWARE, CONDITION_FIXED_WASHOUT
from .sim_participant import PopulationPrior


def coarse_carryover(fine: bool = False) -> CarryoverConfig:
    """A smaller ``(lambda, beta, g)`` grid for Monte-Carlo runs.

    The full grid (12x11x9) costs ~0.4 s per marginal-likelihood fit, which at 24 participants
    x 40 simulated studies x 2 conditions is over an hour. The coarse grid (7x6x5) costs ~0.1 s
    and shifts the estimated power by well under its own Monte-Carlo error. Use ``--fine`` for
    the final memo.
    """

    if fine:
        return CarryoverConfig()
    return CarryoverConfig(n_lambda=7, n_beta=6, n_g=5)


@dataclass
class PowerConfig:
    n_participants: Tuple[int, ...] = (8, 12, 16, 24)
    n_sims: int = 40
    alpha: float = 0.05
    reference_condition: str = CONDITION_CARRYOVER_AWARE
    comparator: str = CONDITION_FIXED_WASHOUT
    trials_per_block: Optional[int] = None      # sweep the per-participant budget too
    effect_scale: float = 1.0                   # multiplies the population prompt gain g
    fine_grid: bool = False
    seed: int = 20260816

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["n_participants"] = list(self.n_participants)
        return d


def _run_study(
    n_participants: int,
    *,
    contract: Phase0Contract,
    protocol: Phase0Protocol,
    prior: PopulationPrior,
    conditions: Sequence[str],
    carryover: CarryoverConfig,
    seed: int,
) -> Dict[str, List[float]]:
    """One simulated study. Returns per-condition, per-participant crossover-weighted MAE."""

    from .run_pilot import run_one

    out: Dict[str, List[float]] = {c: [] for c in conditions}
    with tempfile.TemporaryDirectory(prefix="rehab_power_") as tmp:
        for i in range(int(n_participants)):
            result, participant = run_one(
                i, contract, protocol,
                prior=prior, log_root=tmp, seed=seed, conditions=list(conditions),
            )
            from .analyze import ParticipantData

            pd = ParticipantData(result.session_dir, carryover_cfg=carryover)
            res = analyze_participant(pd)
            for c in conditions:
                row = res.get("conditions", {}).get(c)
                out[c].append(float(row["mae"]) if row else float("nan"))
    return out


def monte_carlo_power(cfg: Optional[PowerConfig] = None, *, quiet: bool = False) -> Dict[str, Any]:
    """Sweep ``N`` (and optionally the per-participant budget); estimate power at each point."""

    cfg = cfg or PowerConfig()
    protocol = Phase0Protocol()
    prior = PopulationPrior()
    if cfg.effect_scale != 1.0:
        prior = PopulationPrior(**{**prior.to_dict(), "g_mean": prior.g_mean * float(cfg.effect_scale)})
    conditions = (cfg.reference_condition, cfg.comparator)
    carry = coarse_carryover(cfg.fine_grid)

    base_contract = Phase0Contract()
    if cfg.trials_per_block:
        base_contract = Phase0Contract(
            budget=BudgetConfig(
                **{**base_contract.budget.to_dict(), "trials_per_block": int(cfg.trials_per_block)}
            )
        )

    rows: List[Dict[str, Any]] = []
    for n in cfg.n_participants:
        detections = 0
        diffs: List[float] = []
        for s in range(int(cfg.n_sims)):
            study = _run_study(
                n, contract=base_contract, protocol=protocol, prior=prior,
                conditions=conditions, carryover=carry, seed=int(cfg.seed) + 10007 * s,
            )
            a, b = study[cfg.reference_condition], study[cfg.comparator]
            pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
            if len(pairs) < 3:
                continue
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            d = paired_difference(xs, ys)
            diffs.append(float(d["mean"]))
            w = wilcoxon_signed_rank(xs, ys)
            # One-sided in the pre-registered direction: the proposed policy should have the
            # LOWER error. A two-sided p that "detects" the wrong sign is not a detection.
            if w["p_value"] <= cfg.alpha and d["mean"] < 0:
                detections += 1
        power = detections / max(1, int(cfg.n_sims))
        dz = float(np.mean(diffs) / np.std(diffs, ddof=1)) if len(diffs) > 1 and np.std(diffs, ddof=1) > 0 else float("nan")
        row = {
            "n_participants": int(n),
            "n_sims": int(cfg.n_sims),
            "power": float(power),
            "mean_difference": float(np.mean(diffs)) if diffs else float("nan"),
            "sd_of_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan"),
            "ci95_of_mean_difference": list(bootstrap_ci(diffs, seed=cfg.seed)) if diffs else [float("nan")] * 2,
        }
        rows.append(row)
        if not quiet:
            print(
                f"[rehab.power] N={n:3d}  power={power:.2f}  mean diff={row['mean_difference']:+.4f} "
                f"(sd {row['sd_of_difference']:.4f}) over {cfg.n_sims} simulated studies"
            )
    return {"config": cfg.to_dict(), "sweep": rows}


def analytic_cross_check(dz: float, *, alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
    """Closed-form paired-t N for a measured Cohen's ``dz``, from the shared power module.

    A cross-check, not the answer: it assumes a paired t on a normal DV, while the study runs a
    signed-rank test on an MAE. Expect it to be *optimistic* relative to the Monte-Carlo sweep.
    """

    n = sample_size_paired(float(dz), alpha, power)
    return {
        "dz": float(dz),
        "alpha": float(alpha),
        "target_power": float(power),
        "n_paired_t": int(n),
        "achieved_power_at_n": float(power_paired_t(float(dz), int(n), alpha)),
        "z_alpha": float(normal_ppf(1.0 - alpha / 2.0)),
        "note": "paired-t approximation; the Monte-Carlo sweep is the primary answer",
    }


def minimum_detectable_effect(sweep: Sequence[Dict[str, Any]], *, target_power: float = 0.8) -> Dict[str, Any]:
    """The smallest ``N`` in the sweep that reaches ``target_power``, if any."""

    ok = [r for r in sweep if float(r.get("power", 0.0)) >= float(target_power)]
    if not ok:
        best = max(sweep, key=lambda r: float(r.get("power", 0.0))) if sweep else {}
        return {
            "reached": False,
            "target_power": float(target_power),
            "best_power": float(best.get("power", float("nan"))) if best else float("nan"),
            "best_n": int(best.get("n_participants", 0)) if best else 0,
            "note": "no N in the swept range reached the target; widen --n or raise the effect",
        }
    smallest = min(ok, key=lambda r: int(r["n_participants"]))
    return {
        "reached": True,
        "target_power": float(target_power),
        "n_recommended": int(smallest["n_participants"]),
        "power_at_n": float(smallest["power"]),
        "mean_difference": float(smallest["mean_difference"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=str, default="8,12,16,24", help="comma-separated participant counts to sweep")
    ap.add_argument("--sims", type=int, default=40, help="simulated studies per N")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-power", type=float, default=0.8)
    ap.add_argument("--effect-scale", type=float, default=1.0, help="multiplies the population prompt gain g")
    ap.add_argument("--trials-per-block", type=int, default=None, help="override T (sweep the per-participant budget)")
    ap.add_argument("--reference-condition", type=str, default=CONDITION_CARRYOVER_AWARE)
    ap.add_argument("--comparator", type=str, default=CONDITION_FIXED_WASHOUT)
    ap.add_argument("--fine", action="store_true", help="full carryover grid (slower; use for the final memo)")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", type=str, default="vla_lab/results/rehab_phase0/power_memo.json")
    args = ap.parse_args(argv)

    cfg = PowerConfig(
        n_participants=tuple(int(x) for x in str(args.n).split(",") if x.strip()),
        n_sims=int(args.sims),
        alpha=float(args.alpha),
        reference_condition=str(args.reference_condition),
        comparator=str(args.comparator),
        trials_per_block=args.trials_per_block,
        effect_scale=float(args.effect_scale),
        fine_grid=bool(args.fine),
        seed=int(args.seed),
    )
    print(f"[rehab.power] {cfg.reference_condition} vs {cfg.comparator}, crossover-weighted MAE, "
          f"alpha={cfg.alpha}, {cfg.n_sims} simulated studies per N")
    print("[rehab.power] assumptions come from sim_participant.PopulationPrior — priors, not "
          "measurements; re-run after the M4 pilot with fitted values (see the module docstring)")
    result = monte_carlo_power(cfg)
    mde = minimum_detectable_effect(result["sweep"], target_power=float(args.target_power))
    result["minimum_detectable"] = mde

    diffs = [r["mean_difference"] for r in result["sweep"] if r["mean_difference"] == r["mean_difference"]]
    sds = [r["sd_of_difference"] for r in result["sweep"] if r["sd_of_difference"] == r["sd_of_difference"]]
    if diffs and sds and max(sds) > 0:
        dz = float(np.mean(diffs) / np.mean(sds))
        result["analytic_cross_check"] = analytic_cross_check(dz, alpha=cfg.alpha, power=float(args.target_power))
        print(f"[rehab.power] analytic cross-check: dz={dz:+.2f} -> n={result['analytic_cross_check']['n_paired_t']} "
              "(paired-t approximation; expect it to be optimistic)")

    if mde.get("reached"):
        print(f"[rehab.power] => N={mde['n_recommended']} reaches power {mde['power_at_n']:.2f} "
              f"at a mean difference of {mde['mean_difference']:+.4f}")
    else:
        print(f"[rehab.power] => target power {args.target_power} NOT reached in the swept range "
              f"(best {mde.get('best_power', float('nan')):.2f} at N={mde.get('best_n')})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"[rehab.power] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

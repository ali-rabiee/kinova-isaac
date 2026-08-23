"""W14/W16 — the synthetic Phase 0 pilot: a full study, end to end, with no robot and no humans.

    python -m vla_lab.rehab.run_pilot --participants 8 --analyze

Runs ``N`` simulated participants through the **real** session code path — the same protocol,
schedulers, observers, safety envelope, phase machine, and event-locked logging a real session
uses — with :class:`~vla_lab.rehab.sim_participant.SimulatedParticipant` behind the observer
seam and :class:`~vla_lab.rehab.apparatus.null.NullApparatus` behind the apparatus seam.

This is what makes every Tier-A work item verifiable before hardware or IRB approval exists
(``rehab.md`` §6), and it is the substrate the power analysis (W16) runs on. It is a
**rehearsal**, not evidence: every number it produces is a consequence of the population prior
in ``sim_participant.py``, which is a set of assumptions the lab pilot (M4) has to replace.

The same command with ``--apparatus real --gen2-socket <path>`` runs a real session; with
``--apparatus twin`` it runs against the Isaac digital twin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...human_study.instruments import NASA_TLX_SUBSCALES, SESSION_BURDEN_ITEMS
from .apparatus import BACKEND_NULL, make_apparatus
from .carryover import CarryoverConfig
from .contract import Phase0Contract
from .protocol import Phase0Protocol
from .scheduler import ALL_CONDITIONS
from .session import OBSERVER_SIMULATED, Phase0Session, SessionConfig, SessionResult
from .sim_participant import PopulationPrior, SimulatedObserver, SimulatedParticipant, draw_participant
from .trial import ManualClock
from .workspace import SIDE_LEFT, SIDE_RIGHT


def synthetic_handedness(nonpreferred_side: str) -> List[int]:
    """A consistent EHI response vector for a synthetic participant."""

    v = 2 if str(nonpreferred_side) == SIDE_LEFT else -2  # nonpreferred left => right-handed
    return [v] * 10


class SyntheticQuestionnaireProvider:
    """Plausible questionnaire responses that drift with session position (fatigue)."""

    def __init__(self, seed: int = 0) -> None:
        import random

        self.rng = random.Random(int(seed))
        self.n_blocks = 0

    def __call__(self, instrument: str, block_kind: str) -> Optional[Any]:
        self.n_blocks += 1
        drift = min(1.0, 0.15 * self.n_blocks)
        if instrument in ("nasa_tlx", "tlx"):
            base = 30.0 + 25.0 * drift
            return {k: max(0.0, min(100.0, base + self.rng.uniform(-8, 8))) for k in NASA_TLX_SUBSCALES}
        if instrument in ("session_burden", "burden"):
            out: Dict[str, float] = {}
            for k in SESSION_BURDEN_ITEMS:
                if k == "willing_to_continue":
                    out[k] = max(0.0, min(10.0, 9.0 - 3.0 * drift + self.rng.uniform(-0.7, 0.7)))
                else:
                    out[k] = max(0.0, min(10.0, 1.5 + 5.0 * drift + self.rng.uniform(-0.8, 0.8)))
            return out
        return None


def run_one(
    participant_idx: int,
    contract: Phase0Contract,
    protocol: Phase0Protocol,
    *,
    prior: Optional[PopulationPrior] = None,
    log_root: str = "logs/rehab_sim",
    seed: int = 0,
    conditions: Optional[Sequence[str]] = None,
    apparatus_backend: str = BACKEND_NULL,
    socket_path: Optional[str] = None,
    misdetect_rate: Optional[float] = None,
) -> tuple[SessionResult, SimulatedParticipant]:
    """Run one synthetic participant and return the session result plus the ground truth."""

    pr = prior or PopulationPrior()
    if misdetect_rate is not None:
        pr = PopulationPrior(**{**pr.to_dict(), "misdetect_rate": float(misdetect_rate)})
    params = draw_participant(participant_idx, pr, seed=seed)
    grid = contract.target_grid()

    n_blocks = 2 + len(conditions or protocol.prospective_conditions)
    total = (
        contract.budget.reference_trials
        + contract.budget.retest_trials
        + contract.budget.trials_per_block * (n_blocks - 2)
    )
    participant = SimulatedParticipant(params, grid, prior=pr, total_trials=total)
    observer = SimulatedObserver(participant)

    clock = ManualClock()
    apparatus = make_apparatus(
        apparatus_backend,
        contract,
        manual_clock=clock,
        socket_path=socket_path,
    )
    # The synthetic session runs on a manual clock, so a full study finishes in seconds while
    # its timestamps stay dimensionally identical to a real session's.
    from .trial import SessionClock

    session_clock = SessionClock(source=clock)
    apparatus.clock = session_clock

    cfg = SessionConfig(
        participant_id=f"S{participant_idx:03d}",
        participant_idx=participant_idx,
        log_root=log_root,
        seed=seed,
        poll_interval_ms=20,
        carryover=CarryoverConfig(),
        conditions=list(conditions) if conditions else None,
    )
    session = Phase0Session(
        contract,
        protocol,
        cfg,
        apparatus=apparatus,
        observer_factory=lambda kind: observer,
        observer_kind=OBSERVER_SIMULATED,
        questionnaire_provider=SyntheticQuestionnaireProvider(seed=seed + participant_idx),
        handedness_responses=synthetic_handedness(params.nonpreferred_side),
        clock=session_clock,
        manual_clock=clock,
    )
    return session.run(), participant


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--log-root", type=str, default="logs/rehab_sim")
    ap.add_argument("--config", type=str, default=None, help="configs/rehab_sim_pilot.yaml")
    ap.add_argument("--conditions", type=str, default=None,
                    help=f"comma-separated; default = the protocol's prospective set. Known: {','.join(ALL_CONDITIONS)}")
    ap.add_argument("--all-conditions", action="store_true",
                    help="run every compared condition prospectively (the synthetic study can afford it; a real session cannot)")
    ap.add_argument("--misdetect-rate", type=float, default=None, help="override the observer misdetection rate (W8 stress test)")
    ap.add_argument("--apparatus", type=str, default=BACKEND_NULL, choices=["null", "twin", "real"])
    ap.add_argument("--gen2-socket", type=str, default=None)
    ap.add_argument("--analyze", action="store_true", help="run rehab.analyze on the produced sessions")
    ap.add_argument("--out-dir", type=str, default="vla_lab/results/rehab_phase0")
    ap.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    prior: Optional[PopulationPrior] = None
    if args.config:
        import yaml

        raw = yaml.safe_load(Path(args.config).read_text()) or {}
        contract = Phase0Contract.from_dict(raw.get("contract", {}))
        protocol = Phase0Protocol.from_dict(raw.get("protocol", {}))
        if raw.get("population_prior"):
            prior = PopulationPrior.from_dict(raw["population_prior"])
    else:
        contract, protocol = Phase0Contract(), Phase0Protocol()

    conditions: Optional[Sequence[str]] = None
    if args.all_conditions:
        from .scheduler import COMPARED_CONDITIONS

        conditions = list(COMPARED_CONDITIONS)
    elif args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    problems = contract.validate() + protocol.validate()
    if problems:
        print("[rehab.pilot] configuration is invalid:")
        for p in problems:
            print(f"  - {p}")
        return 2

    truth_path = Path(args.log_root) / "ground_truth.json"
    truth: Dict[str, Any] = {}
    dirs: List[str] = []
    for i in range(int(args.participants)):
        result, participant = run_one(
            i, contract, protocol,
            prior=prior,
            log_root=args.log_root, seed=int(args.seed), conditions=conditions,
            apparatus_backend=args.apparatus, socket_path=args.gen2_socket,
            misdetect_rate=args.misdetect_rate,
        )
        dirs.append(str(result.session_dir))
        truth[result.plan.participant_id] = {
            "params": participant.p.to_dict(),
            "pi_star": {str(k): round(float(v), 6) for k, v in participant.pi_star_map().items()},
        }
        if not args.quiet:
            s = result.summary()
            print(
                f"[rehab.pilot] {s['participant_id']}: {s['n_trials']} trials "
                f"(COACH {s['by_action']['COACH']}, ASSESS {s['by_action']['ASSESS']}, WAIT {s['by_action']['WAIT']}), "
                f"{s['n_observations']} observations, {s['n_halts']} halts -> {result.session_dir}"
            )

    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"[rehab.pilot] {len(dirs)} sessions under {args.log_root} (ground truth: {truth_path})")

    if args.analyze:
        from .analyze import main as analyze_main

        return analyze_main([
            "--session-root", str(args.log_root),
            "--out-dir", str(args.out_dir),
            "--format", str(args.format),
            "--ground-truth", str(truth_path),
        ])
    return 0


if __name__ == "__main__":
    sys.exit(main())

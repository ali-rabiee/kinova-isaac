"""The one session runner. Every condition, every backend, every tier goes through here.

There is deliberately a single implementation of the slot loop. A study whose baseline
conditions run through different code than its proposed condition is not comparing schedules;
it is comparing code paths, and the difference is invisible in the results.

One design decision worth stating, because it shows up in the numbers: when an utterance
cannot be grounded, the robot still has to do something. It executes the **value-optimal**
strategy for that scene and the slot contributes **no observation** to the estimand. Both
halves matter. Executing something is what a deployed system would do, and the choice of the
value-optimal fallback is the neutral one -- falling back to whatever was last demonstrated
would inject the very bias the study is measuring, and falling back at random would add noise
to the execution outcomes for no reason. Contributing no observation is what keeps a hedge
("do what you did last time") out of the preference estimate, where it would masquerade as
compliance.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED, WAIT
from .apparatus.base import Apparatus, Grounder, SupervisorChannel
from .contract import Contract
from .estimand import Observation, sequence_from_records
from .logging import CONTRACT_FILE, PROTOCOL_FILE, TRUTH_FILE, SessionLogger
from .narration import coach_narration, counter_query, probe_query, wait_filler
from .protocol import BLOCK_CONDITION, BLOCK_REFERENCE, BLOCK_RETEST, Block, Protocol
from .scheduler import build_scheduler
from .scheduler.base import BlockBudget, DeltaModel, History, HistoryRecord, Scheduler, Slot
from .strategies import get_axis


@dataclass
class BlockResult:
    """One block's records, plus whatever belief the policy ended with."""

    block_index: int
    kind: str
    condition: str
    records: List[Dict[str, Any]]
    scheduler: Dict[str, Any]
    belief: Optional[Dict[str, Any]] = None

    def observations(self, contract: Contract) -> List[Observation]:
        return sequence_from_records(self.records, contract.grid)


@dataclass
class SessionResult:
    supervisor_id: str
    contract_hash: str
    blocks: List[BlockResult]
    root: Optional[Path] = None
    truth: Optional[Dict[str, Any]] = None

    def block(self, kind: str) -> Optional[BlockResult]:
        return next((b for b in self.blocks if b.kind == kind), None)

    def condition_blocks(self) -> List[BlockResult]:
        return [b for b in self.blocks if b.kind == BLOCK_CONDITION]


def run_block(
    *,
    contract: Contract,
    block: Block,
    apparatus: Apparatus,
    channel: SupervisorChannel,
    grounder: Grounder,
    scheduler: Scheduler,
    logger: Optional[SessionLogger] = None,
    seed: int = 0,
    second_grounder: Optional[Grounder] = None,
) -> BlockResult:
    """Run one block, slot by slot.

    ``second_grounder`` is optional and never authoritative: when supplied, its label is logged
    alongside the primary one so that grounder agreement can be reported. The scheduler acts on
    the primary label only.
    """
    axis = get_axis(contract.axis)
    dm = contract.delta_model()
    dose = contract.dose_spec()
    budget = block.budget
    rng = random.Random(int(seed) * 7919 + block.index)
    scheduler.reset(budget)
    history = History()
    records: List[Dict[str, Any]] = []

    coach_seen = 0
    for i in range(budget.n_slots):
        is_coach = i in set(budget.coach_slots)
        scene_id = (
            budget.coach_scene_sequence[coach_seen]
            if is_coach and coach_seen < len(budget.coach_scene_sequence)
            else budget.scene_sequence[i]
        )
        slot = Slot(
            index=i,
            scene_id=int(scene_id),
            is_coach_slot=is_coach,
            coach_direction=budget.direction_at(i),
            coach_strength=float(budget.coach_strength),
            free_remaining=budget.free_remaining_at(i),
            session_progress=i / max(1, budget.n_slots - 1),
        )
        decision = scheduler.decide(history, slot)
        scene = contract.grid.by_id(int(decision.scene_id))
        apparatus.reset_scene(scene)

        row: Dict[str, Any] = {
            "slot": i,
            "block": block.index,
            "block_kind": block.kind,
            "condition": block.condition,
            "action": decision.action,
            "scene_id": int(scene.scene_id),
            "c": float(scene.c),
            "clutter": int(scene.clutter),
            "margin_m": float(scene.margin_m),
            "coach_direction": int(slot.coach_direction),
            "coach_strength": float(slot.coach_strength),
        }
        truth_row: Dict[str, Any] = {"slot": i, "block": block.index}

        if decision.action == COACH:
            strategy = STRATEGY_A if slot.coach_direction > 0 else STRATEGY_B
            for rep in range(int(dose.repeats)):
                text = coach_narration(axis, strategy, dose=dose, rng=rng)
                apparatus.say(text)
                outcome = apparatus.execute(scene, strategy)
                channel.observe_demonstration(scene, strategy, text, strength=float(slot.coach_strength))
            row.update(
                {
                    "demonstrated_strategy": strategy,
                    "narration": text,
                    "repeats": int(dose.repeats),
                    "executed_strategy": strategy,
                    "success": bool(outcome.success),
                    "duration_s": float(outcome.duration_s) * int(dose.repeats) + contract.timing.inter_slot_s,
                }
            )

        elif decision.action == WAIT:
            text = wait_filler(rng)
            apparatus.say(text)
            row.update({"narration": text, "duration_s": float(contract.timing.wait_s + contract.timing.inter_slot_s)})

        else:  # PROBE or COUNTER
            if decision.action == COUNTER:
                query = counter_query(axis, STRATEGY_A if history.last_coach_direction() > 0 else STRATEGY_B, rng=rng)
            else:
                query = probe_query()
            apparatus.say(query)
            turn = channel.ask(query, scene, action=decision.action, session_progress=slot.session_progress)
            # A learned grounder is given the same carryover context it was trained on, built
            # from the scheduler's live belief rather than from ground truth. Grounders that do
            # not want it (the lexical reference channel) simply do not implement the hook.
            _supply_context(grounder, scheduler, history, scene)
            _supply_context(second_grounder, scheduler, history, scene)
            grounded = grounder.ground(turn.utterance, scene)
            executed = grounded if grounded in (STRATEGY_A, STRATEGY_B) else contract.grid.physics.optimal_strategy(
                scene.margin_m
            )
            outcome = apparatus.execute(scene, executed)
            row.update(
                {
                    "query": query,
                    "utterance": turn.utterance,
                    "instructed_strategy": grounded if grounded in (STRATEGY_A, STRATEGY_B) else None,
                    "grounded": grounded,
                    "grounder": grounder.name,
                    "fallback_used": grounded not in (STRATEGY_A, STRATEGY_B),
                    "executed_strategy": executed,
                    "success": bool(outcome.success),
                    "duration_s": float(
                        outcome.duration_s
                        + turn.latency_s
                        + (
                            contract.timing.counter_overhead_s
                            if decision.action == COUNTER
                            else contract.timing.probe_overhead_s
                        )
                        + contract.timing.inter_slot_s
                    ),
                }
            )
            if second_grounder is not None:
                row["grounded_secondary"] = second_grounder.ground(turn.utterance, scene)
                row["grounder_secondary"] = second_grounder.name
            if turn.truth:
                truth_row.update(turn.truth)

        delta = dm.for_action(decision.action)
        if str(contract.carryover.decay_mode) == "time":
            delta = float(row.get("duration_s", dm.duration_s(decision.action))) / max(
                1e-9, float(contract.carryover.time_unit_s)
            )
        row["delta"] = float(delta)
        row["rationale"] = decision.rationale

        rec = HistoryRecord(
            slot=i,
            action=decision.action,
            scene_id=int(scene.scene_id),
            delta=float(delta),
            coach_direction=int(slot.coach_direction),
            coach_strength=float(slot.coach_strength),
            instructed=row.get("instructed_strategy"),
            c=float(scene.c),
            clutter=int(scene.clutter),
            duration_s=float(row.get("duration_s", 0.0)),
        )
        history.append(rec)
        scheduler.observe(rec)
        channel.elapse(float(delta))
        if is_coach:
            coach_seen += 1

        records.append(row)
        if logger is not None:
            logger.trial({k: v for k, v in row.items() if k != "rationale"})
            logger.belief({"slot": i, "block": block.index, "action": decision.action, **decision.rationale})
            if truth_row.keys() - {"slot", "block"}:
                logger.event("supervisor_truth", truth_row)

    belief = None
    if hasattr(scheduler, "belief_summary"):
        belief = scheduler.belief_summary()  # type: ignore[attr-defined]
    return BlockResult(block.index, block.kind, block.condition, records, scheduler.describe(), belief)


def run_session(
    *,
    contract: Contract,
    protocol: Protocol,
    apparatus: Apparatus,
    channel: SupervisorChannel,
    grounder: Grounder,
    seed: int = 0,
    log_root: Optional[Path] = None,
    truth: Optional[Dict[str, Any]] = None,
    second_grounder: Optional[Grounder] = None,
    inter_block_rest_deltas: float = 12.0,
) -> SessionResult:
    """Run every block of one supervisor's protocol, in order.

    ``inter_block_rest_deltas`` is the enforced between-block rest, in decay units. It is large
    on purpose: blocks must not contaminate each other, or the counterbalanced ordering stops
    protecting the comparison. The terminal retest block is what checks that it worked.
    """
    logger: Optional[SessionLogger] = None
    if log_root is not None:
        logger = SessionLogger(Path(log_root), supervisor_id=protocol.supervisor_id)
        contract.save(Path(log_root) / CONTRACT_FILE)
        protocol.save(Path(log_root) / PROTOCOL_FILE)

    blocks: List[BlockResult] = []
    for blk in protocol.blocks:
        if logger is not None:
            logger.event("block_start", {"index": blk.index, "kind": blk.kind, "condition": blk.condition})
        scheduler = build_scheduler(
            blk.condition,
            contract.grid,
            carryover_cfg=contract.carryover,
            delta_model=contract.delta_model(),
            seed=int(seed) + blk.index,
        )
        blocks.append(
            run_block(
                contract=contract,
                block=blk,
                apparatus=apparatus,
                channel=channel,
                grounder=grounder,
                scheduler=scheduler,
                logger=logger,
                seed=int(seed),
                second_grounder=second_grounder,
            )
        )
        channel.elapse(float(inter_block_rest_deltas))
        if logger is not None:
            logger.event("block_end", {"index": blk.index, "rest_deltas": float(inter_block_rest_deltas)})

    if logger is not None:
        if truth is not None:
            logger.write_json(TRUTH_FILE, truth)
        logger.close(
            {
                "contract_hash": contract.hash(),
                "narration_hash": contract.narration_hash(),
                "apparatus": apparatus.describe(),
                "channel": channel.describe(),
                "grounder": grounder.name,
                "blocks": [{"index": b.block_index, "kind": b.kind, "condition": b.condition} for b in blocks],
            }
        )
        apparatus.close()

    return SessionResult(
        supervisor_id=protocol.supervisor_id,
        contract_hash=contract.hash(),
        blocks=blocks,
        root=Path(log_root) if log_root else None,
        truth=truth,
    )


def _supply_context(grounder: Optional[Any], scheduler: Scheduler, history: History, scene) -> None:
    """Hand a context-aware grounder the belief the policy currently holds."""
    if grounder is None or not hasattr(grounder, "set_context"):
        return
    from ..policy.context import CarryoverContext

    post = getattr(scheduler, "posterior", None)
    mean = post.mean() if post is not None else {}
    cont = post.contamination() if post is not None else {"mean": 0.0, "sd": 0.0}
    recent: List[Tuple[str, int, bool]] = []
    ago = 0
    for rec in reversed(history):
        ago += 1
        if rec.action == COACH:
            recent.append((STRATEGY_A if rec.coach_direction > 0 else STRATEGY_B, ago, True))
            if len(recent) >= 4:
                break
    grounder.set_context(CarryoverContext(
        kappa=float(cont.get("mean", 0.0)),
        kappa_sd=float(cont.get("sd", 0.0)),
        lambda_hat=float(mean.get("lambda", 0.5)),
        beta_g_hat=float(mean.get("beta_g", 0.0)),
        slots_since_coach=int(history.n_since_last_coach()),
        recent=tuple(recent),
        scene_c=float(scene.c),
    ))


__all__ = ["BlockResult", "SessionResult", "run_block", "run_session"]

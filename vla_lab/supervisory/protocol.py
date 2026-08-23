"""Block layout, counterbalancing, and the matched budget.

**Reference-first / retest-last.** The estimand is latent, so "estimation error" needs an
operational reference, and where that reference sits in the session is not a detail:

1. a **no-coach reference block** at session start, uncontaminated by construction, defining
   the reference map;
2. the **compared condition blocks**, in balanced-Latin-square order with an enforced
   inter-block rest;
3. a terminal **no-coach retest block**.

The retest does two things without which the primary outcome cannot be read. It gives a
**test-retest reliability** estimate for the reference map, which bounds how much of any
measured error is irreducible within-session drift rather than contamination. And it gives a
**residual-contamination check**: if the terminal reference differs systematically from the
initial one, the residue outlived the between-block rest and the analysis must say so.

The cost -- the reference block's position is confounded with session position -- is accepted,
because the reference is a *measurement* rather than a compared condition and the terminal
retest quantifies exactly the confound it introduces. The alternative, counterbalancing the
reference into contaminated positions, would make the reference itself unreliable.

**Budget matching is structural.** The scene sequence, the demonstration slot positions, and
the demonstration directions are drawn **once per supervisor** and reused by every compared
condition. ``T``, ``C``, the scenes, and the directions are therefore identical by
construction, not by bookkeeping; the realized-budget manipulation check can then only fail if
something is genuinely broken.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..human_study.protocol import balanced_latin_square
from .contract import Contract
from .scheduler import COMPARED_CONDITIONS, CONDITION_CARRYOVER_AWARE, CONDITION_NO_COACH, PRIMARY_COMPARATOR
from .scheduler.base import BlockBudget

PROTOCOL_SCHEMA = "vla_lab_supervisory_protocol/v1"

BLOCK_REFERENCE = "reference"
BLOCK_RETEST = "retest"
BLOCK_CONDITION = "condition"


@dataclass
class Block:
    """One block: a condition, a budget, and where it sits in the session."""

    index: int
    kind: str
    condition: str
    budget: BlockBudget

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "condition": self.condition, "budget": self.budget.to_dict()}


@dataclass
class Protocol:
    """One supervisor's fully-determined session plan, written to disk before slot zero."""

    supervisor_id: str
    seed: int
    contract_hash: str
    blocks: List[Block]
    prospective_conditions: List[str]
    session_sign: int = 1
    schema: str = PROTOCOL_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "supervisor_id": self.supervisor_id,
            "seed": int(self.seed),
            "contract_hash": self.contract_hash,
            "prospective_conditions": list(self.prospective_conditions),
            "session_sign": int(self.session_sign),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def condition_blocks(self) -> List[Block]:
        return [b for b in self.blocks if b.kind == BLOCK_CONDITION]

    def reference_block(self) -> Block:
        return next(b for b in self.blocks if b.kind == BLOCK_REFERENCE)

    def retest_block(self) -> Optional[Block]:
        return next((b for b in self.blocks if b.kind == BLOCK_RETEST), None)


def _draw_scene_sequence(rng: random.Random, contract: Contract, n: int) -> Tuple[int, ...]:
    """Scenes for the free slots: a shuffled repeat of the probe set, band-weighted.

    Weighting by the crossover-band weights concentrates the budget where the estimand
    actually has information, which is the same reason the grid is densified there.
    """
    scenes = contract.grid.probe_scenes()
    weights = contract.grid.band_weights(crossover_weighted=True)
    ids = [s.scene_id for s in scenes]
    w = [max(weights.get(i, 0.0), 1e-6) for i in ids]
    # Systematic sampling: allocate proportionally, then shuffle. Deterministic counts beat
    # multinomial draws here -- with ~30 slots a random draw can miss a band scene entirely,
    # and a scene with no observations is a hole in the map, not noise.
    # Floor of one slot per scene whenever the budget allows it: a scene with zero
    # observations is a hole in the map that only the psychometric extrapolation can fill, and
    # the pooled estimator would be scoring its prior there rather than any data.
    alloc: List[int] = list(ids) if n >= len(ids) else []
    remaining = n - len(alloc)
    total = float(sum(w))
    for i, wi in zip(ids, w):
        alloc.extend([i] * int(round(remaining * wi / total)))
    while len(alloc) < n:
        alloc.append(ids[rng.randrange(len(ids))])
    if len(alloc) > n:
        # Trim from the most-allocated scenes first so the floor survives.
        from collections import Counter
        counts = Counter(alloc)
        while len(alloc) > n:
            worst = max(counts, key=lambda k: counts[k])
            alloc.remove(worst)
            counts[worst] -= 1
    rng.shuffle(alloc)
    return tuple(alloc)


def _coach_slots(rng: random.Random, n_slots: int, n_coach: int) -> Tuple[int, ...]:
    """Demonstration positions: evenly spaced with jitter, never adjacent, never last.

    Even spacing matters: gaps of wildly different length would confound "how long since the
    demonstration" with "which demonstration", and the decay estimate lives on exactly that
    contrast. Jitter matters too -- a perfectly regular schedule gives every gap the same
    length, and a design in which the elapsed time never varies cannot identify ``lambda`` at
    all. This is the single most important line in the protocol for whether the study can
    answer its own question.
    """
    if n_coach <= 0:
        return ()
    stride = n_slots / float(n_coach)
    slots: List[int] = []
    for k in range(n_coach):
        base = k * stride
        jitter = rng.uniform(-0.28, 0.28) * stride
        pos = int(round(base + jitter))
        pos = max(0, min(n_slots - 3, pos))
        while pos in slots or (pos - 1) in slots or (pos + 1) in slots:
            pos += 1
            if pos > n_slots - 3:
                pos = 0
                while pos in slots:
                    pos += 1
                break
        slots.append(pos)
    return tuple(sorted(set(slots)))


def _coach_scene_ids(rng: random.Random, contract: Contract, directions: Sequence[int]) -> Tuple[int, ...]:
    """A demonstration scene where the demonstrated strategy is unambiguously correct."""
    coach = contract.grid.coach_scenes()
    pos = [s.scene_id for s in coach if s.c > 0]
    neg = [s.scene_id for s in coach if s.c < 0]
    out: List[int] = []
    for d in directions:
        pool = pos if d > 0 else neg
        out.append(pool[rng.randrange(len(pool))] if pool else coach[0].scene_id)
    return tuple(out)


def coach_directions(
    regime: str,
    n: int,
    *,
    session_sign: int = 1,
    run_length: int = 3,
) -> Tuple[int, ...]:
    """Demonstration directions for one block. See :class:`BudgetConfig` for the regimes."""
    r = str(regime)
    if r == "alternating":
        return tuple(session_sign * (1 if i % 2 == 0 else -1) for i in range(n))
    if r == "runs":
        L = max(1, int(run_length))
        return tuple(session_sign * (1 if (i // L) % 2 == 0 else -1) for i in range(n))
    return tuple(int(session_sign) for _ in range(n))


def build_budget(
    rng: random.Random,
    contract: Contract,
    *,
    n_slots: int,
    n_coach: int,
    session_sign: int = 1,
) -> BlockBudget:
    slots = _coach_slots(rng, n_slots, n_coach)
    directions = coach_directions(
        contract.budget.coach_regime,
        len(slots),
        session_sign=int(session_sign),
        run_length=int(contract.budget.coach_run_length),
    )
    return BlockBudget(
        n_slots=int(n_slots),
        coach_slots=slots,
        coach_directions=directions,
        scene_sequence=_draw_scene_sequence(rng, contract, n_slots),
        coach_scene_sequence=_coach_scene_ids(rng, contract, directions),
        coach_strength=float(contract.dose_spec().carryover_scale),
    )


def build_protocol(
    *,
    supervisor_id: str,
    contract: Contract,
    seed: int,
    conditions: Optional[Sequence[str]] = None,
    order_index: int = 0,
    include_retest: bool = True,
    session_sign: Optional[int] = None,
) -> Protocol:
    """Draw one supervisor's session plan.

    ``conditions`` defaults to the confirmatory pair -- the proposed policy and the
    pre-specified strongest baseline. Running every condition prospectively would be stronger
    evidence but is not affordable in session time and would risk cross-block contamination;
    the remaining baselines are evaluated off-policy on the supervisor's fitted model and
    reported separately, labelled model-based.
    """
    rng = random.Random(int(seed))
    # Direction is counterbalanced ACROSS participants, by the same index that counterbalances
    # condition order. Within a session the coaching is one-sided, as deployment is; across the
    # cohort it is balanced, so a cohort-level drift cannot masquerade as carryover.
    sign = int(session_sign) if session_sign is not None else (1 if int(order_index) % 2 == 0 else -1)
    conds = list(conditions) if conditions else [CONDITION_CARRYOVER_AWARE, PRIMARY_COMPARATOR]
    square = balanced_latin_square(len(conds))
    ordered = [conds[i] for i in square[int(order_index) % len(square)]]

    n_slots = int(contract.budget.slots_per_block)
    n_coach = int(contract.budget.coach_per_block)

    # ONE budget, shared by every compared condition. This is the matched-budget commitment.
    shared = build_budget(rng, contract, n_slots=n_slots, n_coach=n_coach, session_sign=sign)

    blocks: List[Block] = []
    ref_budget = build_budget(rng, contract, n_slots=int(contract.budget.reference_slots), n_coach=0, session_sign=sign)
    blocks.append(Block(0, BLOCK_REFERENCE, CONDITION_NO_COACH, ref_budget))
    for i, cond in enumerate(ordered):
        blocks.append(Block(len(blocks), BLOCK_CONDITION, cond, shared))
    if include_retest:
        retest_budget = build_budget(rng, contract, n_slots=int(contract.budget.retest_slots), n_coach=0, session_sign=sign)
        blocks.append(Block(len(blocks), BLOCK_RETEST, CONDITION_NO_COACH, retest_budget))

    return Protocol(
        supervisor_id=str(supervisor_id),
        seed=int(seed),
        contract_hash=contract.hash(),
        blocks=blocks,
        prospective_conditions=ordered,
        session_sign=sign,
    )


__all__ = [
    "PROTOCOL_SCHEMA",
    "coach_directions",
    "BLOCK_REFERENCE",
    "BLOCK_RETEST",
    "BLOCK_CONDITION",
    "Block",
    "Protocol",
    "build_budget",
    "build_protocol",
]

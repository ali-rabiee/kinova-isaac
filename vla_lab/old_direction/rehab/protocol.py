"""W7 — block structure, counterbalancing, and the reference/retest layout.

The reference map is what "estimation error" is measured against, so its position in the
session is not a detail — get it wrong and the primary outcome is uninterpretable
(``rehab.md`` §6/W7, §12.2).

**Reference-first / retest-last.**

1. A no-prompt **reference** block at session start defines ``tilde-pi*``, uncontaminated by
   construction (zero COACH).
2. The compared **schedule blocks**, in balanced-Latin-square order, with an enforced
   inter-block washout.
3. A terminal no-prompt **retest** block, which yields both a test-retest reliability
   estimate for ``pi*`` and a residual-contamination check.

The cost — reference-block position is confounded with session position — is accepted because
the reference is a *measurement*, not a compared condition, and the terminal retest quantifies
the confound. A study that cannot show test-retest stability of ``tilde-pi*`` cannot interpret
its primary outcome, so both are reported, retest first.

**Budget matching (§1.3, §12.1).** The target sequence and the COACH slot positions are drawn
**once per participant** and reused by every compared condition, so ``T``, ``C``, and the
presented targets are identical by construction rather than by bookkeeping. Conditions differ
only in what the scheduler does with the non-COACH slots.

**§12.6 hybrid.** ``prospective_conditions`` defaults to the confirmatory pair (the proposed
policy and the strongest baseline) plus reference/retest; the full baseline set is evaluated
**off-policy** on each participant's fitted carryover model as a clearly-labelled secondary
analysis (:mod:`vla_lab.rehab.analyze`). Which baseline is "strongest" is chosen from the
synthetic study (W6) and fixed in the preregistration — the default here is a placeholder
that the preregistration must overwrite.

The balanced Latin square is imported from :mod:`vla_lab.human_study.protocol` rather than
duplicated: that helper is already condition-agnostic, and the coexistence rules (§3) allow
Phase 0 to import *upward* from shared utilities.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ...human_study.protocol import balanced_latin_square
from .carryover import CarryoverConfig
from .contract import Phase0Contract
from .scheduler import (
    CONDITION_CARRYOVER_AWARE,
    CONDITION_FIXED_WASHOUT,
    CONDITION_NO_PROMPT,
    COMPARED_CONDITIONS,
    population_washout_slots,
)
from .scheduler.base import BlockBudget, DeltaModel
from .workspace import TargetGrid

PROTOCOL_SCHEMA = "vla_lab_rehab_protocol/v1"

BLOCK_REFERENCE = "reference"
BLOCK_COMPARED = "compared"
BLOCK_RETEST = "retest"


# ---------------------------------------------------------------------------
# Plan objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    """One protocol-fixed slot. Identical across every compared condition."""

    slot_idx: int
    target_id: int
    is_coach_slot: bool
    effort_level: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlockPlan:
    block_idx: int
    kind: str          # reference | compared | retest
    condition: str
    slots: List[SlotSpec]
    washout_before_ms: int = 0

    @property
    def coach_slots(self) -> Tuple[int, ...]:
        return tuple(s.slot_idx for s in self.slots if s.is_coach_slot)

    def budget(self) -> BlockBudget:
        return BlockBudget(
            slots_total=len(self.slots),
            coach_slots=self.coach_slots,
            condition=self.condition,
            target_sequence=tuple(s.target_id for s in self.slots),
            effort_levels=tuple(s.effort_level for s in self.slots),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_idx": int(self.block_idx),
            "kind": str(self.kind),
            "condition": str(self.condition),
            "washout_before_ms": int(self.washout_before_ms),
            "n_slots": len(self.slots),
            "n_coach": len(self.coach_slots),
            "slots": [s.to_dict() for s in self.slots],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlockPlan":
        return cls(
            block_idx=int(d["block_idx"]),
            kind=str(d.get("kind", BLOCK_COMPARED)),
            condition=str(d.get("condition", "")),
            slots=[
                SlotSpec(
                    slot_idx=int(s["slot_idx"]),
                    target_id=int(s["target_id"]),
                    is_coach_slot=bool(s["is_coach_slot"]),
                    effort_level=str(s.get("effort_level", "none")),
                )
                for s in d.get("slots", [])
            ],
            washout_before_ms=int(d.get("washout_before_ms", 0)),
        )


@dataclass
class SessionPlan:
    """Written to ``protocol.json`` **before trial 1** (§10)."""

    participant_id: str
    participant_idx: int
    seed: int
    contract_hash: str
    nonpreferred_side: str
    condition_order: List[str]
    blocks: List[BlockPlan]
    fixed_w: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "participant_id": self.participant_id,
            "participant_idx": int(self.participant_idx),
            "seed": int(self.seed),
            "contract_hash": self.contract_hash,
            "nonpreferred_side": self.nonpreferred_side,
            "condition_order": list(self.condition_order),
            "fixed_w": int(self.fixed_w),
            "n_blocks": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionPlan":
        return cls(
            participant_id=str(d["participant_id"]),
            participant_idx=int(d.get("participant_idx", 0)),
            seed=int(d.get("seed", 0)),
            contract_hash=str(d.get("contract_hash", "")),
            nonpreferred_side=str(d.get("nonpreferred_side", "left")),
            condition_order=list(d.get("condition_order", [])),
            blocks=[BlockPlan.from_dict(b) for b in d.get("blocks", [])],
            fixed_w=int(d.get("fixed_w", 0)),
            notes=str(d.get("notes", "")),
        )

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SessionPlan":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- checks ------------------------------------------------------------
    def budget_table(self) -> Dict[str, Dict[str, int]]:
        """Per-condition ``T`` and ``C``. Must be identical across compared conditions."""

        out: Dict[str, Dict[str, int]] = {}
        for b in self.blocks:
            if b.kind != BLOCK_COMPARED:
                continue
            out[b.condition] = {"trials": len(b.slots), "coach": len(b.coach_slots)}
        return out

    def validate(self) -> List[str]:
        problems: List[str] = []
        table = self.budget_table()
        if len(set((v["trials"], v["coach"]) for v in table.values())) > 1:
            problems.append(f"budget is not matched across compared conditions: {table}")
        for b in self.blocks:
            if b.kind in (BLOCK_REFERENCE, BLOCK_RETEST) and b.coach_slots:
                problems.append(
                    f"block {b.block_idx} ({b.kind}) contains {len(b.coach_slots)} COACH slots; "
                    "reference and retest blocks must contain zero (§12.2)"
                )
            if b.kind == BLOCK_COMPARED and not b.coach_slots:
                problems.append(f"compared block {b.block_idx} ({b.condition}) has no COACH slots")
        seqs = {
            b.condition: tuple(s.target_id for s in b.slots)
            for b in self.blocks
            if b.kind == BLOCK_COMPARED
        }
        if len(set(seqs.values())) > 1:
            problems.append(
                "compared conditions do not share an identical target sequence (§12.1): "
                f"{ {k: len(v) for k, v in seqs.items()} }"
            )
        return problems


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Phase0Protocol:
    """Session-level design parameters."""

    #: Conditions run **prospectively** (§12.6 hybrid). The rest are evaluated off-policy.
    prospective_conditions: Tuple[str, ...] = (CONDITION_CARRYOVER_AWARE, CONDITION_FIXED_WASHOUT)
    n_participants: int = 24
    seed: int = 20260816
    #: Population carryover parameters used to derive B2's ``w``. Priors until the pilot
    #: measures them (M4); ``population_source`` records which.
    population_lambda: float = 0.72
    population_beta: float = 1.0
    population_g: float = 0.8
    population_source: str = "prior"
    instruments: Tuple[str, ...] = ("edinburgh_handedness", "nasa_tlx", "session_burden")
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["prospective_conditions"] = list(self.prospective_conditions)
        d["instruments"] = list(self.instruments)
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Phase0Protocol":
        d = dict(d or {})
        if "prospective_conditions" in d and d["prospective_conditions"] is not None:
            d["prospective_conditions"] = tuple(str(x) for x in d["prospective_conditions"])
        if "instruments" in d and d["instruments"] is not None:
            d["instruments"] = tuple(str(x) for x in d["instruments"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Phase0Protocol":
        import yaml

        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(d.get("protocol", d))

    def validate(self) -> List[str]:
        unknown = [c for c in self.prospective_conditions if c not in COMPARED_CONDITIONS]
        return (
            [f"unknown prospective condition(s) {unknown}; known: {list(COMPARED_CONDITIONS)}"]
            if unknown
            else []
        )


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def balanced_target_sequence(grid: TargetGrid, n_slots: int, rng: random.Random) -> List[int]:
    """A length-``n_slots`` target order with as-equal-as-possible coverage.

    Cycles through a shuffled copy of the full target list, so after ``k`` full cycles every
    target has been presented exactly ``k`` times. Uniform sampling would leave some targets
    unvisited at realistic budgets, and an unvisited target has no estimate to score.
    """

    ids = grid.ids()
    out: List[int] = []
    while len(out) < int(n_slots):
        cycle = list(ids)
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[: int(n_slots)]


def coach_slot_positions(n_slots: int, n_coach: int, rng: random.Random, *, min_gap: int = 3) -> Tuple[int, ...]:
    """Evenly spread COACH slots with a seeded jitter, never adjacent.

    Even spread keeps the contamination process comparable across the block; the minimum gap
    guarantees every condition has room to place at least ``min_gap - 1`` free slots between
    prompts, so a fixed washout is *feasible* rather than truncated by the layout.
    """

    n_slots, n_coach = int(n_slots), int(n_coach)
    if n_coach <= 0:
        return ()
    if n_coach * min_gap > n_slots:
        raise ValueError(
            f"cannot place {n_coach} COACH slots in {n_slots} slots with min_gap={min_gap}: "
            "raise trials_per_block or lower coach_per_block"
        )
    spacing = n_slots / float(n_coach)
    out: List[int] = []
    for i in range(n_coach):
        centre = int(round((i + 0.5) * spacing))
        jitter = rng.randint(-1, 1)
        pos = min(n_slots - 1, max(0, centre + jitter))
        while out and pos - out[-1] < min_gap:
            pos += 1
        pos = min(n_slots - 1, pos)
        if out and pos <= out[-1]:
            pos = out[-1] + min_gap
        if pos >= n_slots:
            raise ValueError(
                f"COACH slot placement overflowed the block ({n_coach} prompts, {n_slots} slots)"
            )
        out.append(pos)
    return tuple(out)


def _slots(
    targets: Sequence[int],
    coach: Sequence[int],
    effort_level: str,
) -> List[SlotSpec]:
    cs = set(int(i) for i in coach)
    return [
        SlotSpec(
            slot_idx=i,
            target_id=int(t),
            is_coach_slot=(i in cs),
            effort_level=(effort_level if i in cs else "none"),
        )
        for i, t in enumerate(targets)
    ]


def derive_fixed_w(
    protocol: Phase0Protocol,
    contract: Phase0Contract,
    *,
    n_slots: int,
    coach_slots: Sequence[int],
    carryover_cfg: Optional[CarryoverConfig] = None,
) -> int:
    """B2's washout constant, from population parameters via the shared objective."""

    cfg = carryover_cfg or CarryoverConfig()
    delta = DeltaModel.from_contract(contract, cfg)
    effort = contract.prompts.effort()
    return population_washout_slots(
        lam=float(protocol.population_lambda),
        beta=float(protocol.population_beta),
        g=float(protocol.population_g) * float(effort.carryover_scale),
        slots_total=int(n_slots),
        coach_slots=coach_slots,
        delta=delta,
        corrected=False,
    )


def generate_session_plan(
    participant_idx: int,
    protocol: Phase0Protocol,
    contract: Phase0Contract,
    *,
    nonpreferred_side: str,
    participant_id: Optional[str] = None,
    seed: Optional[int] = None,
    conditions: Optional[Sequence[str]] = None,
    carryover_cfg: Optional[CarryoverConfig] = None,
) -> SessionPlan:
    """Build one participant's reference-first / counterbalanced / retest-last plan.

    Deterministic in ``(participant_id, seed)``: the same pair always yields the same
    assignment, so the plan can be regenerated and checked against the realized session.
    """

    pid = participant_id or f"P{int(participant_idx):03d}"
    base_seed = int(seed if seed is not None else protocol.seed) + int(participant_idx)
    rng = random.Random(base_seed)
    grid = contract.target_grid()
    budget = contract.budget
    conds = list(conditions if conditions is not None else protocol.prospective_conditions)

    # One target sequence and one COACH layout, shared by every compared condition (§12.1).
    T = int(budget.trials_per_block)
    C = int(budget.coach_per_block)
    shared_targets = balanced_target_sequence(grid, T, rng)
    shared_coach = coach_slot_positions(T, C, rng)
    effort_level = str(contract.prompts.coach_effort_level)
    fixed_w = derive_fixed_w(
        protocol, contract, n_slots=T, coach_slots=shared_coach, carryover_cfg=carryover_cfg
    )

    # Counterbalance the compared conditions.
    order = list(range(len(conds)))
    if len(conds) > 1:
        square = balanced_latin_square(len(conds))
        order = square[int(participant_idx) % len(square)]
    condition_order = [conds[i] for i in order]

    blocks: List[BlockPlan] = []
    ref_targets = balanced_target_sequence(grid, int(budget.reference_trials), rng)
    blocks.append(
        BlockPlan(
            block_idx=0,
            kind=BLOCK_REFERENCE,
            condition=CONDITION_NO_PROMPT,
            slots=_slots(ref_targets, (), effort_level),
            washout_before_ms=0,
        )
    )
    for k, cond in enumerate(condition_order):
        blocks.append(
            BlockPlan(
                block_idx=len(blocks),
                kind=BLOCK_COMPARED,
                condition=cond,
                slots=_slots(shared_targets, shared_coach, effort_level),
                washout_before_ms=int(budget.inter_block_washout_ms),
            )
        )
    retest_targets = balanced_target_sequence(grid, int(budget.retest_trials), rng)
    blocks.append(
        BlockPlan(
            block_idx=len(blocks),
            kind=BLOCK_RETEST,
            condition=CONDITION_NO_PROMPT,
            slots=_slots(retest_targets, (), effort_level),
            washout_before_ms=int(budget.inter_block_washout_ms),
        )
    )

    return SessionPlan(
        participant_id=pid,
        participant_idx=int(participant_idx),
        seed=base_seed,
        contract_hash=contract.contract_hash(),
        nonpreferred_side=str(nonpreferred_side),
        condition_order=condition_order,
        blocks=blocks,
        fixed_w=int(fixed_w),
        notes=protocol.notes,
    )


def n_wait_for_static(plan: SessionPlan) -> int:
    """B3's WAIT count: the same split B2 produces, placed independently of history."""

    for b in plan.blocks:
        if b.kind == BLOCK_COMPARED:
            return int(min(len(b.slots) - len(b.coach_slots), len(b.coach_slots) * max(0, plan.fixed_w)))
    return 0


__all__ = [
    "PROTOCOL_SCHEMA",
    "BLOCK_REFERENCE",
    "BLOCK_COMPARED",
    "BLOCK_RETEST",
    "SlotSpec",
    "BlockPlan",
    "SessionPlan",
    "Phase0Protocol",
    "generate_session_plan",
    "balanced_target_sequence",
    "coach_slot_positions",
    "derive_fixed_w",
    "n_wait_for_static",
]

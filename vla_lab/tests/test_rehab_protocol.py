"""W7 — tests for ``vla_lab.rehab.protocol``.

"Done when" (``rehab.md`` §6/W7): budget and COACH count identical across conditions; Latin
square balanced across the planned N; the same participant ID + seed always produces the same
assignment; **reference and retest blocks contain zero COACH actions**.
"""

from __future__ import annotations

from collections import Counter

from vla_lab.rehab.contract import BudgetConfig, Phase0Contract
from vla_lab.rehab.protocol import (
    BLOCK_COMPARED,
    BLOCK_REFERENCE,
    BLOCK_RETEST,
    Phase0Protocol,
    SessionPlan,
    balanced_target_sequence,
    coach_slot_positions,
    derive_fixed_w,
    generate_session_plan,
    n_wait_for_static,
)
from vla_lab.rehab.scheduler import COMPARED_CONDITIONS, CONDITION_NO_PROMPT

CONTRACT = Phase0Contract()
PROTOCOL = Phase0Protocol()


def _plan(idx: int = 0, conditions=None, seed: int = 1):
    return generate_session_plan(
        idx, PROTOCOL, CONTRACT, nonpreferred_side="left", seed=seed,
        conditions=list(conditions) if conditions else None,
    )


# --------------------------------------------------------------------------- layout


def test_layout_is_reference_first_and_retest_last():
    plan = _plan(conditions=COMPARED_CONDITIONS)
    kinds = [b.kind for b in plan.blocks]
    assert kinds[0] == BLOCK_REFERENCE
    assert kinds[-1] == BLOCK_RETEST
    assert all(k == BLOCK_COMPARED for k in kinds[1:-1])


def test_reference_and_retest_blocks_contain_zero_coach():
    plan = _plan(conditions=COMPARED_CONDITIONS)
    for b in plan.blocks:
        if b.kind in (BLOCK_REFERENCE, BLOCK_RETEST):
            assert b.coach_slots == (), (b.kind, b.coach_slots)
            assert b.condition == CONDITION_NO_PROMPT


def test_inter_block_washout_is_enforced_between_compared_blocks():
    plan = _plan(conditions=COMPARED_CONDITIONS)
    for b in plan.blocks[1:]:
        assert b.washout_before_ms == CONTRACT.budget.inter_block_washout_ms


# --------------------------------------------------------------------------- matched budget


def test_budget_is_matched_across_compared_conditions():
    plan = _plan(conditions=COMPARED_CONDITIONS)
    table = plan.budget_table()
    assert len(table) == len(COMPARED_CONDITIONS)
    assert len({(v["trials"], v["coach"]) for v in table.values()}) == 1, table
    assert plan.validate() == []


def test_every_compared_condition_gets_the_identical_target_sequence():
    plan = _plan(conditions=COMPARED_CONDITIONS)
    seqs = {
        b.condition: tuple(s.target_id for s in b.slots)
        for b in plan.blocks if b.kind == BLOCK_COMPARED
    }
    assert len(set(seqs.values())) == 1, {k: len(v) for k, v in seqs.items()}
    coach = {b.condition: b.coach_slots for b in plan.blocks if b.kind == BLOCK_COMPARED}
    assert len(set(coach.values())) == 1


def test_validate_catches_a_coach_leaking_into_the_reference_block():
    from vla_lab.rehab.protocol import SlotSpec

    plan = _plan(conditions=COMPARED_CONDITIONS)
    ref = plan.blocks[0]
    ref.slots[3] = SlotSpec(slot_idx=3, target_id=ref.slots[3].target_id, is_coach_slot=True)
    problems = plan.validate()
    assert any("reference" in p or "zero" in p for p in problems), problems


# --------------------------------------------------------------------------- counterbalancing


def test_condition_order_is_a_balanced_latin_square_across_participants():
    n = len(COMPARED_CONDITIONS)
    orders = [_plan(i, conditions=COMPARED_CONDITIONS).condition_order for i in range(n)]
    for o in orders:
        assert sorted(o) == sorted(COMPARED_CONDITIONS)
    for position in range(n):
        column = Counter(o[position] for o in orders)
        assert set(column) == set(COMPARED_CONDITIONS), (position, column)
        assert all(v == 1 for v in column.values()), (position, column)


def test_the_same_participant_and_seed_reproduce_the_assignment():
    a = _plan(5, conditions=COMPARED_CONDITIONS, seed=99)
    b = _plan(5, conditions=COMPARED_CONDITIONS, seed=99)
    assert a.to_dict() == b.to_dict()
    c = _plan(5, conditions=COMPARED_CONDITIONS, seed=100)
    assert c.to_dict() != a.to_dict()


def test_plan_round_trips_through_json():
    a = _plan(2, conditions=COMPARED_CONDITIONS)
    b = SessionPlan.from_dict(a.to_dict())
    assert b.to_dict() == a.to_dict()


# --------------------------------------------------------------------------- slot construction


def test_coach_slots_are_spread_and_never_adjacent():
    import random

    for seed in range(8):
        pos = coach_slot_positions(40, 8, random.Random(seed), min_gap=3)
        assert len(pos) == 8
        assert list(pos) == sorted(pos)
        assert all(b - a >= 3 for a, b in zip(pos, pos[1:])), pos
        assert max(pos) < 40


def test_coach_slot_placement_refuses_an_impossible_layout():
    import random

    try:
        coach_slot_positions(10, 8, random.Random(0), min_gap=3)
        raise AssertionError("should refuse to place 8 prompts in 10 slots with a gap of 3")
    except ValueError as exc:
        assert "min_gap" in str(exc)


def test_target_sequence_covers_every_target_evenly():
    import random

    grid = CONTRACT.target_grid()
    seq = balanced_target_sequence(grid, len(grid) * 3, random.Random(0))
    counts = Counter(seq)
    assert set(counts) == set(grid.ids())
    assert set(counts.values()) == {3}


def test_derived_washout_is_a_nonnegative_integer():
    import random

    coach = coach_slot_positions(40, 8, random.Random(0))
    w = derive_fixed_w(PROTOCOL, CONTRACT, n_slots=40, coach_slots=coach)
    assert isinstance(w, int) and w >= 0
    plan = _plan(conditions=COMPARED_CONDITIONS)
    assert n_wait_for_static(plan) >= 0


def test_stronger_carryover_implies_a_longer_derived_washout():
    import random

    coach = coach_slot_positions(40, 8, random.Random(0))
    weak = Phase0Protocol(population_lambda=0.72, population_beta=0.2, population_g=0.2)
    strong = Phase0Protocol(population_lambda=0.9, population_beta=2.0, population_g=1.6)
    w_weak = derive_fixed_w(weak, CONTRACT, n_slots=40, coach_slots=coach)
    w_strong = derive_fixed_w(strong, CONTRACT, n_slots=40, coach_slots=coach)
    assert w_strong >= w_weak, (w_weak, w_strong)


def test_protocol_rejects_an_unknown_condition():
    p = Phase0Protocol(prospective_conditions=("not_a_condition",))
    assert p.validate()


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_protocol") else 0)

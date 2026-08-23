"""Contract, protocol, and the design commitments they are supposed to enforce."""

from __future__ import annotations

from collections import Counter

from vla_lab.supervisory.contract import BudgetConfig, Contract
from vla_lab.supervisory.protocol import (
    BLOCK_CONDITION,
    BLOCK_REFERENCE,
    BLOCK_RETEST,
    build_protocol,
    coach_directions,
)
from vla_lab.supervisory.scenes import ScenePhysics, build_scene_grid
from vla_lab.tests import assert_raises


def test_the_default_contract_is_runnable():
    assert Contract().check() == []


def test_the_contract_hash_changes_when_anything_load_bearing_changes():
    a = Contract()
    b = Contract()
    assert a.hash() == b.hash()
    b.budget.coach_per_block += 1
    assert a.hash() != b.hash()


def test_a_contract_with_no_ambiguous_scene_is_refused():
    c = Contract(grid=build_scene_grid(physics=ScenePhysics(p_b_asym=0.02, b_slope=1.0)))
    assert c.check()


def test_a_contract_whose_demonstrations_are_ambiguous_is_refused():
    c = Contract(grid=build_scene_grid(coach_c=0.1, crossover_halfwidth=1.0))
    assert any("crossover band" in p for p in c.check())


def test_every_compared_condition_shares_one_budget_object():
    p = build_protocol(supervisor_id="S", contract=Contract(), seed=1,
                       conditions=["carryover_aware", "fixed_washout"])
    blocks = p.condition_blocks()
    assert len(blocks) == 2
    assert blocks[0].budget is blocks[1].budget


def test_the_session_has_a_reference_first_and_a_retest_last():
    p = build_protocol(supervisor_id="S", contract=Contract(), seed=1)
    kinds = [b.kind for b in p.blocks]
    assert kinds[0] == BLOCK_REFERENCE and kinds[-1] == BLOCK_RETEST
    assert BLOCK_CONDITION in kinds


def test_the_reference_and_retest_blocks_never_demonstrate():
    p = build_protocol(supervisor_id="S", contract=Contract(), seed=1)
    for b in p.blocks:
        if b.kind in (BLOCK_REFERENCE, BLOCK_RETEST):
            assert b.budget.n_coach == 0


def test_demonstration_gaps_vary_so_the_decay_rate_is_identifiable_at_all():
    # A perfectly regular demonstration schedule gives every gap the same length, and a design
    # in which the elapsed time never varies cannot identify lambda.
    b = build_protocol(supervisor_id="S", contract=Contract(), seed=3).condition_blocks()[0].budget
    slots = list(b.coach_slots)
    gaps = [y - x for x, y in zip(slots, slots[1:])]
    assert len(set(gaps)) > 1, gaps


def test_no_two_demonstrations_are_adjacent():
    b = build_protocol(supervisor_id="S", contract=Contract(), seed=7).condition_blocks()[0].budget
    slots = sorted(b.coach_slots)
    assert all(y - x >= 2 for x, y in zip(slots, slots[1:])), slots


def test_every_probe_scene_gets_at_least_one_slot():
    c = Contract()
    b = build_protocol(supervisor_id="S", contract=c, seed=2).condition_blocks()[0].budget
    counts = Counter(b.scene_sequence)
    for s in c.grid.probe_scenes():
        assert counts[s.scene_id] >= 1, s.scene_id


def test_the_crossover_band_gets_the_larger_share_of_the_budget():
    c = Contract()
    b = build_protocol(supervisor_id="S", contract=c, seed=2).condition_blocks()[0].budget
    counts = Counter(b.scene_sequence)
    band = sum(counts[s.scene_id] for s in c.grid.probe_scenes() if c.grid.in_crossover_band(s))
    flank = sum(counts[s.scene_id] for s in c.grid.probe_scenes() if not c.grid.in_crossover_band(s))
    assert band > flank, (band, flank)


def test_demonstration_direction_is_counterbalanced_across_supervisors_not_within_a_session():
    c = Contract()
    signs = []
    for i in range(6):
        p = build_protocol(supervisor_id=f"S{i}", contract=c, seed=1, order_index=i)
        dirs = set(p.condition_blocks()[0].budget.coach_directions)
        assert len(dirs) == 1, dirs   # one-sided within a session: the deployment regime
        signs.append(p.session_sign)
    assert set(signs) == {1, -1}      # balanced across the cohort


def test_the_alternating_regime_is_available_for_the_identification_analysis():
    assert coach_directions("alternating", 6) == (1, -1, 1, -1, 1, -1)
    assert coach_directions("runs", 6, run_length=3) == (1, 1, 1, -1, -1, -1)
    assert coach_directions("one_sided", 4, session_sign=-1) == (-1, -1, -1, -1)


def test_demonstrations_are_presented_on_scenes_where_that_strategy_is_correct():
    c = Contract()
    b = build_protocol(supervisor_id="S", contract=c, seed=1, order_index=0).condition_blocks()[0].budget
    for d, sid in zip(b.coach_directions, b.coach_scene_sequence):
        scene = c.grid.by_id(sid)
        assert (scene.c > 0) == (d > 0), (d, scene.c)


def test_a_budget_whose_scene_sequence_is_too_short_is_refused():
    from vla_lab.supervisory.scheduler.base import BlockBudget

    assert_raises(ValueError, lambda: BlockBudget(n_slots=10, coach_slots=(0,), coach_directions=(1,),
                                                  scene_sequence=(1, 2)))


def test_mismatched_demonstration_slots_and_directions_are_refused():
    from vla_lab.supervisory.scheduler.base import BlockBudget

    assert_raises(ValueError, lambda: BlockBudget(n_slots=10, coach_slots=(0, 4), coach_directions=(1,),
                                                  scene_sequence=tuple(range(10))))

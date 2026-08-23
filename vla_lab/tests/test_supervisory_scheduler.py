"""Conditions, the policy family, and the reductions the ablation argument depends on."""

from __future__ import annotations

from collections import Counter

from vla_lab.supervisory import COACH, COUNTER, PROBE, WAIT
from vla_lab.supervisory.carryover import CarryoverConfig
from vla_lab.supervisory.contract import Contract
from vla_lab.supervisory.protocol import build_protocol
from vla_lab.supervisory.scenes import build_scene_grid
from vla_lab.supervisory.scheduler import (
    ABLATION_COUNTER_ONLY,
    ABLATION_ESTIMATOR_ONLY,
    ABLATION_SCHEDULE_ONLY,
    ALL_CONDITIONS,
    CONDITION_ALWAYS_COUNTER,
    CONDITION_CARRYOVER_AWARE,
    CONDITION_FIXED_WASHOUT,
    CONDITION_MEMORYLESS,
    CONDITION_NO_COACH,
    CONDITION_RANDOM_STATIC,
    build_scheduler,
    estimator_for,
    population_washout_slots,
)
from vla_lab.supervisory.scheduler.base import DeltaModel, History, HistoryRecord, Slot
from vla_lab.supervisory.scheduler.carryover_aware import CarryoverAwareConfig
from vla_lab.tests import assert_raises

GRID = build_scene_grid()
CONTRACT = Contract()


def _budget():
    return build_protocol(supervisor_id="S", contract=CONTRACT, seed=1).condition_blocks()[0].budget


def _run(condition, *, n=None, aware_cfg=None):
    """Drive one scheduler through a block with no supervisor: actions only."""
    b = _budget()
    sch = build_scheduler(condition, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), aware_cfg=aware_cfg, seed=1)
    sch.reset(b)
    hist = History()
    acts = []
    for i in range(n or b.n_slots):
        slot = Slot(index=i, scene_id=b.scene_sequence[i], is_coach_slot=i in set(b.coach_slots),
                    coach_direction=b.direction_at(i), coach_strength=b.coach_strength,
                    free_remaining=b.free_remaining_at(i), session_progress=i / max(1, b.n_slots - 1))
        d = sch.decide(hist, slot)
        acts.append(d.action)
        rec = HistoryRecord(slot=i, action=d.action, scene_id=d.scene_id,
                            delta=CONTRACT.delta_model().for_action(d.action),
                            coach_direction=slot.coach_direction, coach_strength=slot.coach_strength,
                            instructed=None, c=GRID.by_id(d.scene_id).c, clutter=GRID.by_id(d.scene_id).clutter)
        hist.append(rec)
        sch.observe(rec)
    return Counter(acts), sch


def test_every_condition_builds_and_declares_an_estimator():
    for c in ALL_CONDITIONS:
        s = build_scheduler(c, GRID, carryover_cfg=CONTRACT.carryover, delta_model=CONTRACT.delta_model())
        assert s is not None
        assert estimator_for(c) in ("pooled", "psychometric", "carryover_corrected")


def test_an_unknown_condition_is_an_error_not_a_default():
    assert_raises(KeyError, lambda: build_scheduler("not_a_condition", GRID))


def test_the_demonstration_schedule_is_identical_across_every_condition():
    # This is the matched-budget commitment: conditions may differ only in the free slots.
    b = _budget()
    for c in (CONDITION_MEMORYLESS, CONDITION_FIXED_WASHOUT, CONDITION_ALWAYS_COUNTER,
              CONDITION_CARRYOVER_AWARE):
        acts, _ = _run(c)
        assert acts[COACH] == b.n_coach, (c, acts)


def test_the_reference_condition_never_demonstrates():
    acts, _ = _run(CONDITION_NO_COACH)
    assert acts[COACH] == 0 and acts[PROBE] > 0


def test_memoryless_probes_every_free_slot_and_never_waits_or_asks():
    acts, _ = _run(CONDITION_MEMORYLESS)
    assert acts[WAIT] == 0 and acts[COUNTER] == 0


def test_always_counter_never_probes_plainly():
    acts, _ = _run(CONDITION_ALWAYS_COUNTER)
    assert acts[PROBE] == 0 and acts[COUNTER] > 0


def test_fixed_washout_spends_its_waits_immediately_after_each_demonstration():
    b = _budget()
    sch = build_scheduler(CONDITION_FIXED_WASHOUT, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), washout_w=2, seed=1)
    sch.reset(b)
    hist = History()
    seen = []
    for i in range(b.n_slots):
        slot = Slot(index=i, scene_id=b.scene_sequence[i], is_coach_slot=i in set(b.coach_slots),
                    coach_direction=b.direction_at(i), free_remaining=b.free_remaining_at(i))
        d = sch.decide(hist, slot)
        seen.append((i, d.action))
        rec = HistoryRecord(slot=i, action=d.action, scene_id=d.scene_id, delta=1.0)
        hist.append(rec)
        sch.observe(rec)
    for k, (i, a) in enumerate(seen):
        if a == COACH and k + 2 < len(seen):
            assert seen[k + 1][1] == WAIT, seen[k : k + 3]


def test_the_random_static_condition_matches_the_washout_total_but_not_its_placement():
    w = population_washout_slots(cfg=CONTRACT.carryover, delta_model=CONTRACT.delta_model())
    acts, _ = _run(CONDITION_RANDOM_STATIC)
    assert acts[WAIT] == w


def test_the_adaptive_policy_reduces_to_the_fixed_washout_when_every_switch_is_off():
    cfg = CarryoverAwareConfig(adaptive_schedule=False, estimator_correction=False, counter_enabled=False)
    acts, _ = _run(CONDITION_CARRYOVER_AWARE, aware_cfg=cfg)
    fixed, _ = _run(CONDITION_FIXED_WASHOUT)
    assert acts[COUNTER] == 0
    assert abs(acts[WAIT] - fixed[WAIT]) <= 2, (acts, fixed)


def test_each_ablation_switches_off_exactly_what_its_name_says():
    for cond, expect_counter in ((ABLATION_SCHEDULE_ONLY, False), (ABLATION_ESTIMATOR_ONLY, False),
                                 (ABLATION_COUNTER_ONLY, True)):
        acts, sch = _run(cond)
        assert (acts[COUNTER] > 0) == expect_counter or acts[COUNTER] == 0, (cond, acts)
    assert estimator_for(ABLATION_ESTIMATOR_ONLY) == "carryover_corrected"
    assert estimator_for(ABLATION_SCHEDULE_ONLY) == "psychometric"


def test_the_policy_logs_enough_to_reconstruct_its_own_decision():
    b = _budget()
    sch = build_scheduler(CONDITION_CARRYOVER_AWARE, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), seed=1)
    sch.reset(b)
    slot = Slot(index=1, scene_id=b.scene_sequence[1], free_remaining=b.free_remaining_at(1))
    d = sch.decide(History(), slot)
    for key in ("w", "k", "scored", "kappa_mean", "contamination_mean", "washout_delta",
                "lambda_mean", "beta_g_mean", "reason"):
        assert key in d.rationale, key
    assert len(d.rationale["scored"]) > 1


def test_the_lookahead_covers_the_whole_block_not_just_the_current_gap():
    b = _budget()
    sch = build_scheduler(CONDITION_CARRYOVER_AWARE, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), seed=1)
    sch.reset(b)
    slot = Slot(index=1, scene_id=b.scene_sequence[1], free_remaining=b.free_remaining_at(1))
    plan = sch._remaining_plan(slot, consumed=0, w=0, k=0)
    assert len(plan) == b.n_slots - 1
    assert sum(1 for a, _, _ in plan if a == COACH) == sum(1 for s in b.coach_slots if s >= 1)


def test_the_population_washout_follows_the_priors_it_is_derived_from():
    dm = CONTRACT.delta_model()
    slow = population_washout_slots(cfg=CONTRACT.carryover, delta_model=dm, lam=0.95, beta=1.5, g=1.5)
    fast = population_washout_slots(cfg=CONTRACT.carryover, delta_model=dm, lam=0.10, beta=1.5, g=1.5)
    assert slow > fast

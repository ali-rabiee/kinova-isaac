"""W5 — tests for ``vla_lab.rehab.scheduler``.

"Done when" (``rehab.md`` §6/W5): each baseline reproduces its defining rule exactly; budget
accounting is identical across conditions given the same protocol; B4 is deterministic given a
seed; **B4 degenerates to B2 when the carryover posterior is forced to a point mass at the
population lambda**.
"""

from __future__ import annotations

from vla_lab.rehab import ASSESS, COACH, WAIT
from vla_lab.rehab.carryover import CarryoverConfig
from vla_lab.rehab.contract import Phase0Contract
from vla_lab.rehab.protocol import Phase0Protocol, generate_session_plan
from vla_lab.rehab.scheduler import (
    ALL_CONDITIONS,
    COMPARED_CONDITIONS,
    CONDITION_ABLATION_ESTIMATOR,
    CONDITION_ABLATION_SCHEDULE,
    CONDITION_CARRYOVER_AWARE,
    CONDITION_FIXED_WASHOUT,
    CONDITION_IMMEDIATE,
    CONDITION_NO_PROMPT,
    CONDITION_RANDOM_STATIC,
    make_scheduler,
)
from vla_lab.rehab.scheduler.base import DeltaModel
from vla_lab.rehab.scheduler.carryover_aware import population_washout_slots
from vla_lab.rehab.trial import History, Trial, TrialRecord, TrialResult

CONTRACT = Phase0Contract()
PROTOCOL = Phase0Protocol()
GRID = CONTRACT.target_grid()
SIDE = "left"


def _plan():
    return generate_session_plan(
        0, PROTOCOL, CONTRACT, nonpreferred_side=SIDE, seed=1, conditions=list(COMPARED_CONDITIONS)
    )


def _compared_block(plan, condition: str):
    return next(b for b in plan.blocks if b.condition == condition)


def _run_block(scheduler, block, *, respond=None):
    """Drive a scheduler through a block; return the emitted actions."""

    scheduler.reset(block.budget())
    hist = History(
        block_idx=block.block_idx, condition=block.condition,
        slots_total=len(block.slots), coach_slots=block.coach_slots,
    )
    actions = []
    for slot in block.slots:
        dec = scheduler.decide(hist, slot)
        actions.append(dec.action)
        rec = TrialRecord(
            trial=Trial(
                trial_idx=slot.slot_idx, block_idx=block.block_idx, condition=block.condition,
                action=dec.action, target_id=dec.target_id, slot_idx=slot.slot_idx,
                is_coach_slot=slot.is_coach_slot,
                effort_level=(slot.effort_level if dec.action == COACH else "none"),
            ),
            result=TrialResult(arm=(respond(slot) if respond else "nonpreferred"), success=True),
        )
        hist.append(rec)
        scheduler.observe(rec)
    return actions, hist


# --------------------------------------------------------------------------- defining rules


def test_b0_refuses_a_coach_slot():
    plan = _plan()
    ref = next(b for b in plan.blocks if b.kind == "reference")
    s = make_scheduler(CONDITION_NO_PROMPT, grid=GRID, nonpreferred_side=SIDE)
    actions, _ = _run_block(s, ref)
    assert set(actions) == {ASSESS}

    blk = _compared_block(plan, CONDITION_IMMEDIATE)
    s2 = make_scheduler(CONDITION_NO_PROMPT, grid=GRID, nonpreferred_side=SIDE)
    s2.reset(blk.budget())
    hist = History(block_idx=0, condition="", slots_total=len(blk.slots), coach_slots=blk.coach_slots)
    coach_slot = next(sl for sl in blk.slots if sl.is_coach_slot)
    try:
        s2.decide(hist, coach_slot)
        raise AssertionError("B0 must refuse a COACH slot")
    except ValueError as exc:
        assert "zero COACH" in str(exc)


def test_b1_never_waits():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_IMMEDIATE)
    actions, _ = _run_block(make_scheduler(CONDITION_IMMEDIATE, grid=GRID, nonpreferred_side=SIDE), blk)
    assert WAIT not in actions
    for i, a in enumerate(actions):
        assert a == (COACH if i in blk.coach_slots else ASSESS)


def test_b1_probes_the_slot_right_after_every_coach():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_IMMEDIATE)
    actions, _ = _run_block(make_scheduler(CONDITION_IMMEDIATE, grid=GRID, nonpreferred_side=SIDE), blk)
    for c in blk.coach_slots:
        if c + 1 < len(actions):
            assert actions[c + 1] == ASSESS


def test_b2_waits_exactly_w_slots_after_each_coach():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_FIXED_WASHOUT)
    w = 3
    actions, _ = _run_block(
        make_scheduler(CONDITION_FIXED_WASHOUT, grid=GRID, nonpreferred_side=SIDE, fixed_w=w), blk
    )
    for c in blk.coach_slots:
        for k in range(1, w + 1):
            if c + k < len(actions) and (c + k) not in blk.coach_slots:
                assert actions[c + k] == WAIT, (c, k, actions[c : c + w + 2])


def test_b3_is_static_and_seeded():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_RANDOM_STATIC)
    a1, _ = _run_block(make_scheduler(CONDITION_RANDOM_STATIC, grid=GRID, nonpreferred_side=SIDE, seed=5, n_wait=8), blk)
    a2, _ = _run_block(make_scheduler(CONDITION_RANDOM_STATIC, grid=GRID, nonpreferred_side=SIDE, seed=5, n_wait=8), blk)
    a3, _ = _run_block(make_scheduler(CONDITION_RANDOM_STATIC, grid=GRID, nonpreferred_side=SIDE, seed=6, n_wait=8), blk)
    assert a1 == a2
    assert a1 != a3
    assert a1.count(WAIT) == 8


# --------------------------------------------------------------------------- matched budget


def test_budget_is_identical_across_every_compared_condition():
    plan = _plan()
    sigs = set()
    for cond in COMPARED_CONDITIONS:
        blk = _compared_block(plan, cond)
        sched = make_scheduler(
            cond, grid=GRID, nonpreferred_side=SIDE, seed=1,
            fixed_w=plan.fixed_w, n_wait=4,
        )
        actions, hist = _run_block(sched, blk)
        spent = hist.budget_spent()
        sigs.add((spent["trials"], spent["coach"]))
    assert len(sigs) == 1, sigs


def test_the_scheduler_never_overrides_a_coach_slot():
    plan = _plan()
    for cond in COMPARED_CONDITIONS:
        blk = _compared_block(plan, cond)
        sched = make_scheduler(cond, grid=GRID, nonpreferred_side=SIDE, seed=2, fixed_w=plan.fixed_w, n_wait=4)
        actions, _ = _run_block(sched, blk)
        for i in blk.coach_slots:
            assert actions[i] == COACH, (cond, i, actions[i])


# --------------------------------------------------------------------------- B4


def test_b4_is_deterministic_given_a_seed():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_CARRYOVER_AWARE)
    a1, _ = _run_block(make_scheduler(CONDITION_CARRYOVER_AWARE, grid=GRID, nonpreferred_side=SIDE, seed=3), blk)
    a2, _ = _run_block(make_scheduler(CONDITION_CARRYOVER_AWARE, grid=GRID, nonpreferred_side=SIDE, seed=3), blk)
    assert a1 == a2


def test_b4_logs_its_belief_for_audit():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_CARRYOVER_AWARE)
    s = make_scheduler(CONDITION_CARRYOVER_AWARE, grid=GRID, nonpreferred_side=SIDE, seed=3)
    s.reset(blk.budget())
    hist = History(block_idx=0, condition="", slots_total=len(blk.slots), coach_slots=blk.coach_slots)
    free = next(sl for sl in blk.slots if not sl.is_coach_slot)
    dec = s.decide(hist, free)
    assert "lambda_mean" in dec.values and "contamination_mean" in dec.values
    assert dec.rationale


def test_b4_schedule_only_ablation_degenerates_to_b2_under_a_point_mass_posterior():
    """The §1.4 degeneracy: with a population point mass and no de-biasing, the adaptive
    schedule *is* a fixed washout, and it is the same one B2 derives."""

    plan = _plan()
    blk = _compared_block(plan, CONDITION_ABLATION_SCHEDULE)
    cfg = CarryoverConfig()
    delta = DeltaModel.from_contract(CONTRACT, cfg)
    s = make_scheduler(
        CONDITION_ABLATION_SCHEDULE, grid=GRID, nonpreferred_side=SIDE, seed=1,
        carryover_cfg=cfg, delta=delta,
    )
    s.reset(blk.budget())
    s.posterior.force_point_mass(
        lam=PROTOCOL.population_lambda, beta=PROTOCOL.population_beta, g=PROTOCOL.population_g
    )
    hist = History(block_idx=0, condition="", slots_total=len(blk.slots), coach_slots=blk.coach_slots)
    w_adaptive, _ = s.best_w(hist, blk.slots[0])
    w_population = population_washout_slots(
        lam=PROTOCOL.population_lambda, beta=PROTOCOL.population_beta, g=PROTOCOL.population_g,
        slots_total=len(blk.slots), coach_slots=blk.coach_slots, delta=delta, corrected=False,
    )
    assert w_adaptive == w_population, (w_adaptive, w_population)
    assert w_population == plan.fixed_w, (w_population, plan.fixed_w)


def test_b4_with_de_biasing_probes_more_than_without():
    """Mechanism (ii): being able to correct a contaminated observation makes probing cheaper,
    so the corrected policy should spend less budget waiting."""

    plan = _plan()
    cfg = CarryoverConfig()
    delta = DeltaModel.from_contract(CONTRACT, cfg)
    waits = {}
    for cond in (CONDITION_CARRYOVER_AWARE, CONDITION_ABLATION_SCHEDULE):
        blk = _compared_block(plan, cond)
        s = make_scheduler(cond, grid=GRID, nonpreferred_side=SIDE, seed=1, carryover_cfg=cfg, delta=delta)
        s.reset(blk.budget())
        s.posterior.force_point_mass(lam=0.85, beta=1.4, g=1.2)  # a clearly contaminating world
        hist = History(block_idx=0, condition="", slots_total=len(blk.slots), coach_slots=blk.coach_slots)
        waits[cond], _ = s.best_w(hist, blk.slots[0])
    assert waits[CONDITION_CARRYOVER_AWARE] <= waits[CONDITION_ABLATION_SCHEDULE], waits


def test_estimator_only_ablation_uses_the_fixed_schedule():
    plan = _plan()
    blk = _compared_block(plan, CONDITION_ABLATION_ESTIMATOR)
    s = make_scheduler(CONDITION_ABLATION_ESTIMATOR, grid=GRID, nonpreferred_side=SIDE, seed=1, fixed_w=2)
    b2 = make_scheduler(CONDITION_FIXED_WASHOUT, grid=GRID, nonpreferred_side=SIDE, seed=1, fixed_w=2)
    a1, _ = _run_block(s, blk)
    a2, _ = _run_block(b2, blk)
    assert a1 == a2
    assert s.uses_estimator_correction and not b2.uses_estimator_correction


def test_condition_registry_is_complete():
    for cond in ALL_CONDITIONS:
        s = make_scheduler(cond, grid=GRID, nonpreferred_side=SIDE)
        assert s.describe()["condition"]
    try:
        make_scheduler("nonsense", grid=GRID, nonpreferred_side=SIDE)
        raise AssertionError("unknown conditions must be rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_scheduler") else 0)

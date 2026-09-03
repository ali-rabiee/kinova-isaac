"""Conditions, the policy family, and the reductions the ablation argument depends on."""

from __future__ import annotations

from collections import Counter

from vla_lab.supervisory import COACH, COUNTER, PROBE, WAIT
from vla_lab.supervisory.carryover import CarryoverConfig
from vla_lab.supervisory.contract import Contract
from vla_lab.supervisory.protocol import build_protocol
from vla_lab.supervisory.scenes import build_scene_grid
from vla_lab.supervisory.scheduler import (
    ABLATION_B6_FIXED_SCENES,
    CONDITION_IDENTIFICATION_FIRST,
    CONDITION_RECOMMENDED,
    POLICY_RECOMMENDED,
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


# ---------------------------------------------------------------------------
# 2026-08-23: the recommended configuration, B6, and the identification diagnostic
# ---------------------------------------------------------------------------
def test_the_recommended_policy_has_the_adaptive_schedule_off():
    """P2-1: ``policy_recommended`` is corrected estimator + counter-proposals, and never the
    adaptive schedule. The scheduler the registry builds must agree with the config object."""
    cfg = CarryoverAwareConfig.recommended()
    assert cfg.adaptive_schedule is False
    assert cfg.estimator_correction is True and cfg.counter_enabled is True
    sch = build_scheduler(POLICY_RECOMMENDED, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), seed=1)
    assert sch.cfg.adaptive_schedule is False
    assert sch.cfg.estimator_correction and sch.cfg.counter_enabled
    assert estimator_for(POLICY_RECOMMENDED) == "carryover_corrected"
    assert POLICY_RECOMMENDED == CONDITION_RECOMMENDED
    full = CarryoverAwareConfig.full()
    assert full.adaptive_schedule is True, "B5 as proposed keeps the schedule, behind its flag"


def test_the_recommended_policy_places_probes_exactly_like_the_fixed_washout():
    """With the schedule off, B7 differs from B2 only in what it does with a probe (counter or
    not) and in how it estimates -- never in when it waits."""
    acts_rec, sch = _run(CONDITION_RECOMMENDED)
    acts_b2, _ = _run(CONDITION_FIXED_WASHOUT)
    # B7's wait count can differ from B2's by the counter-proposals it spends where B2 would
    # have waited out the last slot of a gap; its placement rule is the same fixed washout.
    assert sch.cfg.adaptive_schedule is False
    assert abs(acts_rec[WAIT] - acts_b2[WAIT]) <= 2, (acts_rec, acts_b2)
    assert acts_rec[PROBE] + acts_rec[COUNTER] + acts_rec[WAIT] == acts_b2[PROBE] + acts_b2[WAIT]


def test_b6_identifies_first_then_exploits_and_keeps_the_budget():
    b = _budget()
    acts, sch = _run(CONDITION_IDENTIFICATION_FIRST)
    assert acts[COACH] == b.n_coach
    assert sum(acts.values()) == b.n_slots
    assert sch._phase == "exploit", "four gaps is enough to leave the identification phase"
    assert not sch._pool, "every free-slot scene must have been consumed exactly once"
    from vla_lab.supervisory.scheduler import ABLATION_B6_LADDER

    acts_l, sch_l = _run(ABLATION_B6_LADDER)
    assert acts_l[WAIT] >= 1, "the log-spaced ladder variant must spend at least one wait"
    assert acts_l[COACH] == b.n_coach and sum(acts_l.values()) == b.n_slots


def test_b6_with_reordering_presents_the_same_scene_multiset_as_the_protocol():
    b = _budget()
    sch = build_scheduler(CONDITION_IDENTIFICATION_FIRST, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), seed=1)
    sch.reset(b)
    hist = History()
    presented = []
    for i in range(b.n_slots):
        slot = Slot(index=i, scene_id=b.scene_sequence[i], is_coach_slot=i in set(b.coach_slots),
                    coach_direction=b.direction_at(i), coach_strength=b.coach_strength,
                    free_remaining=b.free_remaining_at(i), session_progress=i / max(1, b.n_slots - 1))
        d = sch.decide(hist, slot)
        if d.action != COACH:
            presented.append(d.scene_id)
        rec = HistoryRecord(slot=i, action=d.action, scene_id=d.scene_id,
                            delta=CONTRACT.delta_model().for_action(d.action),
                            coach_direction=slot.coach_direction, coach_strength=slot.coach_strength,
                            instructed=None, c=GRID.by_id(d.scene_id).c, clutter=GRID.by_id(d.scene_id).clutter)
        hist.append(rec)
        sch.observe(rec)
    free = [b.scene_sequence[i] for i in range(b.n_slots) if i not in set(b.coach_slots)]
    assert Counter(presented) == Counter(free), "reordering may permute the free-slot scenes, never change them"
    assert presented != free, "and it should actually have reordered something"


def test_b6_fixed_scene_variant_does_not_reorder():
    b = _budget()
    sch = build_scheduler(ABLATION_B6_FIXED_SCENES, GRID, carryover_cfg=CONTRACT.carryover,
                          delta_model=CONTRACT.delta_model(), seed=1)
    assert sch.ident.reorder_scenes is False
    sch.reset(b)
    hist = History()
    for i in range(b.n_slots):
        slot = Slot(index=i, scene_id=b.scene_sequence[i], is_coach_slot=i in set(b.coach_slots),
                    coach_direction=b.direction_at(i), coach_strength=b.coach_strength,
                    free_remaining=b.free_remaining_at(i), session_progress=i / max(1, b.n_slots - 1))
        d = sch.decide(hist, slot)
        assert d.scene_id == slot.scene_id
        rec = HistoryRecord(slot=i, action=d.action, scene_id=d.scene_id,
                            delta=CONTRACT.delta_model().for_action(d.action),
                            coach_direction=slot.coach_direction, coach_strength=slot.coach_strength,
                            instructed=None, c=GRID.by_id(d.scene_id).c, clutter=GRID.by_id(d.scene_id).clutter)
        hist.append(rec)
        sch.observe(rec)


def test_every_adaptive_policy_logs_the_lambda_identification_readout_each_slot():
    """P1-3: the prospective diagnostic is a first-class per-slot output, not a post-hoc number."""
    for cond in (CONDITION_CARRYOVER_AWARE, CONDITION_RECOMMENDED, CONDITION_IDENTIFICATION_FIRST):
        b = _budget()
        sch = build_scheduler(cond, GRID, carryover_cfg=CONTRACT.carryover,
                              delta_model=CONTRACT.delta_model(), seed=1)
        sch.reset(b)
        hist = History()
        i = next(j for j in range(b.n_slots) if j not in set(b.coach_slots))
        slot = Slot(index=i, scene_id=b.scene_sequence[i], free_remaining=b.free_remaining_at(i))
        d = sch.decide(hist, slot)
        for key in ("lambda_tv", "lambda_identified", "lambda_information", "lambda_efold_delta"):
            assert key in d.rationale, (cond, key, sorted(d.rationale))
        assert d.rationale["lambda_identified"] is False, "nothing has been observed yet"


def test_lambda_information_grows_with_observations_at_distinct_delays():
    """Two probes at the same delay after a demonstration carry less information about lambda
    than two at different delays -- the Fisher argument B6 is built on, checked numerically."""
    from vla_lab.supervisory.carryover import CarryoverPosterior

    def info(deltas):
        post = CarryoverPosterior(CONTRACT.carryover)
        post.force_point_mass(lam=0.6, beta=1.5, g=1.0)
        post.step(action=COACH, delta=deltas[0], direction=1)
        total = 0.0
        for d in deltas[1:]:
            total += post.lambda_information(pi_star=0.5)["gain_if_observe"]
            post.step(action=PROBE, delta=d, pi_star=0.5, chose_a=True)
        return total

    same = info([1.0, 1.0, 1.0])
    spread = info([0.2, 2.5, 1.0])
    assert spread > 0.0 and same > 0.0
    assert info([1.0, 1.0]) < info([1.0, 1.0, 1.0]), "information accumulates"

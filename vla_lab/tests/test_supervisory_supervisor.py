"""The generative supervisor: does it produce the phenomenon the study is designed to find?"""

from __future__ import annotations

import numpy as np

from vla_lab.supervisory import COUNTER, PROBE, STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED
from vla_lab.supervisory.carryover import CarryoverConfig
from vla_lab.supervisory.narration import ground
from vla_lab.supervisory.scenes import build_scene_grid
from vla_lab.supervisory.strategies import PLAN_AXIS
from vla_lab.supervisory.supervisor import (
    SimulatedSupervisor,
    SupervisorParams,
    SupervisorPopulation,
    draw_cohort,
    draw_supervisor,
)
from vla_lab.tests import approx

GRID = build_scene_grid()


def _sup(**kw):
    base = dict(supervisor_id="S", a=1.5, c0=0.0, d=0.0, beta=1.2, g=1.0, lam=0.6,
                phi=0.0, lapse=0.0, ungrounded=0.0, latency_s=2.0)
    base.update(kw)
    return SimulatedSupervisor(SupervisorParams(**base), axis="plan", cfg=CarryoverConfig(), seed=0)


def test_the_preference_map_is_monotone_and_saturating():
    s = _sup()
    m = s.pi_star_map(GRID)
    cs = sorted(GRID.probe_scenes(), key=lambda x: x.c)
    vals = [m[x.scene_id] for x in cs]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])), vals
    assert vals[0] < 0.15 and vals[-1] > 0.85


def test_coaching_moves_the_answer_in_the_direction_it_was_coached():
    s = _sup()
    scene = GRID.by_id(7)
    cold = s.respond(scene).p_a
    s.apply_coach(+1)
    up = s.respond(scene).p_a
    s.kappa = 0.0
    s.apply_coach(-1)
    down = s.respond(scene).p_a
    assert down < cold < up, (down, cold, up)


def test_a_counter_proposal_pulls_the_answer_back_toward_the_unprompted_one():
    cfg = CarryoverConfig(rho_counter=0.4)
    s = SimulatedSupervisor(SupervisorParams(supervisor_id="S", a=1.5, c0=0.0, d=0.0, beta=1.5, g=1.4,
                                             lam=0.6, phi=0.0, lapse=0.0, ungrounded=0.0, latency_s=2.0),
                            axis="plan", cfg=cfg, seed=0)
    scene = GRID.by_id(7)
    s.apply_coach(+1)
    probe = s.respond(scene, action=PROBE)
    counter = s.respond(scene, action=COUNTER)
    assert abs(counter.p_a - counter.pi_star) < abs(probe.p_a - probe.pi_star)


def test_a_non_complier_is_unmoved_by_any_amount_of_coaching():
    s = _sup(beta=0.0)
    scene = GRID.by_id(7)
    cold = s.respond(scene).p_a
    for _ in range(5):
        s.apply_coach(+1)
    assert approx(s.respond(scene).p_a, cold, tol=1e-9)


def test_the_residue_decays_toward_nothing():
    s = _sup(lam=0.5)
    s.apply_coach(+1)
    k0 = s.kappa_eff
    for _ in range(8):
        s.decay(1.0)
    assert 0.0 < s.kappa_eff < 0.05 * k0


def test_session_drift_is_present_and_is_not_carryover():
    # Drift pushes a whole session one way; carryover pushes each gap toward what was just
    # shown. If drift were absent the design would never be stress-tested against its own
    # worst confound.
    s = _sup(phi=-1.0)
    scene = GRID.by_id(7)
    early = s.respond(scene, session_progress=0.0).p_a
    late = s.respond(scene, session_progress=1.0).p_a
    assert late < early


def test_every_utterance_a_supervisor_produces_is_either_grounded_or_a_hedge():
    s = _sup(ungrounded=0.3)
    seen = set()
    for i in range(300):
        r = s.respond(GRID.by_id(7))
        g = ground(r.utterance, PLAN_AXIS)
        seen.add(g)
        # When the utterance grounds at all, it must ground to what they meant.
        if g != STRATEGY_UNRESOLVED:
            assert g == r.strategy, (r.utterance, g, r.strategy)
    assert STRATEGY_UNRESOLVED in seen


def test_lapses_pull_toward_chance_not_toward_a_side():
    s = _sup(lapse=1.0)
    n_a = sum(1 for _ in range(600) if s.respond(GRID.by_id(0)).strategy == STRATEGY_A)
    assert 0.4 < n_a / 600 < 0.6, n_a / 600


def test_a_cohort_is_reproducible_and_heterogeneous():
    a = draw_cohort(12, seed=4)
    b = draw_cohort(12, seed=4)
    assert [x.p.to_dict() for x in a] == [x.p.to_dict() for x in b]
    bgs = [x.p.beta * x.p.g for x in a]
    assert max(bgs) - min(bgs) > 0.5


def test_the_population_contains_people_who_simply_do_not_comply():
    pop = SupervisorPopulation(p_noncomplier=0.5)
    import random

    rng = random.Random(0)
    n0 = sum(1 for i in range(200) if draw_supervisor(rng, pop, supervisor_id=f"S{i}").beta == 0.0)
    assert 60 < n0 < 140, n0


def test_the_misspecified_population_does_not_decay_as_one_exponential():
    pop = SupervisorPopulation().misspecify(decay_shape="double")
    import random

    p = draw_supervisor(random.Random(0), pop, supervisor_id="S")
    assert p.decay_shape == "double"
    s = SimulatedSupervisor(p, axis="plan", cfg=CarryoverConfig(), seed=0)
    s.apply_coach(+1)
    ks = []
    for _ in range(6):
        s.decay(1.0)
        ks.append(s.kappa_eff)
    ratios = [b / a for a, b in zip(ks, ks[1:]) if a > 1e-9]
    # A single exponential has a constant decay ratio; a two-component residue does not.
    assert max(ratios) - min(ratios) > 1e-3, ratios

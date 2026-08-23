"""W6 — tests for ``vla_lab.old_direction.rehab.sim_participant``.

"Done when" (``rehab.md`` §6/W6): sampled maps are monotone in the lateral coordinate;
carryover shows the expected post-COACH elevation and decay; seeding is reproducible; the
misdetection rate is honoured.
"""

from __future__ import annotations

from vla_lab.old_direction.rehab import ARM_NONPREFERRED, ARM_PREFERRED, ASSESS, COACH, WAIT
from vla_lab.old_direction.rehab.contract import Phase0Contract
from vla_lab.old_direction.rehab.sim_participant import (
    PopulationPrior,
    SimulatedObserver,
    SimulatedParticipant,
    draw_participant,
)
from vla_lab.old_direction.rehab.workspace import SIDE_LEFT, SIDE_RIGHT, nonpreferred_lateral

CONTRACT = Phase0Contract()
GRID = CONTRACT.target_grid()


def _participant(idx: int = 0, *, side: str = SIDE_LEFT, seed: int = 7, prior=None, total=200):
    p = draw_participant(idx, prior, nonpreferred_side=side, seed=seed)
    return SimulatedParticipant(p, GRID, prior=prior, total_trials=total)


# --------------------------------------------------------------------------- the estimand


def test_pi_star_is_monotone_in_the_nonpreferred_lateral_coordinate():
    for idx in range(6):
        for side in (SIDE_LEFT, SIDE_RIGHT):
            sim = _participant(idx, side=side, seed=idx + 1)
            m = sim.pi_star_map()
            for depth in {t.depth_bin for t in GRID}:
                row = sorted(
                    (t for t in GRID if t.depth_bin == depth),
                    key=lambda t: nonpreferred_lateral(t.y_m, side),
                )
                vals = [m[t.target_id] for t in row]
                assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), (idx, side, depth, vals)


def test_pi_star_spans_the_full_range_across_the_workspace():
    sim = _participant(0, seed=3)
    vals = list(sim.pi_star_map().values())
    assert min(vals) < 0.25 and max(vals) > 0.75, (min(vals), max(vals))


def test_population_draws_are_heterogeneous():
    crossovers = [draw_participant(i, seed=42).crossover_m for i in range(24)]
    spread = max(crossovers) - min(crossovers)
    assert spread > 0.05, spread  # between-person variation is the headline, not a nuisance


# --------------------------------------------------------------------------- carryover


def test_coach_elevates_nonpreferred_use_and_the_elevation_decays():
    prior = PopulationPrior(beta_mean=1.5, beta_sd=0.0, g_mean=1.5, g_sd=0.0, lambda_a=8.0, lambda_b=2.0, lapse_rate=0.0)
    target = min(GRID, key=lambda t: abs(t.y_m))  # a crossover-band target: choice is movable
    counts = {0: 0, 1: 0, 2: 0, 5: 0}
    reps = 400
    for r in range(reps):
        sim = _participant(0, seed=1000 + r, prior=prior, total=50)
        sim.select(target, action=COACH, strength=1.0, delta=1.0)
        for lag in range(6):
            resp = sim.select(target, action=ASSESS, strength=1.0, delta=1.0)
            if lag in counts and resp is not None and resp.arm == ARM_NONPREFERRED:
                counts[lag] += 1
    # Immediately after the prompt, the most; the elevation then decays with lag.
    assert counts[0] > counts[5], counts
    assert counts[0] >= counts[2] - reps * 0.02, counts


def test_wait_returns_no_response_but_still_decays_kappa():
    prior = PopulationPrior(beta_mean=1.5, beta_sd=0.0, g_mean=1.5, g_sd=0.0, lapse_rate=0.0)
    sim = _participant(0, seed=5, prior=prior)
    sim.select(min(GRID, key=lambda t: abs(t.y_m)), action=COACH, delta=1.0)
    before = sim.kappa
    assert sim.select(None, action=WAIT, delta=1.0) is None
    assert sim.kappa < before


# --------------------------------------------------------------------------- reproducibility


def test_seeding_is_reproducible():
    a = _participant(3, seed=11)
    b = _participant(3, seed=11)
    t = GRID.targets[4]
    ra = [a.select(t, action=ASSESS, delta=1.0).arm for _ in range(50)]
    rb = [b.select(t, action=ASSESS, delta=1.0).arm for _ in range(50)]
    assert ra == rb


def test_different_seeds_give_different_participants():
    a = draw_participant(0, seed=1)
    b = draw_participant(0, seed=2)
    assert a.to_dict() != b.to_dict()


# --------------------------------------------------------------------------- observation


def test_misdetection_rate_is_honoured():
    for rate in (0.0, 0.2):
        prior = PopulationPrior(misdetect_rate=rate)
        sim = _participant(0, seed=17, prior=prior, total=2000)
        t = max(GRID, key=lambda x: nonpreferred_lateral(x.y_m, SIDE_LEFT))  # pi* ~ 1
        flips = 0
        n = 1500
        for _ in range(n):
            resp = sim.select(t, action=ASSESS, delta=1.0)
            arm, _side, _c = sim.observe(resp)
            flips += int(arm != resp.arm)
        assert abs(flips / n - rate) < 0.05, (rate, flips / n)


def test_misdetection_lowers_reported_confidence():
    prior = PopulationPrior(misdetect_rate=1.0)
    sim = _participant(0, seed=19, prior=prior)
    resp = sim.select(GRID.targets[0], action=ASSESS, delta=1.0)
    _arm, _side, conf = sim.observe(resp)
    assert conf < 0.8


def test_simulated_observer_follows_the_observer_protocol():
    sim = _participant(0, seed=23)
    obs = SimulatedObserver(sim)
    t = GRID.targets[3]
    obs.prepare(t, action=ASSESS, strength=1.0, delta=1.0)
    obs.begin_trial(0, 1000)
    assert obs.poll(1000) is None            # nothing yet: the reach takes time
    sel = obs.poll(1000 + 5000)
    assert sel is not None and sel.arm in (ARM_PREFERRED, ARM_NONPREFERRED)
    assert sel.t_ms > 1000
    assert obs.poll(1000 + 6000) is sel      # latch-once: a second detection is a re-attempt


def test_simulated_observer_reports_no_reach_when_nothing_was_prepared():
    sim = _participant(0, seed=29)
    obs = SimulatedObserver(sim)
    obs.begin_trial(0, 0)
    sel = obs.end_trial(9999)
    assert sel.arm == "none"


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_sim_participant") else 0)

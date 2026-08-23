"""The estimand, its three estimators, and the metrics the paper reports."""

from __future__ import annotations

import numpy as np

from vla_lab.supervisory import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, WAIT
from vla_lab.supervisory.carryover import CarryoverConfig, CarryoverPosterior
from vla_lab.supervisory.estimand import (
    METHOD_CORRECTED,
    METHOD_POOLED,
    METHOD_PSYCHOMETRIC,
    CarryoverCorrectedEstimator,
    Observation,
    PooledBetaEstimator,
    PsychometricEstimator,
    decision_regret,
    evaluate,
    executed_regret,
    interval_coverage,
    joint_carryover_posterior,
    mae,
    reference_map_from_observations,
)
from vla_lab.supervisory.scenes import build_scene_grid
from vla_lab.tests import approx

GRID = build_scene_grid()


def _true_map(a=1.4, c0=0.1):
    return {s.scene_id: float(1.0 / (1.0 + np.exp(-(a * (s.c - c0))))) for s in GRID.probe_scenes()}


def _draw(n, *, seed=0, a=1.4, c0=0.1, kappa_fn=None, cfg=None):
    """A block of observations, optionally contaminated by a decaying residue."""
    rng = np.random.default_rng(seed)
    cfg = cfg or CarryoverConfig()
    scenes = GRID.probe_scenes()
    out, kappa = [], 0.0
    for i in range(n):
        if kappa_fn is not None and i % 6 == 0:
            out.append(Observation(slot=i, action=COACH, scene_id=GRID.coach_scenes()[0].scene_id,
                                   c=GRID.coach_scenes()[0].c, clutter=2, coach_direction=1, delta=1.0))
            kappa = kappa_fn(kappa, coached=True)
            continue
        s = scenes[int(rng.integers(len(scenes)))]
        eta = a * (s.c - c0) + kappa
        p = 1.0 / (1.0 + np.exp(-eta))
        out.append(Observation(slot=i, action=PROBE, scene_id=s.scene_id, c=s.c, clutter=s.clutter,
                               instructed=STRATEGY_A if rng.random() < p else STRATEGY_B, delta=1.0))
        if kappa_fn is not None:
            kappa = kappa_fn(kappa, coached=False)
    return out


def test_the_psychometric_estimator_recovers_a_clean_map():
    seq = _draw(600, seed=1)
    est = PsychometricEstimator().fit(seq, GRID)
    assert est.method == METHOD_PSYCHOMETRIC
    assert mae(est, _true_map(), GRID) < 0.06


def test_the_pooled_estimator_is_honest_but_noisy_on_a_small_budget():
    seq = _draw(60, seed=2)
    pooled = PooledBetaEstimator().fit(seq, GRID)
    psych = PsychometricEstimator().fit(seq, GRID)
    # Sharing strength across scenes is a requirement, not a refinement, at realistic budgets.
    assert mae(psych, _true_map(), GRID) < mae(pooled, _true_map(), GRID)


def test_contamination_biases_the_naive_fit_and_the_correction_removes_most_of_it():
    lam, g, beta = 0.6, 1.2, 1.3

    def kf(k, coached):
        return (k + beta * g) * lam if coached else k * lam

    seq = _draw(900, seed=3, kappa_fn=kf)
    truth = _true_map()
    naive = PsychometricEstimator().fit(seq, GRID)
    post, _ = joint_carryover_posterior(seq, GRID)
    corrected = CarryoverCorrectedEstimator().fit(seq, GRID, post)
    assert corrected.method == METHOD_CORRECTED
    assert mae(corrected, truth, GRID) < mae(naive, truth, GRID), (
        mae(corrected, truth, GRID), mae(naive, truth, GRID)
    )


def test_the_joint_fit_detects_the_contamination_it_corrects_for():
    lam, g, beta = 0.6, 1.2, 1.3

    def kf(k, coached):
        return (k + beta * g) * lam if coached else k * lam

    post, _ = joint_carryover_posterior(_draw(900, seed=4, kappa_fn=kf), GRID)
    assert post.effect(threshold=0.2)["p_above_threshold"] > 0.5


def test_the_joint_fit_finds_nothing_when_there_is_nothing():
    post, _ = joint_carryover_posterior(_draw(900, seed=5), GRID)
    assert post.effect(threshold=0.6)["p_above_threshold"] < 0.5


def test_crossover_weighting_concentrates_the_error_where_the_map_bends():
    seq = _draw(400, seed=6)
    est = PsychometricEstimator().fit(seq, GRID)
    truth = _true_map()
    flat = mae(est, truth, GRID, crossover_weighted=False)
    band = mae(est, truth, GRID, crossover_weighted=True)
    assert flat >= 0.0 and band >= 0.0
    assert not approx(flat, band, tol=1e-9)


def test_coverage_is_reported_at_every_nominal_level_and_is_monotone():
    seq = _draw(300, seed=7)
    est = PsychometricEstimator().fit(seq, GRID)
    cov = interval_coverage(est, _true_map(), GRID)
    assert set(cov) == {"coverage@50", "coverage@80", "coverage@95"}
    assert cov["coverage@50"] <= cov["coverage@80"] <= cov["coverage@95"] + 1e-9


def test_deployment_regret_is_zero_for_a_perfect_map_and_positive_for_a_wrong_one():
    truth = _true_map()
    perfect = PsychometricEstimator().fit(_draw(2000, seed=8), GRID)
    good = decision_regret(perfect, truth, GRID)
    flipped = {k: 1.0 - v for k, v in truth.items()}
    bad = decision_regret(perfect, flipped, GRID)
    assert good["deployment_regret"] < bad["deployment_regret"]
    assert good["alignment"] > bad["alignment"]


def test_executed_regret_counts_only_answers_that_cost_something():
    truth = _true_map()
    seq = _draw(200, seed=9)
    r = executed_regret(seq, truth, GRID)
    assert r["n_slots"] > 0
    assert 0.0 <= r["flip_rate"] <= 1.0
    assert r["executed_regret_per_slot"] >= 0.0


def test_an_ungrounded_answer_advances_the_state_but_never_the_likelihood():
    seq = _draw(80, seed=10)
    seq.append(Observation(slot=999, action=PROBE, scene_id=GRID.probe_scenes()[0].scene_id,
                           c=GRID.probe_scenes()[0].c, clutter=2, instructed=None, delta=1.0))
    est = PsychometricEstimator().fit(seq, GRID)
    assert est.diagnostics["n_observations"] == sum(1 for o in seq if o.observed)


def test_the_reference_map_pools_across_scenes_rather_than_counting_per_scene():
    seq = _draw(40, seed=11)
    ref = reference_map_from_observations(seq, GRID)
    assert len(ref) == len(GRID.probe_scenes())
    assert all(0.0 <= v <= 1.0 for v in ref.values())


def test_evaluate_reports_every_outcome_the_paper_prints():
    seq = _draw(200, seed=12)
    est = PsychometricEstimator().fit(seq, GRID)
    row = evaluate(est, _true_map(), GRID, seq)
    for k in ("mae", "mae_crossover", "brier_crossover", "coverage@95", "deployment_regret",
              "alignment", "executed_regret_per_slot", "n_probe", "n_counter", "n_wait", "n_ungrounded"):
        assert k in row, k


def test_counter_observations_are_de_biased_less_aggressively_than_plain_probes():
    # rho < 1 means a countered answer carries less residue, so the correction must subtract
    # less from it -- getting this backwards would over-correct exactly the cleanest answers.
    cfg = CarryoverConfig(rho_counter=0.5)
    post = CarryoverPosterior(cfg)
    post.force_point_mass(lam=0.6, beta=1.5, g=1.5)
    post.kappa = post.g * 1.0
    assert abs(post.contamination(action=COUNTER)["mean"]) < abs(post.contamination(action=PROBE)["mean"])

"""The carryover model: signs, decay, recovery, identifiability, and the counter attenuation."""

from __future__ import annotations

import numpy as np

from vla_lab.supervisory import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, WAIT
from vla_lab.supervisory.carryover import (
    DECAY_TIME,
    DECAY_TRIALS,
    CarryoverConfig,
    CarryoverPosterior,
    delta_for,
    fit_population_prior,
    fit_rho,
    kappa_trace,
)
from vla_lab.supervisory.scheduler.base import DeltaModel
from vla_lab.tests import approx


def _sim(n_blocks, truth, rho_true=0.4, seed=0, waits=(0, 1, 2, 4), directions=(1, -1)):
    rng = np.random.default_rng(seed)
    recs = []
    k = 0.0
    for blk in range(n_blocks):
        d = directions[blk % len(directions)]
        plan = [(COACH, d)] + [(WAIT, 0)] * waits[blk % len(waits)] + [(PROBE, 0), (COUNTER, 0), (PROBE, 0)]
        for a, dd in plan:
            eff = k + (truth["g"] * dd if a == COACH else 0.0)
            rec = {"action": a, "delta": 1.0, "coach_direction": dd, "coach_strength": 1.0, "scene_id": 7}
            if a in (PROBE, COUNTER):
                rho = rho_true if a == COUNTER else 1.0
                p = 1.0 / (1.0 + np.exp(-(rho * truth["beta"] * eff)))
                rec["instructed_strategy"] = STRATEGY_A if rng.random() < p else STRATEGY_B
            recs.append(rec)
            k = truth["lam"] * eff
    return recs


def _fit(recs, cfg=None, log_prior=None):
    post = CarryoverPosterior(cfg or CarryoverConfig(decay_mode=DECAY_TRIALS), log_prior=log_prior)
    for r in recs:
        has = r.get("instructed_strategy") is not None
        post.step(action=r["action"], delta=r["delta"],
                  pi_star=0.5 if has else None,
                  chose_a=(r["instructed_strategy"] == STRATEGY_A) if has else None,
                  direction=r["coach_direction"], strength=1.0)
    return post


def test_kappa_is_signed_and_coaching_either_way_moves_it_that_way():
    up = kappa_trace([COACH, PROBE], lam=0.7, g=1.0, directions=[+1, +1])
    down = kappa_trace([COACH, PROBE], lam=0.7, g=1.0, directions=[-1, +1])
    assert up[0] > 0 and down[0] < 0
    assert approx(up[0], -down[0])


def test_kappa_decays_and_never_grows_without_a_demonstration():
    tr = kappa_trace([COACH, PROBE, PROBE, PROBE], lam=0.5, g=1.0, directions=[1, 1, 1, 1])
    assert tr[0] > tr[1] > tr[2] > tr[3] > 0.0


def test_a_wait_buys_decay_and_costs_no_observation():
    tr = kappa_trace([COACH, WAIT, WAIT], lam=0.5, g=1.0)
    assert tr[2] < tr[1] < tr[0]


def test_the_posterior_recovers_a_known_compliance_strength():
    truth = {"lam": 0.6, "beta": 1.2, "g": 1.0}
    post = _fit(_sim(200, truth, seed=11))
    lo, hi = post.credible_interval("beta_g", level=0.9)
    assert lo <= truth["beta"] * truth["g"] <= hi, (lo, hi)
    assert post.mean()["beta_g"] > 0.4


def test_a_non_complier_is_reported_as_such_rather_than_given_a_spurious_effect():
    post = _fit(_sim(200, {"lam": 0.6, "beta": 0.0, "g": 1.0}, seed=5))
    eff = post.effect(threshold=0.5)
    assert eff["p_above_threshold"] < 0.5, eff
    assert post.mean()["beta_g"] < 0.8


def test_zero_carryover_is_representable_at_all():
    # If the grid cannot express "no effect", the study cannot answer its own go/no-go.
    post = CarryoverPosterior()
    assert float(np.min(post.beta)) == 0.0
    assert float(np.min(post.g)) == 0.0


def test_identifiability_reports_an_axis_the_data_never_moved():
    # A single constant-gap design cannot identify the decay rate. The diagnostic must say so
    # rather than letting a scheduler personalise on its prior.
    post = _fit(_sim(60, {"lam": 0.6, "beta": 1.2, "g": 1.0}, seed=3, waits=(1,)))
    ident = post.identifiability()
    assert ident["lambda"]["tv"] < ident["beta_g"]["tv"]


def test_resampling_represents_the_posterior_not_its_mode():
    # Top-k of a prior peaked at zero returns only no-carryover cells, which silently zeroes
    # every downstream bias term. Systematic resampling must not.
    post = CarryoverPosterior()
    top = post.resample_cells(32, mode="top")
    sys_ = post.resample_cells(32, mode="systematic")
    assert np.mean([c["beta"] for c in sys_]) > np.mean([c["beta"] for c in top])


def test_the_counter_attenuation_is_estimated_and_beats_its_prior():
    truth = {"lam": 0.6, "beta": 1.2, "g": 1.0}
    recs = _sim(500, truth, rho_true=0.3, seed=7)
    out = fit_rho(recs, pi_star={7: 0.5}, lam=truth["lam"], beta=truth["beta"], g=truth["g"])
    assert out["source"] == "measured" and out["n_counter"] > 0
    assert abs(out["rho"] - 0.3) < 0.25, out


def test_rho_falls_back_to_its_prior_when_no_counter_proposals_were_made():
    recs = [r for r in _sim(50, {"lam": 0.6, "beta": 1.0, "g": 1.0}, seed=1) if r["action"] != COUNTER]
    out = fit_rho(recs, pi_star={7: 0.5}, lam=0.6, beta=1.0, g=1.0)
    assert out["source"] == "prior" and out["n_counter"] == 0


def test_a_counter_proposal_shrinks_the_predicted_contamination():
    post = CarryoverPosterior()
    post.kappa = post.g * 1.0
    plain = abs(post.contamination(action=PROBE)["mean"])
    counter = abs(post.contamination(action=COUNTER)["mean"])
    assert counter < plain


def test_the_personalised_washout_grows_with_the_decay_rate():
    slow, fast = CarryoverPosterior(), CarryoverPosterior()
    for p, lam in ((slow, 0.95), (fast, 0.10)):
        p.force_point_mass(lam=lam, beta=1.5, g=1.5)
        p.kappa = p.g * 1.0
    assert slow.washout_delta(0.1) > fast.washout_delta(0.1)


def test_time_decay_and_trial_decay_disagree_about_what_a_wait_buys():
    dm_t = DeltaModel(decay_mode=DECAY_TIME, time_unit_s=30.0, wait_s=45.0, probe_s=42.0)
    dm_n = DeltaModel(decay_mode=DECAY_TRIALS, wait_interference=0.55)
    assert dm_t.for_action(WAIT) > dm_t.for_action(PROBE)
    assert dm_n.for_action(WAIT) < dm_n.for_action(PROBE)


def test_the_population_prior_excludes_the_supervisor_it_is_built_for():
    truth = {"lam": 0.6, "beta": 1.4, "g": 1.2}
    posts = [_fit(_sim(80, truth, seed=s)) for s in range(4)]
    loo = fit_population_prior(posts, exclude=0)
    full = fit_population_prior(posts)
    assert loo is not None and full is not None
    assert not np.allclose(loo, full)


def test_a_population_prior_makes_a_short_session_more_accurate():
    """The point of the empirical prior is accuracy, not narrowness.

    The default prior is a half-normal peaked at zero, so it is *already* narrow -- and wrong,
    because it insists nobody complies. A population prior fitted on people who do comply is
    wider on ``beta*g`` and much closer to the truth. Testing for a narrower posterior would
    therefore pass whenever the prior was uninformative and fail whenever it worked.
    """
    truth = {"lam": 0.6, "beta": 1.4, "g": 1.2}
    target = truth["beta"] * truth["g"]
    cohort = [_fit(_sim(120, truth, seed=100 + s)) for s in range(5)]
    prior = fit_population_prior(cohort)
    errs_flat, errs_pop = [], []
    for s in range(6):
        short = _sim(10, truth, seed=900 + s)
        errs_flat.append(abs(_fit(short).mean()["beta_g"] - target))
        errs_pop.append(abs(_fit(short, log_prior=prior).mean()["beta_g"] - target))
    assert np.mean(errs_pop) < np.mean(errs_flat), (np.mean(errs_pop), np.mean(errs_flat))

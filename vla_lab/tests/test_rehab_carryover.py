"""W3 — tests for ``vla_lab.rehab.carryover``.

"Done when" (``rehab.md`` §6/W3): parameter recovery on data generated from the model
(posterior mean within tolerance, credible intervals covering at the nominal rate); the
posterior concentrates monotonically with more data; degenerate cases (``g=0``, ``beta=0``) are
identified as such rather than producing spurious confidence.
"""

from __future__ import annotations

import math
import random

from vla_lab.rehab import ASSESS, COACH, WAIT
from vla_lab.rehab.carryover import (
    DECAY_TIME,
    DECAY_TRIALS,
    CarryoverConfig,
    CarryoverPosterior,
    delta_for,
    kappa_trace,
    logit,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _generate(lam: float, beta: float, g: float, *, n: int = 500, seed: int = 0, coach_every: int = 5):
    """Data from the model itself: recovery tests are about the estimator, not a mismatch."""

    rng = random.Random(seed)
    post = CarryoverPosterior(CarryoverConfig(decay_mode=DECAY_TRIALS))
    kappa = 0.0
    for t in range(n):
        pi = 0.25 + 0.5 * rng.random()  # a spread of target difficulties
        action = COACH if (t % coach_every == 0) else ASSESS
        eff = kappa + (g if action == COACH else 0.0)
        p = _sigmoid(logit(pi) + beta * eff)
        y = rng.random() < p
        post.step(action=action, delta=1.0, pi_star=pi, chose_nonpreferred=y)
        kappa = (lam ** 1.0) * eff
    return post


# --------------------------------------------------------------------------- dynamics


def test_kappa_trace_matches_the_documented_recursion():
    actions = [COACH, ASSESS, ASSESS, COACH, ASSESS]
    lam, g = 0.5, 1.0
    tr = kappa_trace(actions, lam=lam, g=g)
    # COACH injects immediately, then everything decays by lam per slot.
    assert abs(tr[0] - 1.0) < 1e-12
    assert abs(tr[1] - 0.5) < 1e-12
    assert abs(tr[2] - 0.25) < 1e-12
    assert abs(tr[3] - (0.125 + 1.0)) < 1e-12
    assert abs(tr[4] - 0.5625) < 1e-12


def test_kappa_decays_to_zero_without_prompts():
    tr = kappa_trace([COACH] + [ASSESS] * 40, lam=0.6, g=1.0)
    assert tr[-1] < 1e-6
    assert all(tr[i] >= tr[i + 1] for i in range(1, len(tr) - 1))


def test_delta_modes_differ():
    trials = CarryoverConfig(decay_mode=DECAY_TRIALS)
    timed = CarryoverConfig(decay_mode=DECAY_TIME, time_unit_s=10.0)
    assert delta_for(trials, dt_ms=99999) == 1.0
    assert abs(delta_for(timed, dt_ms=7400) - 0.74) < 1e-9


# --------------------------------------------------------------------------- recovery


def test_parameter_recovery_on_data_from_the_model():
    for lam, beta, g in [(0.7, 1.2, 0.9), (0.85, 0.9, 1.1)]:
        post = _generate(lam, beta, g, n=600, seed=int(lam * 1000))
        m = post.mean()
        # beta and g are only weakly separable; their PRODUCT is what is identified.
        assert abs(m["beta_g"] - beta * g) < 0.6, (lam, beta, g, m)
        lo, hi = post.credible_interval("beta_g", 0.9)
        assert lo <= beta * g <= hi, (beta * g, lo, hi)
        lo_l, hi_l = post.credible_interval("lambda", 0.9)
        assert lo_l <= lam <= hi_l, (lam, lo_l, hi_l)


def test_credible_intervals_cover_at_about_the_nominal_rate():
    hits = 0
    trials = 20
    for s in range(trials):
        lam, beta, g = 0.7, 1.0, 1.0
        post = _generate(lam, beta, g, n=400, seed=1000 + s)
        lo, hi = post.credible_interval("beta_g", 0.9)
        hits += int(lo <= beta * g <= hi)
    assert hits >= int(0.7 * trials), f"coverage {hits}/{trials} well below nominal 90%"


def test_posterior_concentrates_with_more_data():
    widths = []
    for n in (100, 300, 900):
        post = _generate(0.75, 1.2, 1.0, n=n, seed=7)
        lo, hi = post.credible_interval("beta_g", 0.9)
        widths.append(hi - lo)
    assert widths[0] > widths[-1], widths
    assert widths == sorted(widths, reverse=True), widths


# --------------------------------------------------------------------------- degeneracy


def test_no_carryover_is_reported_as_no_carryover():
    post = _generate(0.7, 0.0, 0.0, n=500, seed=3)
    eff = post.effect()
    assert not eff["detected"], eff
    assert eff["beta_g_ci"][0] <= 0.15, eff
    ident = post.identifiability()
    assert not ident["lambda"]["identified"], ident["lambda"]


def test_lambda_is_unidentified_without_any_coach():
    post = CarryoverPosterior(CarryoverConfig())
    rng = random.Random(0)
    for _ in range(200):
        post.step(action=ASSESS, delta=1.0, pi_star=0.5, chose_nonpreferred=rng.random() < 0.5)
    ident = post.identifiability()
    assert not ident["lambda"]["identified"]
    assert "no COACH" in str(ident["lambda"].get("note", "")) or "not credibly" in str(ident.get("note", ""))


def test_strong_effect_is_detected():
    post = _generate(0.75, 1.4, 1.2, n=600, seed=11)
    assert post.effect()["detected"]


# --------------------------------------------------------------------------- contamination


def test_contamination_decays_with_waiting_and_rises_with_a_prompt():
    post = _generate(0.7, 1.2, 1.0, n=200, seed=5)
    now, _ = post.predict_contamination(0.0)
    later, _ = post.predict_contamination(5.0)
    assert later < now or now < 1e-6
    with_coach = post.contamination(if_coach=True)["mean"]
    assert with_coach > post.contamination()["mean"]


def test_washout_delta_is_monotone_in_the_threshold():
    post = _generate(0.8, 1.2, 1.2, n=300, seed=9)
    strict = post.washout_delta(tau=0.05)
    loose = post.washout_delta(tau=0.4)
    assert strict >= loose


# --------------------------------------------------------------------------- resampling


def test_resample_cells_represents_the_posterior_not_its_mode():
    """Under a diffuse prior the highest-density cells all sit at beta ~ 0, g ~ 0.

    Selecting the top-k by weight would report "no carryover" regardless of the data — the bug
    this resampling exists to prevent.
    """

    post = CarryoverPosterior(CarryoverConfig())
    cells = post.resample_cells(24)
    assert len(cells) == 24
    assert abs(sum(c["weight"] for c in cells) - 1.0) < 1e-9
    mean_bg = sum(c["beta"] * c["g"] * c["weight"] for c in cells)
    assert mean_bg > 0.1, mean_bg  # a mode-collapsed selection would be ~0
    # And it is deterministic.
    assert [c["index"] for c in cells] == [c["index"] for c in post.resample_cells(24)]


def test_force_point_mass_collapses_the_posterior():
    post = CarryoverPosterior(CarryoverConfig())
    post.force_point_mass(lam=0.72, beta=1.0, g=0.8)
    m = post.mean()
    assert abs(m["lambda"] - 0.72) < 0.1
    sd = post.sd()
    assert sd["lambda"] < 1e-6 and sd["beta"] < 1e-6


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_carryover") else 0)

"""W4 — tests for ``vla_lab.rehab.estimand``.

"Done when" (``rehab.md`` §6/W4): on synthetic clean data all three estimators agree; on
synthetic contaminated data the carryover-corrected one is unbiased while the other two are
biased **in the predicted direction**; interval coverage is at the nominal level; metrics match
hand-computed values on a small fixture.
"""

from __future__ import annotations

import math
import random

import numpy as np

from vla_lab.rehab import ASSESS, COACH
from vla_lab.rehab.carryover import CarryoverConfig
from vla_lab.rehab.contract import Phase0Contract
from vla_lab.rehab.estimand import (
    METHOD_CORRECTED,
    METHOD_POOLED,
    METHOD_SPATIAL,
    CarryoverCorrectedEstimator,
    PooledBetaEstimator,
    SpatialLogisticEstimator,
    TrialObservation,
    brier_vs_outcomes,
    brier_vs_reference,
    fit_all,
    interval_coverage,
    joint_carryover_posterior,
    mae,
)
from vla_lab.rehab.workspace import nonpreferred_lateral

CONTRACT = Phase0Contract()
GRID = CONTRACT.target_grid()
SIDE = "left"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _truth(steepness: float = 12.0, crossover: float = 0.01):
    return {
        t.target_id: _sigmoid(steepness * (nonpreferred_lateral(t.y_m, SIDE) - crossover))
        for t in GRID
    }


def _generate(lam: float, beta: float, g: float, *, n: int, seed: int, coach_every: int = 5, truth=None):
    """A contaminated block. ``coach_every=0`` gives a clean (reference-like) block."""

    rng = random.Random(seed)
    truth = truth or _truth()
    seq, kappa = [], 0.0
    for t in range(n):
        tg = GRID.targets[rng.randrange(len(GRID.targets))]
        action = COACH if (coach_every and t % coach_every == 0) else ASSESS
        eff = kappa + (g if action == COACH else 0.0)
        p0 = truth[tg.target_id]
        p = _sigmoid(math.log(p0 / (1 - p0)) + beta * eff)
        seq.append(
            TrialObservation(
                trial_idx=t, action=action, target_id=tg.target_id,
                s_m=nonpreferred_lateral(tg.y_m, SIDE), depth_m=tg.x_m,
                y=(rng.random() < p), delta=1.0,
            )
        )
        kappa = lam * eff
    return seq


def _band_bias(est, truth) -> float:
    band = {t.target_id for t in GRID.crossover_targets()}
    m = est.mean()
    return sum(m[t] - truth[t] for t in band) / len(band)


# --------------------------------------------------------------------------- clean data


def test_all_three_estimators_agree_on_clean_data():
    truth = _truth()
    seq = _generate(0.7, 0.0, 0.0, n=400, seed=1, coach_every=0, truth=truth)
    fits = fit_all(seq, GRID, SIDE)
    for name in (METHOD_POOLED, METHOD_SPATIAL, METHOD_CORRECTED):
        assert abs(_band_bias(fits[name], truth)) < 0.08, (name, _band_bias(fits[name], truth))
    spatial = fits[METHOD_SPATIAL].mean()
    corrected = fits[METHOD_CORRECTED].mean()
    assert max(abs(spatial[k] - corrected[k]) for k in spatial) < 0.08


def test_spatial_beats_pooled_when_the_budget_per_target_is_thin():
    """Sharing strength across nearby targets is a requirement, not a refinement."""

    truth = _truth()
    seq = _generate(0.7, 0.0, 0.0, n=60, seed=4, coach_every=0, truth=truth)
    pooled = PooledBetaEstimator().fit(seq, GRID)
    spatial = SpatialLogisticEstimator().fit(seq, GRID, SIDE)
    assert mae(spatial, truth, grid=GRID) < mae(pooled, truth, grid=GRID)


# --------------------------------------------------------------------------- contaminated data


def test_corrected_estimator_removes_the_bias_the_others_keep():
    truth = _truth()
    ref = _generate(0.75, 1.2, 1.0, n=60, seed=100, coach_every=0, truth=truth)   # clean anchor
    blk = _generate(0.75, 1.2, 1.0, n=300, seed=2, coach_every=5, truth=truth)
    seq = ref + blk
    fits = fit_all(seq, GRID, SIDE)

    bias_spatial = _band_bias(fits[METHOD_SPATIAL], truth)
    bias_pooled = _band_bias(fits[METHOD_POOLED], truth)
    bias_corrected = _band_bias(fits[METHOD_CORRECTED], truth)

    # The predicted direction: COACH pushes toward the nonpreferred arm, so an uncorrected
    # estimator over-states pi*.
    assert bias_spatial > 0.05, bias_spatial
    assert bias_pooled > 0.05, bias_pooled
    assert abs(bias_corrected) < 0.5 * abs(bias_spatial), (bias_corrected, bias_spatial)
    assert mae(fits[METHOD_CORRECTED], truth, grid=GRID) < mae(fits[METHOD_SPATIAL], truth, grid=GRID)


def test_corrected_estimator_covers_where_the_naive_one_does_not():
    truth = _truth()
    seq = _generate(0.75, 1.2, 1.0, n=60, seed=101, coach_every=0, truth=truth) + \
          _generate(0.75, 1.2, 1.0, n=300, seed=3, coach_every=5, truth=truth)
    fits = fit_all(seq, GRID, SIDE)
    cov_corrected = interval_coverage(fits[METHOD_CORRECTED], truth, level=0.9, grid=GRID)["coverage"]
    cov_spatial = interval_coverage(fits[METHOD_SPATIAL], truth, level=0.9, grid=GRID)["coverage"]
    assert cov_corrected > cov_spatial, (cov_corrected, cov_spatial)


def test_joint_posterior_beats_the_online_plugin_at_recovering_the_effect():
    """The reason :func:`joint_carryover_posterior` exists (see its docstring)."""

    from vla_lab.rehab.carryover import CarryoverPosterior

    truth = _truth()
    seq = _generate(0.75, 1.2, 1.0, n=60, seed=102, coach_every=0, truth=truth) + \
          _generate(0.75, 1.2, 1.0, n=300, seed=2, coach_every=5, truth=truth)

    joint = joint_carryover_posterior(seq, GRID)
    plugin_pi = SpatialLogisticEstimator().fit(seq, GRID, SIDE).mean()
    plugin = CarryoverPosterior(CarryoverConfig())
    for o in seq:
        plugin.step(
            action=o.action, delta=o.delta,
            pi_star=(plugin_pi.get(o.target_id) if o.observed else None),
            chose_nonpreferred=(o.y if o.observed else None),
        )
    assert joint.mean()["beta_g"] > plugin.mean()["beta_g"] + 0.2, (
        joint.mean()["beta_g"], plugin.mean()["beta_g"]
    )


def test_interval_coverage_is_near_nominal_on_clean_data():
    hits, total = 0, 0
    for s in range(8):
        truth = _truth()
        seq = _generate(0.7, 0.0, 0.0, n=300, seed=200 + s, coach_every=0, truth=truth)
        est = SpatialLogisticEstimator().fit(seq, GRID, SIDE)
        c = interval_coverage(est, truth, level=0.9, grid=GRID)
        hits += c["coverage"] * c["n_targets"]
        total += c["n_targets"]
    assert hits / total >= 0.7, hits / total


# --------------------------------------------------------------------------- metrics


def test_metrics_match_hand_computed_values():
    from vla_lab.rehab.estimand import PiStarPosterior, _prob_grid

    grid_vals = _prob_grid()

    def spike(p: float):
        d = np.zeros_like(grid_vals)
        d[int(np.argmin(np.abs(grid_vals - p)))] = 1.0
        return d

    ids = GRID.ids()[:2]
    est = PiStarPosterior(
        target_ids=list(ids), grid=grid_vals,
        density={ids[0]: spike(0.60), ids[1]: spike(0.20)}, method="fixture",
    )
    ref = {ids[0]: 0.50, ids[1]: 0.30}
    w = GRID.crossover_weights()
    m = est.mean()
    expect_mae = (
        w[ids[0]] * abs(m[ids[0]] - 0.50) + w[ids[1]] * abs(m[ids[1]] - 0.30)
    ) / (w[ids[0]] + w[ids[1]])
    assert abs(mae(est, ref, grid=GRID) - expect_mae) < 1e-6
    expect_brier = (
        w[ids[0]] * (m[ids[0]] - 0.50) ** 2 + w[ids[1]] * (m[ids[1]] - 0.30) ** 2
    ) / (w[ids[0]] + w[ids[1]])
    assert abs(brier_vs_reference(est, ref, grid=GRID) - expect_brier) < 1e-6


def test_brier_vs_outcomes_is_the_classical_score():
    assert abs(brier_vs_outcomes([1.0, 0.0], [True, False])) < 1e-12
    assert abs(brier_vs_outcomes([0.5, 0.5], [True, False]) - 0.25) < 1e-12


def test_unweighted_and_crossover_weighted_mae_differ():
    truth = _truth()
    seq = _generate(0.7, 0.0, 0.0, n=200, seed=6, coach_every=0, truth=truth)
    est = SpatialLogisticEstimator().fit(seq, GRID, SIDE)
    a = mae(est, truth, grid=GRID, crossover_weighted=True)
    b = mae(est, truth, grid=GRID, crossover_weighted=False)
    assert a == a and b == b and abs(a - b) > 0  # both finite, and weighting changes the answer


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_estimand") else 0)

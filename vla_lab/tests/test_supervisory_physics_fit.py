"""The value-model fit, its uncertainty, and the invariants the 2026-08-23 brief added.

Each test here pins something that has already gone wrong once or that the paper now claims:
the per-metre ridge prior that flattened the first fit (defect xii), the two sign changes in the
measured value gap, the bootstrap interval the study runner can rebuild its grid under, the
counterfactual physics the flip diagnostic sweeps, and the refusal to emit a correlation
coefficient on too few units.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np

from vla_lab.supervisory import STRATEGY_A, STRATEGY_B
from vla_lab.supervisory.physics_fit import (
    FIT_LAPSE,
    IsotonicCurve,
    bootstrap_physics,
    fit_lapse_logistic,
    fit_physics_lapse,
    fit_physics_legacy,
    isotonic_physics_summary,
    pava_increasing,
)
from vla_lab.supervisory.scenes import ScenePhysics, build_scene_grid


def _rollouts(truth: ScenePhysics, *, reps: int = 24, gaps=None, seed: int = 0):
    rng = random.Random(seed)
    gaps = gaps or [i / 100.0 for i in range(0, 13)]
    rows = []
    for m in gaps:
        for s in (STRATEGY_A, STRATEGY_B):
            for _ in range(reps):
                rows.append({"strategy": s, "margin_m": m, "success": rng.random() < truth.p_success(s, m),
                             "duration_s": truth.duration_s(s, m) + rng.gauss(0, 0.3)})
    return rows


#: A steep scene like the measured one: transitions a centimetre wide, a real floor on A.
STEEP = ScenePhysics(p_a_asym=1.0, a_slope=250.0, a_mid=0.012, p_a_floor=0.15,
                     p_b_asym=1.0, b_slope=120.0, b_mid=0.04, p_b_floor=0.1,
                     t_a=13.4, t_b=9.0, t_b_tight=0.0)


def test_the_lapse_fit_recovers_a_steep_curve_the_legacy_fit_flattens():
    """Defect (xii): a precision-0.01 ridge on a per-metre slope cannot reach 120/m."""
    rows = _rollouts(STEEP, reps=40, seed=1)
    new, fits = fit_physics_lapse(rows)
    old = fit_physics_legacy(rows)
    assert new.fit_method == FIT_LAPSE
    assert abs(new.b_slope - STEEP.b_slope) < 0.35 * STEEP.b_slope, (new.b_slope, STEEP.b_slope)
    assert old.b_slope < 0.5 * STEEP.b_slope, "the legacy fit should visibly under-estimate the slope"
    assert abs(new.transition_width_m() - STEEP.transition_width_m()) < 0.004
    assert abs(new.p_a_floor - STEEP.p_a_floor) <= 0.1, new.p_a_floor


def test_the_fit_is_unit_aware():
    """Same data in metres must give the same curve as the per-centimetre parameterisation implies."""
    rows = _rollouts(STEEP, reps=30, seed=2)
    m = np.array([r["margin_m"] for r in rows if r["strategy"] == STRATEGY_B])
    y = np.array([1.0 if r["success"] else 0.0 for r in rows if r["strategy"] == STRATEGY_B])
    fit = fit_lapse_logistic(m, y)
    assert abs(fit["slope_per_m"] - 100.0 * fit["slope_per_cm"]) < 1e-9
    assert fit["slope_per_m"] > 60.0, fit


def test_the_crossover_is_the_upper_one_where_direct_overtakes():
    """The measured value gap changes sign twice; the coordinate is defined on the second."""
    p = STEEP
    assert not p.is_degenerate()
    m_star = p.crossover_margin()
    assert 0.03 < m_star < 0.09, m_star
    assert p.value_gap(m_star - 0.005) > 0 > p.value_gap(m_star + 0.005)
    lower = p.lower_crossover_margin()
    assert lower is not None and lower < 0.015, lower
    # The prior physics has one crossing and no lower one; its behaviour is unchanged.
    prior = ScenePhysics()
    assert prior.lower_crossover_margin() is None
    assert abs(prior.crossover_margin() - 0.0690) < 1e-3


def test_the_grid_never_reaches_the_trivial_lower_crossing():
    g = build_scene_grid(physics=STEEP)
    lower = STEEP.lower_crossover_margin() or 0.0
    assert all(s.margin_m > lower + 0.01 for s in g.scenes), [s.margin_m for s in g.scenes]


def test_pava_is_monotone_and_weighted():
    fit = pava_increasing([0.2, 0.1, 0.5, 0.4, 0.9], [10, 10, 10, 30, 10])
    assert all(fit[i] <= fit[i + 1] + 1e-12 for i in range(len(fit) - 1))
    assert abs(fit[0] - 0.15) < 1e-9 and abs(fit[1] - 0.15) < 1e-9
    assert abs(fit[2] - 0.425) < 1e-9 and abs(fit[3] - 0.425) < 1e-9


def test_isotonic_width_equals_one_over_slope_for_an_exact_logistic():
    phys = ScenePhysics(p_b_asym=1.0, p_b_floor=0.0, b_slope=80.0, b_mid=0.05)
    ms = [i / 1000.0 for i in range(0, 130, 2)]
    curve = IsotonicCurve(ms, [phys.p_success(STRATEGY_B, m) for m in ms])
    assert abs(curve.transition_width_m() - 1.0 / 80.0) < 0.0015, curve.transition_width_m()


def test_isotonic_summary_agrees_with_the_parametric_fit_on_clean_data():
    rows = _rollouts(STEEP, reps=60, seed=3)
    phys, _ = fit_physics_lapse(rows)
    iso = isotonic_physics_summary(rows, phys)
    assert abs(iso["crossover_margin_m"] - phys.crossover_margin()) < 0.012, (iso["crossover_margin_m"], phys.crossover_margin())
    assert abs(iso["transition_width_m"] - phys.transition_width_m()) < 0.006


def test_the_bootstrap_brackets_the_truth_and_returns_usable_physics():
    rows = _rollouts(STEEP, reps=24, seed=4)
    b = bootstrap_physics(rows, n_boot=60, seed=0)
    ci = b["crossover_margin_m"]
    assert ci["p2.5"] <= STEEP.crossover_margin() <= ci["p97.5"], (ci, STEEP.crossover_margin())
    wci = b["transition_width_m"]
    assert wci["p2.5"] < wci["p97.5"]
    for tag in ("lower", "point", "upper"):
        q = ScenePhysics.from_dict(b["quantile_physics"][tag])
        assert q.quantile == tag
        assert not q.is_degenerate()
        assert len(build_scene_grid(physics=q)) == len(build_scene_grid(physics=STEEP))
    lo = ScenePhysics.from_dict(b["quantile_physics"]["lower"]).transition_width_m()
    hi = ScenePhysics.from_dict(b["quantile_physics"]["upper"]).transition_width_m()
    assert lo < hi


def test_counterfactual_physics_moves_one_thing_at_a_time():
    p = STEEP
    q = p.with_transition_width(0.02)
    assert abs(q.transition_width_m() - 0.02) < 1e-9
    assert abs(q.crossover_margin() - p.crossover_margin()) < 5e-4, "the crossover must stay put"
    assert q.quantile.startswith("assumed_w")
    r = p.with_crossover(0.07)
    assert abs(r.crossover_margin() - 0.07) < 5e-4
    assert abs(r.transition_width_m() - p.transition_width_m()) < 1e-12


def test_physics_json_round_trip_keeps_floors_and_provenance():
    with tempfile.TemporaryDirectory() as td:
        from vla_lab.supervisory.scenes import load_physics, save_physics

        path = Path(td) / "physics.json"
        save_physics(STEEP, path)
        back = load_physics(path)
        assert back.p_a_floor == STEEP.p_a_floor and back.p_b_floor == STEEP.p_b_floor
        assert back.quantile == "point"
    # Old files without the new fields load with zero floors.
    old = ScenePhysics.from_dict({"p_b_asym": 1.0, "b_slope": 32.0, "b_mid": 0.03, "source": "measured"})
    assert old.p_a_floor == 0.0 and old.fit_method == "prior"


def test_the_fit_report_carries_old_and_new_residuals_and_the_bootstrap():
    from vla_lab.supervisory.apparatus.measure import fit_from_rollouts

    rows = _rollouts(STEEP, reps=16, seed=5)
    phys, report = fit_from_rollouts(rows, n_boot=20)
    for key in ("fit_residuals", "legacy_fit", "isotonic", "bootstrap", "lower_crossover_margin_m"):
        assert key in report, key
    assert report["fit_residuals"]["worst_abs_dp_in_band"] < 0.15
    assert "worst_abs_dp_in_band" in report["legacy_fit"]["fit_residuals"]
    assert report["legacy_fit"]["fit_method"] == "scaled_logistic_legacy"
    json.dumps(report, default=float)                         # must be serialisable


def test_correlations_below_the_minimum_n_are_withheld():
    from vla_lab.stats_utils import MIN_N_FOR_CORRELATION, guarded_correlation

    r = guarded_correlation(range(7), [3, 1, 4, 1, 5, 9, 2])
    assert r["rho"] is None and not r["reported"] and "withheld" in r["reason"]
    r = guarded_correlation(range(MIN_N_FOR_CORRELATION), range(MIN_N_FOR_CORRELATION))
    assert r["reported"] and abs(r["rho"] - 1.0) < 1e-12
    r = guarded_correlation(range(7), [3, 1, 4, 1, 5, 9, 2], override=True)
    assert r["reported"] and r["rho"] is not None and "OVERRIDE" in r["reason"]


def test_the_deployed_facts_never_contain_a_coefficient_on_seven_cells():
    from vla_lab.supervisory.deployed_figure import deployed_facts

    def summary(maes):
        d = {"lexical": {"conditions": {"carryover_aware": {"mae_crossover": {"mean": 0.11}}}, "grounder": {}}}
        for ctx, mae, ab in maes:
            d[f"policy_unprompted@x__{ctx}"] = {"conditions": {"carryover_aware": {"mae_crossover": {"mean": mae}}},
                                                "grounder": {"abstain_rate_band": ab}}
        return d

    models = [{"context": c, "debias_gain_brier": g, "debias_kappa_corr": 0.1} for c, g in
              (("none", 0.09), ("token", 0.11), ("film", 0.135))]
    runs = [("TinyVLA-2M", summary([("none", 0.118, 0.01), ("token", 0.113, 0.03), ("film", 0.150, 0.116)]), models),
            ("SmolVLA-450M", summary([("none", 0.110, 0.027), ("token", 0.104, 0.044), ("film", 0.104, 0.05),
                                      ("text", 0.112, 0.013)]), models + [{"context": "text", "debias_gain_brier": 0.08,
                                                                            "debias_kappa_corr": 0.2}])]
    facts = deployed_facts(runs)
    assert facts["n_cells"] == 7
    assert facts["offline_gain_vs_deployed"]["rho"] is None
    assert facts["per_backbone"]["TinyVLA-2M"]["best_offline_deployed_rank"] == 3
    assert facts["worst_is_most_abstaining"]
    assert facts["cells_above_11pct_band_abstention"] == ["TinyVLA-2M/film"]


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(globals(), label="test_supervisory_physics_fit") else 0)

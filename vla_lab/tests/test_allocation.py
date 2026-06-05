"""Tests for ``vla_lab.allocation`` — the act/compute/query allocator and its parts."""

from __future__ import annotations

import math

import numpy as np

from vla_lab.allocation import build_controller
from vla_lab.allocation.allocator import ActComputeQueryAllocator, AllocatorFit
from vla_lab.allocation.conformal import (
    MondrianConformal,
    SplitConformal,
    bucketize,
    conformal_quantile,
    coverage_vs_bucket,
    empirical_coverage,
)
from vla_lab.allocation.linear_model import fit_logistic
from vla_lab.allocation.query import QueryAnswer, ScriptedQueryInterface, refine_instruction
from vla_lab.allocation.transparency import TransparencyConfig, render_message, uncertainty_type_for
from vla_lab.allocation.value_of_information import (
    ACT,
    COMPUTE,
    QUERY,
    CostModel,
    ValueParams,
    allocate,
    value_of_computation,
    value_of_information,
)
from vla_lab.tests.mocks import MockComputeBranch, make_occlusion_fit


# --------------------------------------------------------------------------- VoC / VoI


def test_value_of_computation_limit_result():
    """The §4.1 limit: compute value → 0 as the action becomes unidentifiable (irr→1)."""
    vp, cost = ValueParams(), CostModel()
    v_reducible = value_of_computation(0.3, irreducible=0.0, k=4, vp=vp, cost=cost)
    v_irreducible = value_of_computation(0.3, irreducible=1.0, k=4, vp=vp, cost=cost)
    assert v_reducible > 0.05, v_reducible
    assert v_irreducible <= 0.0, v_irreducible  # only the (negative) compute penalty remains


def test_value_of_information_threshold():
    vp = ValueParams(voi_gain=0.8, voi_floor=0.1)
    cost = CostModel(query_cost_s=2.0, value_scale=0.02)  # penalty = 0.04
    assert value_of_information(0.0, vp, cost) < 0.0      # never worth it when identifiable
    assert value_of_information(0.9, vp, cost) > 0.0      # worth it when irreducible


def test_allocate_act_compute_query():
    vp = ValueParams(act_dispersion_tau=0.02, voi_gain=0.8, voi_floor=0.1)
    cost = CostModel(query_cost_s=2.0, value_scale=0.02)
    assert allocate(0.001, 0.0, 4, vp, cost)[0] == ACT      # confident & identifiable
    assert allocate(0.30, 0.05, 4, vp, cost)[0] == COMPUTE  # reducible ambiguity
    assert allocate(0.30, 0.95, 4, vp, cost)[0] == QUERY    # irreducible


# --------------------------------------------------------------------------- allocator


def test_allocator_decision_priority():
    """Clean→act, mild occlusion→compute, heavy occlusion→query, with a fitted detector."""
    cb = MockComputeBranch()
    q = ScriptedQueryInterface(oracle_answer="red", accuracy=1.0, sim_latency_s=1.0)
    alloc = ActComputeQueryAllocator(cb, q, fit=make_occlusion_fit(), transparency=TransparencyConfig(signal_type=True))

    d_clean = alloc.decide({"occlusion_fraction": 0.0}, instruction="pick the red block", options=["red", "blue"], oracle_answer="red")
    d_mild = alloc.decide({"occlusion_fraction": 0.2}, instruction="pick the red block", options=["red", "blue"], oracle_answer="red")
    d_heavy = alloc.decide({"occlusion_fraction": 0.5}, instruction="pick the red block", options=["red", "blue"], oracle_answer="red")

    assert d_clean.action == ACT, d_clean.action
    assert d_mild.action == COMPUTE, d_mild.action
    assert d_heavy.action == QUERY, d_heavy.action
    # Query executed and the oracle answer was correct + folded back in.
    assert d_heavy.queried and d_heavy.query_answer_correct is True
    assert d_heavy.chunk is not None
    # Type-aware transparency distinguishes the two non-act steps.
    assert d_mild.uncertainty_type == "reducible"
    assert d_heavy.uncertainty_type == "irreducible"
    assert d_mild.transparency_message != d_heavy.transparency_message


def test_allocator_default_never_queries_without_oracle_occlusion():
    """Uncalibrated default fit + no known occlusion → behaves like compute-gated TTC."""
    cb = MockComputeBranch()
    q = ScriptedQueryInterface(oracle_answer="red")
    alloc = ActComputeQueryAllocator(cb, q, fit=AllocatorFit.default())
    actions = set()
    for disp in [0.0, 0.05, 0.2, 0.5, 1.0]:
        d = alloc.decide({"mock_dispersion": disp}, instruction="pick the red block", options=["red", "blue"])
        actions.add(d.action)
    assert QUERY not in actions, actions
    assert actions <= {ACT, COMPUTE}


def test_allocator_decision_is_json_serializable():
    cb = MockComputeBranch()
    alloc = ActComputeQueryAllocator(cb, ScriptedQueryInterface(oracle_answer="red"), fit=make_occlusion_fit())
    d = alloc.decide({"occlusion_fraction": 0.5}, instruction="pick", options=["red", "blue"], oracle_answer="red")
    import json

    json.dumps(d.to_record())  # must not raise; chunk is dropped


# --------------------------------------------------------------------------- baselines


def test_build_all_controllers_run():
    cb = MockComputeBranch()
    q = ScriptedQueryInterface(oracle_answer="red")
    fit = make_occlusion_fit()
    for kind in ["autonomy", "fixed_compute", "compute_gated", "scale", "knowno", "insight", "allocator"]:
        ctrl = build_controller(kind, compute_branch=cb, query_interface=q, fit=fit)
        d = ctrl.decide({"occlusion_fraction": 0.5}, instruction="pick the red block", options=["red", "blue"], oracle_answer="red")
        assert d.action in (ACT, COMPUTE, QUERY), (kind, d.action)
        assert d.chunk is not None, kind


def test_autonomy_never_computes_or_queries():
    cb = MockComputeBranch()
    ctrl = build_controller("autonomy", compute_branch=cb)
    for occ in [0.0, 0.5]:
        d = ctrl.decide({"occlusion_fraction": occ}, instruction="pick")
        assert d.action == ACT
    assert cb.calls["compute"] == 0 and cb.calls["probe"] == 0


def test_knowno_is_binary_act_or_query():
    """KnowNo asks when its conformal flags the step; it never uses the compute branch."""
    cb = MockComputeBranch()
    q = ScriptedQueryInterface(oracle_answer="red")
    fit = AllocatorFit.default()
    # Calibrate a dispersion detector on low-dispersion "clean" data so high dispersion flags.
    fit.knowno_conformal = SplitConformal(alpha=0.1).calibrate([0.001, 0.002, 0.003, 0.004, 0.005] * 5)
    ctrl = build_controller("knowno", compute_branch=cb, query_interface=q, fit=fit)
    d_low = ctrl.decide({"mock_dispersion": 0.0005}, instruction="pick", options=["red", "blue"], oracle_answer="red")
    d_high = ctrl.decide({"mock_dispersion": 0.9}, instruction="pick", options=["red", "blue"], oracle_answer="red")
    assert d_low.action == ACT
    assert d_high.action == QUERY
    assert cb.calls["compute"] == 0  # binary: no compute branch ever


# --------------------------------------------------------------------------- conformal


def test_split_conformal_coverage():
    rng = np.random.default_rng(0)
    calib = rng.normal(size=2000).tolist()
    sc = SplitConformal(alpha=0.1).calibrate(calib)
    test = rng.normal(size=5000).tolist()
    cov = empirical_coverage(test, sc.q_hat)
    assert abs(cov - 0.9) < 0.03, cov
    assert sc.is_anomalous(10.0) and not sc.is_anomalous(-10.0)


def test_conformal_quantile_too_few_points_is_inf():
    # ceil((n+1)(1-alpha))/n > 1 when n is small relative to coverage -> never fires.
    assert math.isinf(conformal_quantile([0.1, 0.2], alpha=0.1))
    assert math.isinf(conformal_quantile([], alpha=0.1))


def test_mondrian_per_bucket_and_fallback():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    buckets = ["lo", "lo", "lo", "lo", "hi", "hi", "hi", "hi"]
    mc = MondrianConformal(alpha=0.2).calibrate(scores, buckets)
    assert "lo" in mc.q_by_bucket and "hi" in mc.q_by_bucket
    assert mc.quantile_for("hi") >= mc.quantile_for("lo")
    # Unknown bucket falls back to the pooled quantile.
    assert mc.quantile_for("unseen") == mc.q_pooled


def test_coverage_degrades_under_shift():
    """One pooled q_hat fitted on clean scores: realized coverage falls as scores shift up (§4.3)."""
    rng = np.random.default_rng(1)
    clean = rng.normal(0.0, 1.0, size=400)
    shifted = rng.normal(1.5, 1.0, size=400)
    scores = np.concatenate([clean, shifted]).tolist()
    buckets = ["clean"] * 400 + ["occluded"] * 400
    q_hat = conformal_quantile(clean.tolist(), alpha=0.1)
    cov = coverage_vs_bucket(scores, buckets, q_hat)
    assert cov["clean"] > cov["occluded"] + 0.2, cov


def test_bucketize_labels():
    assert bucketize(0.05, [0.1, 0.2, 0.3]) == "[0.00,0.10)"
    assert bucketize(0.25, [0.1, 0.2, 0.3]) == "[0.20,0.30)"
    assert bucketize(0.9, [0.1, 0.2, 0.3]) == "[0.30,inf)"


# --------------------------------------------------------------------------- logistic model


def test_logistic_fit_separable():
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(-2, 0.5, size=(100, 1)), rng.normal(2, 0.5, size=(100, 1))])
    y = np.array([0.0] * 100 + [1.0] * 100)
    model = fit_logistic(X, y, ["x"], epochs=2000)
    assert model.fitted
    assert model.predict_proba_features({"x": 2.0}) > 0.8
    assert model.predict_proba_features({"x": -2.0}) < 0.2


def test_logistic_single_class_is_unfitted_with_baserate_bias():
    X = np.zeros((10, 1))
    y = np.ones(10)  # single class
    model = fit_logistic(X, y, ["x"])
    assert model.fitted is False
    # bias reflects the (clipped) base rate -> high P even though "unfitted".
    assert model.bias > 0.0


# --------------------------------------------------------------------------- query / transparency


def test_scripted_query_accuracy():
    correct = ScriptedQueryInterface(oracle_answer="red", accuracy=1.0)
    ans = correct.ask("which?", ["red", "blue"])
    assert ans.text == "red" and ans.correct is True
    wrong = ScriptedQueryInterface(oracle_answer="red", accuracy=0.0)
    ans2 = wrong.ask("which?", ["red", "blue"])
    assert ans2.text == "blue" and ans2.correct is False


def test_refine_instruction_folds_answer():
    out = refine_instruction("pick up the block", QueryAnswer(text="red"))
    assert "red" in out.lower()
    # Idempotent when the answer is already present.
    assert refine_instruction("pick the red block", QueryAnswer(text="red")) == "pick the red block"


def test_transparency_conditions():
    assert uncertainty_type_for("compute") == "reducible"
    assert uncertainty_type_for("query") == "irreducible"
    off = TransparencyConfig(enabled=False)
    assert render_message("query", off) == ""
    magnitude = TransparencyConfig(enabled=True, signal_type=False)
    # cond4: same generic message for compute and query.
    assert render_message("compute", magnitude) == render_message("query", magnitude)
    typed = TransparencyConfig(enabled=True, signal_type=True)
    # cond5: distinct messages.
    assert render_message("compute", typed) != render_message("query", typed)


# --------------------------------------------------------------------------- fit serialization


def test_allocator_fit_roundtrip():
    fit = make_occlusion_fit()
    fit.conformal = MondrianConformal(alpha=0.1).calibrate([0.1, 0.2, 0.3, 0.4] * 5, ["a", "b"] * 10)
    fit.conformal_kind = "mondrian"
    fit.knowno_conformal = SplitConformal(alpha=0.2).calibrate([0.01, 0.02, 0.03] * 10)
    restored = AllocatorFit.from_dict(fit.to_dict())
    assert restored.conformal_kind == "mondrian"
    assert isinstance(restored.conformal, MondrianConformal)
    assert restored.knowno_conformal is not None
    assert restored.irreducibility.fitted == fit.irreducibility.fitted
    assert abs(restored.value_params.voi_gain - fit.value_params.voi_gain) < 1e-9


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_allocation") else 0)

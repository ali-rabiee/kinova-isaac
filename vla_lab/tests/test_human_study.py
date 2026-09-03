"""Tests for ``vla_lab.human_study`` — protocol, instruments, reliance, power, sim pipeline."""

from __future__ import annotations


from vla_lab.human_study.instruments import (
    NASA_TLX_SUBSCALES,
    jian_trust_score,
    mdmt_score,
    nasa_tlx_raw,
    nasa_tlx_weighted,
)
from vla_lab.human_study.power import (
    normal_cdf,
    normal_ppf,
    sample_size_paired,
    sample_size_two_proportion,
)
from vla_lab.human_study.protocol import (
    StudyProtocol,
    balanced_latin_square,
    generate_session_plan,
)
from vla_lab.human_study.reliance import (
    OVER_TRUST,
    UNDER_TRUST,
    CORRECT_RELIANCE,
    CORRECT_INTERVENTION,
    RelianceTrial,
    classify_reliance,
    reliance_summary,
    trust_calibration,
)
from vla_lab.human_study.session import SessionRunner, SimEpisodeRunner
from vla_lab.human_study.reliance import group_by


# --------------------------------------------------------------------------- protocol


def test_balanced_latin_square_is_balanced():
    for n in (4, 5):
        sq = balanced_latin_square(n)
        for row in sq:
            assert sorted(row) == list(range(n))  # each row is a permutation
        # First-order (column) balance over the first n rows.
        for col in range(n):
            assert sorted(sq[i][col] for i in range(n)) == list(range(n))


def test_within_session_plan_size_and_blocks():
    proto = StudyProtocol(design="within", reps_per_cell=1)
    plan = generate_session_plan(0, proto)
    assert len(plan.trials) == len(proto.conditions) * len(proto.observability)
    assert len({t.block_idx for t in plan.trials}) == len(proto.conditions)


def test_between_session_plan_is_single_condition():
    proto = StudyProtocol(design="between")
    plan = generate_session_plan(2, proto)
    assert len({t.condition_name for t in plan.trials}) == 1
    assert len(plan.trials) == len(proto.observability) * proto.reps_per_cell


def test_session_plan_is_reproducible():
    proto = StudyProtocol(design="within")
    a = generate_session_plan(7, proto)
    b = generate_session_plan(7, proto)
    assert [t.to_dict() for t in a.trials] == [t.to_dict() for t in b.trials]


# --------------------------------------------------------------------------- instruments


def test_nasa_tlx_raw_and_weighted():
    scores = {k: 60.0 for k in NASA_TLX_SUBSCALES}
    assert abs(nasa_tlx_raw(scores) - 60.0) < 1e-9
    # Weight only mental_demand=80 vs others=20 -> weighted mean pulled toward 80.
    scores2 = dict.fromkeys(NASA_TLX_SUBSCALES, 20.0)
    scores2["mental_demand"] = 80.0
    weights = {k: 0 for k in NASA_TLX_SUBSCALES}
    weights["mental_demand"] = 5
    assert abs(nasa_tlx_weighted(scores2, weights) - 80.0) < 1e-9


def test_jian_trust_reverse_coding():
    # distrust items low (1), trust items high (7) -> high overall trust.
    high = jian_trust_score([1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 4])
    # distrust items high (7), trust items low (1) -> low overall trust.
    low = jian_trust_score([7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 4])
    assert high["overall"] > 6.0 and low["overall"] < 2.0
    assert high["trust"] > high["distrust"]


def test_mdmt_subscales():
    resp = {"reliable": [6, 7], "capable": [6, 6], "ethical": [5, 5], "sincere": [4, 6]}
    out = mdmt_score(resp)
    assert "capacity_trust" in out and "moral_trust" in out and "overall" in out
    assert out["capacity_trust"] > out["moral_trust"]  # capability rated higher here


# --------------------------------------------------------------------------- reliance


def test_reliance_classification_quadrants():
    assert classify_reliance(RelianceTrial(robot_correct=True, human_intervened=False)) == CORRECT_RELIANCE
    assert classify_reliance(RelianceTrial(robot_correct=False, human_intervened=True)) == CORRECT_INTERVENTION
    assert classify_reliance(RelianceTrial(robot_correct=False, human_intervened=False)) == OVER_TRUST
    assert classify_reliance(RelianceTrial(robot_correct=True, human_intervened=True)) == UNDER_TRUST


def test_reliance_summary_rates():
    trials = [
        RelianceTrial(robot_correct=True, human_intervened=False),    # appropriate
        RelianceTrial(robot_correct=False, human_intervened=True),    # appropriate (caught)
        RelianceTrial(robot_correct=False, human_intervened=False),   # over-trust
        RelianceTrial(robot_correct=True, human_intervened=True),     # under-trust
    ]
    s = reliance_summary(trials)
    assert s["n"] == 4
    assert abs(s["appropriate_reliance_rate"] - 0.5) < 1e-9
    assert abs(s["over_trust_rate"] - 0.25) < 1e-9
    assert abs(s["under_trust_rate"] - 0.25) < 1e-9


def test_trust_calibration_discrimination():
    # Human intervenes iff the robot is wrong -> perfect discrimination.
    trials = [RelianceTrial(robot_correct=False, human_intervened=True) for _ in range(5)]
    trials += [RelianceTrial(robot_correct=True, human_intervened=False) for _ in range(5)]
    cal = trust_calibration(trials)
    assert abs(cal["discrimination"] - 1.0) < 1e-9
    assert abs(cal["phi_intervene_wrong"] - 1.0) < 1e-9


# --------------------------------------------------------------------------- power


def test_normal_ppf_cdf_roundtrip():
    assert abs(normal_ppf(0.975) - 1.959963985) < 1e-3
    assert abs(normal_cdf(1.959963985) - 0.975) < 1e-3
    for p in (0.05, 0.3, 0.8, 0.99):
        assert abs(normal_cdf(normal_ppf(p)) - p) < 1e-6


def test_sample_size_monotone_in_effect():
    big_effect = sample_size_two_proportion(0.5, 0.8)
    small_effect = sample_size_two_proportion(0.5, 0.55)
    assert small_effect > big_effect
    # A medium paired effect lands in the commonly-reported HRI range.
    n = sample_size_paired(0.5)
    assert 25 <= n <= 40, n


# --------------------------------------------------------------------------- offline sim pipeline


def test_sim_pipeline_reproduces_overtrust_ordering():
    """The memo's headline (§5.4): the type-aware allocator reduces over-trust vs. baselines."""
    proto = StudyProtocol(design="within", reps_per_cell=2)
    trials = []
    for pi in range(10):
        plan = generate_session_plan(pi, proto)
        res = SessionRunner(plan, SimEpisodeRunner(seed=pi)).run()
        trials += res.reliance_trials

    by_cond = {k: reliance_summary(v) for k, v in group_by(trials, lambda t: t.condition_name).items()}
    ot = {k: v["over_trust_rate"] for k, v in by_cond.items()}
    appr = {k: v["appropriate_reliance_rate"] for k, v in by_cond.items()}

    # Allocator conditions beat the autonomous / fixed-compute / gated baselines on over-trust.
    assert ot["allocator_type"] < ot["autonomy"]
    assert ot["allocator_type"] < ot["fixed_compute"]
    assert ot["allocator_type"] < ot["compute_gated"]
    # ...and on appropriate reliance.
    assert appr["allocator_type"] > appr["autonomy"]
    # Every condition produced a full set of trials.
    assert all(v["n"] > 0 for v in by_cond.values())


def test_sim_episode_runner_query_is_collaboration_not_intervention():
    """A trial where the robot queried should not also be counted as a human takeover."""
    runner = SimEpisodeRunner(seed=0)
    saw_query = False
    for _ in range(500):
        from vla_lab.human_study.protocol import Trial

        t = Trial(0, 0, 4, "allocator", "allocator", True, False, "bottom_strip", 0.5)
        r = runner(t)
        if r.robot_queried:
            saw_query = True
            assert not r.human_intervened  # answering a query != taking control
    assert saw_query  # the allocator does query under heavy occlusion


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_human_study") else 0)


# ---------------------------------------------------------------------------
# 2026-08-23: the phrase-corpus tooling (P3-1). Fixture utterances only -- no corpus exists.
# ---------------------------------------------------------------------------
_FIXTURE = [
    ("p1", "clear the blue one out of the way first", "A"),
    ("p1", "just grab it directly", "B"),
    ("p2", "hmm, either way I guess", None),
    ("p2", "if the gap is big enough go straight for it, otherwise clear it", None),
    ("p3", "lift it over the top of the blue one", None),
    ("p3", "push the blocker aside then take the red one", "A"),
    ("p4", "reach around and take it", "B"),
    ("p4", "do something sensible", None),
]


def _write_fixture(path):
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["participant_id", "packet_index", "scene_id", "response_text", "modality"])
        for i, (pid, text, _lab) in enumerate(_FIXTURE):
            w.writerow([pid, 1 + i % 3, 7, text, "typed"])


def test_phrase_corpus_evaluate_enumerates_every_failure_mode():
    import tempfile
    from pathlib import Path

    from vla_lab.human_study.phrase_corpus import evaluate

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "responses.csv"
        _write_fixture(p)
        rep = evaluate(p)
    assert rep["n_responses"] == 8 and rep["n_participants"] == 4
    assert abs(rep["grounding_rate"] - 4 / 8) < 1e-9
    assert not rep["meets_threshold"] and rep["verdict"].startswith("no-go")
    fm = rep["failure_modes"]
    assert fm.get("hedge") == 1 and fm.get("conditional") == 1
    assert fm.get("unimplemented_strategy") == 1 and fm.get("names_neither") == 1
    assert fm.get("multi_clause_both_sides") == 1            # the conditional names both sides


def test_phrase_corpus_rebuild_installs_an_empirical_axis_that_the_contract_hashes():
    import tempfile
    from pathlib import Path

    from vla_lab.human_study.phrase_corpus import install_empirical_axis, rebuild
    from vla_lab.supervisory import strategies, supervisor
    from vla_lab.supervisory.contract import Contract
    from vla_lab.supervisory.narration import ground

    before_axis, before_hedges = strategies.AXES["plan"], supervisor._HEDGES
    before_hash = Contract().narration_hash()
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "responses.csv"
            _write_fixture(p)
            rep = rebuild(p)
            f = Path(td) / "axis.json"
            import json

            f.write_text(json.dumps(rep))
            info = install_empirical_axis(f)
        assert info["n_phrases_a"] == 2 and info["n_phrases_b"] == 2 and info["n_hedges"] == 4
        assert abs(info["ungrounded_rate"] - 0.5) < 1e-9
        axis = strategies.get_axis("plan")
        assert all(ground(ph, axis) == "A" for ph in axis.phrases_a), "every sampled phrase must still ground"
        assert all(ground(ph, axis) == "B" for ph in axis.phrases_b)
        assert Contract().narration_hash() != before_hash, "the rebuilt supervisor must not pool with the scripted one"
    finally:
        strategies.AXES["plan"] = before_axis
        supervisor._HEDGES = before_hedges

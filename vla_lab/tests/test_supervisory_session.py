"""The session runner and the gate: does a session record what it claims to?"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from vla_lab.supervisory import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, WAIT
from vla_lab.supervisory.apparatus import LexicalGrounder, SimulatedSupervisorChannel, SurrogateApparatus
from vla_lab.supervisory.apparatus.isaac import fidelity_report
from vla_lab.supervisory.contract import Contract
from vla_lab.supervisory.estimand import sequence_from_records
from vla_lab.supervisory.logging import BELIEFS_FILE, EVENTS_FILE, META_FILE, TRIALS_FILE, read_jsonl
from vla_lab.supervisory.protocol import build_protocol
from vla_lab.supervisory.run_study import run_one_supervisor
from vla_lab.supervisory.scheduler import CONDITION_CARRYOVER_AWARE, CONDITION_MEMORYLESS
from vla_lab.supervisory.session import run_session
from vla_lab.supervisory.supervisor import draw_cohort

CONTRACT = Contract()


def _session(log_root=None, conditions=None):
    sup = draw_cohort(1, seed=5)[0]
    proto = build_protocol(supervisor_id=sup.p.supervisor_id, contract=CONTRACT, seed=5,
                           conditions=conditions or [CONDITION_CARRYOVER_AWARE])
    return run_session(contract=CONTRACT, protocol=proto,
                       apparatus=SurrogateApparatus(CONTRACT.grid, seed=5),
                       channel=SimulatedSupervisorChannel(sup),
                       grounder=LexicalGrounder(CONTRACT.axis), seed=5, log_root=log_root,
                       truth={"params": sup.p.to_dict()})


def test_a_session_runs_every_block_of_its_protocol_in_order():
    one = _session()
    assert [b.kind for b in one.blocks] == ["reference", "condition", "retest"]
    two = _session(conditions=[CONDITION_CARRYOVER_AWARE, CONDITION_MEMORYLESS])
    assert [b.kind for b in two.blocks] == ["reference", "condition", "condition", "retest"]


def test_the_realized_budget_matches_the_contract():
    res = _session()
    for b in res.blocks:
        if b.kind == "condition":
            assert len(b.records) == CONTRACT.budget.slots_per_block
            assert sum(1 for r in b.records if r["action"] == COACH) == CONTRACT.budget.coach_per_block


def test_every_slot_emits_exactly_one_action_from_the_action_set():
    res = _session()
    for b in res.blocks:
        for r in b.records:
            assert r["action"] in (COACH, PROBE, WAIT, COUNTER)


def test_answered_slots_carry_an_utterance_and_a_grounding():
    res = _session()
    for b in res.blocks:
        for r in b.records:
            if r["action"] in (PROBE, COUNTER):
                assert "utterance" in r and "grounded" in r
                assert r["instructed_strategy"] in (STRATEGY_A, STRATEGY_B, None)


def test_an_ungrounded_answer_falls_back_to_the_value_optimal_strategy_and_is_flagged():
    # Falling back to whatever was last demonstrated would inject the very bias being measured.
    res = _session()
    for b in res.blocks:
        for r in b.records:
            if r.get("fallback_used"):
                assert r["instructed_strategy"] is None
                scene = CONTRACT.grid.by_id(int(r["scene_id"]))
                assert r["executed_strategy"] == CONTRACT.grid.physics.optimal_strategy(scene.margin_m)


def test_demonstrations_never_produce_an_observation():
    res = _session()
    for b in res.blocks:
        for r in b.records:
            if r["action"] == COACH:
                assert r.get("instructed_strategy") is None
                assert "demonstrated_strategy" in r


def test_the_session_writes_every_documented_artifact():
    with TemporaryDirectory() as td:
        root = Path(td) / "S000"
        _session(log_root=root)
        for name in (TRIALS_FILE, BELIEFS_FILE, EVENTS_FILE, META_FILE, "contract.json", "protocol.json",
                     "truth.json"):
            assert (root / name).exists(), name
        meta = json.loads((root / META_FILE).read_text())
        assert meta["contract_hash"] == CONTRACT.hash()
        assert meta["narration_hash"] == CONTRACT.narration_hash()


def test_ground_truth_is_kept_out_of_the_trial_log():
    with TemporaryDirectory() as td:
        root = Path(td) / "S000"
        _session(log_root=root)
        trials = read_jsonl(root / TRIALS_FILE)
        blob = json.dumps(trials)
        for forbidden in ("pi_star", "kappa_true", "intended_strategy", "p_a"):
            assert forbidden not in blob, forbidden


def test_the_scheduler_belief_is_logged_every_slot_for_audit():
    with TemporaryDirectory() as td:
        root = Path(td) / "S000"
        _session(log_root=root)
        beliefs = read_jsonl(root / BELIEFS_FILE)
        assert len(beliefs) > 0
        assert all("action" in b for b in beliefs)


def test_the_clock_offset_is_recorded_once_per_session():
    with TemporaryDirectory() as td:
        root = Path(td) / "S000"
        _session(log_root=root)
        meta = json.loads((root / META_FILE).read_text())
        assert "t0_wall_unix" in meta["clock"] and meta["clock"]["monotonic_source"] == "time.monotonic"


def test_the_full_per_supervisor_pipeline_produces_every_reported_outcome():
    row = run_one_supervisor(index=0, contract=CONTRACT,
                             conditions=[CONDITION_MEMORYLESS, CONDITION_CARRYOVER_AWARE], seed=3)
    row.pop("_identification_posterior", None)
    assert set(row["conditions"]) == {CONDITION_MEMORYLESS, CONDITION_CARRYOVER_AWARE}
    for cond, cell in row["conditions"].items():
        for k in ("mae_crossover", "coverage@95", "deployment_regret", "alignment", "vs_truth"):
            assert k in cell, (cond, k)
    assert "mae_crossover" in row["test_retest"]
    assert len(row["reference_map"]) == len(CONTRACT.grid.probe_scenes())


def test_conditions_face_an_identical_supervisor_so_the_comparison_is_paired():
    a = run_one_supervisor(index=0, contract=CONTRACT, conditions=[CONDITION_MEMORYLESS], seed=3)
    b = run_one_supervisor(index=0, contract=CONTRACT, conditions=[CONDITION_MEMORYLESS], seed=3)
    a.pop("_identification_posterior", None)
    b.pop("_identification_posterior", None)
    assert a["reference_map"] == b["reference_map"]
    assert a["conditions"][CONDITION_MEMORYLESS]["mae_crossover"] == b["conditions"][CONDITION_MEMORYLESS]["mae_crossover"]


def test_the_fidelity_check_flags_a_surrogate_that_does_not_match_the_closed_loop():
    grid = CONTRACT.grid
    scene = grid.probe_scenes()[3]
    # Every closed-loop rollout succeeds; the physics predicts it usually fails.
    rolls = [{"strategy": STRATEGY_B, "scene_id": scene.scene_id, "success": True, "duration_s": 15.0}
             for _ in range(20)]
    rep = fidelity_report(rolls, grid)
    assert rep["n_cells"] == 1
    if grid.physics.p_success(STRATEGY_B, scene.margin_m) < 0.9:
        assert "NOT faithful" in rep["verdict"]


def test_the_fidelity_check_passes_when_the_surrogate_matches():
    import random

    grid = CONTRACT.grid
    rng = random.Random(0)
    rolls = []
    for s in grid.probe_scenes()[:4]:
        for strat in (STRATEGY_A, STRATEGY_B):
            for _ in range(400):
                rolls.append({"strategy": strat, "scene_id": s.scene_id,
                              "success": rng.random() < grid.physics.p_success(strat, s.margin_m),
                              "duration_s": grid.physics.duration_s(strat, s.margin_m)})
    rep = fidelity_report(rolls, grid)
    assert rep["fraction_within_tolerance"] == 1.0, rep["verdict"]

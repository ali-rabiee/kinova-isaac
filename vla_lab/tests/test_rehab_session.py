"""W14/W15 — end-to-end session tests and the Phase 0 gate's failure-mode fixtures.

W15's "done when" (``rehab.md`` §6/W15) is *"each failure mode has a fixture that triggers it
and a passing session that does not"* — that is what most of this file is. The passing session
is a real synthetic run through :class:`~vla_lab.rehab.session.Phase0Session`, so the gate is
tested against output the pipeline actually produces rather than against hand-written JSON.

W14's "done when" is also checked here: a full synthetic study runs end to end in seconds, and
**an induced halt mid-block leaves a session the gate accepts as *partial* rather than
corrupt** — the participant's right to stop must never produce an unusable file (§11).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from vla_lab.rehab import ASSESS, COACH, WAIT
from vla_lab.rehab.contract import BudgetConfig, Phase0Contract
from vla_lab.rehab.logging import SessionReader
from vla_lab.rehab.protocol import BLOCK_COMPARED, BLOCK_REFERENCE, BLOCK_RETEST, Phase0Protocol
from vla_lab.rehab.run_pilot import run_one, synthetic_handedness
from vla_lab.rehab.scheduler import COMPARED_CONDITIONS
from vla_lab.rehab.verify_session import Thresholds, verify_pool, verify_session

# A small budget: these tests are about structure, not statistics.
CONTRACT = Phase0Contract(
    budget=BudgetConfig(trials_per_block=20, coach_per_block=4, reference_trials=16, retest_trials=16)
)
PROTOCOL = Phase0Protocol()


def _session(tmp: str, idx: int = 0, *, conditions=None, misdetect: Optional[float] = None):
    result, participant = run_one(
        idx, CONTRACT, PROTOCOL, log_root=tmp, seed=7,
        conditions=list(conditions) if conditions else None,
        misdetect_rate=misdetect,
    )
    return result, participant


# --------------------------------------------------------------------------- the runner


def test_a_full_synthetic_session_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        result, participant = _session(tmp, conditions=COMPARED_CONDITIONS)
        assert result.completed
        s = result.summary()
        expected = (
            CONTRACT.budget.reference_trials
            + CONTRACT.budget.retest_trials
            + CONTRACT.budget.trials_per_block * len(COMPARED_CONDITIONS)
        )
        assert s["n_trials"] == expected, (s["n_trials"], expected)
        assert s["n_halts"] == 0


def test_the_session_writes_every_documented_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        result, _ = _session(tmp)
        d = result.session_dir
        for name in ("contract.json", "participant.json", "protocol.json", "trials.jsonl", "events.jsonl", "observers.jsonl"):
            assert (d / name).exists(), name
        assert (d / "media").is_dir()


def test_the_budget_is_matched_in_the_realized_session():
    with tempfile.TemporaryDirectory() as tmp:
        result, _ = _session(tmp, conditions=COMPARED_CONDITIONS)
        by_block = Counter()
        coach = Counter()
        kinds = {int(b.block_idx): b.kind for b in result.plan.blocks}
        for r in result.records:
            by_block[r.trial.block_idx] += 1
            if r.trial.action == COACH:
                coach[r.trial.block_idx] += 1
        compared = [b for b, k in kinds.items() if k == BLOCK_COMPARED]
        assert len({by_block[b] for b in compared}) == 1
        assert len({coach[b] for b in compared}) == 1
        for b, k in kinds.items():
            if k in (BLOCK_REFERENCE, BLOCK_RETEST):
                assert coach[b] == 0


def test_the_handedness_inventory_gates_the_session():
    from vla_lab.rehab.apparatus import NullApparatus
    from vla_lab.rehab.session import Phase0Session, SessionConfig

    with tempfile.TemporaryDirectory() as tmp:
        s = Phase0Session(
            CONTRACT, PROTOCOL, SessionConfig(participant_id="P", log_root=tmp),
            apparatus=NullApparatus(CONTRACT.timing),
            observer_factory=lambda kind: None,
        )
        try:
            s.run()
            raise AssertionError("a session without a handedness inventory must refuse to start")
        except ValueError as exc:
            assert "handedness" in str(exc)


def test_mixed_handedness_is_an_exclusion_not_a_coin_flip():
    from vla_lab.rehab.apparatus import NullApparatus
    from vla_lab.rehab.session import Phase0Session, SessionConfig

    with tempfile.TemporaryDirectory() as tmp:
        s = Phase0Session(
            CONTRACT, PROTOCOL, SessionConfig(participant_id="P", log_root=tmp),
            apparatus=NullApparatus(CONTRACT.timing),
            observer_factory=lambda kind: None,
            handedness_responses=[0] * 10,  # no preference on any item
        )
        try:
            s.run()
            raise AssertionError("mixed handedness must be refused")
        except ValueError as exc:
            assert "mixed" in str(exc)


def test_the_scheduler_belief_is_logged_for_audit():
    with tempfile.TemporaryDirectory() as tmp:
        result, _ = _session(tmp, conditions=["carryover_aware"])
        events = SessionReader(result.session_dir).events().rows
        decisions = [e for e in events if e.get("type") == "scheduler_decision"]
        assert decisions, "an adaptive policy's decisions must be recoverable post hoc"
        assert "values" in decisions[0]["data"]
        rows = SessionReader(result.session_dir).trials()
        assert any(r.trial.scheduler_rationale for r in rows)


# --------------------------------------------------------------------------- the gate: passing


def test_a_clean_session_passes_the_gate():
    with tempfile.TemporaryDirectory() as tmp:
        result, _ = _session(tmp, conditions=COMPARED_CONDITIONS)
        rep = verify_session(result.session_dir)
        assert rep.ok, rep.failures
        assert not rep.partial


def test_sessions_from_one_contract_pool():
    with tempfile.TemporaryDirectory() as tmp:
        a, _ = _session(tmp, 0)
        b, _ = _session(tmp, 1)
        rep = verify_pool([a.session_dir, b.session_dir])
        assert rep.ok, rep.failures


# --------------------------------------------------------------------------- the gate: fixtures


def _mutate(session_dir: Path, name: str, fn) -> None:
    p = session_dir / name
    d = json.loads(p.read_text())
    p.write_text(json.dumps(fn(d), indent=2))


def test_fixture_contract_hash_drift_fails_the_pool():
    with tempfile.TemporaryDirectory() as tmp:
        a, _ = _session(tmp, 0)
        b, _ = _session(tmp, 1)
        _mutate(b.session_dir, "contract.json", lambda d: {**d, "contract_hash": "0" * 64})
        rep = verify_pool([a.session_dir, b.session_dir])
        assert not rep.ok
        assert any("poolable" in f for f in rep.failures)


def test_fixture_a_tampered_contract_fails_its_own_session():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        _mutate(r.session_dir, "contract.json", lambda d: {**d, "workspace": {**d["workspace"], "n_lateral": 5}})
        rep = verify_session(r.session_dir)
        assert not rep.ok
        assert any("edited after the session" in f for f in rep.failures)


def test_fixture_prompt_wording_drift_inside_a_session():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        lines = p.read_text().splitlines()
        row = json.loads(lines[-1])
        row["prompt_hash"] = "f" * 64
        lines[-1] = json.dumps(row)
        p.write_text("\n".join(lines) + "\n")
        rep = verify_session(r.session_dir)
        assert any("drifted" in f or "prompt" in f for f in rep.failures), rep.failures


def test_fixture_too_many_unresolved_selections():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for row in rows[: len(rows) // 3]:
            row["selection"] = {**row["selection"], "arm": "none"}
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        rep = verify_session(r.session_dir)
        assert any("usable arm selection" in f for f in rep.failures), rep.failures


def test_fixture_ambiguous_selections_above_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for row in rows[: len(rows) // 6]:
            row["selection"] = {**row["selection"], "arm": "ambiguous"}
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        rep = verify_session(r.session_dir)
        assert any("ambiguous" in f for f in rep.failures), rep.failures


def test_fixture_thin_crossover_band_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        grid = CONTRACT.target_grid()
        band = {t.target_id for t in grid.crossover_targets()}
        victim = sorted(band)[0]
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows = [x for x in rows if x.get("target_id") != victim]
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        rep = verify_session(r.session_dir)
        assert any("crossover-band" in f for f in rep.failures), rep.failures


def test_fixture_coach_leaking_into_the_reference_block():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows[0]["action"] = COACH
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        rep = verify_session(r.session_dir)
        assert any("must contain zero" in f for f in rep.failures), rep.failures


def test_fixture_an_unexplained_clock_jump():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        # Inside a block, not at a boundary: an inter-block gap is annotated and excused.
        cut = next(
            i for i in range(len(rows) - 2, 1, -1)
            if rows[i]["block_idx"] == rows[i - 1]["block_idx"]
        )
        for row in rows[cut:]:
            for k in ("t_present_ms", "t_settled_ms", "t_go_ms"):
                if row.get(k) is not None:
                    row[k] = int(row[k]) + 3_600_000  # an unlogged hour
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        rep = verify_session(r.session_dir)
        assert any("clock jump" in f for f in rep.failures), rep.failures


def test_fixture_a_backwards_clock():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows[5]["t_go_ms"] = -1000
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        rep = verify_session(r.session_dir)
        assert any("backwards" in f for f in rep.failures), rep.failures


def test_fixture_a_missing_handedness_inventory():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        _mutate(r.session_dir, "participant.json", lambda d: {**d, "handedness": {}, "nonpreferred_side": ""})
        rep = verify_session(r.session_dir)
        assert any("handedness" in f for f in rep.failures), rep.failures


def test_fixture_a_truncated_trials_file():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "trials.jsonl"
        text = p.read_text()
        p.write_text(text[: int(len(text) * 0.8)])
        rep = verify_session(r.session_dir)
        assert any("truncated" in f for f in rep.failures), rep.failures


def test_fixture_a_halt_without_a_reason():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        p = r.session_dir / "events.jsonl"
        with p.open("a") as fp:
            fp.write(json.dumps({"t_ms": 1, "type": "safety_halt", "data": {"reason": ""}}) + "\n")
        rep = verify_session(r.session_dir)
        assert any("no reason" in f for f in rep.failures), rep.failures


def test_fixture_a_missing_protocol_file():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        (r.session_dir / "protocol.json").unlink()
        rep = verify_session(r.session_dir)
        assert any("protocol.json" in f for f in rep.failures), rep.failures


def test_fixture_low_observer_agreement_fails_the_gate():
    from vla_lab.rehab import ARM_NONPREFERRED, ARM_PREFERRED
    from vla_lab.rehab.logging import OBSERVERS_FILE

    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp)
        with (r.session_dir / OBSERVERS_FILE).open("a") as fp:
            for i in range(40):
                # A coder who systematically disagrees with the online label.
                fp.write(json.dumps({
                    "trial_idx": i, "observer": "coder_a", "source": "coded",
                    "arm": (ARM_PREFERRED if i % 2 else ARM_NONPREFERRED), "confidence": 1.0,
                }) + "\n")
        rep = verify_session(r.session_dir)
        assert any("kappa" in f for f in rep.failures), rep.failures


# --------------------------------------------------------------------------- partial sessions


def test_a_partial_session_is_accepted_as_partial_not_corrupt():
    """The participant's right to stop must never produce a *corrupt* file (§11).

    It may still be too thin to analyze — a half-finished block genuinely cannot estimate
    ``pi*`` over the crossover band, and the gate should say so. What must NOT happen is the
    stop being reported as corruption, or the resulting budget mismatch being treated as a
    design violation.
    """

    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp, conditions=COMPARED_CONDITIONS)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        keep = rows[: int(len(rows) * 0.55)]
        p.write_text("\n".join(json.dumps(x) for x in keep) + "\n")
        with (r.session_dir / "events.jsonl").open("a") as fp:
            fp.write(json.dumps({
                "t_ms": 999999, "type": "session_stopped",
                "data": {"reason": "participant_request"},
            }) + "\n")
        rep = verify_session(r.session_dir)

        assert rep.partial, "a stopped session must be flagged partial"
        assert any("accepted as partial" in n for n in rep.info), rep.info
        # No corruption findings: the file is intact, the contract holds, the clock is sane.
        for word in ("truncated", "contract", "prompt", "clock", "protocol.json", "handedness"):
            assert not any(word in f for f in rep.failures), (word, rep.failures)
        # And the budget mismatch caused by stopping early is a WARNING, not a failure.
        assert any("budget" in w for w in rep.warnings), rep.warnings
        assert not any("budget" in f for f in rep.failures), rep.failures


def test_an_unexplained_partial_session_warns():
    with tempfile.TemporaryDirectory() as tmp:
        r, _ = _session(tmp, conditions=COMPARED_CONDITIONS)
        p = r.session_dir / "trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        p.write_text("\n".join(json.dumps(x) for x in rows[: len(rows) // 2]) + "\n")
        rep = verify_session(r.session_dir)
        assert rep.partial
        assert any("no recorded stop or halt" in w for w in rep.warnings), rep.warnings


# --------------------------------------------------------------------------- analysis wiring


def test_the_analysis_runs_on_a_synthetic_session():
    from vla_lab.rehab.analyze import ParticipantData, aggregate, analyze_participant
    from vla_lab.rehab.carryover import CarryoverConfig

    coarse = CarryoverConfig(n_lambda=5, n_beta=4, n_g=4)
    with tempfile.TemporaryDirectory() as tmp:
        results = []
        for i in range(3):
            r, participant = _session(tmp, i, conditions=["carryover_aware", "fixed_washout"])
            pd = ParticipantData(r.session_dir, carryover_cfg=coarse)
            results.append(analyze_participant(pd, ground_truth=participant.pi_star_map()))
        summary = aggregate(results)
        assert summary["n_participants"] == 3
        assert set(summary["conditions"]) == {"carryover_aware", "fixed_washout"}
        assert summary["budget_matched"]
        assert summary["primary_contrast"]["n"] == 3
        assert "test_retest" in summary
        for row in results:
            for cond in ("carryover_aware", "fixed_washout"):
                assert 0.0 <= row["conditions"][cond]["mae"] <= 1.0


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_session") else 0)

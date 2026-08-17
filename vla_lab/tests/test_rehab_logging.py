"""W2 — tests for ``vla_lab.rehab.trial`` and ``vla_lab.rehab.logging``.

"Done when" (``rehab.md`` §6/W2): the phase machine rejects out-of-order transitions; every
trial record round-trips; clock offsets are recorded; **a truncated session file is detected
rather than silently parsed**.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vla_lab.rehab import ARM_NONPREFERRED, ASSESS, COACH, WAIT
from vla_lab.rehab.contract import Phase0Contract, TimingConfig
from vla_lab.rehab.logging import (
    CONTRACT_FILE,
    OBSERVERS_FILE,
    PROTOCOL_FILE,
    SessionReader,
    SessionWriter,
    find_sessions,
    read_jsonl,
    session_dir_for,
)
from vla_lab.rehab.trial import (
    PHASE_DWELL,
    PHASE_GO,
    PHASE_LOG,
    PHASE_PRESENT,
    PHASE_REACH,
    PHASE_RETURN,
    PHASE_SELECT,
    PHASE_SETTLE,
    History,
    ManualClock,
    PhaseError,
    SessionClock,
    Trial,
    TrialPhaseMachine,
    TrialRecord,
    TrialResult,
)

TIMING = TimingConfig()


# --------------------------------------------------------------------------- phase machine


def test_the_presented_sequence_runs_in_order():
    pm = TrialPhaseMachine(ASSESS, TIMING)
    for i, phase in enumerate(
        [PHASE_PRESENT, PHASE_SETTLE, PHASE_GO, PHASE_REACH, PHASE_SELECT, PHASE_RETURN, PHASE_LOG]
    ):
        pm.enter(phase, i * 100)
    assert pm.complete
    assert [e["phase"] for e in pm.to_list()][-1] == PHASE_LOG


def test_out_of_order_transitions_are_rejected():
    pm = TrialPhaseMachine(ASSESS, TIMING)
    pm.enter(PHASE_PRESENT, 0)
    try:
        pm.enter(PHASE_GO, 10)  # skipped SETTLE
        raise AssertionError("skipping a phase must raise")
    except PhaseError as exc:
        assert "out-of-order" in str(exc)


def test_a_wait_trial_uses_its_own_short_sequence():
    pm = TrialPhaseMachine(WAIT, TIMING)
    pm.enter(PHASE_DWELL, 0)
    pm.enter(PHASE_LOG, 6000)
    assert pm.complete
    pm2 = TrialPhaseMachine(WAIT, TIMING)
    try:
        pm2.enter(PHASE_PRESENT, 0)
        raise AssertionError("a WAIT trial has no PRESENT phase")
    except PhaseError:
        pass


def test_a_backwards_clock_is_rejected():
    pm = TrialPhaseMachine(ASSESS, TIMING)
    pm.enter(PHASE_PRESENT, 1000)
    try:
        pm.enter(PHASE_SETTLE, 900)
        raise AssertionError("a backwards timestamp must raise")
    except PhaseError as exc:
        assert "backwards" in str(exc)


def test_halt_is_terminal():
    pm = TrialPhaseMachine(ASSESS, TIMING)
    pm.enter(PHASE_PRESENT, 0)
    pm.halt(50, "estop_participant")
    try:
        pm.enter(PHASE_SETTLE, 60)
        raise AssertionError("no phases may follow a halt")
    except PhaseError:
        pass


def test_phase_timeouts_are_checked_against_the_contract():
    pm = TrialPhaseMachine(ASSESS, TIMING)
    pm.enter(PHASE_PRESENT, 0)
    assert not pm.timed_out(PHASE_PRESENT, TIMING.present_timeout_ms - 1)
    assert pm.timed_out(PHASE_PRESENT, TIMING.present_timeout_ms + 1)


# --------------------------------------------------------------------------- clock


def test_manual_clock_drives_the_session_clock_deterministically():
    mc = ManualClock()
    clock = SessionClock(source=mc)
    assert clock.now_ms() == 0
    mc.advance_ms(1500)
    assert clock.now_ms() == 1500
    assert "wall_clock_epoch_s_at_t0" in clock.offset()


# --------------------------------------------------------------------------- records


def _record(idx: int = 0, action: str = ASSESS) -> TrialRecord:
    return TrialRecord(
        trial=Trial(
            trial_idx=idx, block_idx=1, condition="carryover_aware", action=action,
            target_id=(3 if action != WAIT else None),
            target_xy_participant_m=((0.34, -0.02) if action != WAIT else None),
            prompt_id=("coach_v1" if action == COACH else None), prompt_hash="abc123",
            effort_level=("moderate" if action == COACH else "none"),
            t_present_ms=1712, t_settled_ms=2480, t_go_ms=2500,
            kappa_prior_mean=0.31, since_last_coach_ms=48200, coach_count_so_far=6,
        ),
        result=TrialResult(
            arm=ARM_NONPREFERRED, t_select_ms=3210, reach_time_ms=710, success=True,
            observer="vision", confidence=0.97,
        ),
        phases=[{"phase": PHASE_PRESENT, "t_ms": 1712}],
    )


def test_trial_record_round_trips():
    rec = _record()
    back = TrialRecord.from_dict(rec.to_dict())
    assert back.to_dict() == rec.to_dict()


def test_the_schema_matches_the_documented_record():
    d = _record().to_dict()
    for key in (
        "trial_idx", "block_idx", "condition", "action", "target_id",
        "target_xy_participant_m", "prompt_id", "prompt_hash", "t_present_ms",
        "t_settled_ms", "t_go_ms", "selection", "reach_time_ms", "success",
        "kappa_prior_mean", "since_last_coach_ms", "coach_count_so_far", "halted", "halt_reason",
    ):
        assert key in d, key
    assert set(d["selection"]) == {"arm", "t_ms", "observer", "confidence"}


def test_only_resolved_selections_count_as_observations():
    assert _record().result.is_observation
    r = TrialResult(arm="none")
    assert not r.is_observation and r.chose_nonpreferred is None
    r2 = TrialResult(arm="ambiguous")
    assert not r2.is_observation
    r3 = TrialResult(arm=ARM_NONPREFERRED, halted=True)
    assert not r3.is_observation


# --------------------------------------------------------------------------- history


def test_history_tracks_budget_and_the_last_coach():
    h = History(slots_total=5, coach_slots=(0,))
    h.append(_record(0, COACH))
    h.append(_record(1, ASSESS))
    h.append(_record(2, WAIT))
    assert h.n_action(COACH) == 1 and h.n_action(WAIT) == 1
    assert h.trials_since_last_coach() == 2
    spent = h.budget_spent()
    assert spent["trials"] == 3 and spent["coach"] == 1 and spent["wait"] == 1


# --------------------------------------------------------------------------- the writer


def test_protocol_json_must_be_written_before_the_first_trial():
    with tempfile.TemporaryDirectory() as tmp:
        w = SessionWriter(Path(tmp) / "s1")
        try:
            w.log_trial(_record())
            raise AssertionError("trials before protocol.json must be refused")
        except RuntimeError as exc:
            assert "protocol.json" in str(exc)
        w.write_protocol({"blocks": []})
        w.log_trial(_record())
        w.close()


def test_contract_json_is_written_unrounded_so_its_hash_still_verifies():
    with tempfile.TemporaryDirectory() as tmp:
        w = SessionWriter(Path(tmp) / "s2")
        c = Phase0Contract()
        w.write_contract(c)
        w.close()
        d = json.loads((Path(tmp) / "s2" / CONTRACT_FILE).read_text())
        assert Phase0Contract.from_dict(d).contract_hash() == d["contract_hash"]


def test_observers_are_never_merged_into_trials():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "s3"
        w = SessionWriter(d)
        w.write_protocol({"blocks": []})
        w.log_trial(_record(0))
        w.log_observation(trial_idx=0, observer="vision", arm=ARM_NONPREFERRED, t_ms=3210, confidence=0.97, physical_side="left")
        w.log_observation(trial_idx=0, observer="keyed", arm="preferred", t_ms=3240, confidence=1.0, physical_side="right")
        w.log_observation(trial_idx=0, observer="coder_a", arm=ARM_NONPREFERRED, t_ms=3212, source="coded")
        w.close()
        rows = read_jsonl(d / OBSERVERS_FILE).rows
        assert len(rows) == 3
        # The disagreement survives on disk instead of being silently resolved.
        assert {r["arm"] for r in rows} == {ARM_NONPREFERRED, "preferred"}
        trials = SessionReader(d).trials()
        assert len(trials) == 1 and trials[0].result.observer == "vision"


def test_a_truncated_file_is_detected_not_silently_parsed():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "s4"
        w = SessionWriter(d)
        w.write_protocol({"blocks": []})
        for i in range(3):
            w.log_trial(_record(i))
        w.close()
        p = d / "trials.jsonl"
        text = p.read_text()
        p.write_text(text[: int(len(text) * 0.85)])  # power loss mid-write
        res = SessionReader(d).trials_raw()
        assert res.truncated_lines >= 1, res
        assert len(res.rows) < 3


def test_find_sessions_walks_the_log_root():
    with tempfile.TemporaryDirectory() as tmp:
        for pid in ("P001", "P002"):
            d = session_dir_for(pid, root=tmp, timestamp="20260901_101500")
            w = SessionWriter(d)
            w.write_protocol({"blocks": []})
            w.log_trial(_record())
            w.close()
        found = find_sessions(tmp)
        assert len(found) == 2
        assert all(SessionReader(f).exists() for f in found)


if __name__ == "__main__":
    import sys

    from vla_lab.tests import run_namespace

    sys.exit(1 if run_namespace(dict(globals()), label="test_rehab_logging") else 0)

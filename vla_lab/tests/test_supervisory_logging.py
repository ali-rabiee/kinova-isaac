"""The session gate: each failure mode has a fixture that triggers it, and one that does not."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from vla_lab.supervisory import COACH, PROBE, STRATEGY_A, WAIT
from vla_lab.supervisory.apparatus import LexicalGrounder, SimulatedSupervisorChannel, SurrogateApparatus
from vla_lab.supervisory.contract import Contract
from vla_lab.supervisory.logging import EVENTS_FILE, META_FILE, TRIALS_FILE, SessionLogger, read_jsonl
from vla_lab.supervisory.protocol import build_protocol
from vla_lab.supervisory.session import run_session
from vla_lab.supervisory.supervisor import draw_cohort
from vla_lab.supervisory.verify_session import verify, verify_pool

CONTRACT = Contract()


def _good_session(root: Path, seed: int = 5) -> Path:
    sup = draw_cohort(1, seed=seed)[0]
    proto = build_protocol(supervisor_id=root.name, contract=CONTRACT, seed=seed)
    run_session(contract=CONTRACT, protocol=proto, apparatus=SurrogateApparatus(CONTRACT.grid, seed=seed),
                channel=SimulatedSupervisorChannel(sup), grounder=LexicalGrounder(CONTRACT.axis),
                seed=seed, log_root=root)
    return root


def _rewrite_trials(root: Path, fn) -> None:
    rows = read_jsonl(root / TRIALS_FILE)
    rows = fn(rows)
    (root / TRIALS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_clean_session_passes_the_gate():
    with TemporaryDirectory() as td:
        assert verify(_good_session(Path(td) / "S000")).ok


def test_a_tampered_contract_fails_its_own_session():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")
        m = json.loads((root / META_FILE).read_text())
        m["contract_hash"] = "deadbeefdeadbeef"
        (root / META_FILE).write_text(json.dumps(m))
        r = verify(root)
        assert not r.ok and any("contract hash drift" in f for f in r.failures)


def test_narration_drift_fails_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")
        m = json.loads((root / META_FILE).read_text())
        m["narration_hash"] = "0000000000000000"
        (root / META_FILE).write_text(json.dumps(m))
        assert any("narration hash drift" in f for f in verify(root).failures)


def test_a_budget_that_is_not_actually_matched_fails_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")
        _rewrite_trials(root, lambda rows: [r for r in rows if not (r.get("block_kind") == "condition"
                                                                    and r["slot"] == 5)])
        r = verify(root)
        assert not r.ok and any("slots" in f for f in r.failures)


def test_a_demonstration_leaking_into_a_reference_block_fails_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")

        def leak(rows: List[Dict[str, Any]]):
            for r in rows:
                if r.get("block_kind") == "reference":
                    r["action"] = COACH
                    break
            return rows

        _rewrite_trials(root, leak)
        r = verify(root)
        assert not r.ok and any("demonstrations" in f for f in r.failures)


def test_too_many_ungrounded_answers_fail_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")

        def blank(rows):
            for r in rows:
                if r["action"] == PROBE:
                    r["instructed_strategy"] = None
            return rows

        _rewrite_trials(root, blank)
        r = verify(root)
        assert not r.ok and any("grounding rate" in f for f in r.failures)


def test_thin_crossover_coverage_fails_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")
        band = {s.scene_id for s in CONTRACT.grid.probe_scenes() if CONTRACT.grid.in_crossover_band(s)}

        def strip(rows):
            return [r for r in rows if int(r.get("scene_id", -1)) not in band]

        _rewrite_trials(root, strip)
        r = verify(root)
        assert not r.ok and any("crossover-band" in f for f in r.failures)


def test_a_backwards_clock_fails_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")

        def scramble(rows):
            rows[3]["t_ms"] = -1
            return rows

        _rewrite_trials(root, scramble)
        r = verify(root)
        assert not r.ok and any("backwards" in f for f in r.failures)


def test_an_unannotated_halt_fails_the_gate():
    with TemporaryDirectory() as td:
        root = _good_session(Path(td) / "S000")
        with (root / EVENTS_FILE).open("a") as fh:
            fh.write(json.dumps({"t_ms": 1, "kind": "halt_unspecified"}) + "\n")
        r = verify(root)
        assert not r.ok and any("halt" in f for f in r.failures)


def test_sessions_from_one_contract_pool_and_from_two_do_not():
    with TemporaryDirectory() as td:
        root = Path(td)
        _good_session(root / "S000", seed=1)
        _good_session(root / "S001", seed=2)
        assert verify_pool(root).ok
        m = json.loads((root / "S001" / META_FILE).read_text())
        m["contract_hash"] = "cafebabecafebabe"
        (root / "S001" / META_FILE).write_text(json.dumps(m))
        assert not verify_pool(root).ok


def test_the_logger_keeps_its_four_streams_apart():
    with TemporaryDirectory() as td:
        root = Path(td) / "S"
        lg = SessionLogger(root, supervisor_id="S")
        lg.trial({"slot": 0, "action": PROBE})
        lg.belief({"slot": 0, "w": 1})
        lg.event("block_start", {"index": 0})
        lg.write_json("truth.json", {"secret": 1})
        lg.close({"contract_hash": "x"})
        assert len(read_jsonl(root / TRIALS_FILE)) == 1
        assert "secret" not in (root / TRIALS_FILE).read_text()
        assert json.loads((root / META_FILE).read_text())["n_trials"] == 1

"""W2/§10 — the Phase 0 session writer and reader.

On-disk layout (``rehab.md`` §10)::

    logs/rehab/participant_<PID>/session_<TS>/
    ├── contract.json      hashed Phase 0 contract + provenance
    ├── participant.json   PID, handedness inventory + score, participant-frame solve,
    │                      condition assignment, clock offset
    ├── protocol.json      block layout, seeds, target sequence — written BEFORE trial 1
    ├── trials.jsonl       one record per trial (the unit of analysis)
    ├── events.jsonl       audit trail: phase transitions, halts, prompts, pauses, faults
    ├── observers.jsonl    per-trial labels from EVERY observer, kept separately
    └── media/             wrist + front video segments per trial (gitignored; §11)

Three design rules from §10 are enforced here rather than left to convention:

1. **``observers.jsonl`` is never collapsed into ``trials.jsonl``.** The online label is what
   the scheduler acted on; the coded label is what the analysis uses. Both stay recoverable,
   and their disagreement is a reported quantity (W8). :meth:`SessionWriter.log_observation`
   is the only way to record an observer label, and it appends — it never rewrites a trial.
2. **``protocol.json`` is written before trial 1**, so a preregistered analysis can be checked
   against the realized assignment. :meth:`SessionWriter.log_trial` refuses to run before
   :meth:`SessionWriter.write_protocol`.
3. **One monotonic clock**, with the wall-clock offset recorded once in ``participant.json``.

Floats are rounded to fixed precision but written as JSON **numbers**, not the zero-padded
strings the VLA tick logger emits — ``trials.jsonl`` is read directly by the analysis, and a
numeric column that has to be re-parsed from a string is an unforced error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from .trial import SessionClock, TrialRecord

SESSION_SCHEMA = "vla_lab_rehab_session/v1"

TRIALS_FILE = "trials.jsonl"
EVENTS_FILE = "events.jsonl"
OBSERVERS_FILE = "observers.jsonl"
CONTRACT_FILE = "contract.json"
PARTICIPANT_FILE = "participant.json"
PROTOCOL_FILE = "protocol.json"
MEDIA_DIR = "media"


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        v = round(obj, ndigits)
        return 0.0 if v == -0.0 else v
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def session_dir_for(
    participant_id: str,
    *,
    root: Union[str, Path] = "logs/rehab",
    timestamp: Optional[str] = None,
) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / f"participant_{participant_id}" / f"session_{ts}"


class SessionWriter:
    """Append-only, event-locked writer for one Phase 0 session."""

    def __init__(
        self,
        session_dir: Union[str, Path],
        *,
        clock: Optional[SessionClock] = None,
        create: bool = True,
    ) -> None:
        self.dir = Path(session_dir)
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / MEDIA_DIR).mkdir(exist_ok=True)
        self.clock = clock or SessionClock()
        self._protocol_written = False
        self._trials = open(self.dir / TRIALS_FILE, "a", buffering=1)
        self._events = open(self.dir / EVENTS_FILE, "a", buffering=1)
        self._observers = open(self.dir / OBSERVERS_FILE, "a", buffering=1)
        self._closed = False

    # -- one-shot documents ------------------------------------------------
    def write_contract(self, contract: Any) -> Path:
        """Write ``contract.json`` **unrounded**.

        This is the hashed artifact: rounding it would make the recorded hash disagree with the
        file's own contents on reload, and the gate would (correctly) report every session as
        tampered with.
        """

        payload = contract.to_dict() if hasattr(contract, "to_dict") else dict(contract)
        p = self.dir / CONTRACT_FILE
        p.write_text(json.dumps(payload, indent=2))
        return p

    def write_participant(self, participant: Dict[str, Any]) -> Path:
        payload = {
            "schema": SESSION_SCHEMA,
            **dict(participant),
            "clock": self.clock.offset(),
        }
        p = self.dir / PARTICIPANT_FILE
        p.write_text(json.dumps(_round_floats(payload, 6), indent=2))
        return p

    def write_protocol(self, protocol: Any) -> Path:
        """Must be called before the first trial (§10). Records the realized assignment."""

        payload = protocol.to_dict() if hasattr(protocol, "to_dict") else dict(protocol)
        p = self.dir / PROTOCOL_FILE
        p.write_text(json.dumps(_round_floats(payload, 6), indent=2))
        self._protocol_written = True
        return p

    # -- streams -----------------------------------------------------------
    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, *, t_ms: Optional[int] = None) -> None:
        rec = {
            "t_ms": int(t_ms if t_ms is not None else self.clock.now_ms()),
            "type": str(event_type),
            "data": _round_floats(dict(data or {}), 4),
        }
        self._events.write(json.dumps(rec) + "\n")

    def log_trial(self, record: TrialRecord) -> None:
        if not self._protocol_written and not (self.dir / PROTOCOL_FILE).exists():
            raise RuntimeError(
                "protocol.json must be written before the first trial (rehab.md §10) so the "
                "preregistered analysis can be checked against the realized assignment"
            )
        self._trials.write(json.dumps(_round_floats(record.to_dict(), 4)) + "\n")

    def log_observation(
        self,
        *,
        trial_idx: int,
        observer: str,
        arm: str,
        t_ms: Optional[int],
        confidence: float = 0.0,
        physical_side: str = "",
        source: str = "online",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one observer's label for one trial. Never merged into ``trials.jsonl``."""

        rec = {
            "trial_idx": int(trial_idx),
            "observer": str(observer),
            "source": str(source),  # online | coded
            "arm": str(arm),
            "physical_side": str(physical_side),
            "t_ms": (int(t_ms) if t_ms is not None else None),
            "confidence": round(float(confidence), 4),
            **({"extra": _round_floats(dict(extra), 4)} if extra else {}),
        }
        self._observers.write(json.dumps(rec) + "\n")

    def media_path(self, trial_idx: int, camera: str, ext: str = "mp4") -> Path:
        """Path for one trial's video segment (created lazily by the capture backend)."""

        return self.dir / MEDIA_DIR / f"trial_{int(trial_idx):04d}_{camera}.{ext}"

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fp in (self._trials, self._events, self._observers):
            try:
                fp.close()
            except Exception:  # noqa: BLE001 - closing must never mask the study outcome
                pass

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass
class ReadResult:
    """Parsed JSONL plus the number of trailing bytes that were **not** valid JSON.

    A truncated session file (power loss mid-write) must be *detected*, not silently
    parsed into a shorter study — so the truncation count is returned, and
    :mod:`vla_lab.rehab.verify_session` fails on it.
    """

    rows: List[Dict[str, Any]]
    truncated_lines: int = 0


def read_jsonl(path: Union[str, Path]) -> ReadResult:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    bad = 0
    if not p.exists():
        return ReadResult(rows, 0)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return ReadResult(rows, bad)


class SessionReader:
    """Read-side view of one session directory."""

    def __init__(self, session_dir: Union[str, Path]) -> None:
        self.dir = Path(session_dir)

    def _json(self, name: str) -> Optional[Dict[str, Any]]:
        p = self.dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    @property
    def contract(self) -> Optional[Dict[str, Any]]:
        return self._json(CONTRACT_FILE)

    @property
    def participant(self) -> Optional[Dict[str, Any]]:
        return self._json(PARTICIPANT_FILE)

    @property
    def protocol(self) -> Optional[Dict[str, Any]]:
        return self._json(PROTOCOL_FILE)

    def trials_raw(self) -> ReadResult:
        return read_jsonl(self.dir / TRIALS_FILE)

    def trials(self) -> List[TrialRecord]:
        return [TrialRecord.from_dict(r) for r in self.trials_raw().rows]

    def events(self) -> ReadResult:
        return read_jsonl(self.dir / EVENTS_FILE)

    def observations(self) -> ReadResult:
        return read_jsonl(self.dir / OBSERVERS_FILE)

    def exists(self) -> bool:
        return self.dir.is_dir() and (self.dir / TRIALS_FILE).exists()


def find_sessions(root: Union[str, Path] = "logs/rehab") -> List[Path]:
    """Every ``session_*`` directory under ``logs/rehab/participant_*/``, sorted."""

    r = Path(root)
    if not r.is_dir():
        return []
    out: List[Path] = []
    for pdir in sorted(r.glob("participant_*")):
        out += sorted(s for s in pdir.glob("session_*") if s.is_dir())
    return out


__all__ = [
    "SESSION_SCHEMA",
    "TRIALS_FILE",
    "EVENTS_FILE",
    "OBSERVERS_FILE",
    "CONTRACT_FILE",
    "PARTICIPANT_FILE",
    "PROTOCOL_FILE",
    "MEDIA_DIR",
    "SessionWriter",
    "SessionReader",
    "ReadResult",
    "read_jsonl",
    "find_sessions",
    "session_dir_for",
]

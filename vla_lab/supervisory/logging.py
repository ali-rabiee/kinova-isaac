"""Event-locked session records.

Timestamp fidelity is a correctness property here, not logging hygiene: the carryover model is
a function of elapsed time since the last demonstration, so clock error propagates directly
into the fitted decay and thence into every scheduling decision. One monotonic source is used
throughout and its offset to wall-clock is recorded once per session.

Four files, deliberately kept apart:

``trials.jsonl``
    One row per slot, containing **only what an estimator is allowed to see**.
``beliefs.jsonl``
    The scheduler's pre-decision posterior and both candidate utilities, every slot. An
    adaptive policy whose reasoning cannot be reconstructed from the log is not reviewable.
``events.jsonl``
    Block boundaries, faults, and anything that interrupted the session.
``truth.json``
    Ground truth -- present only for simulated supervisors, in its own file so that no analysis
    can read it by accident. Loading it is an explicit act.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TRIALS_FILE = "trials.jsonl"
BELIEFS_FILE = "beliefs.jsonl"
EVENTS_FILE = "events.jsonl"
TRUTH_FILE = "truth.json"
META_FILE = "meta.json"
CONTRACT_FILE = "contract.json"
PROTOCOL_FILE = "protocol.json"


class SessionClock:
    """One monotonic source, with the wall-clock offset recorded once."""

    def __init__(self) -> None:
        self.t0_monotonic = time.monotonic()
        self.t0_wall = time.time()

    def now_ms(self) -> int:
        return int(round((time.monotonic() - self.t0_monotonic) * 1000.0))

    def to_dict(self) -> Dict[str, Any]:
        return {"t0_wall_unix": self.t0_wall, "monotonic_source": "time.monotonic"}


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


class SessionLogger:
    """Writes one session directory. Append-only; nothing is rewritten after the fact."""

    def __init__(self, root: Path, *, supervisor_id: str, condition: Optional[str] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = SessionClock()
        self.supervisor_id = str(supervisor_id)
        self.condition = condition
        self._trials = (self.root / TRIALS_FILE).open("w")
        self._beliefs = (self.root / BELIEFS_FILE).open("w")
        self._events = (self.root / EVENTS_FILE).open("w")
        self.n_trials = 0
        self.event("session_open", {"supervisor_id": self.supervisor_id, "condition": condition})

    # -- writes -------------------------------------------------------------
    def trial(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("t_ms", self.clock.now_ms())
        self._trials.write(json.dumps(row) + "\n")
        self._trials.flush()
        self.n_trials += 1

    def belief(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("t_ms", self.clock.now_ms())
        self._beliefs.write(json.dumps(row, default=float) + "\n")
        self._beliefs.flush()

    def event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._events.write(json.dumps({"t_ms": self.clock.now_ms(), "kind": str(kind), **(payload or {})}) + "\n")
        self._events.flush()

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        (self.root / name).write_text(json.dumps(payload, indent=2, default=float) + "\n")

    def close(self, meta: Optional[Dict[str, Any]] = None) -> None:
        self.event("session_close", {"n_trials": self.n_trials})
        payload = {
            "supervisor_id": self.supervisor_id,
            "condition": self.condition,
            "n_trials": self.n_trials,
            "clock": self.clock.to_dict(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        payload.update(meta or {})
        self.write_json(META_FILE, payload)
        for f in (self._trials, self._beliefs, self._events):
            f.close()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


__all__ = [
    "TRIALS_FILE",
    "BELIEFS_FILE",
    "EVENTS_FILE",
    "TRUTH_FILE",
    "META_FILE",
    "CONTRACT_FILE",
    "PROTOCOL_FILE",
    "SessionClock",
    "SessionLogger",
    "read_jsonl",
]

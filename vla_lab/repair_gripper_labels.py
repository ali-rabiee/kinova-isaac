"""Backfill gripper state/action labels in collected sessions from events.jsonl.

Why this exists: until 2026-06 the tick logger inferred gripper state from the
mean of *all* "finger" joints (including the tip joints, which stay near 0)
against a 0.5 rad threshold. Fingers wrapped around a box stall well below
that, so every tick of a successful grasp was logged as ``state: "open"`` and
``action_from_prev.gripper_action`` stayed 0 — the model never sees a close
signal. The logger is fixed for future sessions; this tool repairs sessions
that were already collected by replaying the GRIPPER_CLOSE / GRIPPER_OPEN
action intervals from each episode's ``events.jsonl`` (wall-clock ``t_ms``,
same clock as ticks).

Usage (from repo root, no Isaac required):
    python -m vla_lab.repair_gripper_labels --session logs/data_collection/session_20260607_115038
    python -m vla_lab.repair_gripper_labels --session ... --dry-run

The original ticks file is kept as ``ticks.jsonl.orig`` (first run only).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _gripper_intervals(events: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """Return [(t_ms, "close"|"open"), ...] transitions, ordered by time.

    A transition is registered at the *end* of each GRIPPER_CLOSE/OPEN action
    (the fingers have moved by then; ticks during the actuation keep the
    previous state, matching what an online detector would have seen).
    """

    transitions: List[Tuple[int, str]] = []
    for ev in events:
        if ev.get("type") != "action_end":
            continue
        action = str((ev.get("data") or {}).get("action", ""))
        if action == "GRIPPER_CLOSE":
            transitions.append((int(ev["t_ms"]), "close"))
        elif action == "GRIPPER_OPEN":
            transitions.append((int(ev["t_ms"]), "open"))
    transitions.sort(key=lambda x: x[0])
    return transitions


def _state_at(t_ms: int, transitions: List[Tuple[int, str]]) -> str:
    state = "open"
    for tt, st in transitions:
        if t_ms >= tt:
            state = st
        else:
            break
    return state


def repair_episode(ep_dir: Path, *, dry_run: bool = False) -> Optional[Dict[str, int]]:
    ticks_path = ep_dir / "ticks.jsonl"
    events_path = ep_dir / "events.jsonl"
    if not ticks_path.exists() or not events_path.exists():
        return None

    transitions = _gripper_intervals(_load_jsonl(events_path))
    ticks = _load_jsonl(ticks_path)
    if not ticks:
        return None

    n_state_fixed = 0
    n_action_fixed = 0
    prev_state: Optional[str] = None
    for rec in ticks:
        t_ms = int(rec.get("t_ms", 0))
        state = _state_at(t_ms, transitions)

        robot = rec.get("robot") or {}
        gripper = robot.get("gripper") or {}
        if gripper.get("state") != state:
            gripper["state"] = state
            n_state_fixed += 1

        afp = (rec.get("policy") or {}).get("action_from_prev")
        if isinstance(afp, dict) and prev_state is not None:
            g = 0
            if prev_state == "open" and state == "close":
                g = -1
            elif prev_state == "close" and state == "open":
                g = +1
            if int(afp.get("gripper_action", 0)) != g:
                afp["gripper_action"] = g
                n_action_fixed += 1
        prev_state = state

    if not dry_run and (n_state_fixed or n_action_fixed):
        backup = ticks_path.with_suffix(".jsonl.orig")
        if not backup.exists():
            shutil.copy2(ticks_path, backup)
        with ticks_path.open("w") as f:
            for rec in ticks:
                f.write(json.dumps(rec) + "\n")

    n_close = sum(
        1
        for rec in ticks
        if isinstance((rec.get("policy") or {}).get("action_from_prev"), dict)
        and int(rec["policy"]["action_from_prev"].get("gripper_action", 0)) == -1
    )
    return {
        "ticks": len(ticks),
        "transitions": len(transitions),
        "state_fixed": n_state_fixed,
        "action_fixed": n_action_fixed,
        "close_labels": n_close,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair gripper labels from events.jsonl")
    parser.add_argument("--session", type=str, required=True, help="Session dir containing episode_* folders")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    session = Path(args.session)
    ep_dirs = sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not ep_dirs:
        ep_dirs = [session]

    total = {"ticks": 0, "transitions": 0, "state_fixed": 0, "action_fixed": 0, "close_labels": 0}
    for ep_dir in ep_dirs:
        stats = repair_episode(ep_dir, dry_run=bool(args.dry_run))
        if stats is None:
            print(f"[repair] {ep_dir.name}: skipped (missing ticks/events)")
            continue
        for k in total:
            total[k] += stats[k]
        print(
            f"[repair] {ep_dir.name}: ticks={stats['ticks']}  gripper_events={stats['transitions']}  "
            f"state_fixed={stats['state_fixed']}  action_fixed={stats['action_fixed']}  close_labels={stats['close_labels']}"
        )

    mode = "DRY RUN — nothing written" if args.dry_run else "written (originals kept as ticks.jsonl.orig)"
    print(f"[repair] total: {total}  [{mode}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

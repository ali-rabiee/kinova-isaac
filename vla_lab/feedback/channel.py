"""Two-way feedback channels: unsolicited interjections + robot-initiated queries.

A channel owns BOTH directions of the human link so experiments can vary one
"human" coherently:

- ``observe(**tick_state)``: feed one policy tick of context (sim channels use
  it to decide whether the human would speak; console channels ignore it).
- ``poll()``: non-blocking; returns interjections delivered since last poll.
- ``ask(question, options)``: robot-initiated query -> QueryAnswer (blocking).

Implementations:
- :class:`NullFeedbackChannel`    — silence (pure-autonomy runs).
- :class:`ScriptedFeedbackChannel`— wraps :class:`SimulatedHuman` (sim/evals).
- :class:`ConsoleFeedbackChannel` — live keyboard: a daemon thread reads stdin
  lines; typed lines between questions are interjections, lines typed after a
  question are its answer.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from ..allocation.query import QueryAnswer
from .sim_human import SimulatedHuman
from .types import Interjection


class FeedbackChannel:
    name = "base"
    supports_ask = False

    def observe(self, **tick_state: Any) -> None:  # noqa: B027 - optional hook
        pass

    def poll(self) -> List[Interjection]:
        return []

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        raise NotImplementedError(f"{self.name} channel cannot ask")

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class NullFeedbackChannel(FeedbackChannel):
    name = "none"


class ScriptedFeedbackChannel(FeedbackChannel):
    """Simulated human: reactive corrections + oracle-graded query answers."""

    name = "scripted"
    supports_ask = True

    def __init__(self, sim_human: SimulatedHuman) -> None:
        self.human = sim_human
        self._due: List[Interjection] = []

    def observe(self, **tick_state: Any) -> None:
        try:
            self._due.extend(
                self.human.observe(
                    tick_idx=int(tick_state["tick_idx"]),
                    ee_pos_b=tick_state["ee_pos_b"],
                    object_pos_b=dict(tick_state.get("object_pos_b") or {}),
                )
            )
        except KeyError:
            pass

    def poll(self) -> List[Interjection]:
        out, self._due = self._due, []
        return out

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        return self.human.ask(question, options)


class ConsoleFeedbackChannel(FeedbackChannel):
    """Live operator keyboard channel (real robot / interactive sim).

    Any line typed while the robot runs is an interjection; when the robot asks
    a question, the next line answers it.
    """

    name = "console"
    supports_ask = True

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()
        print("[feedback] console channel live — type corrections at any time (e.g. 'no, the red one').")

    def _reader(self) -> None:
        while not self._stop.is_set():
            line = sys.stdin.readline()
            if not line:  # EOF
                return
            line = line.strip()
            if line:
                self._q.put(line)

    def poll(self) -> List[Interjection]:
        out: List[Interjection] = []
        while True:
            try:
                out.append(Interjection(text=self._q.get_nowait(), source="human"))
            except queue.Empty:
                return out

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        t0 = time.time()
        print(f"\n[robot asks] {question}")
        if options:
            for i, opt in enumerate(options):
                print(f"   {i}) {opt}")
        print("[your answer — number or text] ", end="", flush=True)
        raw = self._q.get()  # blocking: next typed line is the answer
        idx: Optional[int] = None
        text = raw
        if options:
            if raw.isdigit() and 0 <= int(raw) < len(options):
                idx = int(raw)
                text = str(options[idx])
            else:
                for i, opt in enumerate(options):
                    if raw.lower() and raw.lower() in str(opt).lower():
                        idx = i
                        text = str(opt)
                        break
        return QueryAnswer(text=text, option_idx=idx, latency_s=time.time() - t0, source="human")

    def close(self) -> None:
        self._stop.set()


def build_feedback_channel(kind: str, sim_human: Optional[SimulatedHuman] = None) -> FeedbackChannel:
    k = str(kind).lower().strip()
    if k in ("none", "", "off"):
        return NullFeedbackChannel()
    if k == "scripted":
        if sim_human is None:
            raise ValueError("scripted feedback channel needs a SimulatedHuman")
        return ScriptedFeedbackChannel(sim_human)
    if k in ("console", "stdin", "keyboard"):
        return ConsoleFeedbackChannel()
    raise ValueError(f"unknown feedback channel {kind!r}")

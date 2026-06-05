"""The query branch: acquiring external information by asking the human.

Per the resolved design decision for this project, querying is **autonomous**: the robot
genuinely escalates and consumes a typed/clicked answer (no Wizard-of-Oz operator in the
loop). The session records the answer source and latency for disclosure and for the
appropriate-reliance analysis.

Interfaces (all share :class:`QueryInterface`):

- :class:`StdinQueryInterface` — the deployment default; prints the question and reads a
  typed answer / option number from the operator-facing console.
- :class:`CallbackQueryInterface` — wraps an arbitrary ``(question, options) -> answer``
  callable (GUI button, speech-to-text, study front-end).
- :class:`ScriptedQueryInterface` — for **sim eval and tests**: answers from an oracle
  (the known target), with a configurable answer-accuracy and simulated human latency so
  query rate / post-query success are measurable without a person present.

The chosen answer is folded back into the instruction via :func:`refine_instruction`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


@dataclass
class QueryAnswer:
    """A human's response to one query."""

    text: str
    option_idx: Optional[int] = None
    latency_s: float = 0.0
    source: str = "human"  # "human" | "oracle" | "callback"
    correct: Optional[bool] = None  # known only in sim/oracle settings


class QueryInterface:
    """Base class. Subclasses implement :meth:`ask`."""

    name: str = "base"

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        raise NotImplementedError


class StdinQueryInterface(QueryInterface):
    """Autonomous deployment default: prompt the operator console for an answer."""

    name = "stdin"

    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print) -> None:
        self._input = input_fn
        self._print = print_fn

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        t0 = time.time()
        self._print(f"\n[robot asks] {question}")
        idx: Optional[int] = None
        if options:
            for i, opt in enumerate(options):
                self._print(f"   {i}) {opt}")
            raw = str(self._input("[your answer — number or text] ")).strip()
            if raw.isdigit() and 0 <= int(raw) < len(options):
                idx = int(raw)
                text = str(options[idx])
            else:
                text = raw
                # Match free text against options if possible.
                for i, opt in enumerate(options):
                    if raw.lower() and raw.lower() in str(opt).lower():
                        idx = i
                        text = str(opt)
                        break
        else:
            text = str(self._input("[your answer] ")).strip()
        return QueryAnswer(text=text, option_idx=idx, latency_s=time.time() - t0, source="human")


class CallbackQueryInterface(QueryInterface):
    """Wrap a ``(question, options) -> str | QueryAnswer`` callable (GUI / study UI)."""

    name = "callback"

    def __init__(self, fn: Callable[[str, Optional[Sequence[str]]], object]) -> None:
        self._fn = fn

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        t0 = time.time()
        ans = self._fn(question, options)
        if isinstance(ans, QueryAnswer):
            if ans.latency_s == 0.0:
                ans.latency_s = time.time() - t0
            return ans
        text = str(ans).strip()
        idx = None
        if options:
            for i, opt in enumerate(options):
                if text.lower() and text.lower() in str(opt).lower():
                    idx = i
                    break
        return QueryAnswer(text=text, option_idx=idx, latency_s=time.time() - t0, source="callback")


@dataclass
class ScriptedQueryInterface(QueryInterface):
    """Oracle-backed answers for sim/tests.

    ``oracle_answer`` is the ground-truth disambiguation (e.g. the target's color). With
    probability ``accuracy`` the returned answer is correct; otherwise a distractor from
    ``options`` is returned (models an imperfect human). ``sim_latency_s`` simulates the
    interruption cost in wall-clock-free runs.
    """

    oracle_answer: str = ""
    accuracy: float = 1.0
    sim_latency_s: float = 3.0
    seed: int = 0
    name: str = field(default="scripted", init=False)
    _calls: int = field(default=0, init=False)

    def ask(self, question: str, options: Optional[Sequence[str]] = None) -> QueryAnswer:
        rng = random.Random(int(self.seed) + self._calls)
        self._calls += 1
        correct = rng.random() < float(self.accuracy)
        opts: List[str] = [str(o) for o in (options or [])]
        if correct or not opts:
            text = str(self.oracle_answer)
        else:
            distractors = [o for o in opts if o.lower() != str(self.oracle_answer).lower()]
            text = rng.choice(distractors) if distractors else str(self.oracle_answer)
        idx = None
        for i, opt in enumerate(opts):
            if text.lower() and text.lower() in opt.lower():
                idx = i
                break
        return QueryAnswer(
            text=text,
            option_idx=idx,
            latency_s=float(self.sim_latency_s),
            source="oracle",
            correct=bool(text.lower() == str(self.oracle_answer).lower()),
        )


def refine_instruction(instruction: str, answer: QueryAnswer) -> str:
    """Fold a human answer back into the instruction for the post-query forward pass.

    Heuristic and deliberately conservative: if the answer text is not already present in
    the instruction, append a disambiguating clause. Real deployments can replace this
    with a structured slot-filling step.
    """

    base = str(instruction).strip()
    ans = str(answer.text).strip()
    if not ans:
        return base
    if ans.lower() in base.lower():
        return base
    sep = " " if base.endswith((".", "!", "?")) else ". "
    return f"{base}{sep}The target is the {ans}."

"""What the robot says, what the supervisor says back, and how either is grounded.

Three jobs, all of which have to be nailed down before the manipulation is a manipulation:

**COACH narration.** A demonstration is a dose, and a dose has to be fixed in content, wording,
and delivery or the comparison between conditions is not a comparison of schedules. The
wording lives here, in code, and its hash goes into the study contract; the session gate
refuses to pool a session whose narration hash differs from the rest. Wording drift between
supervisors is then detectable rather than invisible.

**The neutral probe and the counter-proposal.** PROBE asks a deliberately option-free question
("How should I approach this one?"). COUNTER asks the same question and then *names the
alternative the robot did not just demonstrate*. The distinction is the entire mechanism of
the study's active de-biasing action, so the two are separate, hashed assets and the counter
template is built from the axis rather than free-typed, which guarantees it always names the
option that was not demonstrated.

**Grounding.** Utterances have to become strategy labels. The grounder is lexical and
deliberately conservative: a phrase grounds to A if it hits an A anchor and no B anchor, to B
symmetrically, and to ``STRATEGY_UNRESOLVED`` otherwise. It never guesses. An ungrounded
utterance advances the carryover state, contributes nothing to the likelihood, and is counted
-- because silently coin-flipping an ambiguous answer would inject noise straight into the
estimand and the study would have no way to see it happening.

The lexical grounder is the **reference** channel, not the only one: in closed-loop runs the
policy's own intent head does the grounding, and the disagreement between the two is a
reported quantity (the sim analogue of an online-versus-coded observer ``kappa``).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED
from .strategies import StrategyAxis, get_axis, other

# ---------------------------------------------------------------------------
# Robot speech
# ---------------------------------------------------------------------------
#: Narration templates for a COACH demonstration. ``{what}`` is filled from the axis. Each
#: carries a rationale, because a demonstration that says only *what* it did is a weaker dose
#: than one that says *why* -- and "why" is what a supervisor is most likely to echo back.
COACH_TEMPLATES: Tuple[str, ...] = (
    "I'll {what}. {why}",
    "{why} So I'll {what}.",
    "Approaching this one as follows: I'll {what}. {why}",
)

COACH_WHAT: Dict[Tuple[str, str], str] = {
    ("plan", STRATEGY_A): "move the blocking box aside first, then pick up the target",
    ("plan", STRATEGY_B): "go straight for the target and leave the other box where it is",
    ("grasp", STRATEGY_A): "come down onto the target from above",
    ("grasp", STRATEGY_B): "come in at the target from the side",
}

COACH_WHY: Dict[Tuple[str, str], str] = {
    ("plan", STRATEGY_A): "Clearing the path first keeps the grasp safe.",
    ("plan", STRATEGY_B): "Going direct saves time and leaves the workspace as it is.",
    ("grasp", STRATEGY_A): "A top-down grasp is the most stable hold.",
    ("grasp", STRATEGY_B): "A side approach keeps the gripper clear of what is above it.",
}

#: The neutral probe. No option is named; naming one would answer the question being asked.
PROBE_QUERY = "How should I approach this one?"

#: The counter-proposal. Filled with the option the robot did *not* just demonstrate.
COUNTER_TEMPLATES: Tuple[str, ...] = (
    "How should I approach this one? I could also {alt} -- which would you prefer?",
    "How should I approach this one? Either works here; I can {alt} instead if you'd rather.",
)

COUNTER_ALT: Dict[Tuple[str, str], str] = {
    ("plan", STRATEGY_A): "clear the blocking box out of the way first",
    ("plan", STRATEGY_B): "go straight for the target without moving anything",
    ("grasp", STRATEGY_A): "come down onto it from above",
    ("grasp", STRATEGY_B): "come in from the side",
}

#: The WAIT filler. A matched *interaction* -- it costs budget and wall-clock like everything
#: else -- that carries no strategy information whatsoever. Silence would not be matched.
WAIT_FILLERS: Tuple[str, ...] = (
    "Give me a moment while I re-check the workspace.",
    "Recalibrating my cameras, one moment.",
    "Logging the last result. Stand by.",
    "Just resetting the scene.",
)


@dataclass(frozen=True)
class CoachDose:
    """One COACH level: what is said and how hard it pushes.

    ``carryover_scale`` is the **mechanical** knob -- it multiplies the gain ``g`` a
    demonstration injects, so a stronger dose moves the current answer more *and* leaves more
    residue, which is the intended causal story (the robot shows a strategy; having just seen
    it work is what carries over). ``delta_logit`` is the **interpretive** number: the expected
    immediate shift in ``logit Pr[A]``, i.e. ``beta * g * carryover_scale``. It is a prior
    until a study measures it, and ``source`` records which -- so no figure presents a prior
    as a measurement.
    """

    name: str
    repeats: int
    narrate_rationale: bool
    carryover_scale: float
    delta_logit: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_DOSES: Tuple[CoachDose, ...] = (
    CoachDose("weak", 1, False, 0.6, 0.35, "One demonstration, action narrated, no rationale."),
    CoachDose("moderate", 1, True, 1.0, 0.60, "One demonstration with its rationale. The default."),
    CoachDose("strong", 3, True, 1.8, 1.05, "Three consecutive demonstrations with rationale. Dose-finding only."),
)
DEFAULT_DOSE = "moderate"


def dose_by_name(name: str, doses: Sequence[CoachDose] = DEFAULT_DOSES) -> CoachDose:
    for d in doses:
        if d.name == str(name):
            return d
    raise KeyError(f"unknown coach dose {name!r}; known: {[d.name for d in doses]}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def coach_narration(axis: StrategyAxis, strategy: str, *, dose: CoachDose, rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random(0)
    what = COACH_WHAT[(axis.name, strategy)]
    if not dose.narrate_rationale:
        return f"I'll {what}."
    why = COACH_WHY[(axis.name, strategy)]
    return rng.choice(COACH_TEMPLATES).format(what=what, why=why)


def probe_query() -> str:
    return PROBE_QUERY


def counter_query(axis: StrategyAxis, demonstrated: str, *, rng: Optional[random.Random] = None) -> str:
    """The counter-proposal, naming the option that was *not* demonstrated."""
    rng = rng or random.Random(0)
    alt = COUNTER_ALT[(axis.name, other(demonstrated))]
    return rng.choice(COUNTER_TEMPLATES).format(alt=alt)


def wait_filler(rng: Optional[random.Random] = None) -> str:
    return (rng or random.Random(0)).choice(WAIT_FILLERS)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
def ground(utterance: str, axis: StrategyAxis) -> str:
    """Map an utterance to ``STRATEGY_A`` / ``STRATEGY_B`` / ``STRATEGY_UNRESOLVED``.

    Conservative by design: hits on both sides, or on neither, resolve to *unresolved* rather
    than to a guess. The count of unresolved answers is reported, and a condition that produces
    many of them is telling you something real about its query wording.
    """
    text = f" {str(utterance).lower().strip()} "
    hit_a = any(k in text for k in axis.keywords_a)
    hit_b = any(k in text for k in axis.keywords_b)
    if hit_a and not hit_b:
        return STRATEGY_A
    if hit_b and not hit_a:
        return STRATEGY_B
    return STRATEGY_UNRESOLVED


def grounding_agreement(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Cohen's kappa between two grounding channels over 3 categories.

    Used for lexical-grounder versus policy-intent-head agreement. Reported, never silently
    reconciled: the label the scheduler acted on and the label the analysis uses have to stay
    separately recoverable.
    """
    if not pairs:
        return {"kappa": float("nan"), "agreement": float("nan"), "n": 0}
    cats = [STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED]
    n = len(pairs)
    obs = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    exp = sum(pa[c] * pb[c] for c in cats)
    kappa = (obs - exp) / (1.0 - exp) if abs(1.0 - exp) > 1e-12 else float("nan")
    return {"kappa": float(kappa), "agreement": float(obs), "n": int(n)}


# ---------------------------------------------------------------------------
# The content hash
# ---------------------------------------------------------------------------
def content_hash(axis_name: str, *, doses: Sequence[CoachDose] = DEFAULT_DOSES) -> str:
    """Stable hash of every word the robot can say on this axis, plus the dose ladder."""
    axis = get_axis(axis_name)
    payload = {
        "axis": axis.to_dict(),
        "coach_templates": list(COACH_TEMPLATES),
        "coach_what": {f"{k[0]}|{k[1]}": v for k, v in sorted(COACH_WHAT.items())},
        "coach_why": {f"{k[0]}|{k[1]}": v for k, v in sorted(COACH_WHY.items())},
        "probe_query": PROBE_QUERY,
        "counter_templates": list(COUNTER_TEMPLATES),
        "counter_alt": {f"{k[0]}|{k[1]}": v for k, v in sorted(COUNTER_ALT.items())},
        "wait_fillers": list(WAIT_FILLERS),
        "keywords_a": list(axis.keywords_a),
        "keywords_b": list(axis.keywords_b),
        "phrases_a": list(axis.phrases_a),
        "phrases_b": list(axis.phrases_b),
        "doses": [d.to_dict() for d in doses],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = [
    "COACH_TEMPLATES",
    "PROBE_QUERY",
    "COUNTER_TEMPLATES",
    "WAIT_FILLERS",
    "CoachDose",
    "DEFAULT_DOSES",
    "DEFAULT_DOSE",
    "dose_by_name",
    "coach_narration",
    "probe_query",
    "counter_query",
    "wait_filler",
    "ground",
    "grounding_agreement",
    "content_hash",
]

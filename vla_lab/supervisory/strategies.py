"""The strategy axes: what the supervisor is actually choosing between.

Two axes ship. They differ in *kind*, deliberately, because a de-biasing result that holds
only for one kind of choice is a much weaker result:

``plan`` (primary, confirmatory)
    **CLEAR_FIRST** -- push the blocking object aside, then grasp the target -- versus
    **DIRECT** -- thread the remaining gap and grasp the target where it stands. A *plan*
    choice: two different sequences of actions reaching the same terminal state, trading time
    against the risk of disturbing the scene. This is the "safety-first vs. speed-first"
    preference the study is about.

``grasp`` (secondary, generalisation)
    **TOP_DOWN** -- approach along the tool axis from above -- versus **LATERAL** -- approach
    from the side. A *motor* choice: one action, two geometries, trading grasp stability
    against reachability under an overhang. Included to test whether de-biasing transfers
    across the kind of decision being biased, not to carry the primary claim.

Both axes obey the package sign convention: member **A** is the cautious one, **B** the
efficient one, and a scene's coordinate ``c`` is positive when A is objectively better.

Nothing here knows about robots, simulators, or models. The axis is a *label system* plus the
vocabulary the supervisor and the robot use to talk about it; the physics lives in
:mod:`vla_lab.supervisory.scenes` and the execution in :mod:`vla_lab.supervisory.apparatus`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import STRATEGY_A, STRATEGY_B

AXIS_PLAN = "plan"
AXIS_GRASP = "grasp"


@dataclass(frozen=True)
class StrategyAxis:
    """One binary strategy choice, with everything needed to talk about it.

    Attributes
    ----------
    name:
        Axis id (``"plan"`` / ``"grasp"``).
    label_a, label_b:
        Human-readable names of the cautious and efficient members.
    coordinate:
        What the signed scene coordinate ``c`` measures, in words. Printed in figures.
    margin_units:
        Physical unit of the raw scene margin ``m`` before normalisation.
    phrases_a, phrases_b:
        Utterances a supervisor might produce to select each member. Used by the generative
        supervisor to *speak*, and by the grounding parser to *listen*. Keeping one list for
        both directions is deliberate: a phrase the simulator can emit but the parser cannot
        ground is a silent observation-loss bug, and the round-trip test in
        ``tests/test_supervisory_scenes.py`` catches it.
    keywords_a, keywords_b:
        Minimal lexical anchors for grounding. A phrase grounds to A if it contains any
        ``keywords_a`` anchor and no ``keywords_b`` anchor (and vice versa); ambiguous or
        empty matches ground to ``STRATEGY_UNRESOLVED`` rather than guessing, because a
        silently-guessed instruction corrupts the estimand.
    """

    name: str
    label_a: str
    label_b: str
    coordinate: str
    margin_units: str
    phrases_a: Tuple[str, ...]
    phrases_b: Tuple[str, ...]
    keywords_a: Tuple[str, ...]
    keywords_b: Tuple[str, ...]
    #: Free-text description of what the robot *does* for each member, for narration and for
    #: the instruction string handed to the policy.
    command_a: str = ""
    command_b: str = ""

    def label(self, strategy: str) -> str:
        return self.label_a if strategy == STRATEGY_A else self.label_b

    def command(self, strategy: str) -> str:
        return self.command_a if strategy == STRATEGY_A else self.command_b

    def phrases(self, strategy: str) -> Tuple[str, ...]:
        return self.phrases_a if strategy == STRATEGY_A else self.phrases_b

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "coordinate": self.coordinate,
            "margin_units": self.margin_units,
            "command_a": self.command_a,
            "command_b": self.command_b,
        }


PLAN_AXIS = StrategyAxis(
    name=AXIS_PLAN,
    label_a="CLEAR_FIRST",
    label_b="DIRECT",
    coordinate="clearance deficit: how much tighter the gap is than a safe direct reach needs",
    margin_units="m",
    phrases_a=(
        "clear the path first",
        "move the blocker out of the way first",
        "push that one aside before you grab it",
        "clear it first, then pick up the target",
        "get the obstacle out of the way",
    ),
    phrases_b=(
        "just grab it directly",
        "reach around and take it",
        "go straight for the target",
        "pick it up directly, leave the other one",
        "thread the gap and grab it",
    ),
    keywords_a=("clear", "aside", "out of the way", "move the", "push", "first"),
    keywords_b=("direct", "directly", "straight", "around", "thread", "leave the", "just grab", "just take"),
    command_a="Move the blocking object aside, then pick up the target.",
    command_b="Pick up the target directly without moving anything else.",
)

GRASP_AXIS = StrategyAxis(
    name=AXIS_GRASP,
    label_a="TOP_DOWN",
    label_b="LATERAL",
    coordinate="overhead deficit: how much less headroom there is than a top-down grasp needs",
    margin_units="m",
    phrases_a=(
        "come down from above",
        "grab it from the top",
        "approach it top-down",
        "take it from overhead",
    ),
    phrases_b=(
        "come in from the side",
        "grab it sideways",
        "approach it laterally",
        "take it from the side",
    ),
    keywords_a=("above", "top", "overhead", "top-down", "downward", "from the top"),
    keywords_b=("side", "sideways", "lateral", "laterally", "from the side", "horizontally"),
    command_a="Grasp the target with a top-down approach.",
    command_b="Grasp the target with a lateral approach.",
)

AXES: Dict[str, StrategyAxis] = {AXIS_PLAN: PLAN_AXIS, AXIS_GRASP: GRASP_AXIS}

#: The axis the confirmatory contrast runs on. The other is a labelled generalisation arm.
PRIMARY_AXIS = AXIS_PLAN


def get_axis(name: str) -> StrategyAxis:
    try:
        return AXES[str(name)]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise KeyError(f"unknown strategy axis {name!r}; known: {sorted(AXES)}") from exc


def other(strategy: str) -> str:
    """The member of the axis that ``strategy`` is not."""
    return STRATEGY_B if strategy == STRATEGY_A else STRATEGY_A


__all__ = [
    "AXIS_PLAN",
    "AXIS_GRASP",
    "AXES",
    "PRIMARY_AXIS",
    "PLAN_AXIS",
    "GRASP_AXIS",
    "StrategyAxis",
    "get_axis",
    "other",
]

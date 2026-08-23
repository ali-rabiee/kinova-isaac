"""W9 — the COACH content library, content-hashed.

COACH content must be **identical across conditions and stable across participants**, or the
manipulation is confounded: a schedule comparison in which the prompts differ is not a
schedule comparison. So the wording lives here, in code, and its hash goes into
``contract.json``; :mod:`vla_lab.rehab.verify_session` refuses to pool a session whose
prompt hash differs from the rest (``rehab.md`` §6/W9).

**Effort manipulation (§12.7).** A bare verbal cue may produce carryover too small to
schedule around in healthy adults — the single biggest threat to the study's viability. The
literature the deck cites shows arm choice is *effort-sensitive* (Nguyen et al. 2023), so
COACH here is a **prompt plus an effort manipulation**: the presented target is made more
costly to reach with the preferred arm (a weighted puck, a precision aperture), which is both
better-motivated than a bare instruction and more likely to leave measurable carryover.

Two numbers describe a level, and they play different roles:

``carryover_scale``
    The **mechanical** knob. It multiplies the prompt gain ``g_p`` that a COACH event injects
    into the latent carryover state, so a stronger manipulation moves the current choice more
    *and* leaves more residue. That is the intended causal story: the prompt plus the effort
    cost gets the participant to use the nonpreferred arm, and it is *having used it* that
    carries over. One knob, both effects — which keeps the simulator and the inference model
    (:mod:`vla_lab.rehab.carryover`) parameterized identically.
``delta_logit``
    The **interpretive** number: the expected immediate shift in ``logit P(nonpreferred)``
    the level should produce, i.e. ``beta_p * g_p * carryover_scale`` under the model. It is
    what the pilot measures and what a reader can reason about, and
    :func:`implied_immediate_shift` is what checks the two stay consistent. It is a
    **prior** until the pilot (M4) estimates it; ``effort_delta_logit_source`` records which
    of the two it currently is, and the analysis surfaces that so no figure presents a prior
    as a measurement.

The neutral WAIT filler exists so that WAIT is a matched *interaction*, not silence: it
consumes budget and wall-clock like the other actions but carries **no arm information**.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Prompt kinds.
PROMPT_COACH = "coach"
PROMPT_NEUTRAL = "neutral"
PROMPT_GO = "go"


@dataclass(frozen=True)
class EffortLevel:
    """One level of the COACH effort manipulation. See the module docstring for the two
    numbers' different roles: ``carryover_scale`` is mechanical, ``delta_logit`` interpretive.
    """

    name: str
    delta_logit: float
    carryover_scale: float
    apparatus_setting: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# The default ladder. `none` is the bare verbal cue (what the deck originally proposed);
# `moderate` is the pre-registered default for Phase 0; `high` exists for the pilot's
# dose-finding, not for the confirmatory contrast.
DEFAULT_EFFORT_LEVELS: Tuple[EffortLevel, ...] = (
    EffortLevel(
        name="none",
        delta_logit=0.0,
        carryover_scale=1.0,
        apparatus_setting="plain_puck",
        description="Verbal prompt only; the target puck is unweighted and wide.",
    ),
    EffortLevel(
        name="moderate",
        delta_logit=0.6,
        carryover_scale=1.0,
        apparatus_setting="weighted_puck_light",
        description=(
            "Verbal prompt + a lightly weighted target puck, raising the cost of the "
            "longer (preferred-arm) reach."
        ),
    ),
    EffortLevel(
        name="high",
        delta_logit=1.2,
        carryover_scale=1.4,
        apparatus_setting="weighted_puck_heavy",
        description=(
            "Verbal prompt + a heavier puck and a narrowed placement aperture. Pilot "
            "dose-finding only (§12.7); not the confirmatory level."
        ),
    ),
)


@dataclass(frozen=True)
class PromptSpec:
    """One utterance, verbatim. ``audio_asset`` pins the rendered file (or TTS voice+version)."""

    prompt_id: str
    kind: str
    text: str
    audio_asset: str = ""
    gesture: str = ""  # optional gesture spec, e.g. "point_at_target"; "" = none

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptLibrary:
    """The full COACH/neutral/GO content set plus the effort ladder, content-hashed."""

    coach: PromptSpec = field(
        default_factory=lambda: PromptSpec(
            prompt_id="coach_v1",
            kind=PROMPT_COACH,
            text="Please use your {nonpreferred_arm_word} hand for this one.",
            audio_asset="tts:piper/en_US-lessac-medium@1.2.0",
            gesture="",
        )
    )
    neutral: PromptSpec = field(
        default_factory=lambda: PromptSpec(
            prompt_id="neutral_v1",
            kind=PROMPT_NEUTRAL,
            text="Take a moment. The next target is coming up.",
            audio_asset="tts:piper/en_US-lessac-medium@1.2.0",
            gesture="",
        )
    )
    go: PromptSpec = field(
        default_factory=lambda: PromptSpec(
            prompt_id="go_v1",
            kind=PROMPT_GO,
            text="Go.",
            audio_asset="tone:1000hz_120ms",
            gesture="",
        )
    )
    effort_levels: Tuple[EffortLevel, ...] = DEFAULT_EFFORT_LEVELS
    # Which effort level COACH uses in the confirmatory protocol.
    coach_effort_level: str = "moderate"
    # "prior" until the pilot measures it; "pilot:<session>" afterwards. Verify/analysis
    # surface this so no figure silently presents a prior as a measurement.
    effort_delta_logit_source: str = "prior"

    # -- lookup ------------------------------------------------------------
    def get(self, prompt_id: str) -> PromptSpec:
        for p in (self.coach, self.neutral, self.go):
            if p.prompt_id == prompt_id:
                return p
        raise KeyError(f"unknown prompt_id {prompt_id!r}")

    def effort(self, name: Optional[str] = None) -> EffortLevel:
        want = str(name if name is not None else self.coach_effort_level)
        for lvl in self.effort_levels:
            if lvl.name == want:
                return lvl
        raise KeyError(f"unknown effort level {want!r}; have {[l.name for l in self.effort_levels]}")

    def render_coach(self, nonpreferred_side: str) -> str:
        """The COACH utterance with the participant's nonpreferred side filled in."""

        word = {"left": "left", "right": "right"}.get(str(nonpreferred_side).lower())
        if word is None:
            raise ValueError(f"nonpreferred_side must be 'left' or 'right'; got {nonpreferred_side!r}")
        return self.coach.text.format(nonpreferred_arm_word=word)

    # -- hashing -----------------------------------------------------------
    def content(self) -> Dict[str, Any]:
        """The exact payload that is hashed. Wording, assets, gestures, effort ladder."""

        return {
            "coach": self.coach.to_dict(),
            "neutral": self.neutral.to_dict(),
            "go": self.go.to_dict(),
            "effort_levels": [e.to_dict() for e in self.effort_levels],
            "coach_effort_level": str(self.coach_effort_level),
        }

    def content_hash(self) -> str:
        """SHA-256 over the canonical content. Changes iff the content changes.

        ``effort_delta_logit_source`` is deliberately **not** hashed: replacing a prior with
        a pilot measurement of the same manipulation does not change what the participant
        experienced, so it must not un-pool already-collected sessions.
        """

        blob = json.dumps(self.content(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.content(),
            "effort_delta_logit_source": str(self.effort_delta_logit_source),
            "content_hash": self.content_hash(),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PromptLibrary":
        d = dict(d or {})
        kw: Dict[str, Any] = {}
        for key in ("coach", "neutral", "go"):
            if isinstance(d.get(key), dict):
                spec = dict(d[key])
                kw[key] = PromptSpec(
                    prompt_id=str(spec.get("prompt_id", f"{key}_v1")),
                    kind=str(spec.get("kind", key)),
                    text=str(spec.get("text", "")),
                    audio_asset=str(spec.get("audio_asset", "")),
                    gesture=str(spec.get("gesture", "")),
                )
        if isinstance(d.get("effort_levels"), list):
            kw["effort_levels"] = tuple(
                EffortLevel(
                    name=str(e.get("name", "")),
                    delta_logit=float(e.get("delta_logit", 0.0)),
                    carryover_scale=float(e.get("carryover_scale", 1.0)),
                    apparatus_setting=str(e.get("apparatus_setting", "")),
                    description=str(e.get("description", "")),
                )
                for e in d["effort_levels"]
            )
        if "coach_effort_level" in d:
            kw["coach_effort_level"] = str(d["coach_effort_level"])
        if "effort_delta_logit_source" in d:
            kw["effort_delta_logit_source"] = str(d["effort_delta_logit_source"])
        return cls(**kw)


def implied_immediate_shift(level: EffortLevel, *, beta: float, g: float) -> float:
    """``beta * g * carryover_scale`` — the immediate logit shift the model implies.

    Compare against ``level.delta_logit``: a large gap means the interpretive number and the
    mechanical one have drifted apart, and one of them is wrong.
    """

    return float(float(beta) * float(g) * float(level.carryover_scale))


DEFAULT_PROMPTS = PromptLibrary()


__all__ = [
    "PROMPT_COACH",
    "PROMPT_NEUTRAL",
    "PROMPT_GO",
    "EffortLevel",
    "PromptSpec",
    "PromptLibrary",
    "DEFAULT_EFFORT_LEVELS",
    "DEFAULT_PROMPTS",
    "implied_immediate_shift",
]

"""Real-time semantic human feedback: two-way channel + quality-aware fusion.

The 2026-07 reframing treats the human as a SENSOR with a quality model, not an
oracle: unsolicited corrections and query answers flow through one channel, get
parsed into typed refinements, and are applied / verified / ignored based on an
online reliability estimate (see vla_lab/fable_report.md).

    channel.poll() -> Interjection -> parser.parse_utterance -> Refinement
        -> fusion.fuse(refinement, reliability, intent_posterior)
        -> apply | verify (confirmation query) | ignore

Sim experiments drive the same path with :class:`SimulatedHuman`, whose
quality knobs (accuracy / latency / specificity / noise) are the manipulated
variables of the input-quality study (`sweep_feedback_quality.sh`).
"""

from .channel import (
    ConsoleFeedbackChannel,
    FeedbackChannel,
    NullFeedbackChannel,
    ScriptedFeedbackChannel,
)
from .fusion import FusionConfig, FusionDecision, ReliabilityTracker, fuse
from .parser import parse_utterance
from .sim_human import SimulatedHuman, SimulatedHumanConfig
from .types import DIRECTION_VECTORS, Interjection, Refinement

__all__ = [
    "ConsoleFeedbackChannel",
    "DIRECTION_VECTORS",
    "FeedbackChannel",
    "FusionConfig",
    "FusionDecision",
    "Interjection",
    "NullFeedbackChannel",
    "Refinement",
    "ReliabilityTracker",
    "ScriptedFeedbackChannel",
    "SimulatedHuman",
    "SimulatedHumanConfig",
    "fuse",
    "parse_utterance",
]

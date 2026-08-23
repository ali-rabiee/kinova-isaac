"""The Carryover-Aware VLA: policies that carry a memory of their own coaching.

The architecture is deliberately **backbone-agnostic**. One wrapper
(:class:`~vla_lab.policy.carryover_vla.CarryoverVLA`) adds the same three things to any vision-
language-action backbone:

1. a **carryover context** -- the robot's own recent demonstration history, injected by one of
   four mechanisms (:mod:`vla_lab.policy.context`);
2. an **intent head** that grounds the supervisor's utterance to a strategy *and* reports a
   posterior over what they would have said uncoached;
3. an **ask gate** that decides whether to execute, or to re-open the option the coaching
   closed with a counter-proposal.

Keeping those three separable from the backbone is what makes the paper's model table an
experiment rather than a list: the same context mechanism can be run on a 2M-parameter
from-scratch encoder, on a pretrained 450M VLA, and on a 2-3B pretrained VLM, and the question
"which architectural features let a model use a memory of its own behaviour" gets an answer
that is not confounded with which codebase each model happens to live in.

See :mod:`vla_lab.policy.registry` for the model roster and their cards.
"""

from __future__ import annotations

from .context import CONTEXT_MODES, CarryoverContext, build_context_encoder
from .heads import AskGateHead, IntentHead
from .registry import MODEL_CARDS, ModelCard, available_models, build_model, describe_models

__all__ = [
    "CONTEXT_MODES",
    "CarryoverContext",
    "build_context_encoder",
    "IntentHead",
    "AskGateHead",
    "ModelCard",
    "MODEL_CARDS",
    "available_models",
    "build_model",
    "describe_models",
]

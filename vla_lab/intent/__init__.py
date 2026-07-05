"""Intent-uncertainty estimation: WHICH goal does the human mean?

2026-07 reframing (see fable_report.md at the repo root): with the wrist camera
removing the single-view deficiency, uncertainty is decomposed by SOURCE —

- perception uncertainty: the policy cannot see well enough to execute
  (occlusion, missing view). Signals: K-sample dispersion, per-view occlusion,
  cross-view disagreement. Remedy: more compute or a better view.
- intent uncertainty: the policy cannot tell WHICH goal the human means
  (ambiguous instruction, conflicting correction). Signals: counterfactual
  instruction sweep + lexical grounding. Remedy: a clarifying query.

The act/compute/query allocator consumes both via ``ProbeResult.features``
(keys ``intent_entropy``, ``intent_top2_margin``, ``cross_view_disagreement``,
``instruction_ambiguous``) and the ``intent_allocator`` controller routes each
source to its matching remedy (vla_lab/allocation/baselines.py).
"""

from .estimator import (
    IntentEstimate,
    IntentEstimator,
    cross_view_disagreement,
    lexical_target_candidates,
)

__all__ = [
    "IntentEstimate",
    "IntentEstimator",
    "cross_view_disagreement",
    "lexical_target_candidates",
]

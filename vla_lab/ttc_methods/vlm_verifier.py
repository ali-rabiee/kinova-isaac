"""Appendix / future work: zero-shot VLM scoring over trajectory hypotheses.

Not wired into the default eval loop yet — kept as a small, import-safe stub so
experiments can grow toward the `docs/new_changes.md` appendix idea without blocking
the partial-observability baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class VLMVerifierStub:
    """Placeholder configuration for a future VLM-as-verifier stage."""

    model_name: str = "hidden"
    prompt_template: str = "Does this trajectory match: {instruction}?"

    def score(self, *, instruction: str, trajectories: List[Any]) -> List[float]:
        del instruction, trajectories
        raise NotImplementedError("VLMVerifierStub: implement with your VLM client.")


def score_trajectory_stub(*, instruction: str, trajectory: Any) -> float:
    del instruction, trajectory
    return 0.0

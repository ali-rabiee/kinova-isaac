"""Apparatus backends. Isaac is imported lazily so the offline suite runs without a simulator."""

from __future__ import annotations

from typing import Any

from .base import Apparatus, ExecutionOutcome, Grounder, SupervisorChannel, SupervisorTurn
from .surrogate import LexicalGrounder, SimulatedSupervisorChannel, SurrogateApparatus


def build_apparatus(kind: str, *args: Any, **kw: Any):
    """``"surrogate"`` or ``"isaac"``. Isaac pulls in the simulator only when asked for."""
    k = str(kind)
    if k == "surrogate":
        return SurrogateApparatus(*args, **kw)
    if k == "isaac":
        from .isaac import IsaacApparatus  # noqa: WPS433 - deliberate lazy import

        return IsaacApparatus(*args, **kw)
    raise KeyError(f"unknown apparatus backend {kind!r} (surrogate | isaac)")


__all__ = [
    "Apparatus",
    "ExecutionOutcome",
    "Grounder",
    "SupervisorChannel",
    "SupervisorTurn",
    "SurrogateApparatus",
    "SimulatedSupervisorChannel",
    "LexicalGrounder",
    "build_apparatus",
]

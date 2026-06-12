"""Act / compute / query allocation for VLA policies under partial observability.

This package implements the methodological core of the HRI-2027 pivot
(see the HRI-2027 pivot memo; removed from the repo — ask Kye): inference-time decision-making for a
stochastic VLA policy is reframed from a **binary** (act vs. compute) gate into a
**trichotomy**:

- ``act``     — execute one forward pass (K=1); ~0 extra cost.
- ``compute`` — draw K samples and select (the existing TTC "compute" branch);
                cost ∝ K (latency).
- ``query``   — acquire **external** information (ask the human); cost = latency +
                workload + interruption.

The governing idea (memo §3, §4): self-generated uncertainty (sample dispersion /
twin-probe disagreement) tells the policy *how* to spend compute, but it **cannot**
tell the policy *when compute is useless* — and the useless-compute regime is
exactly where an external information source is required. The allocator therefore
combines

1. a cheap **dispersion** probe (``uncertainty``) — reducible-looking ambiguity,
2. a calibrated **irreducibility detector** (``conformal`` + a fitted model) — is the
   optimal action even identifiable from the current (masked) view?, and
3. a **value-of-computation vs. value-of-information** rule (``value_of_information``)

to choose act / compute / query, optionally signalling the *type* of uncertainty to
the human (``transparency``) — the HRI condition that isolates "I should think" from
"I must ask you."

Design note: the *core* (``uncertainty``, ``conformal``, ``value_of_information``,
``query``, ``transparency``, ``allocator``, ``baselines``) is pure-Python/NumPy and
operates on scalar signals plus opaque "chunk" objects, so it is unit-testable with a
mock policy and has **no torch / Isaac / LeRobot import at module load**. The concrete
policy adapters that turn a TinyVLA / SmolVLA into a :class:`ComputeBranch` live in
``vla_lab.allocation.policy_adapters`` and import torch lazily.
"""

from __future__ import annotations

from .allocator import ActComputeQueryAllocator, AllocatorFit, Decision
from .baselines import (
    AutonomyController,
    ComputeGatedController,
    Controller,
    FixedComputeController,
    InsightController,
    KnowNoController,
    ScaleController,
    build_controller,
)
from .conformal import (
    MondrianConformal,
    SplitConformal,
    coverage_vs_bucket,
    empirical_coverage,
    weighted_quantile,
)
from .query import (
    CallbackQueryInterface,
    QueryAnswer,
    QueryInterface,
    ScriptedQueryInterface,
    StdinQueryInterface,
    refine_instruction,
)
from .transparency import TransparencyConfig, render_message
from .uncertainty import (
    ComputeResult,
    ProbeResult,
    chunk_dispersion,
    twin_dispersion,
)
from .value_of_information import (
    CostModel,
    ValueParams,
    allocate,
    value_of_computation,
    value_of_information,
)

__all__ = [
    "ActComputeQueryAllocator",
    "AllocatorFit",
    "Decision",
    "Controller",
    "AutonomyController",
    "FixedComputeController",
    "ComputeGatedController",
    "KnowNoController",
    "InsightController",
    "ScaleController",
    "build_controller",
    "SplitConformal",
    "MondrianConformal",
    "empirical_coverage",
    "coverage_vs_bucket",
    "weighted_quantile",
    "QueryInterface",
    "QueryAnswer",
    "StdinQueryInterface",
    "CallbackQueryInterface",
    "ScriptedQueryInterface",
    "refine_instruction",
    "TransparencyConfig",
    "render_message",
    "ProbeResult",
    "ComputeResult",
    "chunk_dispersion",
    "twin_dispersion",
    "CostModel",
    "ValueParams",
    "allocate",
    "value_of_computation",
    "value_of_information",
]

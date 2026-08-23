"""The Tier-1 backend: a calibrated surrogate of the robot, and a simulated supervisor.

**Why a surrogate exists at all.** The scheduling study needs thousands of sessions -- a
cohort times conditions times seeds times the ablation grid -- and a closed-loop Isaac episode
costs a minute or two. Running the whole grid in the simulator is not a budget problem to be
solved with patience; it is a design mistake, because it spends compute re-measuring the same
execution outcomes over and over.

So the study runs in two tiers, and the paper reports both:

**Tier 1 (this module).** Execution outcomes are *sampled* from success and duration curves
that were **measured once** from Isaac rollouts of the scripted experts and the trained policy
across a margin sweep. The scheduling, belief, and estimation code paths are the real ones.
**Tier 2.** A smaller number of complete sessions run end-to-end in Isaac with the policy in
the loop, and the fidelity check compares the two.

The honest reading is that Tier 1 is a *variance-reduction* device, not a shortcut: it holds
the execution layer at its measured distribution so that the contrast between scheduling
conditions is not swamped by rollout noise. Its validity rests entirely on the fidelity check,
which is why that check is reported before the results that depend on it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .. import PROBE
from ..narration import ground
from ..scenes import SceneGrid, ScenePhysics, SceneSpec
from ..strategies import StrategyAxis, get_axis
from ..supervisor import SimulatedSupervisor
from .base import ExecutionOutcome, SupervisorTurn


class SurrogateApparatus:
    """Samples execution outcomes from the (ideally measured) scene physics."""

    def __init__(self, grid: SceneGrid, *, seed: int = 0, duration_sd_s: float = 3.0) -> None:
        self.grid = grid
        self.physics: ScenePhysics = grid.physics
        self.rng = random.Random(int(seed))
        self.duration_sd_s = float(duration_sd_s)
        self.n_executions = 0
        self.transcript: list = []

    def reset_scene(self, scene: SceneSpec) -> None:
        return None

    def execute(self, scene: SceneSpec, strategy: str) -> ExecutionOutcome:
        p = self.physics.p_success(strategy, scene.margin_m)
        d = max(1.0, self.rng.gauss(self.physics.duration_s(strategy, scene.margin_m), self.duration_sd_s))
        self.n_executions += 1
        return ExecutionOutcome(
            strategy=str(strategy),
            success=bool(self.rng.random() < p),
            duration_s=float(d),
            notes={"p_success": float(p), "physics_source": str(self.physics.source)},
        )

    def say(self, text: str) -> None:
        self.transcript.append(str(text))

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "surrogate",
            "physics": self.physics.to_dict(),
            "physics_source": str(self.physics.source),
            "n_measured": int(self.physics.n_measured),
        }

    def close(self) -> None:
        return None


class SimulatedSupervisorChannel:
    """Wraps a :class:`~vla_lab.supervisory.supervisor.SimulatedSupervisor` as a channel."""

    def __init__(self, supervisor: SimulatedSupervisor) -> None:
        self.sup = supervisor
        self.n_turns = 0

    def ask(self, query: str, scene: SceneSpec, *, action: str = PROBE, session_progress: float = 0.0) -> SupervisorTurn:
        r = self.sup.respond(scene, action=action, session_progress=session_progress)
        self.n_turns += 1
        return SupervisorTurn(
            utterance=r.utterance,
            latency_s=r.latency_s,
            truth={
                "intended_strategy": r.strategy,
                "p_a": r.p_a,
                "pi_star": r.pi_star,
                "kappa_eff": r.kappa_eff,
                "lapsed": r.lapsed,
                "contaminated_shift": r.contaminated_shift,
            },
        )

    def observe_demonstration(self, scene: SceneSpec, strategy: str, narration: str, *, strength: float = 1.0) -> None:
        self.sup.apply_coach(+1 if str(strategy) == "A" else -1, strength=float(strength))

    def elapse(self, delta: float) -> None:
        self.sup.decay(float(delta))

    def describe(self) -> Dict[str, Any]:
        return {"channel": "simulated", **self.sup.describe()}


class LexicalGrounder:
    """The reference grounder: conservative keyword matching, never a guess."""

    name = "lexical"

    def __init__(self, axis: str) -> None:
        self.axis: StrategyAxis = get_axis(axis)

    def ground(self, utterance: str, scene: SceneSpec) -> str:
        return ground(utterance, self.axis)


__all__ = ["SurrogateApparatus", "SimulatedSupervisorChannel", "LexicalGrounder"]

"""Glue between the Isaac eval loop and the act/compute/query controllers.

Keeps :mod:`vla_lab.eval_isaaclab` edits small: it builds the right
:class:`ComputeBranch` over the already-loaded TinyVLA / SmolVLA policy, constructs the
selected controller (baseline or allocator) with its fit + transparency + query interface,
runs one decision per policy tick, and accumulates a per-run aggregate. In sim the query
interface is a :class:`ScriptedQueryInterface` backed by the episode's true target, so query
rate and post-query success are measurable without a person; on the real robot the autonomous
:class:`StdinQueryInterface` consumes a typed/clicked answer.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .allocator import AllocatorFit, Decision
from .baselines import build_controller
from .query import ScriptedQueryInterface, StdinQueryInterface
from .transparency import TransparencyConfig


def build_compute_branch(
    policy_backend: str,
    *,
    tiny: Optional[Dict[str, Any]] = None,
    smol: Any = None,
    device: Any = None,
    ttc_cfg: Optional[Dict[str, Any]] = None,
):
    """Construct a ComputeBranch from the eval's loaded policy objects (torch imported here)."""

    from .policy_adapters import SmolVLAComputeBranch, TTCComputeBranch

    ttc_cfg = dict(ttc_cfg or {})
    selection = str(ttc_cfg.get("selection", ttc_cfg.get("scoring", "naive_consensus")))
    if policy_backend == "tiny":
        if not tiny:
            raise ValueError("tiny backend requires the tiny={model,tokenizer,action_stats} dict")
        return TTCComputeBranch(
            model=tiny["model"],
            tokenizer=tiny["tokenizer"],
            device=device,
            action_stats=tiny.get("action_stats"),
            noise_std=float(ttc_cfg.get("noise_std", 0.1)),
            gating_noise_std=float(ttc_cfg.get("gating_noise_std", 0.05)),
            selection=selection,
        )
    if policy_backend == "smolvla":
        if smol is None:
            raise ValueError("smolvla backend requires the loaded SmolVLAIsaacPolicy")
        return SmolVLAComputeBranch(smol, selection=selection)
    raise ValueError(f"unknown policy_backend {policy_backend!r}")


def build_query_interface(alloc_cfg: Dict[str, Any]):
    """Scripted oracle (sim default) or autonomous stdin (real)."""

    mode = str(alloc_cfg.get("query_interface", "scripted")).lower()
    if mode in ("stdin", "console", "autonomous", "human"):
        return StdinQueryInterface()
    return ScriptedQueryInterface(
        oracle_answer="",
        accuracy=float(alloc_cfg.get("query_accuracy", 1.0)),
        sim_latency_s=float(alloc_cfg.get("query_latency_s", 3.0)),
    )


class EvalControllerHarness:
    """Owns the controller + query interface; one ``decide`` per policy tick + an aggregate."""

    def __init__(self, controller, query_interface, name: str) -> None:
        self.controller = controller
        self.query_interface = query_interface
        self.name = name
        self.decisions: List[Decision] = []

    def set_oracle(self, answer: Optional[str]) -> None:
        if hasattr(self.query_interface, "oracle_answer") and answer is not None:
            self.query_interface.oracle_answer = str(answer)

    def decide(self, obs: Dict[str, Any], *, instruction: str, options=None, oracle_answer=None) -> Decision:
        d = self.controller.decide(obs, instruction=instruction, options=options, oracle_answer=oracle_answer)
        self.decisions.append(d)
        return d

    def aggregate(self) -> Dict[str, Any]:
        ds = self.decisions
        if not ds:
            return {}
        n = len(ds)
        actions = [d.action for d in ds]
        ks = [int(d.k_used) for d in ds]
        lat = [float(d.latency_ms) for d in ds]
        disp = [float(d.dispersion) for d in ds]
        return {
            "controller": self.name,
            "num_policy_queries": n,
            "frac_act": actions.count("act") / n,
            "frac_compute": actions.count("compute") / n,
            "frac_query": actions.count("query") / n,
            "query_rate": sum(1 for d in ds if d.queried) / n,
            "k_mean": float(statistics.mean(ks)),
            "k_median": float(statistics.median(ks)),
            "latency_median_ms": float(statistics.median(lat)),
            "dispersion_median": float(statistics.median(disp)),
            "num_queries": int(sum(1 for d in ds if d.queried)),
            "query_answer_accuracy": (
                float(statistics.mean([1.0 if d.query_answer_correct else 0.0 for d in ds if d.queried]))
                if any(d.queried for d in ds)
                else None
            ),
        }


def build_eval_harness(
    *,
    controller_kind: str,
    policy_backend: str,
    tiny: Optional[Dict[str, Any]] = None,
    smol: Any = None,
    device: Any = None,
    ttc_cfg: Optional[Dict[str, Any]] = None,
    alloc_cfg: Optional[Dict[str, Any]] = None,
    fit_path: Optional[str] = None,
    query_interface: Any = None,
) -> EvalControllerHarness:
    """One-call construction used by ``eval_isaaclab.py``.

    ``query_interface`` overrides the alloc_cfg-built one — used when a
    feedback channel is active so robot-initiated queries and unsolicited
    corrections come from the SAME (simulated or live) human.
    """

    alloc_cfg = dict(alloc_cfg or {})
    cb = build_compute_branch(policy_backend, tiny=tiny, smol=smol, device=device, ttc_cfg=ttc_cfg)
    qi = query_interface if query_interface is not None else build_query_interface(alloc_cfg)
    fit = AllocatorFit.load(fit_path) if fit_path else AllocatorFit.default()
    transp = TransparencyConfig.from_dict(alloc_cfg.get("transparency", {})) if alloc_cfg.get("transparency") else TransparencyConfig(enabled=False)
    controller = build_controller(
        controller_kind, compute_branch=cb, query_interface=qi, fit=fit, transparency=transp
    )
    return EvalControllerHarness(controller, qi, name=str(controller_kind))

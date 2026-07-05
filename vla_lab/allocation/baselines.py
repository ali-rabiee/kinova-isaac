"""Baseline controllers (memo §5.5) sharing the allocator's :class:`Controller` interface.

These let the evaluation compare every method on identical scenes/seeds:

- :class:`AutonomyController` — K=1, never asks (the "act-only" anchor).
- :class:`FixedComputeController` — best-of-N, never asks (prior-art fixed TTC).
- :class:`ComputeGatedController` — the project's existing twin-probe gate (act/compute).
- :class:`ScaleController` — SCALE-style autonomous self-uncertainty adaptive compute
  (K scales with dispersion); never asks.
- :class:`KnowNoController` — KnowNo-style **binary** conformal ask-for-help (act/ask),
  no compute branch.
- :class:`InsightController` — INSIGHT-style learned help trigger from uncertainty
  features (act/ask), no compute branch.

The full act/compute/query :class:`ActComputeQueryAllocator` lives in ``allocator.py`` and
is selected via :func:`build_controller` with ``kind="allocator"``.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from .allocator import ActComputeQueryAllocator, AllocatorFit, Controller, Decision, formulate_question
from .conformal import SplitConformal, bucketize
from .linear_model import LogisticModel
from .query import QueryAnswer, QueryInterface, refine_instruction
from .transparency import TransparencyConfig, render_message, uncertainty_type_for
from .value_of_information import ACT, COMPUTE, QUERY


def _execute_query(
    cb,
    obs: Any,
    query_interface: QueryInterface,
    instruction: str,
    options: Optional[Sequence[str]],
    oracle_answer: Optional[str],
    question: Optional[str] = None,
):
    """Ask, fold the answer into the instruction, then take one forward on the refined task."""

    default_q, opts = formulate_question(instruction, options)
    question = question or default_q
    ans: QueryAnswer = query_interface.ask(question, opts)
    refined = refine_instruction(instruction, ans)
    res = cb.act(obs, refined)
    correct = (
        bool(str(ans.text).lower() == str(oracle_answer).lower()) if oracle_answer is not None else ans.correct
    )
    return res, question, ans, correct


class AutonomyController(Controller):
    """K=1, never asks."""

    def __init__(self, compute_branch, transparency: Optional[TransparencyConfig] = None, name: str = "autonomy") -> None:
        self.cb = compute_branch
        self.transparency = transparency or TransparencyConfig(enabled=False)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        res = self.cb.act(obs, instruction)
        return Decision(
            action=ACT, chunk=res.chunk, dispersion=0.0, k_used=1,
            transparency_message=render_message(ACT, self.transparency),
            uncertainty_type="none", latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


class FixedComputeController(Controller):
    """Best-of-K every step, never asks."""

    def __init__(self, compute_branch, k: int = 4, transparency: Optional[TransparencyConfig] = None, name: str = "fixed_compute") -> None:
        self.cb = compute_branch
        self.k = max(1, int(k))
        self.transparency = transparency or TransparencyConfig(enabled=False)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        res = self.cb.compute(obs, instruction, self.k)
        return Decision(
            action=COMPUTE, chunk=res.chunk, dispersion=float(res.dispersion), k_used=int(res.k_used),
            transparency_message=render_message(COMPUTE, self.transparency), uncertainty_type="reducible",
            latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


class ComputeGatedController(Controller):
    """Twin-probe gate: act when confident, else best-of-K. Never asks (the existing method)."""

    def __init__(self, compute_branch, tau: float = 0.02, k: int = 4, transparency: Optional[TransparencyConfig] = None, name: str = "compute_gated") -> None:
        self.cb = compute_branch
        self.tau = float(tau)
        self.k = max(1, int(k))
        self.transparency = transparency or TransparencyConfig(enabled=False)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        probe = self.cb.probe(obs, instruction)
        if probe.dispersion <= self.tau:
            res = self.cb.act(obs, instruction)
            action, k_used, disp = ACT, 1, float(probe.dispersion)
        else:
            res = self.cb.compute(obs, instruction, self.k)
            action, k_used, disp = COMPUTE, int(res.k_used), float(res.dispersion)
        return Decision(
            action=action, chunk=res.chunk, dispersion=disp, k_used=k_used,
            transparency_message=render_message(action, self.transparency),
            uncertainty_type=uncertainty_type_for(action), latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


class ScaleController(Controller):
    """SCALE-style: autonomous, self-uncertainty-conditioned adaptive K. Never asks."""

    def __init__(self, compute_branch, tau: float = 0.02, k_min: int = 1, k_max: int = 8, disp_scale: float = 0.1, transparency: Optional[TransparencyConfig] = None, name: str = "scale") -> None:
        self.cb = compute_branch
        self.tau = float(tau)
        self.k_min = max(1, int(k_min))
        self.k_max = max(self.k_min, int(k_max))
        self.disp_scale = max(1e-6, float(disp_scale))
        self.transparency = transparency or TransparencyConfig(enabled=False)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        probe = self.cb.probe(obs, instruction)
        if probe.dispersion <= self.tau:
            res = self.cb.act(obs, instruction)
            return Decision(
                action=ACT, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1,
                transparency_message=render_message(ACT, self.transparency), uncertainty_type="none",
                latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
            )
        frac = min(1.0, float(probe.dispersion) / self.disp_scale)
        k_eff = int(round(self.k_min + frac * (self.k_max - self.k_min)))
        k_eff = max(self.k_min, min(self.k_max, k_eff))
        res = self.cb.compute(obs, instruction, k_eff)
        return Decision(
            action=COMPUTE, chunk=res.chunk, dispersion=float(res.dispersion), k_used=int(res.k_used),
            transparency_message=render_message(COMPUTE, self.transparency), uncertainty_type="reducible",
            latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


class KnowNoController(Controller):
    """KnowNo-style binary conformal ask-for-help (act/ask); no compute branch."""

    def __init__(self, compute_branch, query_interface: QueryInterface, conformal: SplitConformal, score_kind: str = "dispersion", transparency: Optional[TransparencyConfig] = None, name: str = "knowno") -> None:
        self.cb = compute_branch
        self.query = query_interface
        self.conformal = conformal
        self.score_kind = score_kind
        self.transparency = transparency or TransparencyConfig(enabled=True, signal_type=False)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        probe = self.cb.probe(obs, instruction)
        feats = dict(probe.features)
        score = float(feats.get("dispersion", probe.dispersion)) if self.score_kind == "dispersion" else float(feats.get("irreducible_score", probe.dispersion))
        ask = self.conformal.is_anomalous(score)
        if not ask:
            res = self.cb.act(obs, instruction)
            return Decision(
                action=ACT, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1,
                nonconformity_score=score, conformal_anomaly=False,
                transparency_message=render_message(ACT, self.transparency), uncertainty_type="none",
                latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
            )
        res, question, ans, correct = _execute_query(self.cb, obs, self.query, instruction, options, oracle_answer)
        return Decision(
            action=QUERY, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1, queried=True,
            query_question=question, query_answer=ans.text, query_answer_correct=correct, query_latency_s=float(ans.latency_s),
            nonconformity_score=score, conformal_anomaly=True,
            transparency_message=render_message(QUERY, self.transparency), uncertainty_type="irreducible",
            latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


class InsightController(Controller):
    """INSIGHT-style learned help trigger on uncertainty features (act/ask); no compute."""

    def __init__(self, compute_branch, query_interface: QueryInterface, model: LogisticModel, threshold: float = 0.5, transparency: Optional[TransparencyConfig] = None, name: str = "insight") -> None:
        self.cb = compute_branch
        self.query = query_interface
        self.model = model
        self.threshold = float(threshold)
        self.transparency = transparency or TransparencyConfig(enabled=True, signal_type=False)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        probe = self.cb.probe(obs, instruction)
        feats = dict(probe.features)
        feats.setdefault("dispersion", probe.dispersion)
        p = self.model.predict_proba_features(feats) if self.model.fitted else float(feats.get("occlusion_fraction", 0.0))
        ask = bool(p >= self.threshold)
        if not ask:
            res = self.cb.act(obs, instruction)
            return Decision(
                action=ACT, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1,
                irreducible_score=float(p), transparency_message=render_message(ACT, self.transparency),
                uncertainty_type="none", latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
            )
        res, question, ans, correct = _execute_query(self.cb, obs, self.query, instruction, options, oracle_answer)
        return Decision(
            action=QUERY, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1, queried=True,
            query_question=question, query_answer=ans.text, query_answer_correct=correct, query_latency_s=float(ans.latency_s),
            irreducible_score=float(p), transparency_message=render_message(QUERY, self.transparency),
            uncertainty_type="irreducible", latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


class IntentRoutedController(Controller):
    """Route by uncertainty SOURCE (2026-07 intent/perception decomposition).

    Uses the per-tick features the eval loop injects into ``obs`` (see
    ``vla_lab.intent``): ``intent_entropy`` (which goal does the human mean?)
    and ``cross_view_disagreement`` (do the overhead and wrist views agree?).

    Priority:
      1. intent uncertainty high  -> QUERY with a goal-clarification question
         (compute cannot resolve WHICH object is meant — only the human can);
      2. perception uncertainty high (probe dispersion or cross-view
         disagreement) -> COMPUTE best-of-K (reducible with samples/views);
      3. both low -> ACT.
    """

    def __init__(
        self,
        compute_branch,
        query_interface: QueryInterface,
        *,
        tau_intent: float = 0.5,
        tau_disp: float = 0.02,
        tau_view: float = 0.08,
        k: int = 4,
        transparency: Optional[TransparencyConfig] = None,
        name: str = "intent_allocator",
    ) -> None:
        self.cb = compute_branch
        self.query = query_interface
        self.tau_intent = float(tau_intent)
        self.tau_disp = float(tau_disp)
        self.tau_view = float(tau_view)
        self.k = max(1, int(k))
        self.transparency = transparency or TransparencyConfig(enabled=True, signal_type=True)
        self.name = name

    def decide(self, obs: Any, *, instruction: str, options=None, oracle_answer=None) -> Decision:
        t0 = time.time()
        probe = self.cb.probe(obs, instruction)
        feats = dict(probe.features)
        intent_u = float(feats.get("intent_entropy", 0.0))
        view_dis = float(feats.get("cross_view_disagreement", 0.0))

        if intent_u >= self.tau_intent:
            question = "I'm not sure which object you mean. Which one should I pick up?"
            res, question, ans, correct = _execute_query(
                self.cb, obs, self.query, instruction, options, oracle_answer, question=question
            )
            return Decision(
                action=QUERY, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1, queried=True,
                query_question=question, query_answer=ans.text, query_answer_correct=correct,
                query_latency_s=float(ans.latency_s), irreducible_score=intent_u,
                transparency_message=render_message(QUERY, self.transparency),
                uncertainty_type="intent", latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
            )

        if float(probe.dispersion) > self.tau_disp or view_dis > self.tau_view:
            res = self.cb.compute(obs, instruction, self.k)
            return Decision(
                action=COMPUTE, chunk=res.chunk, dispersion=float(res.dispersion), k_used=int(res.k_used),
                transparency_message=render_message(COMPUTE, self.transparency),
                uncertainty_type="perception", latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
            )

        res = self.cb.act(obs, instruction)
        return Decision(
            action=ACT, chunk=res.chunk, dispersion=float(probe.dispersion), k_used=1,
            transparency_message=render_message(ACT, self.transparency),
            uncertainty_type="none", latency_ms=(time.time() - t0) * 1000.0, controller=self.name,
        )


def build_controller(
    kind: str,
    *,
    compute_branch,
    query_interface: Optional[QueryInterface] = None,
    fit: Optional[AllocatorFit] = None,
    transparency: Optional[TransparencyConfig] = None,
    **overrides,
) -> Controller:
    """Factory: map a string ``kind`` to a configured controller.

    ``kind`` ∈ {autonomy, fixed_compute, compute_gated, scale, knowno, insight, allocator,
    intent_allocator}. KnowNo / INSIGHT / allocator / intent_allocator require a
    ``query_interface``; KnowNo/INSIGHT and the allocator read their conformal / trigger
    model from ``fit``.
    """

    k = str(kind).lower().strip().replace("-", "_")
    fit = fit or AllocatorFit.default()
    if k == "autonomy":
        return AutonomyController(compute_branch, transparency=transparency)
    if k in ("fixed_compute", "fixed", "best_of_n", "bon"):
        return FixedComputeController(compute_branch, k=int(overrides.get("k", fit.k_compute)), transparency=transparency)
    if k in ("compute_gated", "gated"):
        return ComputeGatedController(
            compute_branch, tau=float(overrides.get("tau", fit.value_params.act_dispersion_tau)),
            k=int(overrides.get("k", fit.k_compute)), transparency=transparency,
        )
    if k == "scale":
        return ScaleController(
            compute_branch, tau=float(overrides.get("tau", fit.value_params.act_dispersion_tau)),
            k_min=int(overrides.get("k_min", 1)), k_max=int(overrides.get("k_max", fit.k_compute * 2)),
            disp_scale=float(overrides.get("disp_scale", 0.1)), transparency=transparency,
        )
    if k == "knowno":
        if query_interface is None:
            raise ValueError("knowno requires a query_interface")
        conf = fit.knowno_conformal
        if conf is None:
            conf = fit.conformal if isinstance(fit.conformal, SplitConformal) else SplitConformal(alpha=float(overrides.get("alpha", 0.2)))
        return KnowNoController(compute_branch, query_interface, conformal=conf, score_kind=str(overrides.get("score_kind", "dispersion")), transparency=transparency)
    if k == "insight":
        if query_interface is None:
            raise ValueError("insight requires a query_interface")
        return InsightController(compute_branch, query_interface, model=fit.irreducibility, threshold=float(overrides.get("threshold", 0.5)), transparency=transparency)
    if k == "allocator":
        if query_interface is None:
            raise ValueError("allocator requires a query_interface")
        return ActComputeQueryAllocator(compute_branch, query_interface, fit=fit, transparency=transparency)
    if k in ("intent_allocator", "intent_routed", "intent"):
        if query_interface is None:
            raise ValueError("intent_allocator requires a query_interface")
        return IntentRoutedController(
            compute_branch,
            query_interface,
            tau_intent=float(overrides.get("tau_intent", 0.5)),
            tau_disp=float(overrides.get("tau_disp", fit.value_params.act_dispersion_tau)),
            tau_view=float(overrides.get("tau_view", 0.08)),
            k=int(overrides.get("k", fit.k_compute)),
            transparency=transparency,
        )
    raise ValueError(f"unknown controller kind: {kind!r}")

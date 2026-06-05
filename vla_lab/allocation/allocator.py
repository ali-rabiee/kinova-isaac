"""The three-way act / compute / query allocator and its fitted parameters.

This is contribution **C3**: a calibrated compute-or-query controller for any stochastic
VLA. It composes the building blocks in this package:

- a :class:`ComputeBranch` (the policy adapter; ``act`` / ``probe`` / ``compute``),
- a fitted **irreducibility detector** (``linear_model.LogisticModel``) estimating
  ``P(compute is useless | uncertainty features)``,
- a **conformal escalation rule** (``conformal``) on the chosen nonconformity score, and
- a **VoC vs. VoI** value rule (``value_of_information``),

and emits a rich :class:`Decision` record per step for the calibration / reliance analyses.

Decision priority (documented and faithful to the memo):

1. ``dispersion <= act_tau`` and the conformal detector does **not** flag the step
   → **act** (confident and in-distribution).
2. conformal **anomaly** and querying is worth its cost (``VoI > 0``)
   → **query** (the calibrated "compute may be useless" escalation).
3. ``VoC > 0`` and ``dispersion > act_tau`` → **compute** (reducible ambiguity).
4. otherwise → **act**.

Out of the box (``AllocatorFit.default()``, no calibration) the query branch stays closed
and the allocator behaves like compute-gated TTC; **fitting** (``vla_lab.fit_allocator``)
on a calibration log is what unlocks calibrated querying.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .conformal import MondrianConformal, SplitConformal, bucketize
from .linear_model import LogisticModel
from .query import QueryAnswer, QueryInterface, refine_instruction
from .transparency import TransparencyConfig, render_message, uncertainty_type_for
from .uncertainty import ComputeResult, ProbeResult
from .value_of_information import ACT, COMPUTE, QUERY, CostModel, ValueParams, allocate

# Default occlusion-fraction bin edges (mask-stratified conformal / bucketing).
DEFAULT_BUCKET_EDGES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]


class ComputeBranch(Protocol):
    """Policy adapter contract. ``obs`` is opaque to the allocator core.

    Implementations: :class:`vla_lab.allocation.policy_adapters.TTCComputeBranch` (TinyVLA),
    ``SmolVLAComputeBranch`` (SmolVLA), and the test mock.
    """

    def act(self, obs: Any, instruction: str) -> ComputeResult:
        """Single deterministic forward (K=1)."""

    def probe(self, obs: Any, instruction: str) -> ProbeResult:
        """Cheap two-forward dispersion probe + the act-candidate + features."""

    def compute(self, obs: Any, instruction: str, k: int) -> ComputeResult:
        """Best-of-K sampling + selection."""


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """Everything one controller step produced (the chunk plus a JSON-able trace)."""

    action: str
    chunk: Any = None
    dispersion: float = 0.0
    irreducible_score: float = 0.0
    k_used: int = 1
    queried: bool = False
    query_question: Optional[str] = None
    query_answer: Optional[str] = None
    query_answer_correct: Optional[bool] = None
    query_latency_s: float = 0.0
    transparency_message: str = ""
    uncertainty_type: str = "none"
    conformal_anomaly: bool = False
    nonconformity_score: float = 0.0
    occlusion_fraction: float = 0.0
    values: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    controller: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """JSON-safe dict (drops the opaque ``chunk``)."""

        return {
            "action": self.action,
            "dispersion": float(self.dispersion),
            "irreducible_score": float(self.irreducible_score),
            "k_used": int(self.k_used),
            "queried": bool(self.queried),
            "query_question": self.query_question,
            "query_answer": self.query_answer,
            "query_answer_correct": self.query_answer_correct,
            "query_latency_s": float(self.query_latency_s),
            "transparency_message": self.transparency_message,
            "uncertainty_type": self.uncertainty_type,
            "conformal_anomaly": bool(self.conformal_anomaly),
            "nonconformity_score": float(self.nonconformity_score),
            "occlusion_fraction": float(self.occlusion_fraction),
            "values": {k: float(v) for k, v in (self.values or {}).items()},
            "latency_ms": float(self.latency_ms),
            "controller": self.controller,
            **{f"extra_{k}": v for k, v in (self.extra or {}).items()},
        }


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------


class Controller:
    """Common interface for every act/compute/query and baseline controller."""

    name: str = "base"

    def reset(self) -> None:  # pragma: no cover - default no-op
        pass

    def decide(
        self,
        obs: Any,
        *,
        instruction: str,
        options: Optional[Sequence[str]] = None,
        oracle_answer: Optional[str] = None,
    ) -> Decision:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------

ConformalT = Union[SplitConformal, MondrianConformal, None]


@dataclass
class AllocatorFit:
    """Serializable parameters produced by ``vla_lab.fit_allocator`` and read at deploy."""

    value_params: ValueParams = field(default_factory=ValueParams)
    cost: CostModel = field(default_factory=CostModel)
    irreducibility: LogisticModel = field(default_factory=LogisticModel)
    conformal: ConformalT = None
    conformal_kind: str = "none"  # "split" | "mondrian" | "none"
    # Dedicated split-conformal detector on *dispersion* for the KnowNo baseline (act/ask).
    knowno_conformal: Optional[SplitConformal] = None
    score_kind: str = "irreducibility"  # nonconformity score: "irreducibility" | "dispersion"
    occlusion_bucket_edges: List[float] = field(default_factory=lambda: list(DEFAULT_BUCKET_EDGES))
    k_compute: int = 4
    k_post_query: int = 1
    default_irreducible: float = 0.0
    allow_voi_query: bool = True

    @classmethod
    def default(cls) -> "AllocatorFit":
        """Heuristic, uncalibrated fit — runs out of the box, query branch effectively off."""

        return cls()

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        conf = self.conformal.to_dict() if self.conformal is not None else None
        return {
            "schema": "vla_lab_allocator_fit/v1",
            "value_params": self.value_params.to_dict(),
            "cost": self.cost.to_dict(),
            "irreducibility": self.irreducibility.to_dict(),
            "conformal": conf,
            "conformal_kind": self.conformal_kind,
            "knowno_conformal": self.knowno_conformal.to_dict() if self.knowno_conformal is not None else None,
            "score_kind": self.score_kind,
            "occlusion_bucket_edges": [float(x) for x in self.occlusion_bucket_edges],
            "k_compute": int(self.k_compute),
            "k_post_query": int(self.k_post_query),
            "default_irreducible": float(self.default_irreducible),
            "allow_voi_query": bool(self.allow_voi_query),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AllocatorFit":
        kind = str(d.get("conformal_kind", "none"))
        conf: ConformalT = None
        cd = d.get("conformal")
        if cd is not None and kind == "split":
            conf = SplitConformal.from_dict(cd)
        elif cd is not None and kind == "mondrian":
            conf = MondrianConformal.from_dict(cd)
        kc = d.get("knowno_conformal")
        knowno_conf = SplitConformal.from_dict(kc) if kc is not None else None
        return cls(
            value_params=ValueParams.from_dict(d.get("value_params", {})),
            cost=CostModel.from_dict(d.get("cost", {})),
            irreducibility=LogisticModel.from_dict(d.get("irreducibility", {})),
            conformal=conf,
            conformal_kind=kind,
            knowno_conformal=knowno_conf,
            score_kind=str(d.get("score_kind", "irreducibility")),
            occlusion_bucket_edges=[float(x) for x in d.get("occlusion_bucket_edges", DEFAULT_BUCKET_EDGES)],
            k_compute=int(d.get("k_compute", 4)),
            k_post_query=int(d.get("k_post_query", 1)),
            default_irreducible=float(d.get("default_irreducible", 0.0)),
            allow_voi_query=bool(d.get("allow_voi_query", True)),
        )

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AllocatorFit":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Irreducibility estimator (model + heuristic fallback)
# ---------------------------------------------------------------------------


class IrreducibilityEstimator:
    """``P(optimal action unidentifiable from this view)`` from uncertainty features.

    Uses the fitted logistic model when available; otherwise falls back to a documented
    heuristic (the known occlusion fraction in sim, or a fixed prior) so the allocator is
    still runnable before any calibration data exists.
    """

    def __init__(self, model: LogisticModel, default_irreducible: float = 0.0) -> None:
        self.model = model
        self.default = float(default_irreducible)

    def predict(self, features: Dict[str, float]) -> float:
        if self.model is not None and self.model.fitted:
            p = self.model.predict_proba_features(features)
            if p == p:  # not NaN
                return float(min(max(p, 0.0), 1.0))
        if "occlusion_fraction" in features:
            return float(min(max(features["occlusion_fraction"], 0.0), 1.0))
        return float(min(max(self.default, 0.0), 1.0))


# ---------------------------------------------------------------------------
# The allocator
# ---------------------------------------------------------------------------


def formulate_question(instruction: str, options: Optional[Sequence[str]]) -> Tuple[str, Optional[List[str]]]:
    """Phrase the human query (and the option list, if any)."""

    if options and len(list(options)) > 1:
        return ("I can't tell which object you mean — which one should I pick?", [str(o) for o in options])
    return ("I can't see the target clearly — what should I pick up?", None)


class ActComputeQueryAllocator(Controller):
    """The calibrated act/compute/query controller (C3)."""

    def __init__(
        self,
        compute_branch: ComputeBranch,
        query_interface: QueryInterface,
        fit: Optional[AllocatorFit] = None,
        transparency: Optional[TransparencyConfig] = None,
        name: str = "allocator",
    ) -> None:
        self.cb = compute_branch
        self.query = query_interface
        self.fit = fit or AllocatorFit.default()
        self.transparency = transparency or TransparencyConfig()
        self.name = name
        self._estimator = IrreducibilityEstimator(self.fit.irreducibility, self.fit.default_irreducible)

    # -- helpers -----------------------------------------------------------
    def _conformal_anomaly(self, score: float, occ_frac: float) -> bool:
        conf = self.fit.conformal
        if conf is None:
            return False
        if isinstance(conf, MondrianConformal):
            bucket = bucketize(occ_frac, self.fit.occlusion_bucket_edges)
            return conf.is_anomalous(score, bucket)
        return conf.is_anomalous(score)

    # -- main --------------------------------------------------------------
    def decide(
        self,
        obs: Any,
        *,
        instruction: str,
        options: Optional[Sequence[str]] = None,
        oracle_answer: Optional[str] = None,
    ) -> Decision:
        t0 = time.time()
        vp = self.fit.value_params
        cost = self.fit.cost

        probe = self.cb.probe(obs, instruction)
        disp = float(probe.dispersion)
        feats: Dict[str, float] = dict(probe.features)
        feats.setdefault("dispersion", disp)
        occ_frac = float(feats.get("occlusion_fraction", 0.0))

        irr = self._estimator.predict(feats)
        score = irr if self.fit.score_kind == "irreducibility" else disp
        anomaly = self._conformal_anomaly(score, occ_frac)

        _, values = allocate(disp, irr, self.fit.k_compute, vp, cost)
        voi = values["query"]
        voc = values["compute"]

        # Decision priority (see module docstring).
        if disp <= vp.act_dispersion_tau and not anomaly:
            final = ACT
        elif anomaly and voi > 0.0:
            final = QUERY
        elif self.fit.allow_voi_query and voi > 0.0 and voi >= voc:
            final = QUERY
        elif voc > 0.0 and disp > vp.act_dispersion_tau:
            final = COMPUTE
        else:
            final = ACT

        # Execute the chosen branch.
        queried = False
        q_question: Optional[str] = None
        q_answer: Optional[str] = None
        q_correct: Optional[bool] = None
        q_latency = 0.0

        if final == ACT:
            res = self.cb.act(obs, instruction)
            chunk, k_used, disp_final = res.chunk, 1, disp
        elif final == COMPUTE:
            res = self.cb.compute(obs, instruction, int(self.fit.k_compute))
            chunk, k_used, disp_final = res.chunk, int(res.k_used), float(res.dispersion)
        else:  # QUERY
            q_question, opts = formulate_question(instruction, options)
            ans: QueryAnswer = self.query.ask(q_question, opts)
            refined = refine_instruction(instruction, ans)
            res = self.cb.compute(obs, refined, int(self.fit.k_post_query)) if self.fit.k_post_query > 1 else self.cb.act(obs, refined)
            chunk, k_used, disp_final = res.chunk, int(getattr(res, "k_used", 1)), disp
            queried = True
            q_answer = ans.text
            q_latency = float(ans.latency_s)
            if oracle_answer is not None:
                q_correct = bool(str(ans.text).lower() == str(oracle_answer).lower())
            else:
                q_correct = ans.correct

        msg = render_message(final, self.transparency)
        return Decision(
            action=final,
            chunk=chunk,
            dispersion=disp_final,
            irreducible_score=irr,
            k_used=k_used,
            queried=queried,
            query_question=q_question,
            query_answer=q_answer,
            query_answer_correct=q_correct,
            query_latency_s=q_latency,
            transparency_message=msg,
            uncertainty_type=uncertainty_type_for(final),
            conformal_anomaly=bool(anomaly),
            nonconformity_score=float(score),
            occlusion_fraction=occ_frac,
            values=values,
            latency_ms=(time.time() - t0) * 1000.0,
            controller=self.name,
        )

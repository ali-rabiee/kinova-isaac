"""Shared test doubles — the "test mock" the allocator's ``ComputeBranch`` doc refers to.

A :class:`MockComputeBranch` implements the allocator's policy-adapter contract
(``act`` / ``probe`` / ``compute``) with **no torch**, returning ``(T, A)`` NumPy chunks and a
controllable dispersion, so the act/compute/query decision can be driven deterministically.
Helpers build synthetic calibration records (with the §4.2 "confidently-wrong" structure)
and a fitted occlusion-aware allocator.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

import numpy as np

from vla_lab.allocation.uncertainty import ComputeResult, ProbeResult


class MockComputeBranch:
    """Deterministic, torch-free :class:`~vla_lab.allocation.allocator.ComputeBranch`.

    Dispersion is read from ``obs["mock_dispersion"]`` if present, else derived from
    ``obs["occlusion_fraction"]`` (more occlusion → more disagreement). ``calls`` counts how
    many times each branch fired (so tests can assert which path the controller took).
    """

    def __init__(self, T: int = 8, A: int = 7, dispersion: Optional[float] = None) -> None:
        self.T, self.A = int(T), int(A)
        self._fixed = dispersion
        self.calls = {"act": 0, "probe": 0, "compute": 0}

    def _disp(self, obs: Any) -> float:
        if self._fixed is not None:
            return float(self._fixed)
        if isinstance(obs, dict) and "mock_dispersion" in obs:
            return float(obs["mock_dispersion"])
        occ = float(obs.get("occlusion_fraction", 0.0)) if isinstance(obs, dict) else 0.0
        return 0.005 + occ

    def _occ(self, obs: Any) -> float:
        return float(obs.get("occlusion_fraction", 0.0)) if isinstance(obs, dict) else 0.0

    def _chunk(self, seed: int) -> np.ndarray:
        return np.random.default_rng(int(seed)).normal(size=(self.T, self.A))

    def act(self, obs: Any, instruction: str) -> ComputeResult:
        self.calls["act"] += 1
        return ComputeResult(chunk=self._chunk(0), dispersion=0.0, k_used=1)

    def probe(self, obs: Any, instruction: str) -> ProbeResult:
        self.calls["probe"] += 1
        d = self._disp(obs)
        feats = {"dispersion": d, "occlusion_fraction": self._occ(obs)}
        # Mirror policy_adapters._features: intent/perception decomposition keys
        # flow from obs into the probe features (2026-07 intent axis).
        if isinstance(obs, dict):
            for key in ("intent_entropy", "intent_top2_margin", "cross_view_disagreement", "instruction_ambiguous"):
                if obs.get(key) is not None:
                    feats[key] = float(obs[key])
        return ProbeResult(
            dispersion=d,
            act_chunk=self._chunk(0),
            features=feats,
        )

    def compute(self, obs: Any, instruction: str, k: int) -> ComputeResult:
        self.calls["compute"] += 1
        return ComputeResult(chunk=self._chunk(1), dispersion=self._disp(obs), k_used=int(k))


def synthetic_records(n: int = 400, seed: int = 0, *, confidently_wrong: bool = True) -> List[Any]:
    """Build ``n`` :class:`CalibrationRecord`s with the §4.2 structure.

    Correctness drops with occlusion (the optimal action stops being identifiable). When
    ``confidently_wrong`` the dispersion stays low regardless of occlusion/correctness — the
    decoupling regime where self-agreement is *not* a correctness signal. Otherwise dispersion
    grows with occlusion (the coupled regime), as a contrast.
    """

    from vla_lab.calibration.records import CalibrationRecord

    rng = random.Random(int(seed))
    recs: List[CalibrationRecord] = []
    for i in range(int(n)):
        occ = rng.choice([0.0, 0.15, 0.25, 0.35, 0.5])
        identifiable = occ < 0.3
        if confidently_wrong:
            disp = abs(rng.gauss(0.01, 0.004))
        else:
            disp = abs(rng.gauss(0.01 + occ * 0.12, 0.004))
        correct = rng.random() < (0.9 if identifiable else 0.3)
        recs.append(
            CalibrationRecord(
                run_id="test",
                episode_idx=i,
                occlusion_mode="bottom_strip",
                occlusion_fraction=occ,
                dispersion=disp,
                nonconformity_score=disp,
                correct=correct,
                identifiable=identifiable,
                success=correct,
            )
        )
    return recs


def make_occlusion_fit(*, query: bool = True):
    """An :class:`AllocatorFit` whose irreducibility detector fires above occ≈0.3.

    Value/cost params are set so the query branch genuinely opens on irreducible (heavily
    occluded) steps while compute wins on reducible (mildly occluded) ones.
    """

    from vla_lab.allocation.allocator import AllocatorFit
    from vla_lab.allocation.linear_model import fit_logistic
    from vla_lab.allocation.value_of_information import CostModel, ValueParams

    occs = [0.0, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.6]
    X = np.array([[0.0, o] for o in occs], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    model = fit_logistic(X, y, ["dispersion", "occlusion_fraction"], epochs=3000)

    fit = AllocatorFit.default()
    fit.irreducibility = model
    fit.value_params = ValueParams(act_dispersion_tau=0.02, voi_gain=0.8, voi_floor=0.1)
    fit.cost = CostModel(query_cost_s=2.0, value_scale=0.02)
    fit.allow_voi_query = bool(query)
    return fit

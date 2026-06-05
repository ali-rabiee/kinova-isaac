"""Value-of-computation (VoC) vs. value-of-information (VoI) allocation rule.

This operationalizes the memo's decision-theoretic account (§3.2) and, crucially, its
**limit result** (§4.1): compute can only resolve ambiguity that ``π(·|õ)`` actually
represents; it cannot manufacture information the mask removed. We encode that bound by
multiplying the value of computation by ``(1 - irreducible)`` — as the optimal action
becomes unidentifiable from the current view, ``VoC -> 0`` while ``VoI`` (a correct
external answer) stays positive, so the rule tips from *compute* to *query*.

Everything is in **success-probability units**; costs (latency, human workload) are
converted into the same units via ``CostModel.value_scale`` (success-equivalent per
second). The rule is deliberately small and interpretable — HRI reviewers punish
decorative math, and the thresholds/gains are fit from calibration data
(``vla_lab.fit_allocator``) rather than hand-tuned in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Allowed decisions, in cost order.
ACT = "act"
COMPUTE = "compute"
QUERY = "query"


@dataclass
class CostModel:
    """Costs of each branch, in seconds (converted to success-equiv via value_scale)."""

    compute_latency_per_sample_s: float = 0.08
    query_cost_s: float = 4.0  # latency + workload + interruption of asking a human
    act_cost_s: float = 0.0
    value_scale: float = 0.02  # success-prob equivalent of one second of cost

    def compute_penalty(self, k: int) -> float:
        return float(max(0, int(k) - 1)) * self.compute_latency_per_sample_s * self.value_scale

    def query_penalty(self) -> float:
        return self.query_cost_s * self.value_scale

    def to_dict(self) -> Dict[str, float]:
        return {
            "compute_latency_per_sample_s": float(self.compute_latency_per_sample_s),
            "query_cost_s": float(self.query_cost_s),
            "act_cost_s": float(self.act_cost_s),
            "value_scale": float(self.value_scale),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CostModel":
        return cls(
            compute_latency_per_sample_s=float(d.get("compute_latency_per_sample_s", 0.08)),
            query_cost_s=float(d.get("query_cost_s", 4.0)),
            act_cost_s=float(d.get("act_cost_s", 0.0)),
            value_scale=float(d.get("value_scale", 0.02)),
        )


@dataclass
class ValueParams:
    """Saturating value-curve parameters (fit from calibration, or sensible defaults)."""

    voc_gain: float = 0.25  # max success gain compute can deliver on fully-reducible ambiguity
    voc_dispersion_beta: float = 8.0  # how fast VoC rises with dispersion
    voc_k_gamma: float = 0.6  # diminishing returns in K
    voi_gain: float = 0.6  # max success gain from a correct human answer
    voi_floor: float = 0.0  # min irreducibility before a query is ever worth it
    act_dispersion_tau: float = 0.02  # below this dispersion just act (cheap path)

    def to_dict(self) -> Dict[str, float]:
        return {
            "voc_gain": float(self.voc_gain),
            "voc_dispersion_beta": float(self.voc_dispersion_beta),
            "voc_k_gamma": float(self.voc_k_gamma),
            "voi_gain": float(self.voi_gain),
            "voi_floor": float(self.voi_floor),
            "act_dispersion_tau": float(self.act_dispersion_tau),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ValueParams":
        return cls(
            voc_gain=float(d.get("voc_gain", 0.25)),
            voc_dispersion_beta=float(d.get("voc_dispersion_beta", 8.0)),
            voc_k_gamma=float(d.get("voc_k_gamma", 0.6)),
            voi_gain=float(d.get("voi_gain", 0.6)),
            voi_floor=float(d.get("voi_floor", 0.0)),
            act_dispersion_tau=float(d.get("act_dispersion_tau", 0.02)),
        )


def value_of_computation(
    dispersion: float, irreducible: float, k: int, vp: ValueParams, cost: CostModel
) -> float:
    """Expected net success gain from drawing ``k`` samples and selecting.

    ``= voc_gain · (1 - irreducible) · (1 - e^{-β·dispersion}) · (1 - e^{-γ·(k-1)}) - penalty(k)``

    The ``(1 - irreducible)`` factor is the limit result: compute cannot recover what the
    view does not contain.
    """

    d = max(0.0, float(dispersion))
    irr = min(max(float(irreducible), 0.0), 1.0)
    kk = max(1, int(k))
    reducible_mass = 1.0 - np.exp(-vp.voc_dispersion_beta * d)
    k_factor = 1.0 - np.exp(-vp.voc_k_gamma * (kk - 1))
    gross = vp.voc_gain * (1.0 - irr) * reducible_mass * k_factor
    return float(gross - cost.compute_penalty(kk))


def value_of_information(irreducible: float, vp: ValueParams, cost: CostModel) -> float:
    """Expected net success gain from a (correct) external answer.

    ``= voi_gain · max(0, irreducible - voi_floor) - query_penalty``. A query is only
    worth its interruption cost when the optimal action is genuinely unidentifiable from
    the current view (``irreducible`` high).
    """

    irr = min(max(float(irreducible), 0.0), 1.0)
    eff = max(0.0, irr - vp.voi_floor)
    return float(vp.voi_gain * eff - cost.query_penalty())


def allocate(
    dispersion: float,
    irreducible: float,
    k: int,
    vp: ValueParams,
    cost: CostModel,
) -> Tuple[str, Dict[str, float]]:
    """Pick act / compute / query by argmax expected value; return (decision, values).

    ``act`` is the zero-value baseline. Below ``act_dispersion_tau`` the policy is
    confident enough that we shortcut to ``act`` (avoids paying compute on easy steps).
    """

    v_act = 0.0
    v_compute = value_of_computation(dispersion, irreducible, k, vp, cost)
    v_query = value_of_information(irreducible, vp, cost)
    values = {"act": v_act, "compute": float(v_compute), "query": float(v_query)}

    if float(dispersion) <= vp.act_dispersion_tau and v_query <= 0.0:
        return ACT, values

    best = max(values, key=lambda key: values[key])
    # Ties / all-nonpositive -> act (the safe, cheap default).
    if values[best] <= 0.0:
        return ACT, values
    return best, values

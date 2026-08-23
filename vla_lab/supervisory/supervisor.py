r"""The generative supervisor -- what makes the whole pipeline testable before any people.

This is the synthetic human the Tier-1 study runs against. It is the substrate for the power
analysis, for every ablation, and for the end-to-end rehearsal of the session code, and it is
the reason a result table can exist before an IRB protocol does.

**It is also the study's biggest interpretive hazard**, so the boundary is stated here and
repeated in the paper: a de-biasing method evaluated against a simulator whose bias *we
injected* is being tested for whether it can invert a process we wrote down. That is a
legitimate and necessary thing to measure -- it is how any estimator is validated, and a
method that cannot recover a known ground truth certainly will not recover an unknown one --
but it is **not** evidence that human supervisors exhibit compliance carryover, and no claim
of that kind may rest on this module. Only participants can supply that.

The generative model, per supervisor ``p``:

.. math::
    \operatorname{logit}\pi^*_p(c) &= a_p\,(c - c_{0,p}) + d_p\,(\text{clutter} - \bar{\text{clutter}}) \\
    \operatorname{logit}\Pr[\text{instructs A at } t] &= \operatorname{logit}\pi^*_p(c_t)
        + \rho_t\,\beta_p\,\kappa^{\mathrm{eff}}_t + \phi_p\,(t/T) \\
    \kappa_{t+1} &= \lambda_p^{\Delta_t}\big(\kappa_t + g_p\,s_t\,d_t\,\mathbb{1}[a_t=\mathrm{COACH}]\big)

with a lapse rate on top (with probability ``lapse`` the supervisor answers at chance), and an
``ungrounded_rate`` of answers the lexical grounder cannot resolve.

``phi_p`` is **session drift**: a slow slide toward the efficient strategy as the supervisor
disengages. It is in the model because it is the confound this design is most exposed to --
drift and carryover both look like "answers changed over the session". The defence is
structural rather than statistical: because COACH is **counterbalanced in direction**, drift
pushes every block the same way while carryover pushes each block toward whatever was just
demonstrated. A method that mistakes one for the other fails visibly on the counterbalanced
contrast, which is why ``phi`` is switched on by default rather than kept for a stress test.

The choice model is deliberately **identical in form** to the inference model in
:mod:`vla_lab.supervisory.carryover`, including the immediate-effect convention on COACH slots.
Recovery tests are therefore about whether the estimator works, not about whether two
different equations happen to agree; misspecification is studied by *perturbing* these
parameters (:class:`SupervisorPopulation.misspecify`), not by baking a mismatch in.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED, WAIT
from ._numerics import sigmoid
from .carryover import CarryoverConfig
from .narration import ground
from .scenes import SceneGrid, SceneSpec
from .strategies import StrategyAxis, get_axis


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
@dataclass
class SupervisorPopulation:
    """The population synthetic supervisors are drawn from.

    Every number here is a **modelling assumption**, not a measurement. The pilot's job is to
    replace the compliance block (``beta``, ``g``, ``lambda``) with estimates from people; the
    preference block (``a``, ``c0``, ``d``) is a prior about how sharply preferences track the
    task-value gradient. The analysis prints the provenance next to any figure that depends on
    them.
    """

    # -- preference map ----------------------------------------------------
    a_range: Tuple[float, float] = (0.8, 2.4)          # slope in c: how sharply preference tracks value
    c0_range: Tuple[float, float] = (-0.7, 0.7)        # personal crossover shift: cautious vs. bold people
    d_range: Tuple[float, float] = (-0.25, 0.25)       # clutter sensitivity
    clutter_ref: float = 3.0
    # -- compliance --------------------------------------------------------
    beta_range: Tuple[float, float] = (0.0, 2.0)       # includes 0: some people simply do not comply
    g_range: Tuple[float, float] = (0.5, 1.5)
    lambda_range: Tuple[float, float] = (0.25, 0.92)   # the heterogeneity a fixed washout cannot serve
    # -- nuisance ----------------------------------------------------------
    phi_range: Tuple[float, float] = (-0.5, 0.1)       # session drift, mostly toward the efficient strategy
    lapse_range: Tuple[float, float] = (0.01, 0.07)
    ungrounded_range: Tuple[float, float] = (0.0, 0.06)
    latency_range_s: Tuple[float, float] = (1.5, 5.0)
    #: Fraction of supervisors drawn with beta == 0 exactly. A population in which *nobody* is
    #: immune to compliance is not a population any real study would meet, and a method that
    #: only works when everyone complies is not a method.
    p_noncomplier: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SupervisorPopulation":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        out = {}
        for k, v in d.items():
            if k not in known:
                continue
            out[k] = tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else v
        return cls(**out)

    def misspecify(self, *, decay_shape: str = "double", scale: float = 1.0) -> "SupervisorPopulation":
        """Return a population whose residue does *not* follow the inference model.

        Used for the misspecification stress test. ``decay_shape="double"`` gives a fast and a
        slow residue component, which no single-exponential fit can represent -- so a null on
        the primary hypothesis under this population would be ambiguous between "no benefit"
        and "wrong model", and the paper says so rather than claiming the model is right.
        """
        out = SupervisorPopulation(**{k: v for k, v in asdict(self).items()})
        out._decay_shape = str(decay_shape)  # type: ignore[attr-defined]
        out._decay_scale = float(scale)  # type: ignore[attr-defined]
        return out


@dataclass
class SupervisorParams:
    """One supervisor's true parameters. Never shown to any estimator."""

    supervisor_id: str
    a: float
    c0: float
    d: float
    beta: float
    g: float
    lam: float
    phi: float
    lapse: float
    ungrounded: float
    latency_s: float
    clutter_ref: float = 3.0
    decay_shape: str = "single"
    #: Slow component of a double-exponential residue (misspecification stress test only).
    lam_slow: float = 0.97
    slow_share: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def draw_supervisor(
    rng: random.Random,
    population: Optional[SupervisorPopulation] = None,
    *,
    supervisor_id: str = "S000",
) -> SupervisorParams:
    pop = population or SupervisorPopulation()

    def u(rng_: random.Random, rng_range: Tuple[float, float]) -> float:
        return float(rng_.uniform(rng_range[0], rng_range[1]))

    beta = 0.0 if rng.random() < float(pop.p_noncomplier) else u(rng, pop.beta_range)
    return SupervisorParams(
        supervisor_id=str(supervisor_id),
        a=u(rng, pop.a_range),
        c0=u(rng, pop.c0_range),
        d=u(rng, pop.d_range),
        beta=beta,
        g=u(rng, pop.g_range),
        lam=u(rng, pop.lambda_range),
        phi=u(rng, pop.phi_range),
        lapse=u(rng, pop.lapse_range),
        ungrounded=u(rng, pop.ungrounded_range),
        latency_s=u(rng, pop.latency_range_s),
        clutter_ref=float(pop.clutter_ref),
        decay_shape=str(getattr(pop, "_decay_shape", "single")),
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
@dataclass
class SupervisorResponse:
    """What the supervisor said, and (for analysis only) why."""

    strategy: str                  # the strategy actually intended
    utterance: str
    grounded: str                  # what the lexical grounder makes of the utterance
    latency_s: float
    p_a: float                     # the contaminated probability the draw came from
    pi_star: float                 # the uncontaminated probability at this scene
    kappa_eff: float
    lapsed: bool = False

    @property
    def contaminated_shift(self) -> float:
        """How far this answer's probability was moved from the unprompted one."""
        return float(self.p_a - self.pi_star)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimulatedSupervisor:
    """A seeded synthetic supervisor: a preference map, a compliance bias, and a mouth."""

    def __init__(
        self,
        params: SupervisorParams,
        *,
        axis: str = "plan",
        cfg: Optional[CarryoverConfig] = None,
        seed: int = 0,
    ) -> None:
        self.p = params
        self.axis: StrategyAxis = get_axis(axis)
        self.cfg = cfg or CarryoverConfig()
        self.rng = random.Random(int(seed) * 1_000_003 + hash(params.supervisor_id) % 100_003)
        self.kappa = 0.0
        self._kappa_slow = 0.0
        self.n_slots = 0

    # -- the estimand -------------------------------------------------------
    def pi_star(self, scene: SceneSpec) -> float:
        eta = self.p.a * (float(scene.c) - self.p.c0) + self.p.d * (float(scene.clutter) - self.p.clutter_ref)
        return float(sigmoid(eta))

    def pi_star_map(self, grid: SceneGrid) -> Dict[int, float]:
        """The ground-truth map. Used to score estimators, never given to one."""
        return {s.scene_id: self.pi_star(s) for s in grid.probe_scenes()}

    # -- the latent state ---------------------------------------------------
    def apply_coach(self, direction: int, *, strength: float = 1.0) -> None:
        """Deposit residue for a demonstration in ``direction`` (+1 = cautious A, -1 = B)."""
        inc = float(self.p.g) * float(strength) * float(direction)
        if self.p.decay_shape == "double":
            self._kappa_slow += inc * float(self.p.slow_share)
            self.kappa += inc * (1.0 - float(self.p.slow_share))
        else:
            self.kappa += inc

    def decay(self, delta: float) -> None:
        self.kappa = (float(self.p.lam) ** float(delta)) * self.kappa
        if self.p.decay_shape == "double":
            self._kappa_slow = (float(self.p.lam_slow) ** float(delta)) * self._kappa_slow

    @property
    def kappa_eff(self) -> float:
        return float(self.kappa + self._kappa_slow)

    # -- speaking -----------------------------------------------------------
    def respond(
        self,
        scene: SceneSpec,
        *,
        action: str = PROBE,
        session_progress: float = 0.0,
    ) -> SupervisorResponse:
        """Answer the robot's query about ``scene``, under the current residue."""
        pi = self.pi_star(scene)
        rho = self.cfg.rho_for(action)
        eta = math.log(max(pi, 1e-9) / max(1.0 - pi, 1e-9))
        eta += rho * float(self.p.beta) * self.kappa_eff
        eta += float(self.p.phi) * float(session_progress)
        p_a = float(sigmoid(eta))

        lapsed = self.rng.random() < float(self.p.lapse)
        draw_p = 0.5 if lapsed else p_a
        strategy = STRATEGY_A if self.rng.random() < draw_p else STRATEGY_B

        if self.rng.random() < float(self.p.ungrounded):
            utterance = self.rng.choice(_HEDGES)
        else:
            utterance = self.rng.choice(self.axis.phrases(strategy))
        return SupervisorResponse(
            strategy=strategy,
            utterance=utterance,
            grounded=ground(utterance, self.axis),
            latency_s=float(max(0.2, self.rng.gauss(self.p.latency_s, 0.6))),
            p_a=p_a,
            pi_star=pi,
            kappa_eff=self.kappa_eff,
            lapsed=lapsed,
        )

    def describe(self) -> Dict[str, Any]:
        return {"axis": self.axis.name, "params": self.p.to_dict(), "kappa": self.kappa_eff}


#: Answers the lexical grounder cannot resolve. Real supervisors hedge; a study that pretends
#: they do not will over-estimate how much signal a probe buys.
_HEDGES: Tuple[str, ...] = (
    "hmm, either way I guess",
    "whatever you think is best",
    "I'm not sure",
    "do what you did last time",
    "up to you",
)


def draw_cohort(
    n: int,
    *,
    seed: int = 0,
    population: Optional[SupervisorPopulation] = None,
    axis: str = "plan",
    cfg: Optional[CarryoverConfig] = None,
) -> List[SimulatedSupervisor]:
    """A reproducible cohort. Supervisor ``k`` is the same person for a given ``seed``."""
    rng = random.Random(int(seed))
    out: List[SimulatedSupervisor] = []
    for k in range(int(n)):
        params = draw_supervisor(rng, population, supervisor_id=f"S{k:03d}")
        out.append(SimulatedSupervisor(params, axis=axis, cfg=cfg, seed=int(seed) + k))
    return out


__all__ = [
    "SupervisorPopulation",
    "SupervisorParams",
    "SupervisorResponse",
    "SimulatedSupervisor",
    "draw_supervisor",
    "draw_cohort",
]

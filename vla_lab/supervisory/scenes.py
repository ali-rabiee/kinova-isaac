r"""Scenes, the ambiguity coordinate ``c``, and the task-value model that defines it.

**Why ``c`` is not just "gap width".** The estimand is a *map*, and the map is only useful if
its horizontal axis means the same thing across scene families, across strategy axes, and --
eventually -- across the sim/real boundary. So the coordinate is defined by the *physics of
the task*, not by a raw distance:

.. math::
    c(\text{scene}) \;=\; \frac{V_A(\text{scene}) - V_B(\text{scene})}{\\text{scale}}

where ``V_A``/``V_B`` are the expected task values of the cautious and efficient strategies.
``c > 0`` means the cautious strategy is objectively better here; ``c < 0`` means the efficient
one is; ``c = 0`` is the **crossover**, where the two are worth the same and the supervisor's
answer is genuinely a preference rather than a correct answer. Three consequences the rest of
the package leans on:

1. Nearly all the information about a person's preferences, and nearly all the room for a
   prompt to move their answer, live near ``c = 0``. Error metrics are crossover-weighted and
   the scene grid is densified there (:func:`build_scene_grid`).
2. Because ``c`` is a value difference, it also gives an objective **regret**: executing the
   wrong strategy at ``|c|`` large costs real task value, so compliance bias is not only a
   measurement artifact but a performance loss. That is what
   :func:`vla_lab.supervisory.estimand.decision_regret` scores.
3. The same definition works for both strategy axes, so a result on ``plan`` and a result on
   ``grasp`` are plotted on comparable axes.

**Priors versus measurements.** ``V_A`` and ``V_B`` come from :class:`ScenePhysics`: success
probability and time cost per strategy as a function of the raw geometric margin ``m``. Those
curves are **measured** by running the scripted experts in Isaac over a margin sweep
(``vla_lab/supervisory/apparatus/measure.py``) and fitting :meth:`ScenePhysics.fit`. Until
that has been done they are the defaults below, and ``ScenePhysics.source`` says which -- the
analysis surfaces it so that no figure presents a prior as a measurement.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import STRATEGY_A, STRATEGY_B
from ._numerics import irls_logistic, sigmoid, sigmoid_np
from .strategies import AXIS_PLAN, StrategyAxis, get_axis

SOURCE_PRIOR = "prior"
SOURCE_MEASURED = "measured"


# ---------------------------------------------------------------------------
# The physics: what each strategy costs and how often it works
# ---------------------------------------------------------------------------
@dataclass
class ScenePhysics:
    """Success probability and time cost of each strategy as a function of margin ``m``.

    ``m`` is the raw geometric margin in metres: for the ``plan`` axis it is the clearance
    between the blocker and the target (how much room a direct reach has); for the ``grasp``
    axis it is the headroom above the target. Larger ``m`` always favours the *efficient*
    strategy B, so ``p_success_B`` is increasing in ``m`` and ``p_success_A`` is roughly flat.

    Parameters
    ----------
    p_a_asym, p_b_asym:
        Ceiling success probabilities. ``A`` is the cautious strategy, so its ceiling is high
        and nearly margin-independent; ``B``'s ceiling is what it achieves with all the room
        in the world.
    b_slope, b_mid:
        Logistic parameters of ``p_success_B(m) = p_b_asym * sigma(b_slope * (m - b_mid))``.
        ``b_mid`` is the margin at which the direct strategy works half as often as it can.
    a_slope, a_mid:
        Same form for ``A``. Defaults make it essentially flat -- clearing the path works
        whether or not the gap was tight -- but the form is there so a *measured* fit can
        contradict that assumption rather than being unable to express it.
    t_a, t_b:
        Wall-clock cost of each strategy in seconds, at nominal margin. Clearing first is
        strictly slower: that is the entire trade.
    t_b_tight:
        Extra seconds the direct strategy spends when the margin is tight (careful threading
        and re-approaches), applied as ``t_b + t_b_tight * (1 - sigma(b_slope*(m-b_mid)))``.
    reward:
        Task value of a completed fetch.
    time_cost:
        Value of one second, in reward units. Sets how much speed is worth relative to
        reliability -- i.e. where the crossover lands. It is a **preference-free** exchange
        rate fixed by the contract, not a claim about what any person values; person-level
        preference is exactly what the estimand measures.
    disturbance_penalty:
        Value charged for having moved a non-target object (only ``A`` does), reflecting that
        rearranging a workspace is not free.
    source:
        ``"prior"`` or ``"measured"``; ``n_measured`` records how many rollouts backed a fit.
    """

    p_a_asym: float = 0.96
    a_slope: float = 90.0
    a_mid: float = -0.09
    p_b_asym: float = 0.97
    b_slope: float = 45.0
    b_mid: float = 0.035
    t_a: float = 26.0
    t_b: float = 15.0
    t_b_tight: float = 9.0
    reward: float = 1.0
    time_cost: float = 0.012
    disturbance_penalty: float = 0.05
    source: str = SOURCE_PRIOR
    n_measured: int = 0

    # -- success / time -----------------------------------------------------
    def p_success(self, strategy: str, m: float) -> float:
        if strategy == STRATEGY_A:
            return float(self.p_a_asym * sigmoid(self.a_slope * (float(m) - self.a_mid)))
        return float(self.p_b_asym * sigmoid(self.b_slope * (float(m) - self.b_mid)))

    def duration_s(self, strategy: str, m: float) -> float:
        if strategy == STRATEGY_A:
            return float(self.t_a)
        tightness = 1.0 - sigmoid(self.b_slope * (float(m) - self.b_mid))
        return float(self.t_b + self.t_b_tight * tightness)

    # -- value --------------------------------------------------------------
    def value(self, strategy: str, m: float) -> float:
        """Expected task value of executing ``strategy`` at margin ``m``."""
        v = self.p_success(strategy, m) * self.reward
        v -= self.time_cost * self.duration_s(strategy, m)
        if strategy == STRATEGY_A:
            v -= self.disturbance_penalty
        return float(v)

    def value_gap(self, m: float) -> float:
        """``V_A - V_B``: positive when the cautious strategy is objectively better."""
        return float(self.value(STRATEGY_A, m) - self.value(STRATEGY_B, m))

    def optimal_strategy(self, m: float) -> str:
        return STRATEGY_A if self.value_gap(m) >= 0.0 else STRATEGY_B

    # -- the coordinate -----------------------------------------------------
    def crossover_margin(self, lo: float = 0.0, hi: float = 0.20) -> float:
        """The margin at which ``V_A = V_B``. Bisection; the gap is monotone decreasing in m."""
        f_lo, f_hi = self.value_gap(lo), self.value_gap(hi)
        if f_lo * f_hi > 0:
            # No sign change in the bracket: the axis is degenerate at these settings. Return
            # the endpoint whose gap is smaller so ``c`` stays finite and the caller's grid is
            # still usable; :meth:`is_degenerate` lets the contract check refuse it loudly.
            return lo if abs(f_lo) < abs(f_hi) else hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.value_gap(lo) * self.value_gap(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def is_degenerate(self, lo: float = 0.0, hi: float = 0.20) -> bool:
        """True when no crossover exists in the bracket -- one strategy always wins."""
        return self.value_gap(lo) * self.value_gap(hi) > 0

    def transition_width_m(self) -> float:
        """Margin change over which the margin-sensitive strategy goes from failing to working.

        This is the natural yardstick for "how ambiguous is this scene": a gap one transition
        width tighter than break-even is meaningfully harder, and a gap five widths tighter is
        not five times more informative -- it is simply hopeless. Taken from the fitted slope
        of ``p_success_B``, so it is a *measured* quantity once the Isaac sweep has run.
        """
        return float(1.0 / max(abs(self.b_slope), 1e-6))

    def coordinate(self, m: float, *, lo: float = 0.0, hi: float = 0.20) -> float:
        """The signed ambiguity coordinate, in transition widths tighter than break-even.

        ``c = (m* - m) / w`` with ``m*`` the value crossover and ``w`` the transition width.
        Positive means tighter than break-even, i.e. the cautious strategy A is objectively
        better; negative means roomier, i.e. the efficient strategy B is. Zero is the
        crossover. Defining it this way -- rather than as the raw value gap -- keeps the
        coordinate symmetric about the crossover, which the value gap is not: past the point
        where the direct strategy simply works, extra clearance buys nothing and the gap
        flattens out, so a value-gap coordinate would compress one flank and stretch the other.
        """
        m_star = self.crossover_margin(lo, hi)
        return float((m_star - float(m)) / self.transition_width_m())

    def margin_for_coordinate(self, c: float, *, lo: float = 0.0, hi: float = 0.20) -> float:
        """Inverse of :meth:`coordinate`."""
        return float(self.crossover_margin(lo, hi) - float(c) * self.transition_width_m())

    # -- fitting from rollouts ---------------------------------------------
    @classmethod
    def fit(
        cls,
        records: Sequence[Dict[str, Any]],
        *,
        base: Optional["ScenePhysics"] = None,
        prior_precision: float = 1e-2,
    ) -> "ScenePhysics":
        """Fit success curves and mean durations from executed rollouts.

        ``records`` are dicts with ``strategy``, ``margin_m``, ``success`` and (optionally)
        ``duration_s`` -- exactly the rows :mod:`vla_lab.supervisory.apparatus.measure` writes
        from an Isaac margin sweep. Everything the rollouts do not constrain (``reward``,
        ``time_cost``, ``disturbance_penalty``) is carried over from ``base``, because those
        are contract constants rather than measurements.
        """
        out = replace(base or cls())
        by_strategy: Dict[str, List[Dict[str, Any]]] = {STRATEGY_A: [], STRATEGY_B: []}
        for r in records:
            st = str(r.get("strategy"))
            if st in by_strategy:
                by_strategy[st].append(r)

        n_used = 0
        for strategy, rows in by_strategy.items():
            if len(rows) < 8:
                continue
            m = np.array([float(r["margin_m"]) for r in rows])
            y = np.array([1.0 if bool(r["success"]) else 0.0 for r in rows])
            asym, slope, mid = _fit_scaled_logistic(m, y, prior_precision=prior_precision)
            durs = [float(r["duration_s"]) for r in rows if r.get("duration_s") is not None]
            n_used += len(rows)
            if strategy == STRATEGY_A:
                out.p_a_asym, out.a_slope, out.a_mid = asym, slope, mid
                if durs:
                    out.t_a = float(np.mean(durs))
            else:
                out.p_b_asym, out.b_slope, out.b_mid = asym, slope, mid
                if durs:
                    tight = [d for d, r in zip(durs, rows) if float(r["margin_m"]) < mid]
                    wide = [d for d, r in zip(durs, rows) if float(r["margin_m"]) >= mid]
                    out.t_b = float(np.mean(wide)) if wide else float(np.mean(durs))
                    out.t_b_tight = float(max(0.0, (np.mean(tight) if tight else out.t_b) - out.t_b))
        out.source = SOURCE_MEASURED if n_used else SOURCE_PRIOR
        out.n_measured = int(n_used)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ScenePhysics":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneSpec:
    """One presentable scene: geometry, the coordinate it induces, and its identity.

    ``scene_id`` is stable and lives in the contract, so the same integer means the same
    geometry in every session and across the sim/real boundary. ``c`` is cached at construction
    from the physics that built it -- a scene carries the coordinate it was *presented* under,
    which matters when the physics is re-measured mid-programme.
    """

    scene_id: int
    axis: str
    margin_m: float
    clutter: int
    c: float
    target_label: str = "red box"
    blocker_label: str = "blue box"
    #: True for the small set of unambiguous scenes reserved for COACH demonstrations.
    coach_scene: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SceneGrid:
    """The contract's scene set, plus the physics that assigns coordinates to it."""

    scenes: Tuple[SceneSpec, ...]
    physics: ScenePhysics
    axis: str = AXIS_PLAN
    crossover_halfwidth: float = 0.35

    def __post_init__(self) -> None:
        self._by_id = {s.scene_id: s for s in self.scenes}

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def by_id(self, scene_id: int) -> SceneSpec:
        return self._by_id[int(scene_id)]

    @property
    def ids(self) -> List[int]:
        return [s.scene_id for s in self.scenes]

    def probe_scenes(self) -> List[SceneSpec]:
        """Scenes eligible for PROBE/COUNTER: everything not reserved for demonstrations."""
        return [s for s in self.scenes if not s.coach_scene]

    def coach_scenes(self) -> List[SceneSpec]:
        """Unambiguous scenes used for COACH.

        COACH must not double as a probe: demonstrating on an ambiguous scene would make the
        demonstration itself an implicit answer to the question the study is asking, and the
        manipulation would no longer be separable from the measurement.
        """
        return [s for s in self.scenes if s.coach_scene]

    def in_crossover_band(self, scene: SceneSpec) -> bool:
        return abs(float(scene.c)) <= float(self.crossover_halfwidth)

    def band_weights(self, crossover_weighted: bool = True) -> Dict[int, float]:
        """Error weights over scenes. Uniform, or concentrated in the crossover band.

        Crossover weighting is not a thumb on the scale: outside the band the estimand is
        saturated, so *every* estimator gets those scenes right and including them dilutes the
        contrast between conditions with cells no method can lose. Both weightings are always
        reported.
        """
        probe = self.probe_scenes()
        if not crossover_weighted:
            w = 1.0 / max(len(probe), 1)
            return {s.scene_id: w for s in probe}
        raw = {s.scene_id: math.exp(-0.5 * (float(s.c) / max(self.crossover_halfwidth, 1e-6)) ** 2) for s in probe}
        z = sum(raw.values()) or 1.0
        return {k: v / z for k, v in raw.items()}

    def coordinates(self) -> np.ndarray:
        return np.array([s.c for s in self.scenes], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "crossover_halfwidth": self.crossover_halfwidth,
            "physics": self.physics.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneGrid":
        return cls(
            scenes=tuple(SceneSpec.from_dict(s) for s in d["scenes"]),
            physics=ScenePhysics.from_dict(d.get("physics")),
            axis=str(d.get("axis", AXIS_PLAN)),
            crossover_halfwidth=float(d.get("crossover_halfwidth", 0.35)),
        )


#: Where the Isaac margin sweep writes the fitted physics. Everything that builds a scene grid
#: reads it from here unless handed one explicitly, so a single measurement propagates to the
#: study, the atlas renderer, the training data and the closed-loop evaluation at once. Passing
#: the path around as a flag was the earlier arrangement and it is a trap: the sweep would write
#: a measurement, one command would be given the flag and the others would not, and the study
#: would run half under measured physics and half under the prior with nothing to say so.
DEFAULT_PHYSICS_PATH = Path("vla_lab/results/physics/physics.json")


def default_physics(path: Optional[Path] = None) -> "ScenePhysics":
    """The measured physics if the sweep has run, otherwise the documented prior.

    Never raises: a missing or unreadable file falls back to the prior, and the returned object
    carries ``source`` so every figure and audit can say which one it got.
    """
    p = Path(path) if path is not None else DEFAULT_PHYSICS_PATH
    try:
        if p.exists():
            return load_physics(p)
    except Exception:                                   # a corrupt file must not break the prior
        pass
    return ScenePhysics()


def build_scene_grid(
    *,
    axis: str = AXIS_PLAN,
    physics: Optional[ScenePhysics] = None,
    n_band: int = 9,
    n_flank: int = 3,
    band_c: float = 1.2,
    flank_c: float = 2.4,
    clutter_levels: Sequence[int] = (2, 4),
    n_coach_scenes: int = 4,
    coach_c: float = 3.2,
    margin_bracket: Tuple[float, float] = (0.0, 0.20),
    crossover_halfwidth: float = 1.0,
    target_label: str = "red box",
    blocker_label: str = "blue box",
) -> SceneGrid:
    """Build the contract scene set: dense in the crossover band, sparse on the flanks.

    The grid is specified in ``c`` and inverted to margins, not the other way round. That is
    the point of defining ``c`` through the value model: a grid laid out on raw gap widths
    would sit in a different place relative to the crossover for every physics setting, and
    two sessions run under re-measured physics would not be poolable.

    ``coach_c`` places the demonstration scenes outside the probe range on **both** sides, so
    a COACH can demonstrate either strategy on a scene where that strategy is unambiguously
    correct. Demonstrating on an ambiguous scene would make the demonstration an implicit
    answer to the question the study asks, and the manipulation would stop being separable
    from the measurement.
    """
    phys = physics if physics is not None else default_physics()
    lo, hi = float(margin_bracket[0]), float(margin_bracket[1])

    targets: List[Tuple[float, bool]] = [(float(c), False) for c in np.linspace(-band_c, band_c, int(n_band))]
    for c in np.linspace(band_c, flank_c, int(n_flank) + 1)[1:]:
        targets.append((float(+c), False))
        targets.append((float(-c), False))
    for k in range(int(n_coach_scenes)):
        sign = 1.0 if k % 2 == 0 else -1.0
        targets.append((float(sign * coach_c), True))

    scenes: List[SceneSpec] = []
    sid = 0
    for c_target, is_coach in sorted(targets, key=lambda t: (t[1], t[0])):
        m = float(np.clip(phys.margin_for_coordinate(c_target, lo=lo, hi=hi), lo, hi))
        clutter = int(clutter_levels[sid % max(len(clutter_levels), 1)]) if clutter_levels else 0
        scenes.append(
            SceneSpec(
                scene_id=sid,
                axis=str(axis),
                margin_m=m,
                clutter=clutter,
                c=float(phys.coordinate(m, lo=lo, hi=hi)),
                target_label=target_label,
                blocker_label=blocker_label,
                coach_scene=bool(is_coach),
            )
        )
        sid += 1
    return SceneGrid(
        scenes=tuple(scenes),
        physics=phys,
        axis=str(axis),
        crossover_halfwidth=float(crossover_halfwidth),
    )


def _fit_scaled_logistic(
    m: np.ndarray,
    y: np.ndarray,
    *,
    prior_precision: float = 1e-2,
    asym_grid: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """Fit ``p(m) = asym * sigma(slope * (m - mid))`` by profiling over the ceiling.

    The ceiling has to be fitted, not guessed. Setting it from the observed mean -- the obvious
    shortcut -- biases it low whenever the sweep spends rollouts at margins where the strategy
    fails, which is most of them by design, and a low ceiling then drags the slope down with it.
    Profiling over a grid of ceilings and taking the best likelihood costs almost nothing and
    recovers all three parameters.
    """
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    grid = list(asym_grid) if asym_grid is not None else list(np.linspace(0.35, 1.0, 27))
    X = np.stack([np.ones_like(m), m], axis=1)
    best = (float(np.clip(y.mean(), 0.05, 0.99)), 1.0, float(np.median(m)))
    best_ll = -np.inf
    for asym in grid:
        a = float(min(max(asym, 1e-3), 1.0))
        w, _ = irls_logistic(X, np.clip(y / a, 0.0, 1.0), prior_precision=prior_precision)
        p = np.clip(a * sigmoid_np(X @ w), 1e-9, 1.0 - 1e-9)
        ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
        if ll > best_ll:
            slope = float(w[1])
            mid = float(-w[0] / w[1]) if abs(w[1]) > 1e-6 else float(np.median(m))
            best, best_ll = (a, slope, mid), ll
    return best


def load_physics(path: Path) -> ScenePhysics:
    return ScenePhysics.from_dict(json.loads(Path(path).read_text()))


def save_physics(physics: ScenePhysics, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(physics.to_dict(), indent=2) + "\n")


__all__ = [
    "SOURCE_PRIOR",
    "SOURCE_MEASURED",
    "ScenePhysics",
    "SceneSpec",
    "SceneGrid",
    "build_scene_grid",
    "load_physics",
    "save_physics",
]

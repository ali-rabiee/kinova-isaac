r"""Fitting the scene physics from rollouts -- and the uncertainty that comes with it.

Every scene coordinate ``c = (m* - m) / w``, every band weight, every MAE-cross and every regret
number in the study inherits whatever error the crossover ``m*`` and the transition width ``w``
carry. Until 2026-08-23 both were treated as constants; this module is what turns them into
estimates with intervals, and it is also where a fitting defect that decided the paper's
physics was found and is recorded.

**Defect (xii): a ridge prior on a per-metre slope.** The first fit
(:func:`vla_lab.supervisory.scenes._fit_scaled_logistic`) regularised the logistic weights with
an isotropic Gaussian of precision ``1e-2``. Margins are in metres, so the success curve the
tight-end data actually want -- clear-first going from 2/12 at 0 cm to 12/12 at 3 cm, i.e. a
slope near 150 per metre -- costs ``0.5 * 0.01 * 150^2 = 112`` nats of prior penalty, and the
fit settled at 30 per metre instead. That flattened curve is where the published ``w = 3.12 cm``
came from, and it is also the whole reason the fit "could not follow a near-step" at the tails:
the two-parameter form was never the limitation, the prior was. In the units the problem is
naturally posed in (per centimetre) the same precision is a prior standard deviation of ten
transitions per centimetre, i.e. no constraint at all. The fitters below put their prior on a
per-centimetre slope, and the legacy fit is kept, labelled, so the two can be shown side by side.

Three fits, all reported:

``lapse``  (the primary)
    ``p(m) = floor + (ceiling - floor) * sigma(slope * (m - mid))``. A four-parameter
    psychometric with a lapse asymptote at both ends, fitted by profiling ``(floor, ceiling)``
    over a grid and Fisher scoring on ``(slope, mid)`` inside. The floor matters here: at a
    0 cm gap the scripted expert still succeeds about one time in six, and a curve forced
    through zero has to buy that with a shallower slope.
``isotonic``  (the model-free check)
    Pool-adjacent-violators on the per-gap success fractions, weighted by cell counts, with the
    transition width read off the fitted step function as ``(m_75 - m_25) / (2 ln 3)`` -- the
    scaling that makes it equal ``1/slope`` for an exact logistic. It carries no shape
    assumption, so where it and the lapse fit agree the shape is not doing the work.
``scaled_legacy``
    The original fit, reproduced exactly, for the "what we had before" comparison.

**The bootstrap** resamples rollouts *within* each ``(strategy, gap)`` cell -- the design is
stratified, so that is the resampling that respects it -- refits the primary and the isotonic
curves, and reports percentile intervals on ``m*`` and ``w``. It also returns two complete
physics objects, ``lower`` and ``upper``, taken from the replicates whose ``w`` sits at the
2.5th and 97.5th percentile; ``run_study --physics-quantile`` rebuilds the whole scene grid under
either, which is how the primary contrasts get reported under the physics interval.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import STRATEGY_A, STRATEGY_B
from ._numerics import sigmoid_np

FIT_LAPSE = "lapse_logistic"
FIT_ISOTONIC = "isotonic"
FIT_LEGACY = "scaled_logistic_legacy"

_LN3 = math.log(3.0)


# ---------------------------------------------------------------------------
# The lapse-parameterised psychometric
# ---------------------------------------------------------------------------
def _loglik(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def fit_lapse_logistic(
    m: np.ndarray,
    y: np.ndarray,
    *,
    floors: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    ceilings: Sequence[float] = (0.85, 0.90, 0.95, 1.0),
    slope_prior_sd_per_cm: float = 10.0,
    mid_prior_sd_cm: float = 50.0,
    init: Optional[Dict[str, float]] = None,
    max_iter: int = 60,
) -> Dict[str, Any]:
    """MAP fit of ``floor + (ceiling - floor) * sigma(slope (m - mid))`` to Bernoulli rollouts.

    ``m`` in metres, ``y`` in {0, 1}. ``(floor, ceiling)`` are profiled over a grid; for each
    pair ``(slope, mid)`` are fitted by damped Fisher scoring on the exact likelihood, in
    **per-centimetre** units with a deliberately weak Gaussian prior (see the module docstring
    for why the units are not a detail). Returns the parameters in per-metre units so they drop
    straight into :class:`~vla_lab.supervisory.scenes.ScenePhysics`.
    """
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x = m * 100.0                                            # centimetres
    X = np.stack([np.ones_like(x), x], axis=1)
    prior_prec = np.diag([1.0 / float(mid_prior_sd_cm) ** 2, 1.0 / float(slope_prior_sd_per_cm) ** 2])

    # A reasonable start: the midpoint of the rise and a one-per-cm slope.
    if init is not None:
        w_init = np.array([-float(init["slope_per_cm"]) * float(init["mid_cm"]), float(init["slope_per_cm"])])
    else:
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        cum = np.cumsum(ys) / max(float(ys.sum()), 1.0)
        mid0 = float(xs[min(len(xs) - 1, int(np.searchsorted(cum, 0.5)))]) if ys.sum() > 0 else float(np.median(xs))
        w_init = np.array([-1.0 * mid0, 1.0])

    best: Optional[Dict[str, Any]] = None
    floor_grid = [float(f) for f in floors]
    ceil_grid = [float(c) for c in ceilings]
    if init is not None:                                     # warm bootstrap: a neighbourhood only
        f0, c0 = float(init["floor"]), float(init["ceiling"])
        floor_grid = sorted({f for f in floor_grid if abs(f - f0) <= 0.101})
        ceil_grid = sorted({c for c in ceil_grid if abs(c - c0) <= 0.051})
    for f in floor_grid:
        for a in ceil_grid:
            if a <= f + 0.05:
                continue
            w = w_init.copy()
            span = a - f

            def objective(wv: np.ndarray) -> float:
                s = sigmoid_np(X @ wv)
                return _loglik(f + span * s, y) - 0.5 * float(wv @ prior_prec @ wv)

            cur = objective(w)
            for _ in range(int(max_iter)):
                s = sigmoid_np(X @ w)
                p = np.clip(f + span * s, 1e-9, 1.0 - 1e-9)
                dp = span * s * (1.0 - s)                      # dp / d eta
                grad = X.T @ (dp * (y / p - (1.0 - y) / (1.0 - p))) - prior_prec @ w
                fisher = X.T @ (X * (dp * dp / (p * (1.0 - p)))[:, None]) + prior_prec
                try:
                    step = np.linalg.solve(fisher, grad)
                except np.linalg.LinAlgError:                 # pragma: no cover
                    step = np.linalg.lstsq(fisher, grad, rcond=None)[0]
                # Damped: halve the step until the penalised likelihood does not fall.
                t = 1.0
                nxt = objective(w + t * step)
                while nxt < cur - 1e-12 and t > 1e-4:
                    t *= 0.5
                    nxt = objective(w + t * step)
                if nxt < cur - 1e-12:
                    break
                w = w + t * step
                if abs(nxt - cur) < 1e-10 and float(np.max(np.abs(t * step))) < 1e-7:
                    cur = nxt
                    break
                cur = nxt
            ll = _loglik(f + span * sigmoid_np(X @ w), y)
            slope_cm = float(w[1])
            mid_cm = float(-w[0] / w[1]) if abs(w[1]) > 1e-9 else float(np.median(x))
            cand = {"floor": f, "ceiling": a, "slope_per_cm": slope_cm, "mid_cm": mid_cm,
                    "slope_per_m": slope_cm * 100.0, "mid_m": mid_cm / 100.0, "loglik": ll,
                    "penalised": cur, "n": int(len(y))}
            if best is None or cur > best["penalised"]:
                best = cand
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA), and the width read off a step function
# ---------------------------------------------------------------------------
def pava_increasing(values: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    """Weighted pool-adjacent-violators for a non-decreasing fit. ``values`` in x order."""
    v = [float(a) for a in values]
    w = [float(b) for b in weights]
    blocks: List[List[float]] = []                           # [mean, weight, count]
    for a, b in zip(v, w):
        blocks.append([a, b, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            tot = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / max(tot, 1e-12), tot, c1 + c2])
    out: List[float] = []
    for mean, _w, count in blocks:
        out.extend([mean] * int(count))
    return np.asarray(out, dtype=float)


class IsotonicCurve:
    """A fitted non-decreasing success curve: step values at the measured gaps, interpolated
    linearly between gap centres and held constant beyond the ends."""

    def __init__(self, margins: Sequence[float], p_fit: Sequence[float]) -> None:
        self.m = np.asarray(margins, dtype=float)
        self.p = np.asarray(p_fit, dtype=float)

    def __call__(self, m: float) -> float:
        return float(np.interp(float(m), self.m, self.p))

    @property
    def floor(self) -> float:
        return float(self.p[0])

    @property
    def ceiling(self) -> float:
        return float(self.p[-1])

    def quantile_margin(self, q: float) -> float:
        """Margin at which the curve reaches ``floor + q (ceiling - floor)``, by interpolation."""
        lo, hi = self.floor, self.ceiling
        if hi - lo < 1e-9:
            return float(np.mean(self.m))
        target = lo + float(q) * (hi - lo)
        fine = np.linspace(float(self.m[0]), float(self.m[-1]), 2001)
        vals = np.interp(fine, self.m, self.p)
        idx = int(np.searchsorted(vals, target))
        return float(fine[min(idx, len(fine) - 1)])

    def transition_width_m(self) -> float:
        """``(m_75 - m_25) / (2 ln 3)``: equals ``1/slope`` for an exact logistic."""
        return float(max(self.quantile_margin(0.75) - self.quantile_margin(0.25), 1e-4) / (2.0 * _LN3))


def fit_isotonic(rows: Sequence[Dict[str, Any]], strategy: str) -> IsotonicCurve:
    cells: Dict[float, List[float]] = {}
    for r in rows:
        if str(r.get("strategy")) == strategy:
            cells.setdefault(round(float(r["margin_m"]), 4), []).append(1.0 if bool(r["success"]) else 0.0)
    margins = sorted(cells)
    means = [float(np.mean(cells[m])) for m in margins]
    counts = [float(len(cells[m])) for m in margins]
    return IsotonicCurve(margins, pava_increasing(means, counts))


def isotonic_physics_summary(rows: Sequence[Dict[str, Any]], base, *, lo: float = 0.0, hi: float = 0.20) -> Dict[str, Any]:
    """``m*`` and ``w`` implied by the isotonic curves, under the same value accounting.

    Uses ``base`` for the contract constants (reward, time cost, disturbance penalty) and its
    fitted durations; only the success curves are replaced by the model-free fit.
    """
    ca, cb = fit_isotonic(rows, STRATEGY_A), fit_isotonic(rows, STRATEGY_B)

    def value(st: str, m: float) -> float:
        p = ca(m) if st == STRATEGY_A else cb(m)
        v = p * base.reward - base.time_cost * base.duration_s(st, m)
        return v - (base.disturbance_penalty if st == STRATEGY_A else 0.0)

    fine = np.linspace(lo, hi, 2001)
    gap = np.array([value(STRATEGY_A, m) - value(STRATEGY_B, m) for m in fine])
    # The upper crossing only -- where the direct strategy overtakes as the gap opens. See
    # ``ScenePhysics._crossings`` for why the lower one (both strategies failing) is not it.
    down = [i for i in range(len(fine) - 1) if gap[i] > 0.0 >= gap[i + 1]]
    if down:
        i = int(down[-1])
        g0, g1 = gap[i], gap[i + 1]
        m_star = float(fine[i] + (fine[i + 1] - fine[i]) * (g0 / (g0 - g1) if g0 != g1 else 0.5))
        degenerate = False
    else:
        m_star = float(fine[int(np.argmin(np.abs(gap)))])
        degenerate = True
    return {
        "method": FIT_ISOTONIC,
        "crossover_margin_m": m_star,
        "transition_width_m": cb.transition_width_m(),
        "transition_width_a_m": ca.transition_width_m(),
        "degenerate": degenerate,
        "curve_a": {"margins_m": ca.m.tolist(), "p": ca.p.tolist()},
        "curve_b": {"margins_m": cb.m.tolist(), "p": cb.p.tolist()},
    }


# ---------------------------------------------------------------------------
# Fitting a ScenePhysics with the lapse model
# ---------------------------------------------------------------------------
def _timed_out(r: Dict[str, Any]) -> bool:
    return str(r.get("failure") or "").startswith("waypoint_timeout")


def fit_physics_lapse(rows: Sequence[Dict[str, Any]], *, base=None, init: Optional[Dict[str, Dict[str, float]]] = None):
    """:class:`ScenePhysics` from rollouts, primary (lapse) fit. ``init`` warm-starts a bootstrap."""
    from .scenes import SOURCE_MEASURED, SOURCE_PRIOR, ScenePhysics

    out = replace(base or ScenePhysics())
    fits: Dict[str, Dict[str, Any]] = {}
    n_used = 0
    for strategy in (STRATEGY_A, STRATEGY_B):
        rs = [r for r in rows if str(r.get("strategy")) == strategy]
        if len(rs) < 8:
            continue
        m = np.array([float(r["margin_m"]) for r in rs])
        y = np.array([1.0 if bool(r["success"]) else 0.0 for r in rs])
        fit = fit_lapse_logistic(m, y, init=(init or {}).get(strategy))
        fits[strategy] = fit
        # Durations: a rollout that hit the follower's step budget reports the budget, not the
        # task (one such rollout ran 270 s against a 13 s median and moved the clear-first mean
        # by over a second on its own). Those are excluded from the time cost and counted.
        kept = [r for r in rs if r.get("duration_s") is not None and not _timed_out(r)]
        fit["n_duration_excluded_timeouts"] = int(sum(1 for r in rs if _timed_out(r)))
        durs = [float(r["duration_s"]) for r in kept]
        n_used += len(rs)
        if strategy == STRATEGY_A:
            out.p_a_asym, out.a_slope, out.a_mid, out.p_a_floor = fit["ceiling"], fit["slope_per_m"], fit["mid_m"], fit["floor"]
            if durs:
                out.t_a = float(np.mean(durs))
        else:
            out.p_b_asym, out.b_slope, out.b_mid, out.p_b_floor = fit["ceiling"], fit["slope_per_m"], fit["mid_m"], fit["floor"]
            if durs:
                mid = fit["mid_m"]
                tight = [d for d, r in zip(durs, kept) if float(r["margin_m"]) < mid]
                wide = [d for d, r in zip(durs, kept) if float(r["margin_m"]) >= mid]
                out.t_b = float(np.mean(wide)) if wide else float(np.mean(durs))
                out.t_b_tight = float(max(0.0, (np.mean(tight) if tight else out.t_b) - out.t_b))
    out.source = SOURCE_MEASURED if n_used else SOURCE_PRIOR
    out.n_measured = int(n_used)
    out.fit_method = FIT_LAPSE
    return out, fits


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------
def _cells(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, float], List[Dict[str, Any]]]:
    by: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((str(r["strategy"]), round(float(r["margin_m"]), 4)), []).append(r)
    return by


def bootstrap_physics(
    rows: Sequence[Dict[str, Any]],
    *,
    base=None,
    n_boot: int = 2000,
    seed: int = 0,
    lo: float = 0.0,
    hi: float = 0.20,
) -> Dict[str, Any]:
    """Within-cell bootstrap of ``m*`` and ``w`` under the lapse fit and the isotonic fit.

    Returns the replicate distributions, percentile intervals, and the two complete physics
    objects (``lower``/``upper`` by ``w``) the study runner can rebuild its scene grid under.
    """
    from .scenes import ScenePhysics

    rng = np.random.default_rng(int(seed))
    cells = _cells(rows)
    point, fits = fit_physics_lapse(rows, base=base)
    init = {st: {"floor": f["floor"], "ceiling": f["ceiling"], "slope_per_cm": f["slope_per_cm"], "mid_cm": f["mid_cm"]}
            for st, f in fits.items()}
    m_star: List[float] = []
    width: List[float] = []
    m_star_iso: List[float] = []
    width_iso: List[float] = []
    params: List[Dict[str, Any]] = []
    for _ in range(int(n_boot)):
        sample: List[Dict[str, Any]] = []
        for _key, rs in cells.items():
            idx = rng.integers(0, len(rs), size=len(rs))
            sample.extend(rs[i] for i in idx)
        phys, _ = fit_physics_lapse(sample, base=base, init=init)
        m_star.append(float(phys.crossover_margin(lo, hi)))
        width.append(float(phys.transition_width_m()))
        params.append(phys.to_dict())
        iso = isotonic_physics_summary(sample, phys, lo=lo, hi=hi)
        m_star_iso.append(float(iso["crossover_margin_m"]))
        width_iso.append(float(iso["transition_width_m"]))

    def pct(a: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(a, dtype=float)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "p2.5": float(np.percentile(arr, 2.5)), "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)), "p75": float(np.percentile(arr, 75)),
                "p97.5": float(np.percentile(arr, 97.5))}

    w_arr = np.asarray(width)
    q_lo, q_hi = np.percentile(w_arr, 2.5), np.percentile(w_arr, 97.5)
    i_lo, i_hi = int(np.argmin(np.abs(w_arr - q_lo))), int(np.argmin(np.abs(w_arr - q_hi)))
    lower = ScenePhysics.from_dict(params[i_lo])
    upper = ScenePhysics.from_dict(params[i_hi])
    lower.quantile, upper.quantile = "lower", "upper"
    point.quantile = "point"
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resampling": "within (strategy, gap) cells",
        "point": {"crossover_margin_m": float(point.crossover_margin(lo, hi)),
                  "transition_width_m": float(point.transition_width_m())},
        "crossover_margin_m": pct(m_star),
        "transition_width_m": pct(width),
        "isotonic": {"crossover_margin_m": pct(m_star_iso), "transition_width_m": pct(width_iso)},
        "replicates": {"crossover_margin_m": [round(v, 6) for v in m_star],
                       "transition_width_m": [round(v, 6) for v in width]},
        "quantile_physics": {"lower": lower.to_dict(), "point": point.to_dict(), "upper": upper.to_dict()},
    }


# ---------------------------------------------------------------------------
# The legacy fit, kept for the comparison
# ---------------------------------------------------------------------------
def fit_physics_legacy(rows: Sequence[Dict[str, Any]], *, base=None):
    """The original scaled-logistic fit with its per-metre ridge prior. Reported, never used."""
    from .scenes import ScenePhysics

    phys = ScenePhysics.fit(rows, base=base, method="scaled_legacy")
    return phys


__all__ = [
    "FIT_LAPSE",
    "FIT_ISOTONIC",
    "FIT_LEGACY",
    "fit_lapse_logistic",
    "pava_increasing",
    "IsotonicCurve",
    "fit_isotonic",
    "isotonic_physics_summary",
    "fit_physics_lapse",
    "bootstrap_physics",
    "fit_physics_legacy",
]

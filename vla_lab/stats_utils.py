"""Lightweight stats helpers shared by the offline eval/analysis CLIs."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple


def mcnemar_test(success_a: Iterable[bool], success_b: Iterable[bool]) -> Optional[Tuple[float, float, int]]:
    """Paired binary McNemar test with continuity correction.

    Returns ``(chi2_statistic, p_value_two_sided, n_discordant)`` or ``None`` if undefined.

    Uses the chi-square approximation (df=1) with continuity correction:
    ``chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)``.
    """

    sa = [bool(x) for x in success_a]
    sb = [bool(x) for x in success_b]
    if len(sa) != len(sb) or not sa:
        return None
    n01 = sum(1 for a, b in zip(sa, sb) if (not a) and b)
    n10 = sum(1 for a, b in zip(sa, sb) if a and (not b))
    nd = int(n01 + n10)
    if nd == 0:
        return 0.0, 1.0, 0
    num = abs(float(n01) - float(n10)) - 1.0
    stat = (num * num) / float(nd)
    # Survival function for chi-square with 1 dof: sf(x) = 2*(1 - Phi(sqrt(x)))
    z = math.sqrt(max(0.0, stat))
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    p = float(min(1.0, max(0.0, p)))
    return float(stat), p, nd

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a Binomial proportion; (nan, nan) when n <= 0."""

    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


#: Below this many units a correlation coefficient is not reported, only the raw pattern. Seven
#: cells with one extreme outlier produced a Spearman rho of -0.46 that the first draft of the
#: paper quoted in its abstract; the number was not wrong so much as not analysable.
MIN_N_FOR_CORRELATION = 15


class UnderpoweredCorrelation(ValueError):
    """Raised when a correlation is requested on fewer than ``MIN_N_FOR_CORRELATION`` units."""


def _rank(x) -> list:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def guarded_correlation(x: Iterable[float], y: Iterable[float], *, method: str = "spearman",
                        min_n: int = MIN_N_FOR_CORRELATION, override: bool = False) -> dict:
    """A correlation coefficient that refuses to exist below a minimum ``n``.

    Returns ``{"rho", "n", "method", "reported"}``. With ``n < min_n`` and no ``override`` the
    coefficient is ``None`` and ``reported`` is ``False`` with a reason; an analysis that wants
    the number anyway has to say so in code (``override=True``), which is the point -- the
    decision to quote an underpowered statistic must be visible, never the default.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    n = min(len(xs), len(ys))
    out = {"n": n, "method": method, "min_n": int(min_n), "rho": None, "reported": False, "reason": None}
    if n < int(min_n) and not override:
        out["reason"] = f"n={n} < {min_n}: coefficient withheld; report the raw pattern instead"
        return out
    if n < 3:
        out["reason"] = "fewer than three units"
        return out
    a, b = (xs, ys) if method == "pearson" else (_rank(xs), _rank(ys))
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((v - ma) ** 2 for v in a))
    sb = math.sqrt(sum((v - mb) ** 2 for v in b))
    if sa == 0 or sb == 0:
        out["reason"] = "degenerate (constant) input"
        return out
    out["rho"] = sum((p - ma) * (q - mb) for p, q in zip(a, b)) / (sa * sb)
    out["reported"] = True
    if n < int(min_n):
        out["reason"] = f"OVERRIDE: reported on n={n} < {min_n}; must be labelled as underpowered"
    return out

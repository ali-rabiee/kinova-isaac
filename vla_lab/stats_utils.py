"""Lightweight stats helpers for offline eval aggregation (Wilson intervals live in `plot_metrics.py`)."""

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

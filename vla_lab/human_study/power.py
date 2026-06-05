"""Power / sample-size analysis from pilot effect sizes (memo §10 July milestone, §11).

Pure standard-library + the normal distribution implemented here (no scipy in the Isaac
env). Covers the comparisons the study actually runs: two-proportion (between-subjects
success / reliance rates), McNemar (paired success bits across conditions), and paired-t
(within-subjects continuous DVs such as trust / TLX), plus a ``recommend_sample_size`` that
turns a pilot into an N.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation; |err| < 1.2e-9)."""

    p = float(p)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _z_alpha(alpha: float, two_sided: bool) -> float:
    return normal_ppf(1.0 - alpha / 2.0) if two_sided else normal_ppf(1.0 - alpha)


# ---------------------------------------------------------------------------
# Two-proportion (between-subjects)
# ---------------------------------------------------------------------------


def power_two_proportion(p1: float, p2: float, n_per_group: int, alpha: float = 0.05, two_sided: bool = True) -> float:
    n = max(1, int(n_per_group))
    eff = abs(p1 - p2)
    if eff == 0:
        return float(alpha)
    pbar = (p1 + p2) / 2.0
    se0 = math.sqrt(2.0 * pbar * (1.0 - pbar) / n)
    se1 = math.sqrt(p1 * (1.0 - p1) / n + p2 * (1.0 - p2) / n)
    za = _z_alpha(alpha, two_sided)
    if se1 <= 0:
        return 1.0
    return float(min(1.0, max(0.0, normal_cdf((eff - za * se0) / se1))))


def sample_size_two_proportion(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True) -> int:
    eff = abs(p1 - p2)
    if eff == 0:
        return 10 ** 9
    pbar = (p1 + p2) / 2.0
    za = _z_alpha(alpha, two_sided)
    zb = normal_ppf(power)
    num = za * math.sqrt(2.0 * pbar * (1.0 - pbar)) + zb * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return int(math.ceil((num * num) / (eff * eff)))


# ---------------------------------------------------------------------------
# McNemar (paired success bits across conditions)
# ---------------------------------------------------------------------------


def sample_size_mcnemar(p01: float, p10: float, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True) -> int:
    """Pairs needed to detect a difference given discordant cell probabilities (Connor 1987)."""

    p_disc = p01 + p10
    if p_disc <= 0 or p01 == p10:
        return 10 ** 9
    psi = max(p01, p10) / max(1e-9, min(p01, p10))  # odds ratio of discordances
    za = _z_alpha(alpha, two_sided)
    zb = normal_ppf(power)
    num = za * (psi + 1.0) + zb * math.sqrt((psi + 1.0) ** 2 - (psi - 1.0) ** 2 * p_disc)
    den = (psi - 1.0) ** 2 * p_disc
    return int(math.ceil((num * num) / den))


# ---------------------------------------------------------------------------
# Paired-t (within-subjects continuous DVs)
# ---------------------------------------------------------------------------


def power_paired_t(d: float, n: int, alpha: float = 0.05, two_sided: bool = True) -> float:
    """Normal approximation to paired-t power for Cohen's ``d`` and ``n`` participants."""

    n = max(2, int(n))
    ncp = abs(d) * math.sqrt(n)
    za = _z_alpha(alpha, two_sided)
    power = normal_cdf(ncp - za)
    if two_sided:
        power += normal_cdf(-ncp - za)
    return float(min(1.0, max(0.0, power)))


def sample_size_paired(d: float, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True) -> int:
    """Participants needed for a paired comparison at Cohen's ``d``."""

    if d == 0:
        return 10 ** 9
    za = _z_alpha(alpha, two_sided)
    zb = normal_ppf(power)
    n = ((za + zb) / abs(d)) ** 2
    return int(math.ceil(n) + 1)  # +1 since paired-t df = n-1


# ---------------------------------------------------------------------------
# Pilot -> recommendation
# ---------------------------------------------------------------------------


@dataclass
class PilotRecommendation:
    p1: float
    p2: float
    effect: float
    cohens_h: float
    design: str
    n_recommended: int
    achieved_power_at_n: float
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "p1": self.p1, "p2": self.p2, "effect": self.effect, "cohens_h": self.cohens_h,
            "design": self.design, "n_recommended": self.n_recommended,
            "achieved_power_at_n": self.achieved_power_at_n, "notes": self.notes,
        }


def cohens_h(p1: float, p2: float) -> float:
    return float(2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2)))))


def recommend_sample_size(
    success_a: Sequence[bool],
    success_b: Sequence[bool],
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    design: str = "between",
    two_sided: bool = True,
) -> PilotRecommendation:
    """Turn two pilot success-bit lists into a recommended N (per group / pairs)."""

    a = [bool(x) for x in success_a]
    b = [bool(x) for x in success_b]
    p1 = sum(a) / max(1, len(a))
    p2 = sum(b) / max(1, len(b))
    eff = abs(p1 - p2)
    if design == "paired" and len(a) == len(b):
        p01 = sum(1 for x, y in zip(a, b) if (not x) and y) / max(1, len(a))
        p10 = sum(1 for x, y in zip(a, b) if x and (not y)) / max(1, len(a))
        n = sample_size_mcnemar(p01, p10, alpha, power, two_sided)
        ach = power  # target; exact paired power requires the realized discordance
        notes = f"paired/McNemar p01={p01:.3f} p10={p10:.3f}"
    else:
        n = sample_size_two_proportion(p1, p2, alpha, power, two_sided)
        ach = power_two_proportion(p1, p2, n, alpha, two_sided)
        notes = "two-proportion (between-subjects)"
    return PilotRecommendation(p1, p2, eff, cohens_h(p1, p2), design, int(n), float(ach), notes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _success_rates_from_log(path: str, cond_a: str, cond_b: str, paired: bool):
    """Pull success bits for two conditions out of a study JSONL (via analyze helpers)."""

    from .analyze import _load, paired_success

    trials, _ = _load([str(path)])
    if not trials:
        raise SystemExit(f"[power] no trials found in {path}")
    if paired:
        a_bits, b_bits = paired_success(trials, cond_a, cond_b)
    else:
        a_bits = [bool(r.get("result", {}).get("success", False)) for r in trials if r.get("condition_name") == cond_a]
        b_bits = [bool(r.get("result", {}).get("success", False)) for r in trials if r.get("condition_name") == cond_b]
    return a_bits, b_bits


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Power / sample-size analysis for the human study")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--one-sided", action="store_true", help="one-sided test (default two-sided)")
    ap.add_argument("--design", type=str, default="between", choices=["between", "paired"])
    # Mode A: a continuous paired DV (trust / TLX) via Cohen's d.
    ap.add_argument("--d", type=float, default=None, help="Cohen's d for a paired continuous DV (within-subjects)")
    # Mode B: explicit success/reliance proportions.
    ap.add_argument("--p1", type=float, default=None, help="proportion in condition 1 (e.g. appropriate-reliance rate)")
    ap.add_argument("--p2", type=float, default=None, help="proportion in condition 2")
    # Mode C: estimate proportions from a pilot study log.
    ap.add_argument("--from-log", type=str, default=None, help="study JSONL to estimate p1,p2 from two conditions")
    ap.add_argument("--cond-a", type=str, default="allocator_type")
    ap.add_argument("--cond-b", type=str, default="autonomy")
    args = ap.parse_args(argv)

    two_sided = not args.one_sided

    if args.d is not None:
        n = sample_size_paired(args.d, args.alpha, args.power, two_sided)
        ach = power_paired_t(args.d, n, args.alpha, two_sided)
        print(f"[power] paired-t  d={args.d:g}  alpha={args.alpha}  target_power={args.power}")
        print(f"[power] => n={n} participants (achieved power≈{ach:.3f})")
        return 0

    if args.from_log is not None:
        a_bits, b_bits = _success_rates_from_log(args.from_log, args.cond_a, args.cond_b, paired=(args.design == "paired"))
        rec = recommend_sample_size(a_bits, b_bits, alpha=args.alpha, power=args.power, design=args.design, two_sided=two_sided)
        print(f"[power] from {args.from_log}: {args.cond_a} (p={rec.p1:.3f}, n={len(a_bits)}) vs "
              f"{args.cond_b} (p={rec.p2:.3f}, n={len(b_bits)})")
        unit = "pairs" if args.design == "paired" else "per group"
        print(f"[power] {rec.notes}  effect={rec.effect:.3f}  h={rec.cohens_h:.3f}")
        print(f"[power] => n={rec.n_recommended} {unit} (achieved power≈{rec.achieved_power_at_n:.3f})")
        return 0

    if args.p1 is not None and args.p2 is not None:
        if args.design == "paired":
            # Without a log we lack discordances; approximate via the effect-size (Cohen's h).
            h = cohens_h(args.p1, args.p2)
            n = sample_size_paired(h, args.alpha, args.power, two_sided)
            ach = power_paired_t(h, n, args.alpha, two_sided)
            print(f"[power] paired proportions (approx via Cohen's h={h:.3f}); for an exact paired/McNemar N, pass --from-log")
            print(f"[power] => n≈{n} pairs (achieved power≈{ach:.3f})")
        else:
            n = sample_size_two_proportion(args.p1, args.p2, args.alpha, args.power, two_sided)
            ach = power_two_proportion(args.p1, args.p2, n, args.alpha, two_sided)
            print(f"[power] two-proportion  p1={args.p1:g}  p2={args.p2:g}  h={cohens_h(args.p1, args.p2):.3f}")
            print(f"[power] => n={n} per group (achieved power≈{ach:.3f})")
        return 0

    ap.error("provide one of: --d, (--p1 and --p2), or --from-log")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

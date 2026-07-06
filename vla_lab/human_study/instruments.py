"""Validated subjective instruments + scoring (NASA-TLX, Jian 2000 trust, MDMT).

Scoring only — administration (showing items, collecting Likert responses) is the study
front-end's job; these functions take the collected responses and return the standard
composites used in the analysis. Item wordings are documented so a front-end can present
them verbatim. No external dependencies.

References
---------
- Hart & Staveland (1988), NASA-TLX (workload).
- Jian, Bisantz & Drury (2000), trust between people and automation (12-item).
- Ullman & Malle (2018/2019), Multi-Dimensional Measure of Trust (MDMT).
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Union

Number = Union[int, float]

# ---------------------------------------------------------------------------
# NASA-TLX
# ---------------------------------------------------------------------------

NASA_TLX_SUBSCALES = (
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
)


def _as_subscale_map(scores: Union[Mapping[str, Number], Sequence[Number]]) -> Dict[str, float]:
    if isinstance(scores, Mapping):
        return {k: float(scores[k]) for k in NASA_TLX_SUBSCALES if k in scores}
    vals = list(scores)
    if len(vals) != len(NASA_TLX_SUBSCALES):
        raise ValueError(f"NASA-TLX expects {len(NASA_TLX_SUBSCALES)} subscales; got {len(vals)}")
    return {k: float(v) for k, v in zip(NASA_TLX_SUBSCALES, vals)}


def nasa_tlx_raw(scores: Union[Mapping[str, Number], Sequence[Number]]) -> float:
    """Raw TLX (RTLX): unweighted mean of the six 0–100 subscales (higher = more workload).

    Convention: every subscale is provided so higher means *more demanding* (the Performance
    scale anchored Good=0 … Poor=100).
    """

    m = _as_subscale_map(scores)
    if len(m) != len(NASA_TLX_SUBSCALES):
        raise ValueError("NASA-TLX raw score needs all six subscales")
    return float(sum(m.values()) / len(NASA_TLX_SUBSCALES))


def tlx_weights_from_comparisons(chosen: Sequence[str]) -> Dict[str, int]:
    """Tally the 15 pairwise-comparison choices into per-subscale weights (each 0–5)."""

    counts = {k: 0 for k in NASA_TLX_SUBSCALES}
    for c in chosen:
        if c in counts:
            counts[c] += 1
    return counts


def nasa_tlx_weighted(
    scores: Union[Mapping[str, Number], Sequence[Number]],
    weights: Mapping[str, Number],
) -> float:
    """Weighted TLX: ``Σ weight_i · rating_i / Σ weight_i`` (weights sum to 15 in the standard)."""

    m = _as_subscale_map(scores)
    total_w = sum(float(weights.get(k, 0.0)) for k in NASA_TLX_SUBSCALES)
    if total_w <= 0:
        return nasa_tlx_raw(scores)
    num = sum(float(weights.get(k, 0.0)) * m.get(k, 0.0) for k in NASA_TLX_SUBSCALES)
    return float(num / total_w)


# ---------------------------------------------------------------------------
# Jian, Bisantz & Drury (2000) trust-in-automation scale
# ---------------------------------------------------------------------------

JIAN_ITEMS = (
    "The system is deceptive",                                    # 1  distrust
    "The system behaves in an underhanded manner",                # 2  distrust
    "I am suspicious of the system's intent, action, or outputs", # 3  distrust
    "I am wary of the system",                                    # 4  distrust
    "The system's actions will have a harmful or injurious outcome",  # 5 distrust
    "I am confident in the system",                               # 6  trust
    "The system provides security",                               # 7  trust
    "The system has integrity",                                   # 8  trust
    "The system is dependable",                                   # 9  trust
    "The system is reliable",                                     # 10 trust
    "I can trust the system",                                     # 11 trust
    "I am familiar with the system",                              # 12 familiarity
)
# 1-indexed item numbers
JIAN_DISTRUST_ITEMS = (1, 2, 3, 4, 5)
JIAN_TRUST_ITEMS = (6, 7, 8, 9, 10, 11)
JIAN_FAMILIARITY_ITEM = 12


def _jian_list(responses: Union[Mapping[int, Number], Sequence[Number]]) -> Dict[int, float]:
    if isinstance(responses, Mapping):
        return {int(k): float(v) for k, v in responses.items()}
    vals = list(responses)
    if len(vals) != 12:
        raise ValueError(f"Jian scale expects 12 responses; got {len(vals)}")
    return {i + 1: float(v) for i, v in enumerate(vals)}


def jian_trust_score(
    responses: Union[Mapping[int, Number], Sequence[Number]],
    *,
    scale_max: int = 7,
    include_familiarity: bool = False,
) -> Dict[str, float]:
    """Score the 12-item scale (1..scale_max Likert).

    Returns the ``trust`` (items 6–11), ``distrust`` (items 1–5), the ``familiarity`` item,
    and an ``overall`` composite that reverse-codes the distrust items
    (``reverse(x) = scale_max + 1 - x``) and averages them with the trust items.
    """

    r = _jian_list(responses)
    trust = sum(r[i] for i in JIAN_TRUST_ITEMS) / len(JIAN_TRUST_ITEMS)
    distrust = sum(r[i] for i in JIAN_DISTRUST_ITEMS) / len(JIAN_DISTRUST_ITEMS)
    rev = [(scale_max + 1 - r[i]) for i in JIAN_DISTRUST_ITEMS]
    composite_items = rev + [r[i] for i in JIAN_TRUST_ITEMS]
    if include_familiarity:
        composite_items.append(r[JIAN_FAMILIARITY_ITEM])
    overall = sum(composite_items) / len(composite_items)
    return {
        "trust": float(trust),
        "distrust": float(distrust),
        "familiarity": float(r[JIAN_FAMILIARITY_ITEM]),
        "overall": float(overall),
    }


# ---------------------------------------------------------------------------
# Multi-Dimensional Measure of Trust (MDMT)
# ---------------------------------------------------------------------------

MDMT_CAPACITY_SUBSCALES = ("reliable", "capable")
MDMT_MORAL_SUBSCALES = ("ethical", "sincere")
MDMT_SUBSCALES = MDMT_CAPACITY_SUBSCALES + MDMT_MORAL_SUBSCALES


def mdmt_score(responses_by_subscale: Mapping[str, Sequence[Number]]) -> Dict[str, float]:
    """Score the MDMT from per-subscale item ratings (each item 0–7).

    Returns each subscale mean plus ``capacity_trust`` (reliable+capable), ``moral_trust``
    (ethical+sincere), and an ``overall`` mean over all provided subscales.
    """

    out: Dict[str, float] = {}
    for sub in MDMT_SUBSCALES:
        items = responses_by_subscale.get(sub)
        if items:
            out[sub] = float(sum(float(x) for x in items) / len(list(items)))
    cap = [out[s] for s in MDMT_CAPACITY_SUBSCALES if s in out]
    mor = [out[s] for s in MDMT_MORAL_SUBSCALES if s in out]
    if cap:
        out["capacity_trust"] = float(sum(cap) / len(cap))
    if mor:
        out["moral_trust"] = float(sum(mor) / len(mor))
    present = [out[s] for s in MDMT_SUBSCALES if s in out]
    if present:
        out["overall"] = float(sum(present) / len(present))
    return out

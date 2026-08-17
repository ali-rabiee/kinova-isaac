"""Validated subjective instruments + scoring (NASA-TLX, Jian 2000 trust, MDMT, EHI).

Scoring only — administration (showing items, collecting Likert responses) is the study
front-end's job; these functions take the collected responses and return the standard
composites used in the analysis. Item wordings are documented so a front-end can present
them verbatim. No external dependencies.

Shared by **both tracks** (``rehab.md`` §5: EXTEND). NASA-TLX is used by the VLA study and by
Phase 0; Jian/MDMT are VLA-only; the Edinburgh Handedness Inventory and the session-burden
scale were added for Phase 0, where the handedness inventory is *required* — it is what
defines "nonpreferred arm", the label the entire estimand is expressed in.

References
---------
- Hart & Staveland (1988), NASA-TLX (workload).
- Jian, Bisantz & Drury (2000), trust between people and automation (12-item).
- Ullman & Malle (2018/2019), Multi-Dimensional Measure of Trust (MDMT).
- Oldfield (1971), Edinburgh Handedness Inventory (laterality quotient).
- Borg (1982), CR10 category-ratio scale — the anchor style of the fatigue item below.
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


# ---------------------------------------------------------------------------
# Edinburgh Handedness Inventory (Oldfield 1971)  —  Phase 0, required
# ---------------------------------------------------------------------------

EHI_ITEMS = (
    "Writing",
    "Drawing",
    "Throwing",
    "Scissors",
    "Toothbrush",
    "Knife (without fork)",
    "Spoon",
    "Broom (upper hand)",
    "Striking a match (match hand)",
    "Opening a box (lid hand)",
)

#: Per-item response scale. Numeric input is accepted directly in [-2, +2].
EHI_SCALE = {
    "always_left": -2,
    "usually_left": -1,
    "no_preference": 0,
    "usually_right": 1,
    "always_right": 2,
}

#: |LQ| below this is "mixed"; the common Oldfield cutoff.
EHI_MIXED_CUTOFF = 40.0


def _ehi_item_scores(responses: Union[Mapping[str, Number], Sequence[Number]]) -> Dict[str, float]:
    """Normalize either a per-item mapping or a 10-long sequence to item -> [-2, +2]."""

    if isinstance(responses, Mapping):
        out: Dict[str, float] = {}
        for k, v in responses.items():
            key = str(k)
            if key not in EHI_ITEMS:
                raise ValueError(f"unknown EHI item {k!r}; expected one of {EHI_ITEMS}")
            out[key] = float(EHI_SCALE[str(v)]) if isinstance(v, str) else float(v)
        return out
    vals = list(responses)
    if len(vals) != len(EHI_ITEMS):
        raise ValueError(f"EHI expects {len(EHI_ITEMS)} item responses; got {len(vals)}")
    return {
        k: (float(EHI_SCALE[str(v)]) if isinstance(v, str) else float(v))
        for k, v in zip(EHI_ITEMS, vals)
    }


def edinburgh_handedness(
    responses: Union[Mapping[str, Number], Sequence[Number]],
    *,
    mixed_cutoff: float = EHI_MIXED_CUTOFF,
) -> Dict[str, object]:
    """Score the EHI. Returns the laterality quotient, the classification, and the arm labels.

    ``LQ = 100 * sum(score) / sum(|score|)``, in ``[-100, +100]``; positive is right-handed.
    Items the respondent skipped (omitted from a mapping) are excluded from both sums, which
    is Oldfield's own handling — ``n_items`` reports how many actually counted.

    ``nonpreferred_arm`` is the label the Phase 0 estimand is expressed in
    (``pi* = P(nonpreferred)``), so it is returned here rather than derived ad hoc at three
    call sites. A **mixed** result is returned as such and not silently forced to a side: a
    participant with no clear preference has no "nonpreferred arm", and enrolling them would
    make their ``pi*`` uninterpretable (an exclusion criterion, not a rounding decision).
    """

    scores = _ehi_item_scores(responses)
    total = sum(scores.values())
    abs_total = sum(abs(v) for v in scores.values())
    if abs_total <= 0:
        lq = 0.0
    else:
        lq = float(100.0 * total / abs_total)
    if lq >= float(mixed_cutoff):
        handedness, nonpreferred, preferred = "right", "left", "right"
    elif lq <= -float(mixed_cutoff):
        handedness, nonpreferred, preferred = "left", "right", "left"
    else:
        handedness, nonpreferred, preferred = "mixed", None, None
    return {
        "lq": lq,
        "handedness": handedness,
        "preferred_arm": preferred,
        "nonpreferred_arm": nonpreferred,
        "n_items": len(scores),
        "mixed_cutoff": float(mixed_cutoff),
        "item_scores": scores,
    }


# ---------------------------------------------------------------------------
# Session burden / fatigue  —  Phase 0
# ---------------------------------------------------------------------------

#: **Not a validated instrument.** A short study-specific set for the secondary "participant
#: burden" outcome, alongside NASA-TLX. The fatigue item uses Borg CR10-style anchors
#: (0 = nothing at all, 10 = maximal); the others are 0-10 with matching anchors. Reported as
#: an ad-hoc scale in the paper, never as a validated measure.
SESSION_BURDEN_ITEMS = (
    "arm_fatigue",        # "How tired do your arms feel right now?"      0 none .. 10 maximal
    "arm_heaviness",      # "How heavy do your arms feel?"                0 none .. 10 maximal
    "effort",             # "How much effort did that block take?"        0 none .. 10 maximal
    "discomfort",         # "Any discomfort in your arms/shoulders?"      0 none .. 10 maximal
    "willing_to_continue",  # "How willing are you to continue?"          0 not at all .. 10 very
)
#: Higher = more burden for every item except this one, which is reverse-coded.
SESSION_BURDEN_REVERSED = ("willing_to_continue",)
SESSION_BURDEN_MAX = 10.0


def session_burden(
    responses: Union[Mapping[str, Number], Sequence[Number]],
    *,
    scale_max: float = SESSION_BURDEN_MAX,
) -> Dict[str, float]:
    """Score the session-burden set: each item, plus a reverse-coded ``burden`` composite.

    Fatigue is both an ethics and a validity issue (``rehab.md`` §11): it drifts arm choice and
    would masquerade as carryover, so it is measured at every block boundary and modelled,
    not assumed away.
    """

    if isinstance(responses, Mapping):
        vals = {str(k): float(v) for k, v in responses.items() if str(k) in SESSION_BURDEN_ITEMS}
    else:
        seq = list(responses)
        if len(seq) != len(SESSION_BURDEN_ITEMS):
            raise ValueError(f"session burden expects {len(SESSION_BURDEN_ITEMS)} items; got {len(seq)}")
        vals = {k: float(v) for k, v in zip(SESSION_BURDEN_ITEMS, seq)}
    if not vals:
        raise ValueError("no recognized session-burden items provided")
    coded = {
        k: (float(scale_max) - v if k in SESSION_BURDEN_REVERSED else v) for k, v in vals.items()
    }
    out: Dict[str, float] = {k: float(v) for k, v in vals.items()}
    out["burden"] = float(sum(coded.values()) / len(coded))
    if "arm_fatigue" in vals:
        out["fatigue"] = float(vals["arm_fatigue"])
    return out

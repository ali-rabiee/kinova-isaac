"""The HRI human-subjects study layer (memo §5) — contribution C4.

Everything here is hardware/human-I/O-agnostic and unit-testable: the robot driver and the
human's responses are *injected* into the session runner, so the protocol, instruments,
reliance metrics, power analysis, and study analysis all run offline (and in CI) and then
drop onto the real Kinova JACO unchanged.

Modules:
  - ``protocol``   — IVs (graded observability × 5 interaction conditions), counterbalancing,
    and a reproducible per-participant :class:`SessionPlan`.
  - ``instruments``— NASA-TLX, the Jian et al. (2000) trust-in-automation scale, and the
    MDMT — item definitions + scoring.
  - ``session``    — an **autonomous** session runner (the robot genuinely escalates and
    consumes a typed/clicked answer) that logs every dependent variable.
  - ``reliance``   — appropriate reliance, the over-trust trap, under-trust, and trust
    calibration — the headline HRI constructs (§5.3, §5.4).
  - ``power``      — power / sample-size analysis from pilot effect sizes (no scipy).
  - ``analyze``    — a CLI that turns a study log into per-condition tables + figures.
"""

from __future__ import annotations

from .instruments import (
    NASA_TLX_SUBSCALES,
    jian_trust_score,
    mdmt_score,
    nasa_tlx_raw,
    nasa_tlx_weighted,
)
from .power import (
    normal_cdf,
    normal_ppf,
    power_two_proportion,
    sample_size_paired,
    sample_size_two_proportion,
)
from .protocol import (
    Condition,
    ObservabilityLevel,
    SessionPlan,
    StudyProtocol,
    Trial,
    balanced_latin_square,
    default_conditions,
    default_observability,
    generate_session_plan,
)
from .reliance import (
    RelianceTrial,
    classify_reliance,
    reliance_summary,
    trust_calibration,
)

__all__ = [
    "Condition",
    "ObservabilityLevel",
    "Trial",
    "SessionPlan",
    "StudyProtocol",
    "balanced_latin_square",
    "default_conditions",
    "default_observability",
    "generate_session_plan",
    "NASA_TLX_SUBSCALES",
    "nasa_tlx_raw",
    "nasa_tlx_weighted",
    "jian_trust_score",
    "mdmt_score",
    "RelianceTrial",
    "classify_reliance",
    "reliance_summary",
    "trust_calibration",
    "normal_cdf",
    "normal_ppf",
    "power_two_proportion",
    "sample_size_two_proportion",
    "sample_size_paired",
]

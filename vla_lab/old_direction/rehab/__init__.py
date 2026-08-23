"""Phase 0 of the post-stroke adaptive-robotics program (see ``vla_lab/rehab.md``).

**Phase 0 (HRI 2027).** With healthy adults and the Kinova Gen2 (JACO 2, ``j2n6s300``),
under a matched interaction budget, can a personalized, carryover-aware COACH / WAIT /
ASSESS policy estimate an individual's unprompted arm-choice map more accurately, and with
better-calibrated uncertainty, than naive assessment schedules?

.. warning::
   **Claim boundary (rehab.md §1.6).** Phase 0 is a *healthy-adult physical-HRI measurement
   and scheduling* result. It is **not** evidence of stroke nonuse, rehabilitation efficacy,
   clinical construct validity, or recovery. "Nonpreferred arm" is a handedness-defined
   label, not a paretic limb.

This package is the Phase 0 track. It coexists with the VLA / act-compute-query track in the
same repository under the rules of ``rehab.md`` §3:

1. Everything Phase 0 lives here and imports *upward* from shared utilities
   (:mod:`vla_lab.calibration`, :mod:`vla_lab.stats_utils`,
   :mod:`vla_lab.human_study.instruments`, :mod:`vla_lab.human_study.power`).
   **Nothing in the VLA track may import from** ``vla_lab.rehab``.
2. Shared modules are extended, never repurposed: where a Phase 0 need is close to an
   existing VLA abstraction (the allocator's three-way decision, the simulated human), the
   design is *borrowed* into a fresh implementation here rather than branched into the
   original.
3. Configs are ``vla_lab/configs/rehab_*.yaml``; scripts are
   ``vla_lab/scripts/rehab_*.sh``; logs are ``logs/rehab/participant_<PID>/session_<TS>/``.

Module map (see ``rehab.md`` §4):

- :mod:`~vla_lab.rehab.workspace`     bilateral target geometry, frames, reachability (W1)
- :mod:`~vla_lab.rehab.contract`      the Phase 0 contract as code, hashed (W1/§9)
- :mod:`~vla_lab.rehab.prompts`       COACH content library + effort manipulation (W9)
- :mod:`~vla_lab.rehab.trial`         Trial/TrialResult + the per-trial phase machine (W2)
- :mod:`~vla_lab.rehab.logging`       event-locked session writer (W2)
- :mod:`~vla_lab.rehab.carryover`     latent carryover model over (lambda, beta, g) (W3)
- :mod:`~vla_lab.rehab.estimand`      pi*(l) estimators + error/calibration metrics (W4)
- :mod:`~vla_lab.rehab.scheduler`     B0-B4 + B4's ablations (W5)
- :mod:`~vla_lab.rehab.sim_participant`  generative participant (W6)
- :mod:`~vla_lab.rehab.protocol`      blocks, counterbalancing, reference/retest (W7)
- :mod:`~vla_lab.rehab.observation`   arm-choice observers + agreement + calibration (W8)
- :mod:`~vla_lab.rehab.apparatus`     null / Isaac twin / real Gen2 backends (W10, W11)
- :mod:`~vla_lab.rehab.safety`        human-proximate interlocks (W12)
- :mod:`~vla_lab.rehab.session`       the one session runner over every backend (W14)
- :mod:`~vla_lab.rehab.verify_session`  the Phase 0 session gate (W15)
- :mod:`~vla_lab.rehab.power`         Monte-Carlo power over sim_participant (W16)
- :mod:`~vla_lab.rehab.analyze`       outcomes, tables, figures (W17)
"""

from __future__ import annotations

__version__ = "0.1.0"

# The Phase 0 action set (rehab.md §1.3). Nothing else is in scope for Phase 0.
COACH = "COACH"
WAIT = "WAIT"
ASSESS = "ASSESS"
ACTIONS = (COACH, WAIT, ASSESS)

# Canonical arm labels used in trials.jsonl. The *physical* side (left/right) is kept in
# observers.jsonl; the estimand is P(nonpreferred), so the analysis label is handedness-
# relative and the raw label is not.
ARM_PREFERRED = "preferred"
ARM_NONPREFERRED = "nonpreferred"
ARM_NONE = "none"          # no reach observed within the GO window
ARM_AMBIGUOUS = "ambiguous"  # observed, but the observer could not resolve a side

CLAIM_BOUNDARY = (
    "Phase 0 is a healthy-adult physical-HRI measurement and scheduling result. It is NOT "
    "evidence of stroke nonuse, rehabilitation efficacy, clinical construct validity, or "
    "recovery. 'Nonpreferred arm' is a handedness-defined label, not a paretic limb."
)

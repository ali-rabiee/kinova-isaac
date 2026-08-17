"""W8/W13 — arm-choice observation and physical calibration.

Three observers behind one protocol (:mod:`~vla_lab.rehab.observation.base`), the agreement
machinery that decides whether the online one may be trusted
(:mod:`~vla_lab.rehab.observation.coding`), and the frame/camera solves the whole apparatus is
indexed against (:mod:`~vla_lab.rehab.observation.calibration`).

The pre-registered acceptance gate: vision-vs-coded Cohen's ``kappa >= 0.9`` at the pilot,
with detection latency inside the trial's SELECT budget. Below that, the study runs
**keyed-only** and vision becomes an exploratory contribution — decided at the pilot, not
later (``rehab.md`` §6/W8).
"""

from __future__ import annotations

from .base import (
    SOURCE_CODED,
    SOURCE_ONLINE,
    ArmChoiceObserver,
    ArmSelection,
    BaseObserver,
    CompositeObserver,
    arm_from_side,
    side_from_arm,
)
from .calibration import (
    CalibrationBundle,
    ParticipantFrame,
    apply_homography,
    homography_residuals,
    pinhole_intrinsics,
    solve_participant_frame,
    solve_table_homography,
)
from .coding import (
    AgreementReport,
    agreement_report,
    cohens_kappa,
    ingest_coded_labels,
    labels_by_observer,
    session_agreement,
)
from .keyed import DEFAULT_KEYMAP, KeyedObserver, ScriptedKeySource
from .vision import (
    HandDetector,
    HandObservation,
    ScriptedHandDetector,
    VisionConfig,
    VisionObserver,
)

#: The pre-registered vision acceptance threshold (W8).
KAPPA_ACCEPTANCE = 0.9

__all__ = [
    "SOURCE_ONLINE",
    "SOURCE_CODED",
    "KAPPA_ACCEPTANCE",
    "ArmSelection",
    "ArmChoiceObserver",
    "BaseObserver",
    "CompositeObserver",
    "arm_from_side",
    "side_from_arm",
    "KeyedObserver",
    "ScriptedKeySource",
    "DEFAULT_KEYMAP",
    "VisionObserver",
    "VisionConfig",
    "HandObservation",
    "HandDetector",
    "ScriptedHandDetector",
    "cohens_kappa",
    "agreement_report",
    "AgreementReport",
    "session_agreement",
    "labels_by_observer",
    "ingest_coded_labels",
    "pinhole_intrinsics",
    "solve_table_homography",
    "apply_homography",
    "homography_residuals",
    "ParticipantFrame",
    "solve_participant_frame",
    "CalibrationBundle",
]

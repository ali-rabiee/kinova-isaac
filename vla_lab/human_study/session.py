"""Autonomous study session runner + dependent-variable logging.

The runner is deliberately I/O-agnostic: the *robot* (one trial → outcome) and the *human*
(questionnaire responses) are **injected**, so the exact same orchestration code runs in CI
against a simulator and on the real Kinova JACO against the hardware loop. Per the resolved
design, querying is autonomous — the robot escalates and consumes a typed/clicked answer via
the allocator's :class:`~vla_lab.allocation.query.QueryInterface`; there is no
Wizard-of-Oz operator.

Inject:
  - ``episode_runner(trial) -> EpisodeResult`` — execute one trial under the trial's
    controller + occlusion, returning task/interaction outcomes (the real JACO loop or the
    bundled :class:`SimEpisodeRunner`).
  - ``questionnaire_provider(instrument, condition_name) -> responses`` — collect raw
    questionnaire responses at block boundaries (a study front-end, or :class:`SimQuestionnaireProvider`).

A :class:`SimEpisodeRunner` + :class:`SimQuestionnaireProvider` are included so the whole
study pipeline (generate → run → analyze → figures) is exercisable offline while debugging
the protocol (the memo's July "pilot to debug the protocol" step).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from . import instruments as _instr
from .protocol import SessionPlan, Trial
from .reliance import RelianceTrial, classify_reliance


# ---------------------------------------------------------------------------
# Outcome containers
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """What one trial produced. ``robot_correct`` is the final-executed-action correctness."""

    success: bool = False
    robot_correct: bool = False
    human_intervened: bool = False
    robot_queried: bool = False
    robot_signaled_uncertainty: bool = False
    n_queries: int = 0
    n_decisions: int = 0
    query_latency_s: float = 0.0
    time_to_complete_s: float = 0.0
    human_idle_s: float = 0.0
    identifiable: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionResult:
    participant_id: str
    trials: List[Dict[str, Any]]
    reliance_trials: List[RelianceTrial]
    questionnaires: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        n = len(self.reliance_trials)
        return {
            "participant_id": self.participant_id,
            "n_trials": n,
            "n_questionnaires": len(self.questionnaires),
        }


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class StudyLogger:
    """Append-only JSONL of ``trial`` and ``questionnaire`` records."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", buffering=1)

    def log_trial(self, trial: Trial, result: EpisodeResult, reliance_label: str, participant_id: str) -> None:
        row = {
            "kind": "trial",
            "participant_id": participant_id,
            **trial.to_dict(),
            "result": asdict(result),
            "reliance_label": reliance_label,
        }
        self._fp.write(json.dumps(row) + "\n")

    def log_questionnaire(self, participant_id: str, condition_name: str, instrument: str, responses: Any, scores: Dict[str, float]) -> None:
        row = {
            "kind": "questionnaire",
            "participant_id": participant_id,
            "condition_name": condition_name,
            "instrument": instrument,
            "responses": responses,
            "scores": scores,
        }
        self._fp.write(json.dumps(row) + "\n")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Questionnaire scoring dispatch
# ---------------------------------------------------------------------------


def score_instrument(instrument: str, responses: Any) -> Dict[str, float]:
    name = str(instrument).lower()
    if name in ("nasa_tlx", "tlx"):
        if isinstance(responses, dict) and "weights" in responses:
            return {"weighted_tlx": _instr.nasa_tlx_weighted(responses.get("subscales", responses), responses["weights"]),
                    "raw_tlx": _instr.nasa_tlx_raw(responses.get("subscales", responses))}
        return {"raw_tlx": _instr.nasa_tlx_raw(responses)}
    if name in ("jian", "jian_trust", "trust"):
        return _instr.jian_trust_score(responses)
    if name in ("mdmt",):
        return _instr.mdmt_score(responses)
    return {}


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class SessionRunner:
    def __init__(
        self,
        plan: SessionPlan,
        episode_runner: Callable[[Trial], EpisodeResult],
        *,
        questionnaire_provider: Optional[Callable[[str, str], Any]] = None,
        instruments: Optional[List[str]] = None,
        logger: Optional[StudyLogger] = None,
    ) -> None:
        self.plan = plan
        self.episode_runner = episode_runner
        self.questionnaire_provider = questionnaire_provider
        self.instruments = list(instruments or ["nasa_tlx", "jian_trust"])
        self.logger = logger

    def run(self) -> SessionResult:
        trials_out: List[Dict[str, Any]] = []
        reliance_trials: List[RelianceTrial] = []
        questionnaires: List[Dict[str, Any]] = []

        # Group trials by block so we can administer questionnaires at block boundaries.
        blocks: Dict[int, List[Trial]] = {}
        for t in self.plan.trials:
            blocks.setdefault(t.block_idx, []).append(t)

        for block_idx in sorted(blocks):
            block_trials = blocks[block_idx]
            condition_name = block_trials[0].condition_name if block_trials else ""
            for trial in block_trials:
                res = self.episode_runner(trial)
                rt = RelianceTrial(
                    robot_correct=bool(res.robot_correct),
                    human_intervened=bool(res.human_intervened),
                    success=bool(res.success),
                    robot_queried=bool(res.robot_queried),
                    robot_signaled_uncertainty=bool(res.robot_signaled_uncertainty),
                    condition_name=trial.condition_name,
                    participant_id=self.plan.participant_id,
                    occlusion_fraction=float(trial.occlusion_fraction),
                )
                label = classify_reliance(rt)
                reliance_trials.append(rt)
                trials_out.append({**trial.to_dict(), "result": asdict(res), "reliance_label": label})
                if self.logger:
                    self.logger.log_trial(trial, res, label, self.plan.participant_id)

            # Questionnaires for this condition block.
            if self.questionnaire_provider is not None:
                for instrument in self.instruments:
                    responses = self.questionnaire_provider(instrument, condition_name)
                    if responses is None:
                        continue
                    scores = score_instrument(instrument, responses)
                    questionnaires.append({"condition_name": condition_name, "instrument": instrument, "scores": scores})
                    if self.logger:
                        self.logger.log_questionnaire(self.plan.participant_id, condition_name, instrument, responses, scores)

        if self.logger:
            self.logger.close()
        return SessionResult(self.plan.participant_id, trials_out, reliance_trials, questionnaires)


# ---------------------------------------------------------------------------
# Offline pilot simulators (debug the protocol + analysis before humans/hardware)
# ---------------------------------------------------------------------------


@dataclass
class SimWorldParams:
    """Knobs of the synthetic world used to pilot the analysis pipeline."""

    base_accuracy_clean: float = 0.9
    accuracy_drop_per_occ: float = 1.1   # accuracy falls with occlusion fraction
    identifiable_drop_per_occ: float = 1.7  # optimal action becomes unidentifiable under occ
    acc_irreducible: float = 0.15        # accuracy when info is absent and no help sought
    query_correct_prob: float = 0.95     # an answered query resolves the target
    # Per-controller query targeting on irreducible (info-absent) steps:
    irr_query_prob_allocator: float = 0.85
    irr_query_prob_asking_baseline: float = 0.3  # KnowNo/INSIGHT gate on agreement → miss irreducible
    # Human model (probability of taking manual control on a robot error):
    human_catch_unaided: float = 0.25
    human_catch_signaled_magnitude: float = 0.55
    human_catch_signaled_type: float = 0.78
    human_false_alarm: float = 0.05      # needless intervention on correct robot actions
    human_fix_success: float = 0.85


class SimEpisodeRunner:
    """A seeded synthetic robot+human producing realistic study outcomes for piloting."""

    QUERYING_BASELINES = {"knowno", "insight"}
    COMPUTE_CONTROLLERS = {"fixed_compute", "compute_gated", "scale"}

    def __init__(self, params: Optional[SimWorldParams] = None, seed: int = 0) -> None:
        self.p = params or SimWorldParams()
        self.rng = random.Random(int(seed))

    def __call__(self, trial: Trial) -> EpisodeResult:
        p = self.p
        occ = float(trial.occlusion_fraction)
        kind = trial.controller_kind
        is_type_aware = bool(trial.transparency_signal_type)
        signals = bool(trial.transparency_enabled)

        base_acc = max(0.1, min(0.95, p.base_accuracy_clean - p.accuracy_drop_per_occ * occ))
        p_ident = max(0.0, min(1.0, 1.0 - p.identifiable_drop_per_occ * occ))
        identifiable = self.rng.random() < p_ident

        queried = False
        n_queries = 0
        if identifiable:
            # Reducible ambiguity: compute helps a little, asking baselines/allocator just act well.
            acc = base_acc + (0.08 if kind in self.COMPUTE_CONTROLLERS or kind == "allocator" else 0.0)
            robot_correct = self.rng.random() < min(0.97, acc)
        else:
            # Irreducible: information absent. Only a query can recover it.
            if kind == "allocator":
                if self.rng.random() < p.irr_query_prob_allocator:
                    queried = True
            elif kind in self.QUERYING_BASELINES:
                if self.rng.random() < p.irr_query_prob_asking_baseline:
                    queried = True
            if queried:
                n_queries = 1
                robot_correct = self.rng.random() < p.query_correct_prob
            else:
                robot_correct = self.rng.random() < p.acc_irreducible

        robot_signaled = bool(signals and (queried or (not identifiable)))
        # Human intervention model.
        intervened = False
        if not queried:  # answering a query is collaboration, not a takeover
            if robot_correct:
                intervened = self.rng.random() < p.human_false_alarm
            else:
                if robot_signaled and is_type_aware:
                    pr = p.human_catch_signaled_type
                elif robot_signaled:
                    pr = p.human_catch_signaled_magnitude
                else:
                    pr = p.human_catch_unaided
                intervened = self.rng.random() < pr

        success = robot_correct or (intervened and self.rng.random() < p.human_fix_success)
        return EpisodeResult(
            success=bool(success),
            robot_correct=bool(robot_correct),
            human_intervened=bool(intervened),
            robot_queried=bool(queried),
            robot_signaled_uncertainty=bool(robot_signaled),
            n_queries=int(n_queries),
            n_decisions=1,
            query_latency_s=3.0 if queried else 0.0,
            time_to_complete_s=float(8.0 + 6.0 * occ + (3.0 if queried else 0.0)),
            identifiable=bool(identifiable),
        )


class SimQuestionnaireProvider:
    """Synthetic questionnaire responses correlated with condition (for pipeline testing)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(int(seed))

    def __call__(self, instrument: str, condition_name: str) -> Any:
        # Better experience (lower workload, higher trust) for type-aware allocator.
        quality = {
            "autonomy": 0.35, "fixed_compute": 0.45, "compute_gated": 0.5,
            "allocator": 0.7, "allocator_type": 0.85,
        }.get(condition_name, 0.5)
        jitter = lambda: self.rng.uniform(-0.08, 0.08)
        if instrument in ("nasa_tlx", "tlx"):
            workload = max(0.0, min(1.0, 1.0 - quality + jitter()))
            return {k: round(100.0 * max(0.0, min(1.0, workload + jitter())), 1) for k in _instr.NASA_TLX_SUBSCALES}
        if instrument in ("jian", "jian_trust", "trust"):
            trust = max(0.0, min(1.0, quality + jitter()))
            resp = {}
            for i in range(1, 6):  # distrust items: low when trust high
                resp[i] = round(1 + 6 * max(0.0, min(1.0, 1.0 - trust + jitter())))
            for i in range(6, 13):  # trust items
                resp[i] = round(1 + 6 * max(0.0, min(1.0, trust + jitter())))
            return resp
        return None

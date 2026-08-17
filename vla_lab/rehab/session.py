"""W14 — the Phase 0 session runner: one code path over every backend.

The same code runs a synthetic pilot, an Isaac twin dry-run, and a real participant session.
Only two things are injected: the **apparatus** (null / twin / real) and the **observer**
(vision / keyed / both / simulated). Everything downstream — the phase machine, the safety
interlocks, the scheduler, the event-locked logging, the questionnaire points — is identical,
which is the whole reason ``rehab_pilot.sh`` is worth trusting as a rehearsal.

Session structure (``rehab.md`` §7, §10, §11):

.. code-block:: text

    handedness inventory        -> defines "nonpreferred arm" BEFORE any trial
    calibration check           -> participant frame, cameras
    protocol.json written       -> BEFORE trial 1, so the analysis plan can be checked
    reference block   (B0)      -> defines tilde-pi*
    [washout] compared blocks   -> counterbalanced, matched budget
    [washout] retest block (B0) -> test-retest reliability + residual-contamination check
    questionnaires at every block boundary

Per trial, for ASSESS/COACH:

.. code-block:: text

    PRESENT -> (arm moves)      safety.begin_motion; refused if a reach is in progress
    SETTLE  -> (arm stopped)    settle verified against the contract tolerance
    [COACH]                     effort level staged, prompt delivered
    GO      -> (cue issued)     safety.issue_go; refused unless stopped AND settled
    REACH   -> (participant)    observers poll; a reach during motion would have halted us
    SELECT  -> (label latched)  every observer's label appended to observers.jsonl
    RETURN  -> LOG

A mid-block halt (e-stop, participant request, driver fault) leaves a **partial** session that
:mod:`vla_lab.rehab.verify_session` accepts as partial rather than corrupt — the participant's
right to stop must never produce an unusable file (§11).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..human_study import instruments as _instr
from . import ARM_NONE, ASSESS, CLAIM_BOUNDARY, COACH, WAIT
from .apparatus.base import HALT_DRIVER_FAULT, ApparatusFault
from .carryover import CarryoverConfig
from .contract import Phase0Contract
from .logging import SessionWriter, session_dir_for
from .observation.base import SOURCE_ONLINE, ArmSelection
from .prompts import PROMPT_COACH, PROMPT_GO, PROMPT_NEUTRAL
from .protocol import BLOCK_COMPARED, BLOCK_REFERENCE, BLOCK_RETEST, BlockPlan, Phase0Protocol, SessionPlan, generate_session_plan, n_wait_for_static
from .safety import SOURCE_EXPERIMENTER, SOURCE_PARTICIPANT, HaltEvent, SafetyEnvelope, SafetyLimits, SafetyViolation
from .scheduler import CONDITION_NO_PROMPT, CONDITION_RANDOM_STATIC, make_scheduler
from .scheduler.base import DeltaModel
from .trial import (
    PHASE_DWELL,
    PHASE_GO,
    PHASE_LOG,
    PHASE_PRESENT,
    PHASE_REACH,
    PHASE_RETURN,
    PHASE_SELECT,
    PHASE_SETTLE,
    History,
    ManualClock,
    SessionClock,
    Trial,
    TrialPhaseMachine,
    TrialRecord,
    TrialResult,
)

OBSERVER_VISION = "vision"
OBSERVER_KEYED = "keyed"
OBSERVER_BOTH = "both"
OBSERVER_SIMULATED = "simulated"


@dataclass
class SessionConfig:
    """Everything the runner needs that is not the contract or the protocol."""

    participant_id: str = "P000"
    participant_idx: int = 0
    #: "left" | "right" — normally derived from the handedness inventory, not set by hand.
    nonpreferred_side: Optional[str] = None
    log_root: Union[str, Path] = "logs/rehab"
    seed: int = 0
    poll_interval_ms: int = 20
    carryover: CarryoverConfig = field(default_factory=CarryoverConfig)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    #: Conditions to run prospectively; ``None`` uses the protocol's default (§12.6 hybrid).
    conditions: Optional[Sequence[str]] = None
    demographics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["log_root"] = str(self.log_root)
        d["carryover"] = self.carryover.to_dict()
        d["safety_limits"] = self.safety_limits.to_dict()
        d["conditions"] = list(self.conditions) if self.conditions else None
        return d


@dataclass
class SessionResult:
    session_dir: Path
    plan: SessionPlan
    records: List[TrialRecord]
    questionnaires: List[Dict[str, Any]]
    halts: List[HaltEvent]
    completed: bool
    stopped_reason: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        by_action = {a: sum(1 for r in self.records if r.trial.action == a) for a in (COACH, WAIT, ASSESS)}
        return {
            "session_dir": str(self.session_dir),
            "participant_id": self.plan.participant_id,
            "n_trials": len(self.records),
            "by_action": by_action,
            "n_observations": sum(1 for r in self.records if r.result.is_observation),
            "n_halts": len(self.halts),
            "completed": bool(self.completed),
            "stopped_reason": self.stopped_reason,
        }


class ParticipantStopped(RuntimeError):
    """The participant (or experimenter) ended the session. A logged event, not a crash."""


class Phase0Session:
    """The runner. Build it, call :meth:`run`."""

    def __init__(
        self,
        contract: Phase0Contract,
        protocol: Phase0Protocol,
        cfg: SessionConfig,
        *,
        apparatus: Any,
        observer_factory: Callable[[str], Any],
        observer_kind: str = OBSERVER_SIMULATED,
        questionnaire_provider: Optional[Callable[[str, str], Any]] = None,
        handedness_responses: Optional[Any] = None,
        calibration: Optional[Any] = None,
        clock: Optional[SessionClock] = None,
        manual_clock: Optional[ManualClock] = None,
        session_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.contract = contract
        self.protocol = protocol
        self.cfg = cfg
        self.apparatus = apparatus
        self.observer_factory = observer_factory
        self.observer_kind = str(observer_kind)
        self.questionnaire_provider = questionnaire_provider
        self.handedness_responses = handedness_responses
        self.calibration = calibration
        self.manual_clock = manual_clock
        if clock is not None:
            self.clock = clock
        elif manual_clock is not None:
            self.clock = SessionClock(source=manual_clock)
        else:
            self.clock = SessionClock()
        # ONE monotonic source per session (§9). The apparatus builds its own clock by default,
        # and two clocks with different origins interleave into timestamps that run backwards —
        # which the phase machine (rightly) refuses. The session owns the clock; the apparatus
        # adopts it here rather than every call site remembering to pass it in.
        if hasattr(apparatus, "clock"):
            apparatus.clock = self.clock
        self.grid = contract.target_grid()
        self.delta_model = DeltaModel.from_contract(contract, cfg.carryover)

        self.records: List[TrialRecord] = []
        self.questionnaires: List[Dict[str, Any]] = []
        self.safety = SafetyEnvelope(cfg.safety_limits, on_halt=self._on_halt)
        self._writer: Optional[SessionWriter] = None
        self._stopped: Optional[str] = None
        self._trial_counter = 0
        self.session_dir = Path(session_dir) if session_dir else session_dir_for(cfg.participant_id, root=cfg.log_root)
        self.handedness: Dict[str, Any] = {}

    # -- helpers -----------------------------------------------------------
    def now(self) -> int:
        return int(self.clock.now_ms())

    def _advance(self, ms: int) -> None:
        if self.manual_clock is not None:
            self.manual_clock.advance_ms(int(ms))
        elif ms > 0:
            time.sleep(float(ms) / 1000.0)

    def _on_halt(self, ev: HaltEvent) -> None:
        if self._writer is not None:
            self._writer.log_event("safety_halt", {**ev.to_dict()}, t_ms=ev.t_ms)
        try:
            self.apparatus.halt(ev.reason)
        except Exception as exc:  # noqa: BLE001 - a failing halt must not hide the halt itself
            if self._writer is not None:
                self._writer.log_event("halt_delivery_failed", {"reason": ev.reason, "error": str(exc)})

    # -- setup -------------------------------------------------------------
    def _resolve_nonpreferred_side(self) -> str:
        """Handedness first, always — it is what "nonpreferred" means (§9)."""

        if self.handedness_responses is not None:
            self.handedness = dict(_instr.edinburgh_handedness(self.handedness_responses))
            side = self.handedness.get("nonpreferred_arm")
            if side is None:
                raise ValueError(
                    f"Edinburgh Handedness LQ={self.handedness['lq']:.1f} is mixed-handed: there is no "
                    "'nonpreferred arm' for this participant, so pi* is undefined. Mixed handedness is "
                    "an exclusion criterion, not a rounding decision (rehab.md §9)."
                )
            return str(side)
        if self.cfg.nonpreferred_side:
            self.handedness = {"handedness": "declared", "nonpreferred_arm": self.cfg.nonpreferred_side}
            return str(self.cfg.nonpreferred_side)
        raise ValueError(
            "the handedness inventory must be administered before any trial: it defines the "
            "nonpreferred arm, which is the label the estimand is expressed in (rehab.md §9)"
        )

    def _build_plan(self, nonpreferred_side: str) -> SessionPlan:
        plan = generate_session_plan(
            self.cfg.participant_idx,
            self.protocol,
            self.contract,
            nonpreferred_side=nonpreferred_side,
            participant_id=self.cfg.participant_id,
            seed=self.cfg.seed or None,
            conditions=self.cfg.conditions,
            carryover_cfg=self.cfg.carryover,
        )
        problems = plan.validate()
        if problems:
            raise ValueError("session plan is invalid:\n  - " + "\n  - ".join(problems))
        return plan

    def _participant_record(self, plan: SessionPlan) -> Dict[str, Any]:
        return {
            "participant_id": plan.participant_id,
            "participant_idx": plan.participant_idx,
            "handedness": self.handedness,
            "nonpreferred_side": plan.nonpreferred_side,
            "condition_order": plan.condition_order,
            "demographics": dict(self.cfg.demographics),
            "calibration": (self.calibration.to_dict() if hasattr(self.calibration, "to_dict") else (self.calibration or {})),
            "observer_kind": self.observer_kind,
            "session_config": self.cfg.to_dict(),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    # -- the run -----------------------------------------------------------
    def run(self) -> SessionResult:
        side = self._resolve_nonpreferred_side()
        plan = self._build_plan(side)

        contract = self.contract.stamped(apparatus_backend=getattr(self.apparatus, "name", "?"))
        problems = contract.validate()
        if problems:
            raise ValueError("contract is invalid:\n  - " + "\n  - ".join(problems))

        self._writer = SessionWriter(self.session_dir, clock=self.clock)
        writer = self._writer
        try:
            writer.write_contract(contract)
            writer.write_participant(self._participant_record(plan))
            writer.write_protocol(plan)  # BEFORE trial 1 (§10)
            writer.log_event("session_start", {
                "contract_hash": contract.contract_hash(),
                "apparatus": getattr(self.apparatus, "name", "?"),
                "observer": self.observer_kind,
                "nonpreferred_side": side,
            })
            if self.calibration is not None and hasattr(self.calibration, "check"):
                cal_problems = self.calibration.check()
                writer.log_event("calibration_check", {"problems": cal_problems})
                if cal_problems:
                    raise ValueError("calibration check failed:\n  - " + "\n  - ".join(cal_problems))

            self.apparatus.connect()
            self.apparatus.home()

            for block in plan.blocks:
                if self._stopped:
                    break
                self._run_block(plan, block, side)
                self._administer_questionnaires(block)
        except ParticipantStopped as exc:
            self._stopped = str(exc)
            writer.log_event("session_stopped", {"reason": str(exc)})
        except ApparatusFault as exc:
            self._stopped = f"apparatus fault: {exc}"
            writer.log_event("session_stopped", {"reason": self._stopped})
        finally:
            try:
                self.apparatus.close()
            except Exception:  # noqa: BLE001
                pass
            writer.log_event("session_end", {
                "n_trials": len(self.records),
                "completed": bool(self._stopped is None),
                "safety": self.safety.summary(),
            })
            writer.close()

        return SessionResult(
            session_dir=self.session_dir,
            plan=plan,
            records=self.records,
            questionnaires=self.questionnaires,
            halts=list(self.safety.halts),
            completed=bool(self._stopped is None),
            stopped_reason=self._stopped,
        )

    # -- blocks ------------------------------------------------------------
    def _make_scheduler(self, plan: SessionPlan, block: BlockPlan, side: str) -> Any:
        effort_strength = {
            lvl.name: float(lvl.carryover_scale) for lvl in self.contract.prompts.effort_levels
        }
        from .scheduler.carryover_aware import CarryoverAwareConfig

        return make_scheduler(
            block.condition,
            grid=self.grid,
            nonpreferred_side=side,
            seed=int(plan.seed) + int(block.block_idx),
            fixed_w=int(plan.fixed_w),
            n_wait=n_wait_for_static(plan) if block.condition == CONDITION_RANDOM_STATIC else 0,
            carryover_cfg=self.cfg.carryover,
            delta=self.delta_model,
            cfg=CarryoverAwareConfig(effort_strength=effort_strength),
        )

    def _run_block(self, plan: SessionPlan, block: BlockPlan, side: str) -> None:
        writer = self._writer
        assert writer is not None
        if block.washout_before_ms:
            writer.log_event("inter_block_washout", {"block_idx": block.block_idx, "ms": block.washout_before_ms})
            self._advance(int(block.washout_before_ms))

        scheduler = self._make_scheduler(plan, block, side)
        scheduler.reset(block.budget())
        history = History(
            block_idx=block.block_idx,
            condition=block.condition,
            slots_total=len(block.slots),
            coach_slots=block.coach_slots,
        )
        writer.log_event("block_start", {
            "block_idx": block.block_idx, "kind": block.kind, "condition": block.condition,
            "n_slots": len(block.slots), "n_coach": len(block.coach_slots),
            "scheduler": scheduler.describe(),
        })

        observer = self.observer_factory(self.observer_kind)
        for slot in block.slots:
            if self._stopped:
                break
            decision = scheduler.decide(history, slot)
            record = self._run_trial(block, slot, decision, observer, history, side)
            history.append(record)
            scheduler.observe(record)
            self.records.append(record)
            writer.log_trial(record)
            if record.result.halted and record.result.halt_reason in (
                "estop_participant", "estop_experimenter", "participant_request",
            ):
                raise ParticipantStopped(str(record.result.halt_reason))

        writer.log_event("block_end", {
            "block_idx": block.block_idx,
            "budget_spent": history.budget_spent(),
        })

    # -- trials ------------------------------------------------------------
    def _run_trial(
        self,
        block: BlockPlan,
        slot: Any,
        decision: Any,
        observer: Any,
        history: History,
        side: str,
    ) -> TrialRecord:
        writer = self._writer
        assert writer is not None
        timing = self.contract.timing
        idx = self._trial_counter
        self._trial_counter += 1

        target = self.grid.get(int(decision.target_id)) if decision.target_id is not None else None
        effort = self.contract.prompts.effort(slot.effort_level) if decision.action == COACH else None
        strength = float(effort.carryover_scale) if effort is not None else 1.0
        trial = Trial(
            trial_idx=idx,
            block_idx=block.block_idx,
            condition=block.condition,
            action=decision.action,
            target_id=(int(target.target_id) if target is not None else None),
            target_xy_participant_m=(target.xy if target is not None else None),
            prompt_id=(self.contract.prompts.coach.prompt_id if decision.action == COACH else None),
            prompt_hash=self.contract.prompts.content_hash(),
            effort_level=(slot.effort_level if decision.action == COACH else "none"),
            slot_idx=int(slot.slot_idx),
            is_coach_slot=bool(slot.is_coach_slot),
            kappa_prior_mean=float(decision.kappa_prior_mean),
            kappa_prior_sd=float(decision.kappa_prior_sd),
            since_last_coach_ms=history.since_last_coach_ms(self.now()),
            coach_count_so_far=history.n_action(COACH),
            scheduler_rationale=str(decision.rationale),
        )
        pm = TrialPhaseMachine(decision.action, timing)
        result = TrialResult(observer=getattr(observer, "name", self.observer_kind))

        try:
            if decision.action == WAIT:
                self._run_wait(pm, trial)
            else:
                self._run_presented(pm, trial, target, decision, observer, strength, result, side)
        except SafetyViolation as exc:
            pm.halt(self.now(), exc.reason)
            result.halted = True
            result.halt_reason = exc.reason
            result.arm = ARM_NONE
            writer.log_event("trial_halted", {"trial_idx": idx, "reason": exc.reason, "detail": str(exc)})
        except ApparatusFault as exc:
            pm.halt(self.now(), HALT_DRIVER_FAULT)
            result.halted = True
            result.halt_reason = HALT_DRIVER_FAULT
            result.arm = ARM_NONE
            writer.log_event("trial_halted", {"trial_idx": idx, "reason": HALT_DRIVER_FAULT, "detail": str(exc)})

        self.safety.end_trial(self.now())
        writer.log_event("trial_phases", {"trial_idx": idx, "action": decision.action, "phases": pm.to_list()})
        if decision.values:
            writer.log_event("scheduler_decision", {"trial_idx": idx, **decision.to_dict()})
        return TrialRecord(trial=trial, result=result, phases=pm.to_list())

    def _run_wait(self, pm: TrialPhaseMachine, trial: Trial) -> None:
        writer = self._writer
        assert writer is not None
        pm.enter(PHASE_DWELL, self.now())
        self.apparatus.prompt(PROMPT_NEUTRAL, self.contract.prompts.neutral.text)
        writer.log_event("wait_dwell", {"trial_idx": trial.trial_idx, "ms": self.contract.timing.wait_dwell_ms})
        self._advance(int(self.contract.timing.wait_dwell_ms))
        pm.enter(PHASE_LOG, self.now())

    def _run_presented(
        self,
        pm: TrialPhaseMachine,
        trial: Trial,
        target: Any,
        decision: Any,
        observer: Any,
        strength: float,
        result: TrialResult,
        side: str,
    ) -> None:
        writer = self._writer
        assert writer is not None
        timing = self.contract.timing

        # --- PRESENT: the arm moves, and nothing else may happen while it does.
        self.safety.begin_motion(self.now())
        pm.enter(PHASE_PRESENT, self.now())
        pres = self.apparatus.present(target)
        trial.t_present_ms = int(pres.t_present_ms)
        self.safety.tick(self.now())
        self.safety.end_motion(self.now(), settled=bool(pres.settled))
        if not pres.settled:
            pm.halt(self.now(), "driver_fault")
            result.halted = True
            result.halt_reason = "driver_fault"
            writer.log_event("settle_failed", {"trial_idx": trial.trial_idx, "pose_error_m": pres.pose_error_m})
            return

        pm.enter(PHASE_SETTLE, int(pres.t_settled_ms))
        trial.t_settled_ms = int(pres.t_settled_ms)

        # --- COACH: stage the effort manipulation, then deliver the prompt.
        if decision.action == COACH:
            needs_experimenter = self.apparatus.configure_effort(trial.effort_level)
            if needs_experimenter:
                writer.log_event("experimenter_action_required", {
                    "trial_idx": trial.trial_idx, "effort_level": trial.effort_level,
                    "setting": self.contract.prompts.effort(trial.effort_level).apparatus_setting,
                })
            text = self.contract.prompts.render_coach(side)
            t_prompt = self.apparatus.prompt(PROMPT_COACH, text)
            writer.log_event("prompt", {
                "trial_idx": trial.trial_idx, "kind": PROMPT_COACH,
                "prompt_id": trial.prompt_id, "prompt_hash": trial.prompt_hash,
                "effort_level": trial.effort_level, "t_ms": int(t_prompt),
            })

        # --- GO: only from a stopped, settled arm.
        self.safety.issue_go(self.now())
        t_go = int(self.apparatus.go_signal())
        pm.enter(PHASE_GO, t_go)
        trial.t_go_ms = t_go
        writer.log_event("go_signal", {"trial_idx": trial.trial_idx, "t_ms": t_go})

        prepare = getattr(observer, "prepare", None)
        if callable(prepare):
            prepare(
                target,
                action=decision.action,
                strength=float(strength),
                delta=float(self.delta_model.for_action(decision.action)),
                effort_index=(float(target.effort_index) if target is not None else None),
            )
        observer.begin_trial(int(trial.trial_idx), t_go)

        # --- REACH: the window in which the participant may reach opens at GO. The safety
        # envelope is told immediately, so the workspace counts as human-occupied from here
        # (a real onset detector can only move this timestamp later, never earlier).
        pm.enter(PHASE_REACH, t_go)
        self.safety.reach_detected(t_go)

        deadline = t_go + int(timing.go_window_ms) + int(timing.reach_timeout_ms)
        sel: Optional[ArmSelection] = None
        while self.now() < deadline:
            sel = observer.poll(self.now())
            if sel is not None:
                break
            self._advance(int(self.cfg.poll_interval_ms))
        if sel is None:
            sel = observer.end_trial(self.now())
        result.timed_out = not sel.resolved

        # --- SELECT. Clamp to the reach window: a detector's own timestamp may predate the
        # last poll, and the phase machine (rightly) refuses a backwards clock.
        t_select = int(sel.t_ms) if sel.t_ms is not None else self.now()
        t_select = max(t_go, min(t_select, self.now()))
        pm.enter(PHASE_SELECT, t_select)

        result.arm = sel.arm
        result.t_select_ms = sel.t_ms
        result.confidence = float(sel.confidence)
        result.observer = sel.observer
        result.reach_time_ms = (int(sel.t_ms) - t_go) if sel.t_ms is not None else None
        # "Clean reach" is a *separate* judgement from arm choice (target touched, no drop, no
        # re-attempt). Only an observer that can see it may assert it; otherwise a resolved
        # selection is the best available evidence.
        result.success = bool((sel.extra or {}).get("clean", sel.resolved))
        self._log_observers(trial.trial_idx, observer, sel)

        # --- RETURN, LOG.
        self.safety.end_reach(self.now())
        pm.enter(PHASE_RETURN, self.now())
        self._advance(int(timing.return_ms))
        pm.enter(PHASE_LOG, self.now())
        self._advance(int(timing.inter_trial_ms))

    def _log_observers(self, trial_idx: int, observer: Any, primary: ArmSelection) -> None:
        """Append **every** observer's label. Never merged into ``trials.jsonl`` (§10)."""

        writer = self._writer
        assert writer is not None
        sels: List[ArmSelection] = [primary]
        for extra in getattr(observer, "observers", []) or []:
            if extra is not observer:
                try:
                    sels.append(extra.end_trial(self.now()))
                except Exception:  # noqa: BLE001 - a secondary observer must not kill a trial
                    continue
        seen = set()
        for s in sels:
            key = (s.observer, s.source)
            if key in seen:
                continue
            seen.add(key)
            writer.log_observation(
                trial_idx=int(trial_idx),
                observer=s.observer,
                arm=s.arm,
                t_ms=s.t_ms,
                confidence=float(s.confidence),
                physical_side=str(s.physical_side),
                source=s.source,
                extra=dict(s.extra) if s.extra else None,
            )

    # -- questionnaires ----------------------------------------------------
    def _administer_questionnaires(self, block: BlockPlan) -> None:
        if self.questionnaire_provider is None or self._writer is None:
            return
        for instrument in self.protocol.instruments:
            if instrument == "edinburgh_handedness":
                continue  # administered once, before trial 1
            responses = self.questionnaire_provider(instrument, block.kind)
            if responses is None:
                continue
            scores = score_instrument(instrument, responses)
            row = {
                "block_idx": block.block_idx,
                "block_kind": block.kind,
                "condition": block.condition,
                "instrument": instrument,
                "scores": scores,
            }
            self.questionnaires.append(row)
            self._writer.log_event("questionnaire", {**row, "responses": responses})


def score_instrument(instrument: str, responses: Any) -> Dict[str, float]:
    """Dispatch to the shared instrument scoring (:mod:`vla_lab.human_study.instruments`)."""

    name = str(instrument).lower()
    if name in ("nasa_tlx", "tlx"):
        if isinstance(responses, dict) and "weights" in responses:
            return {
                "weighted_tlx": _instr.nasa_tlx_weighted(responses.get("subscales", responses), responses["weights"]),
                "raw_tlx": _instr.nasa_tlx_raw(responses.get("subscales", responses)),
            }
        return {"raw_tlx": _instr.nasa_tlx_raw(responses)}
    if name in ("session_burden", "burden"):
        return _instr.session_burden(responses)
    if name in ("edinburgh_handedness", "ehi"):
        return {k: v for k, v in _instr.edinburgh_handedness(responses).items() if isinstance(v, (int, float))}
    return {}


__all__ = [
    "OBSERVER_VISION",
    "OBSERVER_KEYED",
    "OBSERVER_BOTH",
    "OBSERVER_SIMULATED",
    "SessionConfig",
    "SessionResult",
    "Phase0Session",
    "ParticipantStopped",
    "score_instrument",
]

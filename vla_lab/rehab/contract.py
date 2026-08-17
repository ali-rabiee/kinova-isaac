"""W1/§9 — the Phase 0 contract as code.

The analogue of the VLA track's model contract (``README.md`` §7). Anything in
:meth:`Phase0Contract.scientific_payload` that changes between participants makes their
sessions **non-poolable**: :meth:`Phase0Contract.contract_hash` hashes it, every session
stamps it into ``contract.json``, and :mod:`vla_lab.rehab.verify_session` refuses sessions
whose hash drifts from the rest.

**What is and is not hashed.** The hash covers the *scientific* contract — geometry, timing,
budget, prompt content, action set, effort ladder. Provenance (git commit, driver and
apparatus versions, backend name, host) is recorded **alongside** the hash but not inside it:
a session recorded from a different backend or after an unrelated code edit is still poolable
if the participant's experience was identical, and pretending otherwise would make every
commit un-pool the whole study. The two are separated so a reviewer can see both.

Per-participant quantities (the solved participant frame, the handedness inventory) live in
``participant.json``, not here — they vary by design.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import ACTIONS
from .prompts import PromptLibrary
from .workspace import PlanarTransform, TargetGrid, WorkspaceConfig

CONTRACT_SCHEMA = "vla_lab_rehab_contract/v1"

#: Robot base pose in the participant frame: across the table, facing the participant.
_DEFAULT_MOUNT: Dict[str, float] = {"tx": 0.75, "ty": 0.0, "tz": 0.0, "yaw_rad": math.pi}


@dataclass
class TimingConfig:
    """Fixed trial timing (§9 "Trial timing"). Milliseconds throughout."""

    present_timeout_ms: int = 8000    # max time for the arm to reach the target pose
    settle_dwell_ms: int = 700        # position tolerance must hold for this long before GO
    go_window_ms: int = 4000          # GO -> reach onset; no reach by then = "none"
    reach_timeout_ms: int = 5000      # reach onset -> selection resolved
    return_ms: int = 1200             # participant returns to the home posture
    inter_trial_ms: int = 1500        # fixed ITI, so elapsed time is not condition-dependent
    wait_dwell_ms: int = 6000         # one WAIT slot's idle dwell (the cost of waiting)
    settle_tolerance_m: float = 0.01  # position tolerance defining "settled"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TimingConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def nominal_trial_ms(self) -> int:
        """Nominal wall-clock cost of one presented (ASSESS/COACH) trial."""

        return int(self.settle_dwell_ms + self.go_window_ms + self.return_ms + self.inter_trial_ms)


@dataclass
class BudgetConfig:
    """The matched interaction budget (§1.3). Identical ``T`` and ``C`` across conditions."""

    trials_per_block: int = 40      # T: total slots per compared block (ASSESS + WAIT + COACH)
    coach_per_block: int = 8        # C: COACH events per compared block
    reference_trials: int = 30      # the no-prompt reference block (zero COACH, by definition)
    retest_trials: int = 30         # the terminal no-prompt retest block (zero COACH)
    inter_block_washout_ms: int = 120000  # enforced rest/washout between compared blocks

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "BudgetConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.coach_per_block >= self.trials_per_block:
            problems.append(
                f"coach_per_block ({self.coach_per_block}) must be < trials_per_block "
                f"({self.trials_per_block}); the scheduler needs non-COACH slots to place probes in"
            )
        if self.coach_per_block <= 0:
            problems.append("coach_per_block must be > 0 for the compared conditions")
        if self.reference_trials <= 0:
            problems.append("reference_trials must be > 0: it defines the reference map (§12.2)")
        return problems


@dataclass
class Provenance:
    """Recorded alongside the hash, deliberately *not* inside it."""

    git_commit: str = ""
    apparatus_backend: str = "null"
    apparatus_version: str = ""
    driver_version: str = ""
    code_version: str = ""
    host: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Provenance":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def git_commit(repo_root: Optional[Union[str, Path]] = None) -> str:
    """Short git SHA of the working tree, ``""`` when unavailable (e.g. a tarball)."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class Phase0Contract:
    """Everything that must be identical for two sessions to be poolable."""

    schema: str = CONTRACT_SCHEMA
    robot: str = "kinova_gen2_j2n6s300"
    # The apparatus mounting pose: participant frame -> robot base frame. Solved per
    # participant from the chair/table calibration; the *definition* is contractual, the
    # solved numbers land in participant.json.
    participant_frame_definition: str = (
        "origin=sternum projection on table plane; +x forward; +y to participant's left"
    )
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    prompts: PromptLibrary = field(default_factory=PromptLibrary)
    action_set: Tuple[str, ...] = ACTIONS
    # Nominal mount: the robot base pose expressed in the participant frame. The arm sits
    # across the table, ~0.75 m in front of the participant's sternum, facing back toward
    # them (yaw = pi). Used by the twin and as the default before per-participant calibration.
    robot_base_in_participant: PlanarTransform = field(
        default_factory=lambda: PlanarTransform(tx=0.75, ty=0.0, tz=0.0, yaw_rad=math.pi)
    )
    # Cameras: the wrist mount is re-aimed for Phase 0 (W11); intrinsics/extrinsics per
    # session live in participant.json, the *nominal* mount is contractual.
    camera_set: Tuple[str, ...] = ("front", "wrist")
    # Which observer's label the online scheduler consumes. "keyed" is the pre-registered
    # fallback if the vision observer misses its kappa >= 0.9 gate at the pilot (W8).
    online_observer: str = "vision"
    provenance: Provenance = field(default_factory=Provenance)

    # -- hashing -----------------------------------------------------------
    def scientific_payload(self) -> Dict[str, Any]:
        """The canonical, hashed description of the participant-facing contract."""

        return {
            "schema": str(self.schema),
            "robot": str(self.robot),
            "participant_frame_definition": str(self.participant_frame_definition),
            "workspace": self.workspace.to_dict(),
            "timing": self.timing.to_dict(),
            "budget": self.budget.to_dict(),
            "prompts_content_hash": self.prompts.content_hash(),
            "action_set": [str(a) for a in self.action_set],
            "robot_base_in_participant": self.robot_base_in_participant.to_dict(),
            "camera_set": [str(c) for c in self.camera_set],
            "online_observer": str(self.online_observer),
        }

    def contract_hash(self) -> str:
        blob = json.dumps(self.scientific_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- convenience -------------------------------------------------------
    def target_grid(self) -> TargetGrid:
        return TargetGrid(self.workspace)

    def participant_to_robot(self) -> PlanarTransform:
        """Transform mapping a participant-frame point into the robot base frame."""

        return self.robot_base_in_participant.inverse()

    def validate(self) -> List[str]:
        """Contract-level problems: budget arithmetic + grid geometry."""

        problems = list(self.budget.validate())
        grid = self.target_grid()
        problems += grid.validate(participant_to_robot=self.participant_to_robot())
        if tuple(self.action_set) != ACTIONS:
            problems.append(
                f"action_set {tuple(self.action_set)} != {ACTIONS}; Phase 0 studies exactly "
                "COACH/WAIT/ASSESS (rehab.md §15)"
            )
        if self.timing.settle_dwell_ms <= 0:
            problems.append("settle_dwell_ms must be > 0: presentation must stop before GO (§9)")
        try:
            self.prompts.effort(self.prompts.coach_effort_level)
        except KeyError as exc:
            problems.append(str(exc))
        return problems

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.scientific_payload(),
            "prompts": self.prompts.to_dict(),
            "contract_hash": self.contract_hash(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Phase0Contract":
        d = dict(d or {})
        return cls(
            schema=str(d.get("schema", CONTRACT_SCHEMA)),
            robot=str(d.get("robot", "kinova_gen2_j2n6s300")),
            participant_frame_definition=str(
                d.get("participant_frame_definition", Phase0Contract.participant_frame_definition)
            ),
            workspace=WorkspaceConfig.from_dict(d.get("workspace")),
            timing=TimingConfig.from_dict(d.get("timing")),
            budget=BudgetConfig.from_dict(d.get("budget")),
            prompts=PromptLibrary.from_dict(d.get("prompts")),
            action_set=tuple(str(a) for a in d.get("action_set", ACTIONS)),
            robot_base_in_participant=PlanarTransform.from_dict(
                d.get("robot_base_in_participant") or _DEFAULT_MOUNT
            ),
            camera_set=tuple(str(c) for c in d.get("camera_set", ("front", "wrist"))),
            online_observer=str(d.get("online_observer", "vision")),
            provenance=Provenance.from_dict(d.get("provenance")),
        )

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Phase0Contract":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Phase0Contract":
        """Load the ``contract:`` section of a ``configs/rehab_*.yaml``."""

        import yaml

        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(d.get("contract", d))

    def stamped(self, **provenance: Any) -> "Phase0Contract":
        """Copy with provenance filled in (git commit auto-detected when not supplied)."""

        prov = Provenance.from_dict({**self.provenance.to_dict(), **provenance})
        if not prov.git_commit:
            prov.git_commit = git_commit()
        if not prov.code_version:
            from . import __version__

            prov.code_version = str(__version__)
        return Phase0Contract(
            schema=self.schema,
            robot=self.robot,
            participant_frame_definition=self.participant_frame_definition,
            workspace=self.workspace,
            timing=self.timing,
            budget=self.budget,
            prompts=self.prompts,
            action_set=self.action_set,
            robot_base_in_participant=self.robot_base_in_participant,
            camera_set=self.camera_set,
            online_observer=self.online_observer,
            provenance=prov,
        )


DEFAULT_CONTRACT = Phase0Contract()


__all__ = [
    "CONTRACT_SCHEMA",
    "TimingConfig",
    "BudgetConfig",
    "Provenance",
    "Phase0Contract",
    "DEFAULT_CONTRACT",
    "git_commit",
]

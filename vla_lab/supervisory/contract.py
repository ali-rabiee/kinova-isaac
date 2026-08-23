"""The study contract: everything that must not vary, hashed.

Anything in here that changes between supervisors makes their sessions non-poolable. The
contract is hashed and stamped into every session record, and the session gate refuses to pool
sessions whose hashes differ. That turns "we didn't change anything mid-study" from a claim
into a check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .carryover import CarryoverConfig
from .narration import DEFAULT_DOSE, DEFAULT_DOSES, CoachDose, content_hash, dose_by_name
from .scenes import SceneGrid, ScenePhysics, build_scene_grid
from .scheduler.base import DeltaModel
from .strategies import PRIMARY_AXIS, get_axis

CONTRACT_SCHEMA = "vla_lab_supervisory_contract/v1"


@dataclass
class TimingConfig:
    """How long each kind of slot takes. Feeds the decay model and the burden accounting.

    The ``*_overhead_s`` values are the *interaction* time on top of whatever the robot's
    motion costs: posing the query, the supervisor reading the scene and answering. A
    counter-proposal costs more than a plain probe because it asks the person to weigh a named
    alternative rather than just say what they think -- and that difference is the whole price
    of the study's active de-biasing action, so it is a contract constant that is measured on
    hardware rather than guessed forever.
    """

    probe_overhead_s: float = 12.0
    counter_overhead_s: float = 26.0
    wait_s: float = 45.0
    inter_slot_s: float = 4.0
    block_rest_s: float = 60.0
    #: Nominal per-action durations used by the decay lookahead, where the realized duration is
    #: not yet known. Kept close to the measured means by ``vla_lab.supervisory.apparatus.measure``.
    probe_s: float = 42.0
    counter_s: float = 58.0
    coach_s: float = 45.0

    @property
    def counter_time_ratio(self) -> float:
        """Extra time a counter-proposal costs, as a fraction of a probe.

        This is what makes the scheduler's burden term principled rather than a tuned penalty:
        a counter-proposal is charged the *opportunity cost of the slot time it consumes*, in
        the same currency as everything else, not an arbitrary constant.
        """
        return float(max(0.0, (self.counter_s - self.probe_s) / max(self.probe_s, 1e-6)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetConfig:
    """The matched interaction budget, per block."""

    slots_per_block: int = 60
    coach_per_block: int = 10
    #: How the demonstration directions are laid out. This is the single most consequential
    #: design choice in the study, so both regimes ship and the paper reports both.
    #:
    #: ``"one_sided"`` (default, and what the compared conditions run)
    #:     Every demonstration in a session pushes the same way, and the direction is
    #:     **counterbalanced across participants** rather than within a session. This is the
    #:     deployment regime: a robot clears paths because the scenes it has been handed needed
    #:     clearing, not because a protocol told it to alternate. Sustained one-sided residue
    #:     is what actually tilts a supervisor's answers, and therefore the fitted map.
    #: ``"alternating"``
    #:     Direction flips every demonstration. Excellent for *identifying* the carryover
    #:     parameters -- drift pushes a whole session one way while carryover pushes each gap
    #:     toward whatever was just shown, so the two separate cleanly -- but it very nearly
    #:     cancels the map bias, which is exactly why it cannot be the regime the scheduling
    #:     comparison runs under. Used for the mechanism (H1/H2) sensitivity analysis.
    #: ``"runs"``
    #:     Alternates every ``coach_run_length`` demonstrations: a compromise that keeps
    #:     within-session balance while allowing residue to accumulate.
    coach_regime: str = "one_sided"
    coach_run_length: int = 3
    reference_slots: int = 40
    retest_slots: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Contract:
    """The hashed Phase-1 contract."""

    axis: str = PRIMARY_AXIS
    grid: SceneGrid = field(default_factory=build_scene_grid)
    carryover: CarryoverConfig = field(default_factory=CarryoverConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    dose: str = DEFAULT_DOSE
    #: Preregistered thresholds. These are gates, not decorations: the analysis refuses to
    #: report a primary contrast from a session that fails them.
    min_grounding_rate: float = 0.85
    min_band_scenes: int = 6
    schema: str = CONTRACT_SCHEMA
    notes: str = ""

    def dose_spec(self) -> CoachDose:
        return dose_by_name(self.dose)

    def delta_model(self) -> DeltaModel:
        return DeltaModel(
            decay_mode=str(self.carryover.decay_mode),
            time_unit_s=float(self.carryover.time_unit_s),
            probe_s=float(self.timing.probe_s),
            counter_s=float(self.timing.counter_s),
            wait_s=float(self.timing.wait_s),
            coach_s=float(self.timing.coach_s),
        )

    def narration_hash(self) -> str:
        return content_hash(self.axis)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "axis": self.axis,
            "axis_spec": get_axis(self.axis).to_dict(),
            "grid": self.grid.to_dict(),
            "carryover": self.carryover.to_dict(),
            "timing": self.timing.to_dict(),
            "budget": self.budget.to_dict(),
            "dose": self.dose,
            "dose_spec": self.dose_spec().to_dict(),
            "narration_hash": self.narration_hash(),
            "min_grounding_rate": float(self.min_grounding_rate),
            "min_band_scenes": int(self.min_band_scenes),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Contract":
        return cls(
            axis=str(d.get("axis", PRIMARY_AXIS)),
            grid=SceneGrid.from_dict(d["grid"]) if "grid" in d else build_scene_grid(),
            carryover=CarryoverConfig.from_dict(d.get("carryover")),
            timing=TimingConfig(**{k: v for k, v in (d.get("timing") or {}).items()
                                   if k in TimingConfig.__dataclass_fields__}),
            budget=BudgetConfig(**{k: v for k, v in (d.get("budget") or {}).items()
                                   if k in BudgetConfig.__dataclass_fields__}),
            dose=str(d.get("dose", DEFAULT_DOSE)),
            min_grounding_rate=float(d.get("min_grounding_rate", 0.85)),
            min_band_scenes=int(d.get("min_band_scenes", 6)),
            schema=str(d.get("schema", CONTRACT_SCHEMA)),
            notes=str(d.get("notes", "")),
        )

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def check(self) -> List[str]:
        """Structural problems that make a contract un-runnable. Empty list means usable."""
        problems: List[str] = []
        if self.grid.physics.is_degenerate():
            problems.append("scene physics has no value crossover: one strategy always wins, so no scene is ambiguous")
        band = [s for s in self.grid.probe_scenes() if self.grid.in_crossover_band(s)]
        if len(band) < int(self.min_band_scenes):
            problems.append(f"only {len(band)} probe scenes in the crossover band (need >= {self.min_band_scenes})")
        if not self.grid.coach_scenes():
            problems.append("no unambiguous scenes reserved for demonstrations")
        for s in self.grid.coach_scenes():
            if self.grid.in_crossover_band(s):
                problems.append(f"demonstration scene {s.scene_id} sits inside the crossover band")
        if self.budget.coach_per_block >= self.budget.slots_per_block:
            problems.append("every slot is a demonstration; no slots left to probe")
        if self.budget.coach_per_block < 2 and self.budget.alternate_directions:
            problems.append("counterbalanced directions need at least two demonstrations per block")
        return problems

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["contract_hash"] = self.hash()
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Contract":
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = ["CONTRACT_SCHEMA", "TimingConfig", "BudgetConfig", "Contract"]

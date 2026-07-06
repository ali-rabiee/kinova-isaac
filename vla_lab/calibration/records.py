"""Per-step calibration record: the unit of analysis for Result 2.

One record per policy query, pairing the policy's **self-generated uncertainty** with a
**correctness** label and the **observability** under which the step was taken. The schema
is intentionally flat and JSON-able so it round-trips through JSONL and is trivial to load
in any analysis environment.

Key fields:

- ``dispersion`` / ``irreducible_score`` — the self-agreement and (if fitted) irreducibility
  signals at this step.
- ``correct`` — did this step do the right thing (e.g. move toward / select the correct
  target)? In sim this is derived from an oracle; on the real robot it can be hand-coded or
  taken from the episode outcome.
- ``identifiable`` — *oracle only*: is the optimal action even recoverable from the masked
  view? This is the ground-truth "reducible vs. irreducible" label the detector is graded
  against; ``None`` when unknown.
- ``occlusion_fraction`` / ``occlusion_mode`` — the manipulated observability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class CalibrationRecord:
    run_id: str = ""
    controller: str = ""
    episode_idx: int = 0
    step_idx: int = 0
    occlusion_mode: str = "none"
    occlusion_fraction: float = 0.0
    dispersion: float = 0.0
    irreducible_score: float = 0.0
    nonconformity_score: float = 0.0
    conformal_anomaly: bool = False
    k_used: int = 1
    action: str = "act"
    uncertainty_type: str = "none"
    queried: bool = False
    query_answer_correct: Optional[bool] = None
    correct: Optional[bool] = None
    identifiable: Optional[bool] = None
    success: Optional[bool] = None
    target_leaf: Optional[str] = None
    instruction: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        base = {k: v for k, v in d.items() if k in known}
        extra = dict(base.get("extra", {}) or {})
        for k, v in d.items():
            if k not in known:
                extra[k] = v
        base["extra"] = extra
        return cls(**base)

    @classmethod
    def from_decision(
        cls,
        decision,
        *,
        run_id: str,
        episode_idx: int,
        step_idx: int,
        occlusion_mode: str,
        occlusion_fraction: float,
        correct: Optional[bool] = None,
        identifiable: Optional[bool] = None,
        success: Optional[bool] = None,
        target_leaf: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> "CalibrationRecord":
        """Build a record from a :class:`vla_lab.allocation.allocator.Decision`."""

        return cls(
            run_id=run_id,
            controller=getattr(decision, "controller", ""),
            episode_idx=int(episode_idx),
            step_idx=int(step_idx),
            occlusion_mode=str(occlusion_mode),
            occlusion_fraction=float(occlusion_fraction),
            dispersion=float(getattr(decision, "dispersion", 0.0)),
            irreducible_score=float(getattr(decision, "irreducible_score", 0.0)),
            nonconformity_score=float(getattr(decision, "nonconformity_score", 0.0)),
            conformal_anomaly=bool(getattr(decision, "conformal_anomaly", False)),
            k_used=int(getattr(decision, "k_used", 1)),
            action=str(getattr(decision, "action", "act")),
            uncertainty_type=str(getattr(decision, "uncertainty_type", "none")),
            queried=bool(getattr(decision, "queried", False)),
            query_answer_correct=getattr(decision, "query_answer_correct", None),
            correct=correct,
            identifiable=identifiable,
            success=success,
            target_leaf=target_leaf,
            instruction=instruction,
        )


def write_jsonl(records: Iterable[CalibrationRecord], path: Union[str, Path], *, append: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(p, mode) as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return p


def read_jsonl(path: Union[str, Path]) -> List[CalibrationRecord]:
    p = Path(path)
    out: List[CalibrationRecord] = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(CalibrationRecord.from_dict(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return out


def read_many(patterns: Iterable[Union[str, Path]]) -> List[CalibrationRecord]:
    """Load records from paths and/or glob patterns (the CLIs' --records inputs)."""

    import glob

    out: List[CalibrationRecord] = []
    for pat in patterns:
        for p in sorted(glob.glob(str(pat))) or [str(pat)]:
            out.extend(read_jsonl(p))
    return out

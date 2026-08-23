"""W8 — offline video coding: the gold standard, and the agreement machinery.

The coded labels are what the **analysis** uses; the online labels are what the **scheduler
acted on**. They are stored separately (``observers.jsonl``) and never merged, because their
disagreement is a reported quantity and because a session in which the two silently diverged
is a session whose scheduler was making decisions on noise.

What lives here:

- :func:`cohens_kappa` — chance-corrected agreement, the pre-registered acceptance metric
  (``kappa >= 0.9`` for vision-vs-coded at the pilot; below that, the study runs keyed-only).
- :func:`agreement_report` — the full confusion table between any two label sources, plus the
  disagreement list for review.
- :func:`load_coded_labels` / :func:`ingest_coded_labels` — read a coder's ``coded.jsonl`` and
  append it into a session's ``observers.jsonl`` under ``source="coded"``.

``cohens_kappa`` is implemented here rather than pulled from a stats package because the repo
has no scipy dependency (and the formula is four lines).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .. import ARM_AMBIGUOUS, ARM_NONE, ARM_NONPREFERRED, ARM_PREFERRED
from .base import SOURCE_CODED, SOURCE_ONLINE

#: Labels that count as a resolved choice. ``none``/``ambiguous`` are excluded from kappa by
#: default: they are *outcomes*, not categories the observers are trying to discriminate, and
#: including them inflates agreement whenever both sources abstain on the same easy trials.
RESOLVED_LABELS = (ARM_PREFERRED, ARM_NONPREFERRED)


def cohens_kappa(a: Sequence[str], b: Sequence[str], *, labels: Optional[Sequence[str]] = None) -> float:
    """Cohen's ``kappa`` between two label sequences. ``nan`` when undefined."""

    pairs = [(str(x), str(y)) for x, y in zip(a, b)]
    if not pairs:
        return float("nan")
    cats = list(labels) if labels is not None else sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        # Perfect chance agreement (one category only): kappa is undefined, and reporting 1.0
        # would claim discrimination that was never tested.
        return float("nan")
    return float((po - pe) / (1.0 - pe))


@dataclass
class AgreementReport:
    n: int
    n_resolved: int
    percent_agreement: float
    kappa: float
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    source_a: str = ""
    source_b: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "n": int(self.n),
            "n_resolved": int(self.n_resolved),
            "percent_agreement": round(float(self.percent_agreement), 5),
            "kappa": (None if self.kappa != self.kappa else round(float(self.kappa), 5)),
            "confusion": self.confusion,
            "n_disagreements": len(self.disagreements),
            "disagreements": self.disagreements,
        }


def agreement_report(
    labels_a: Dict[int, str],
    labels_b: Dict[int, str],
    *,
    source_a: str = "a",
    source_b: str = "b",
    resolved_only: bool = True,
) -> AgreementReport:
    """Confusion table, percent agreement, and Cohen's kappa over shared trials."""

    shared = sorted(set(labels_a) & set(labels_b))
    rows = [(t, str(labels_a[t]), str(labels_b[t])) for t in shared]
    if resolved_only:
        used = [(t, x, y) for t, x, y in rows if x in RESOLVED_LABELS and y in RESOLVED_LABELS]
        cats: Sequence[str] = RESOLVED_LABELS
    else:
        used = rows
        cats = (ARM_PREFERRED, ARM_NONPREFERRED, ARM_NONE, ARM_AMBIGUOUS)

    confusion: Dict[str, Dict[str, int]] = {c: {d: 0 for d in cats} for c in cats}
    for _, x, y in used:
        if x in confusion and y in confusion[x]:
            confusion[x][y] += 1
    agree = sum(1 for _, x, y in used if x == y)
    return AgreementReport(
        n=len(rows),
        n_resolved=len(used),
        percent_agreement=(agree / len(used)) if used else float("nan"),
        kappa=cohens_kappa([x for _, x, _ in used], [y for _, _, y in used], labels=cats),
        confusion=confusion,
        disagreements=[
            {"trial_idx": int(t), source_a: x, source_b: y} for t, x, y in used if x != y
        ],
        source_a=source_a,
        source_b=source_b,
    )


# ---------------------------------------------------------------------------
# Session-level helpers
# ---------------------------------------------------------------------------


def labels_by_observer(observer_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[int, str]]:
    """Group ``observers.jsonl`` rows into ``{(observer, source): {trial_idx: arm}}``."""

    out: Dict[Tuple[str, str], Dict[int, str]] = {}
    for r in observer_rows:
        key = (str(r.get("observer", "")), str(r.get("source", SOURCE_ONLINE)))
        out.setdefault(key, {})[int(r.get("trial_idx", -1))] = str(r.get("arm", ARM_NONE))
    return out


def session_agreement(
    observer_rows: Iterable[Dict[str, Any]],
    *,
    resolved_only: bool = True,
) -> Dict[str, Any]:
    """Every pairwise agreement in one session, with the coded source as the reference.

    Reports vision-vs-coded and keyed-vs-coded (the acceptance metrics), plus
    vision-vs-keyed (the real-time cross-check) and coder-vs-coder when two coders exist.
    """

    grouped = labels_by_observer(observer_rows)
    reports: Dict[str, Any] = {}
    keys = sorted(grouped)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            name = f"{ka[0]}:{ka[1]}__vs__{kb[0]}:{kb[1]}"
            reports[name] = agreement_report(
                grouped[ka], grouped[kb],
                source_a=f"{ka[0]}:{ka[1]}", source_b=f"{kb[0]}:{kb[1]}",
                resolved_only=resolved_only,
            ).to_dict()
    return reports


def load_coded_labels(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a coder's ``coded.jsonl``.

    Expected fields per line: ``trial_idx``, ``arm`` (or ``physical_side``), ``coder``, and
    optionally ``t_ms`` and ``notes``.
    """

    p = Path(path)
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def ingest_coded_labels(
    session_dir: Union[str, Path],
    coded_path: Union[str, Path],
    *,
    nonpreferred_side: str,
    coder: Optional[str] = None,
) -> int:
    """Append a coder's labels into the session's ``observers.jsonl``. Returns the row count.

    Appends only — it never rewrites ``trials.jsonl``. If the coded label disagrees with the
    online one, both remain on disk and :mod:`vla_lab.rehab.verify_session` reports it.
    """

    from ..logging import OBSERVERS_FILE
    from .base import arm_from_side

    rows = load_coded_labels(coded_path)
    out_path = Path(session_dir) / OBSERVERS_FILE
    n = 0
    with out_path.open("a", buffering=1) as fp:
        for r in rows:
            arm = r.get("arm")
            side = str(r.get("physical_side", ""))
            if arm is None and side:
                arm = arm_from_side(side, nonpreferred_side)
            rec = {
                "trial_idx": int(r.get("trial_idx", -1)),
                "observer": str(coder or r.get("coder", "coder")),
                "source": SOURCE_CODED,
                "arm": str(arm or ARM_NONE),
                "physical_side": side,
                "t_ms": r.get("t_ms"),
                "confidence": float(r.get("confidence", 1.0)),
                **({"extra": {"notes": r["notes"]}} if r.get("notes") else {}),
            }
            fp.write(json.dumps(rec) + "\n")
            n += 1
    return n


__all__ = [
    "RESOLVED_LABELS",
    "AgreementReport",
    "cohens_kappa",
    "agreement_report",
    "labels_by_observer",
    "session_agreement",
    "load_coded_labels",
    "ingest_coded_labels",
]

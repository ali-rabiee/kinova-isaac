"""W15 — the Phase 0 session gate. Run this after **every** session, before any analysis.

    python -m vla_lab.rehab.verify_session logs/rehab/participant_P001/session_20260901_101500
    python -m vla_lab.rehab.verify_session --root logs/rehab --pool

The VLA track's ``vla_lab/verify_session.py`` is the most valuable discipline in the
repository — a hard gate before you analyze anything — and this is its Phase 0 counterpart
with Phase 0's own failure modes (``rehab.md`` §6/W15).

Exit 1 (do not analyze) on:

1. **contract-hash drift** — pooled sessions must share the scientific contract (§9);
2. **prompt-wording drift** — a different COACH utterance is a different manipulation;
3. **missing or ambiguous arm selections** above threshold — every unresolved trial is a
   Bernoulli draw the estimand never got;
4. **classifier-vs-coder kappa below threshold** — the online label drove the scheduler, so a
   bad one corrupts the decisions as well as the outcome (W8);
5. **target coverage below the minimum per crossover bin** — the crossover band is where the
   primary outcome lives; a thin bin there is a hole in the estimate, not noise;
6. **COACH-count mismatch across compared conditions** — the budget is matched by design, so a
   mismatch means the session did not run the design;
7. **unexplained clock jumps** — the carryover model is a function of elapsed time;
8. **un-annotated safety halts** — every halt must carry a reason from the taxonomy;
9. **a missing handedness inventory** — without it "nonpreferred arm" is undefined and the
   estimand has no meaning.

**Partial sessions are accepted as partial, not rejected as corrupt.** The participant may end
a trial, a block, or the session at any time (§11); the resulting file is *supposed* to exist,
and a gate that threw it away would put pressure on exactly the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import ARM_AMBIGUOUS, ARM_NONE, COACH, WAIT
from .contract import Phase0Contract
from .logging import SessionReader, find_sessions
from .observation import KAPPA_ACCEPTANCE, session_agreement
from .protocol import BLOCK_COMPARED, BLOCK_REFERENCE, BLOCK_RETEST, SessionPlan
from .workspace import TargetGrid


class Report:
    """Findings for one session (or a pool). ``failures`` are hard; ``warnings`` are not."""

    def __init__(self, label: str) -> None:
        self.label = str(label)
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.partial: bool = False

    def fail(self, msg: str) -> None:
        self.failures.append(str(msg))

    def warn(self, msg: str) -> None:
        self.warnings.append(str(msg))

    def note(self, msg: str) -> None:
        self.info.append(str(msg))

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "ok": self.ok,
            "partial": self.partial,
            "failures": self.failures,
            "warnings": self.warnings,
            "info": self.info,
            "stats": self.stats,
        }

    def print(self) -> None:
        print(f"[rehab.verify] {self.label}")
        for k, v in self.stats.items():
            print(f"[rehab.verify]   {k}: {v}")
        for m in self.info:
            print(f"[rehab.verify][INFO] {m}")
        for m in self.warnings:
            print(f"[rehab.verify][WARN] {m}")
        for m in self.failures:
            print(f"[rehab.verify][FAIL] {m}")
        if self.failures:
            print("[rehab.verify] RESULT: FAIL — do not analyze this session as-is.")
        else:
            tag = " (partial)" if self.partial else ""
            extra = " (with warnings)" if self.warnings else ""
            print(f"[rehab.verify] RESULT: OK{tag}{extra}")


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


class Thresholds:
    """Everything the gate is willing to tolerate, in one place so it can be preregistered."""

    max_unresolved_fraction: float = 0.10   # arm="none"/"ambiguous" among presented trials
    max_ambiguous_fraction: float = 0.05
    min_kappa_vs_coded: float = KAPPA_ACCEPTANCE
    min_trials_per_crossover_bin: int = 4
    max_clock_jump_ms: int = 600_000        # 10 min between consecutive trials is unexplained
    min_block_completion: float = 0.5       # below this a block is reported as partial


# ---------------------------------------------------------------------------
# Single-session checks
# ---------------------------------------------------------------------------


def verify_session(
    session_dir: Union[str, Path],
    *,
    thresholds: Optional[Thresholds] = None,
) -> Report:
    th = thresholds or Thresholds()
    reader = SessionReader(session_dir)
    rep = Report(str(session_dir))

    if not reader.exists():
        rep.fail(f"no trials.jsonl under {session_dir}")
        return rep

    # --- files and truncation ------------------------------------------------
    raw = reader.trials_raw()
    if raw.truncated_lines:
        rep.fail(
            f"{raw.truncated_lines} unparseable line(s) in trials.jsonl — the session file is "
            "truncated or corrupt (power loss mid-write?); do not analyze a silently-shortened study"
        )
    events = reader.events()
    if events.truncated_lines:
        rep.warn(f"{events.truncated_lines} unparseable line(s) in events.jsonl")

    contract_d = reader.contract
    participant = reader.participant
    protocol_d = reader.protocol
    if contract_d is None:
        rep.fail("contract.json is missing — the session's scientific contract is unrecoverable")
    if protocol_d is None:
        rep.fail("protocol.json is missing — it must be written BEFORE trial 1 (§10)")
    if participant is None:
        rep.fail("participant.json is missing")

    records = reader.trials()
    rep.stats["n_trials"] = len(records)
    if not records:
        rep.fail("zero trials recorded")
        return rep

    # --- (9) handedness ------------------------------------------------------
    hand = (participant or {}).get("handedness") or {}
    side = (participant or {}).get("nonpreferred_side")
    if not hand or not side:
        rep.fail(
            "no handedness inventory in participant.json — 'nonpreferred arm' is undefined, so "
            "pi* has no meaning (§9)"
        )
    elif str(hand.get("handedness")) == "mixed":
        rep.fail("handedness is mixed: this participant has no nonpreferred arm and must be excluded")
    else:
        rep.stats["handedness"] = f"{hand.get('handedness')} (LQ={hand.get('lq')}), nonpreferred={side}"

    # --- (1)(2) contract and prompt hashes -----------------------------------
    if contract_d is not None:
        rep.stats["contract_hash"] = str(contract_d.get("contract_hash", ""))[:12]
        try:
            rebuilt = Phase0Contract.from_dict(contract_d)
            if rebuilt.contract_hash() != str(contract_d.get("contract_hash", "")):
                rep.fail(
                    "contract.json's recorded hash does not match its own contents — the file was "
                    "edited after the session"
                )
        except Exception as exc:  # noqa: BLE001
            rep.warn(f"could not re-hash contract.json: {exc}")
        prompt_hashes = {r.trial.prompt_hash for r in records if r.trial.prompt_hash}
        rep.stats["prompt_hashes"] = sorted(h[:12] for h in prompt_hashes)
        if len(prompt_hashes) > 1:
            rep.fail(f"{len(prompt_hashes)} different prompt hashes inside one session — COACH wording drifted mid-session")
        contract_prompt = str(((contract_d.get("prompts") or {}).get("content_hash", "")))
        if prompt_hashes and contract_prompt and contract_prompt not in prompt_hashes:
            rep.fail("trial prompt hashes do not match contract.json's prompt content hash")

    # --- (8) safety halts ----------------------------------------------------
    from .apparatus.base import HALT_REASONS

    halted = [r for r in records if r.result.halted]
    rep.stats["n_halted_trials"] = len(halted)
    bad_reasons = [r.result.halt_reason for r in halted if r.result.halt_reason not in HALT_REASONS]
    if bad_reasons:
        rep.fail(f"{len(bad_reasons)} halted trial(s) carry a reason outside the taxonomy: {sorted(set(map(str, bad_reasons)))}")
    halt_events = [e for e in events.rows if e.get("type") == "safety_halt"]
    if halt_events:
        rep.note(f"{len(halt_events)} safety halt event(s): {dict(Counter(str(e.get('data', {}).get('reason')) for e in halt_events))}")
        for e in halt_events:
            if not str((e.get("data") or {}).get("reason", "")):
                rep.fail("a safety_halt event has no reason recorded")

    # --- (3) unresolved selections -------------------------------------------
    presented = [r for r in records if r.trial.action != WAIT and not r.result.halted]
    n_pres = len(presented)
    n_none = sum(1 for r in presented if r.result.arm == ARM_NONE)
    n_amb = sum(1 for r in presented if r.result.arm == ARM_AMBIGUOUS)
    rep.stats["presented_trials"] = n_pres
    rep.stats["unresolved"] = f"{n_none} none + {n_amb} ambiguous"
    if n_pres:
        f_unres = (n_none + n_amb) / n_pres
        f_amb = n_amb / n_pres
        if f_unres > th.max_unresolved_fraction:
            rep.fail(
                f"{100*f_unres:.1f}% of presented trials produced no usable arm selection "
                f"(limit {100*th.max_unresolved_fraction:.0f}%)"
            )
        if f_amb > th.max_ambiguous_fraction:
            rep.fail(
                f"{100*f_amb:.1f}% of presented trials were ambiguous (limit "
                f"{100*th.max_ambiguous_fraction:.0f}%) — the observer cannot resolve this setup"
            )

    # --- (4) observer agreement ----------------------------------------------
    obs_rows = reader.observations().rows
    if obs_rows:
        reports = session_agreement(obs_rows)
        rep.stats["observer_pairs"] = sorted(reports)
        for name, r in reports.items():
            if ":coded" not in name:
                continue
            k = r.get("kappa")
            if k is None:
                rep.warn(f"{name}: kappa undefined (only one label category present)")
            elif float(k) < th.min_kappa_vs_coded:
                rep.fail(
                    f"{name}: Cohen's kappa {float(k):.3f} < {th.min_kappa_vs_coded:.2f} — the online "
                    "label the scheduler acted on does not match the coded gold standard (W8)"
                )
            else:
                rep.note(f"{name}: kappa={float(k):.3f} over {r.get('n_resolved')} resolved trials")
        if not any(":coded" in n for n in reports):
            rep.warn("no coded labels ingested yet — the analysis gold standard is missing (see rehab.observation.coding)")
    else:
        rep.warn("observers.jsonl is empty — per-observer labels were not recorded")

    # --- (5) target coverage in the crossover band ----------------------------
    if contract_d is not None:
        try:
            grid = TargetGrid(Phase0Contract.from_dict(contract_d).workspace)
            band = {t.target_id for t in grid.crossover_targets()}
            counts = Counter(int(r.trial.target_id) for r in records if r.trial.target_id is not None and r.result.is_observation)
            thin = {tid: counts.get(tid, 0) for tid in sorted(band) if counts.get(tid, 0) < th.min_trials_per_crossover_bin}
            rep.stats["crossover_coverage"] = (
                f"{len(band) - len(thin)}/{len(band)} band targets have >= "
                f"{th.min_trials_per_crossover_bin} observations"
            )
            if thin:
                rep.fail(
                    f"{len(thin)} crossover-band target(s) below {th.min_trials_per_crossover_bin} "
                    f"observations: {thin} — the primary outcome is weighted toward this band"
                )
        except Exception as exc:  # noqa: BLE001
            rep.warn(f"could not check target coverage: {exc}")

    # --- (6) matched budget ---------------------------------------------------
    # Keyed by *block*, not condition: the reference and retest blocks share the condition name
    # ``no_prompt``, and collapsing them would hide a COACH leaking into one of them.
    planned_blocks: Dict[int, Dict[str, Any]] = {}
    if protocol_d is not None:
        for b in protocol_d.get("blocks", []):
            planned_blocks[int(b.get("block_idx", -1))] = dict(b)
    by_block: Dict[int, Counter] = defaultdict(Counter)
    for r in records:
        by_block[int(r.trial.block_idx)][r.trial.action] += 1
        by_block[int(r.trial.block_idx)]["slots"] += 1

    def kind_of(bi: int) -> str:
        return str(planned_blocks.get(bi, {}).get("kind", BLOCK_COMPARED))

    def cond_of(bi: int) -> str:
        return str(planned_blocks.get(bi, {}).get("condition", ""))

    rep.stats["budget_by_block"] = {
        f"{bi}:{kind_of(bi)}:{cond_of(bi)}": {k: int(v[k]) for k in ("slots", COACH, "ASSESS", WAIT)}
        for bi, v in sorted(by_block.items())
    }
    compared = {bi: v for bi, v in by_block.items() if kind_of(bi) == BLOCK_COMPARED}
    if len(compared) > 1:
        signatures = {(v["slots"], v[COACH]) for v in compared.values()}
        if len(signatures) > 1:
            complete = all(
                int(v["slots"]) >= int(planned_blocks.get(bi, {}).get("n_slots", 0))
                for bi, v in compared.items()
            )
            msg = (
                "compared conditions do not share an identical budget (slots, COACH): "
                f"{ {cond_of(bi): (v['slots'], v[COACH]) for bi, v in compared.items()} }"
            )
            if complete:
                rep.fail(msg)
            else:
                rep.partial = True
                rep.warn(msg + " — but at least one block is incomplete, so this is a partial session")
    for bi, v in by_block.items():
        if kind_of(bi) in (BLOCK_REFERENCE, BLOCK_RETEST) and v[COACH]:
            rep.fail(f"block {bi} ({kind_of(bi)}) ran {v[COACH]} COACH trial(s); it must contain zero (§12.2)")

    # --- (7) clock -----------------------------------------------------------
    anchors = [r.trial.t_go_ms if r.trial.t_go_ms is not None else r.trial.t_present_ms for r in records]
    anchors = [a for a in anchors if a is not None]
    jumps = [(i, int(anchors[i + 1]) - int(anchors[i])) for i in range(len(anchors) - 1)]
    backwards = [j for j in jumps if j[1] < 0]
    big = [j for j in jumps if j[1] > th.max_clock_jump_ms]
    if backwards:
        rep.fail(f"{len(backwards)} backwards clock step(s) between trials — timestamps are not from one monotonic source")
    if big:
        annotated = {int((e.get("data") or {}).get("ms", 0)) for e in events.rows if e.get("type") == "inter_block_washout"}
        unexplained = [j for j in big if int(j[1]) not in annotated and not _near_block_boundary(records, j[0])]
        if unexplained:
            rep.fail(
                f"{len(unexplained)} unexplained clock jump(s) over {th.max_clock_jump_ms/1000:.0f}s "
                f"between consecutive trials: {unexplained[:5]}"
            )
        else:
            rep.note(f"{len(big)} long inter-trial gap(s), all at annotated block boundaries")

    # --- completeness ---------------------------------------------------------
    if protocol_d is not None:
        planned = sum(int(b.get("n_slots", 0)) for b in protocol_d.get("blocks", []))
        rep.stats["completion"] = f"{len(records)}/{planned} planned slots"
        if planned and len(records) < planned:
            rep.partial = True
            stops = [e for e in events.rows if e.get("type") in ("session_stopped", "safety_halt")]
            if stops:
                rep.note(f"partial session, explained by {len(stops)} stop/halt event(s) — accepted as partial (§11)")
            else:
                rep.warn(f"partial session ({len(records)}/{planned} slots) with no recorded stop or halt event")

    return rep


def _near_block_boundary(records: Sequence[Any], i: int) -> bool:
    return 0 <= i < len(records) - 1 and records[i].trial.block_idx != records[i + 1].trial.block_idx


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def verify_pool(session_dirs: Sequence[Union[str, Path]]) -> Report:
    """Cross-session checks: contract and prompt hashes must agree to be poolable (§9)."""

    rep = Report(f"pool of {len(session_dirs)} session(s)")
    contracts: Dict[str, List[str]] = defaultdict(list)
    prompts: Dict[str, List[str]] = defaultdict(list)
    for d in session_dirs:
        c = SessionReader(d).contract or {}
        contracts[str(c.get("contract_hash", "<missing>"))].append(str(d))
        prompts[str((c.get("prompts") or {}).get("content_hash", "<missing>"))].append(str(d))
    rep.stats["contract_hashes"] = {k[:12]: len(v) for k, v in contracts.items()}
    rep.stats["prompt_hashes"] = {k[:12]: len(v) for k, v in prompts.items()}
    if len(contracts) > 1:
        biggest = max(contracts, key=lambda k: len(contracts[k]))
        odd = [d for k, v in contracts.items() if k != biggest for d in v]
        rep.fail(
            f"{len(contracts)} different contract hashes across the pool — these sessions are NOT "
            f"poolable (§9). Odd ones out: {odd[:5]}"
        )
    if len(prompts) > 1:
        rep.fail(f"{len(prompts)} different COACH prompt hashes across the pool — the manipulation differed")
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="*", help="session directory/directories")
    ap.add_argument("--root", type=str, default=None, help="verify every session under this log root")
    ap.add_argument("--pool", action="store_true", help="also run the cross-session poolability check")
    ap.add_argument("--json", type=str, default=None, help="write the full report to this path")
    args = ap.parse_args(argv)

    dirs: List[Path] = [Path(s) for s in args.session]
    if args.root:
        dirs += find_sessions(args.root)
    if not dirs:
        ap.error("give at least one session directory, or --root")

    reports = [verify_session(d) for d in dirs]
    for r in reports:
        r.print()
        print()
    failed = sum(1 for r in reports if not r.ok)

    pool_rep: Optional[Report] = None
    if args.pool and len(dirs) > 1:
        pool_rep = verify_pool(dirs)
        pool_rep.print()
        print()
        failed += 0 if pool_rep.ok else 1

    if args.json:
        payload = {
            "sessions": [r.to_dict() for r in reports],
            **({"pool": pool_rep.to_dict()} if pool_rep else {}),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"[rehab.verify] wrote {args.json}")

    n_partial = sum(1 for r in reports if r.partial)
    print(
        f"[rehab.verify] {len(reports) - sum(1 for r in reports if not r.ok)}/{len(reports)} session(s) "
        f"passed ({n_partial} partial)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

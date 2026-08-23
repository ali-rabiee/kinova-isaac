"""The session gate. Run it after every session; a session that fails it is not poolable.

Every failure mode below cost this project, or its arm-choice predecessor, a session at some
point. The gate exists so that each one is caught by a check with a name rather than by
noticing an odd number three weeks later:

* **contract drift** -- two sessions run under different geometry, timing, or narration are not
  measuring the same thing and must not be pooled;
* **budget mismatch** -- a matched-budget comparison that is not actually matched in the data
  is not a comparison;
* **thin crossover coverage** -- a session whose probes missed the band estimated the flat part
  of the map, which every method gets right;
* **low grounding rate** -- if too many answers could not be resolved, the estimand is built on
  whatever was left;
* **a demonstration leaking into a reference block** -- the reference is defined by having none;
* **clock anomalies** -- the decay model is a function of elapsed time, so a backwards or
  jumping clock corrupts every fitted parameter downstream.

    python -m vla_lab.supervisory.verify_session logs/supervisory/S000/carryover_aware
    python -m vla_lab.supervisory.verify_session --root logs/supervisory --pool
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import COACH, COUNTER, PROBE, STRATEGY_A, STRATEGY_B, WAIT
from .contract import Contract
from .logging import EVENTS_FILE, META_FILE, TRIALS_FILE, read_jsonl


@dataclass
class GateResult:
    root: Path
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = f"{'PASS' if self.ok else 'FAIL'}  {self.root}"
        lines = [head]
        for f in self.failures:
            lines.append(f"  [FAIL] {f}")
        for w in self.warnings:
            lines.append(f"  [warn] {w}")
        if self.stats:
            lines.append("  " + "  ".join(f"{k}={v}" for k, v in self.stats.items()))
        return "\n".join(lines)


def verify(root: Path, contract: Optional[Contract] = None) -> GateResult:
    root = Path(root)
    res = GateResult(root=root)
    trials = read_jsonl(root / TRIALS_FILE)
    if not trials:
        res.failures.append(f"no trials in {TRIALS_FILE}")
        return res

    meta_path = root / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cpath = root / "contract.json"
    c = contract or (Contract.load(cpath) if cpath.exists() else Contract())

    if meta.get("contract_hash") and meta["contract_hash"] != c.hash():
        res.failures.append(f"contract hash drift: session {meta['contract_hash']} vs contract {c.hash()}")
    if meta.get("narration_hash") and meta["narration_hash"] != c.narration_hash():
        res.failures.append("narration hash drift: the robot did not say the same words as the rest of the study")

    # --- budget --------------------------------------------------------------
    by_block: Dict[int, List[Dict[str, Any]]] = {}
    for t in trials:
        by_block.setdefault(int(t.get("block", 0)), []).append(t)
    for bidx, rows in sorted(by_block.items()):
        kind = str(rows[0].get("block_kind", "condition"))
        n_coach = sum(1 for r in rows if r["action"] == COACH)
        if kind == "condition":
            if len(rows) != c.budget.slots_per_block:
                res.failures.append(f"block {bidx}: {len(rows)} slots, contract says {c.budget.slots_per_block}")
            if n_coach != c.budget.coach_per_block:
                res.failures.append(f"block {bidx}: {n_coach} demonstrations, contract says {c.budget.coach_per_block}")
        elif n_coach:
            res.failures.append(f"block {bidx} is a {kind} block but contains {n_coach} demonstrations")

    # --- observation quality --------------------------------------------------
    asked = [r for r in trials if r["action"] in (PROBE, COUNTER)]
    grounded = [r for r in asked if r.get("instructed_strategy") in (STRATEGY_A, STRATEGY_B)]
    rate = len(grounded) / max(len(asked), 1)
    if rate < float(c.min_grounding_rate):
        res.failures.append(f"grounding rate {rate:.2f} below the preregistered {c.min_grounding_rate:.2f}")

    band_ids = {s.scene_id for s in c.grid.probe_scenes() if c.grid.in_crossover_band(s)}
    covered = {int(r["scene_id"]) for r in grounded if int(r["scene_id"]) in band_ids}
    if len(covered) < int(c.min_band_scenes):
        res.failures.append(
            f"only {len(covered)} crossover-band scenes were answered (need {c.min_band_scenes}); "
            "the session estimated the saturated flanks, which every method gets right"
        )

    # --- clock ----------------------------------------------------------------
    ts = [int(t["t_ms"]) for t in trials if "t_ms" in t]
    if ts and any(b < a for a, b in zip(ts, ts[1:])):
        res.failures.append("the trial clock runs backwards; every fitted decay downstream is corrupt")
    if ts:
        jumps = [b - a for a, b in zip(ts, ts[1:]) if b - a > 10 * 60 * 1000]
        if jumps:
            res.warnings.append(f"{len(jumps)} inter-slot gap(s) over 10 minutes; annotate them or exclude the session")

    # --- events ---------------------------------------------------------------
    events = read_jsonl(root / EVENTS_FILE)
    halts = [e for e in events if str(e.get("kind", "")).startswith("halt")]
    for h in halts:
        if not h.get("reason"):
            res.failures.append("a halt was logged without a typed reason")

    res.stats = {
        "slots": len(trials),
        "blocks": len(by_block),
        "asked": len(asked),
        "grounded": f"{rate:.2f}",
        "band_scenes": len(covered),
        "coach": sum(1 for r in trials if r["action"] == COACH),
        "counter": sum(1 for r in trials if r["action"] == COUNTER),
        "wait": sum(1 for r in trials if r["action"] == WAIT),
    }
    return res


def verify_pool(root: Path) -> GateResult:
    """Are these sessions poolable with each other? One contract hash, or they are not."""
    root = Path(root)
    sessions = sorted(p.parent for p in root.rglob(TRIALS_FILE))
    res = GateResult(root=root)
    hashes = set()
    for s in sessions:
        m = s / META_FILE
        if m.exists():
            hashes.add(json.loads(m.read_text()).get("contract_hash"))
    if len(hashes) > 1:
        res.failures.append(f"{len(hashes)} distinct contract hashes across {len(sessions)} sessions: {sorted(hashes)}")
    res.stats = {"sessions": len(sessions), "contract_hashes": len(hashes)}
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?", default=None)
    ap.add_argument("--root", type=Path, default=None, help="verify every session under this directory")
    ap.add_argument("--pool", action="store_true", help="also check that the sessions are poolable")
    args = ap.parse_args(argv)

    results: List[GateResult] = []
    if args.session:
        results.append(verify(args.session))
    if args.root:
        for p in sorted(Path(args.root).rglob(TRIALS_FILE)):
            results.append(verify(p.parent))
        if args.pool:
            results.append(verify_pool(args.root))
    if not results:
        ap.error("give a session directory or --root")

    for r in results:
        print(r.render())
    n_bad = sum(1 for r in results if not r.ok)
    print(f"\n{len(results) - n_bad}/{len(results)} passed")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-collection sanity checks for a `collect_v3` (vla_v1) session.

Run this after EVERY data collection run, before training:

    python -m vla_lab.verify_session logs/data_collection/session_YYYYMMDD_HHMMSS

It validates the properties that have silently failed in the past:

1. Episode outcomes      — success rate, truncations, grasp attempts.
2. Scene diversity       — object layouts must differ across episodes (catches the
                           frozen-respawn teleport bug that poisoned a whole session).
3. Target diversity      — every object/color should appear as the target; warns when
                           the distribution is degenerate (e.g. 2 targets alternating).
4. Action health         — per-tick EE delta magnitudes, leading-idle frames,
                           gripper close labels present (catches the gripper-state
                           logger bug).
5. Logging contract      — log_rate_hz consistent across episodes, images on disk.

Exit code 0 = session looks usable; 1 = hard failure (do not train on it); the
report explains each finding. Pure stdlib — no torch / Isaac required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def verify_session(session_dir: Path, *, min_success_rate: float = 0.7, idle_dp_m: float = 0.002) -> int:
    session_dir = Path(session_dir)
    ep_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not ep_dirs:
        print(f"[verify] FAIL: no episode_* folders under {session_dir}")
        return 1

    hard_failures: List[str] = []
    warnings: List[str] = []

    # ---------------------------------------------------------------- outcomes
    n_success = 0
    n_truncated = 0
    n_unknown = 0
    attempts: List[int] = []
    targets: List[str] = []
    instructions: List[str] = []
    rates: List[int] = []
    layouts: List[tuple] = []  # one hashable XY layout per episode (from respawn readback)
    respawn_mismatch_events = 0

    total_ticks = 0
    total_images_missing = 0
    leading_idle: List[int] = []
    dp_mags: List[float] = []
    close_labels_per_ep: List[int] = []
    close_state_ticks = 0

    for ep_dir in ep_dirs:
        # outcome: episode_summary.json preferred, else events.jsonl
        success: Optional[bool] = None
        truncated = False
        summary_p = ep_dir / "episode_summary.json"
        events = _read_jsonl(ep_dir / "events.jsonl")
        if summary_p.exists():
            try:
                s = json.loads(summary_p.read_text())
                success = bool(s.get("success", False))
                truncated = bool(s.get("truncated", False))
                attempts.append(int(s.get("grasp_attempts", 0)))
            except Exception:
                success = None
        if success is None:
            last_ok: Optional[bool] = None
            for e in events:
                if e.get("type") == "grasp_result":
                    last_ok = bool(e.get("data", {}).get("ok", False))
                elif e.get("type") == "episode_end" and e.get("data", {}).get("truncated"):
                    truncated = True
            success = last_ok if last_ok is not None else None

        if success is True and not truncated:
            n_success += 1
        elif success is None:
            n_unknown += 1
        if truncated:
            n_truncated += 1

        # respawn layout + mismatch events
        for e in events:
            if e.get("type") == "object_respawn_actual":
                xy = e.get("data", {}).get("actual_xy", {})
                try:
                    layouts.append(tuple(sorted((k, round(_f(v[0]) or 0.0, 3), round(_f(v[1]) or 0.0, 3)) for k, v in xy.items())))
                except Exception:
                    pass
            elif e.get("type") == "object_respawn_mismatch":
                respawn_mismatch_events += 1

        # instruction / target
        instr_p = ep_dir / "instruction.json"
        if instr_p.exists():
            try:
                d = json.loads(instr_p.read_text())
                targets.append(str(d.get("target_label", "")))
                instructions.append(str(d.get("language_command", "")))
            except Exception:
                pass

        # metadata
        meta_p = ep_dir / "metadata.json"
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text())
                rates.append(int(m.get("config", {}).get("log_rate_hz", 0)))
            except Exception:
                pass

        # ticks
        ticks = _read_jsonl(ep_dir / "ticks.jsonl")
        total_ticks += len(ticks)
        n_close_labels = 0
        first_moving: Optional[int] = None
        for i, t in enumerate(ticks):
            afp = (t.get("policy") or {}).get("action_from_prev")
            if isinstance(afp, dict):
                dp = [(_f(v) or 0.0) for v in afp.get("ee_delta_pos_b", [0, 0, 0])]
                mag = math.sqrt(sum(v * v for v in dp))
                dp_mags.append(mag)
                if first_moving is None and mag > idle_dp_m:
                    first_moving = i
                if int(_f(afp.get("gripper_action", 0)) or 0) == -1:
                    n_close_labels += 1
            st = ((t.get("robot") or {}).get("gripper") or {}).get("state")
            if st == "close":
                close_state_ticks += 1
            img = (t.get("image") or {}).get("path")
            if img:
                if not (ep_dir / img).exists():
                    total_images_missing += 1
            else:
                total_images_missing += 1
        if first_moving is not None:
            leading_idle.append(first_moving)
        close_labels_per_ep.append(n_close_labels)

    n_eps = len(ep_dirs)
    print(f"[verify] session: {session_dir}")
    print(f"[verify] episodes: {n_eps}  ticks: {total_ticks}")

    # outcomes
    rate = n_success / max(1, n_eps)
    print(f"[verify] success: {n_success}/{n_eps} ({100*rate:.0f}%)  truncated: {n_truncated}  unknown-outcome: {n_unknown}")
    if attempts:
        print(f"[verify] grasp attempts histogram: {dict(sorted(Counter(attempts).items()))}")
    if rate < min_success_rate:
        warnings.append(
            f"success rate {100*rate:.0f}% < {100*min_success_rate:.0f}% — failed episodes are excluded from "
            "training (success_only), so you may need more episodes than planned."
        )

    # scene diversity (the frozen-respawn check)
    if respawn_mismatch_events:
        hard_failures.append(
            f"{respawn_mismatch_events} object_respawn_mismatch events — the teleport respawn failed; "
            "object layouts did not change as intended."
        )
    if len(layouts) >= 2:
        n_unique_layouts = len(set(layouts))
        print(f"[verify] unique object layouts: {n_unique_layouts}/{len(layouts)} respawn readbacks")
        if n_unique_layouts == 1:
            hard_failures.append(
                "ALL episodes share ONE object layout (frozen respawn). This session has zero scene "
                "diversity — do not train on it."
            )
        elif n_unique_layouts < max(2, len(layouts) // 2):
            warnings.append(f"only {n_unique_layouts} unique layouts across {len(layouts)} respawns — check respawn cadence.")
    elif n_eps > 1:
        warnings.append("no object_respawn_actual events found — cannot verify scene diversity (old session format?).")

    # target diversity
    if targets:
        counts = Counter(targets)
        print(f"[verify] target distribution: {dict(counts)}")
        n_unique = len(counts)
        if n_unique <= 2 and n_eps >= 8:
            hard_failures.append(
                f"only {n_unique} unique targets in {n_eps} episodes — language/target coverage is degenerate "
                "(this is the farthest_no_repeat alternation failure; use --target-selection cycle)."
            )
        else:
            most = counts.most_common(1)[0]
            if most[1] > 2 * (n_eps / max(1, n_unique)):
                warnings.append(f"target '{most[0]}' is over-represented ({most[1]}/{n_eps}).")
    else:
        warnings.append("no instruction.json files found — language conditioning will be empty.")

    # action health
    if dp_mags:
        dp_sorted = sorted(dp_mags)
        mean_dp = sum(dp_mags) / len(dp_mags)
        p95 = dp_sorted[int(0.95 * (len(dp_sorted) - 1))]
        mx = dp_sorted[-1]
        print(f"[verify] |delta_pos| per tick: mean {1000*mean_dp:.1f} mm  p95 {1000*p95:.1f} mm  max {1000*mx:.1f} mm")
        if mx > 0.08:
            warnings.append(f"max per-tick EE delta {1000*mx:.0f} mm is suspiciously large (teleport/discontinuity in logs?).")
    if leading_idle:
        avg_idle = sum(leading_idle) / len(leading_idle)
        print(f"[verify] leading idle ticks per episode: avg {avg_idle:.1f}  max {max(leading_idle)}")
        if avg_idle > 5:
            warnings.append(
                f"episodes start with ~{avg_idle:.0f} idle ticks — stabilize frames are being logged "
                "(collect with the current collect_v3.sh, which skips them)."
            )
    n_eps_no_close = sum(1 for c in close_labels_per_ep if c == 0)
    print(f"[verify] gripper: close-labels per episode min/max {min(close_labels_per_ep)}/{max(close_labels_per_ep)}  close-state ticks total {close_state_ticks}")
    if close_state_ticks == 0:
        hard_failures.append(
            "ZERO gripper close-state ticks in the whole session — gripper labels are broken "
            "(pre-2026-06 logger bug; run vla_lab.repair_gripper_labels or recollect)."
        )
    elif n_eps_no_close > n_eps // 2:
        warnings.append(f"{n_eps_no_close}/{n_eps} episodes have no gripper close action label.")

    # logging contract
    if rates:
        if len(set(rates)) > 1:
            hard_failures.append(f"inconsistent log_rate_hz across episodes: {sorted(set(rates))}")
        else:
            print(f"[verify] log_rate_hz: {rates[0]} (eval policy_rate_hz must match)")
    if total_images_missing:
        msg = f"{total_images_missing} ticks have missing image files"
        if total_images_missing > total_ticks // 10:
            hard_failures.append(msg + " (>10% of ticks).")
        else:
            warnings.append(msg + ".")

    # ------------------------------------------------------------------ report
    print()
    for w in warnings:
        print(f"[verify][WARN] {w}")
    for h in hard_failures:
        print(f"[verify][FAIL] {h}")
    if hard_failures:
        print(f"\n[verify] RESULT: FAIL — fix the issues above before training on this session.")
        return 1
    print(f"\n[verify] RESULT: OK{' (with warnings)' if warnings else ''} — session looks usable for training.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=str, help="Path to logs/data_collection/session_YYYYMMDD_HHMMSS")
    p.add_argument("--min-success-rate", type=float, default=0.7, help="Warn below this success fraction.")
    args = p.parse_args(argv)
    return verify_session(Path(args.session), min_success_rate=float(args.min_success_rate))


if __name__ == "__main__":
    raise SystemExit(main())

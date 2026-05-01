"""Quick sanity-check tool for `data_collection.collect_data` outputs.

Usage:
    python -m vla_lab.inspect_data --data-roots logs/data_collection
    python -m vla_lab.inspect_data --data-roots logs/data_collection --print-instructions

This tool does not require Isaac Lab and is safe to run in any Python env
that has Pillow + PyTorch installed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List

from .dataset import discover_episodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect collect_data sessions")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        type=str,
        default=["logs/data_collection"],
    )
    parser.add_argument("--print-instructions", action="store_true")
    parser.add_argument("--max-episodes-shown", type=int, default=5)
    args = parser.parse_args()

    roots: List[Path] = [Path(p) for p in args.data_roots]
    episodes = discover_episodes(roots)
    if not episodes:
        print(f"[inspect] no episodes found under {[str(r) for r in roots]}")
        return 1

    total_ticks = sum(len(ep.ticks) for ep in episodes)
    ticks_with_image = sum(
        sum(1 for t in ep.ticks if t.image_rel) for ep in episodes
    )
    ticks_with_action = sum(
        sum(1 for t in ep.ticks if t.action_from_prev is not None) for ep in episodes
    )
    instr_count = Counter(ep.instruction for ep in episodes if ep.instruction)
    target_count = Counter(ep.target_label for ep in episodes if ep.target_label)

    print(f"[inspect] episodes:        {len(episodes)}")
    print(f"[inspect] ticks total:     {total_ticks}")
    print(f"[inspect] ticks w/ image:  {ticks_with_image}  ({100.0 * ticks_with_image / max(1, total_ticks):.1f}%)")
    print(f"[inspect] ticks w/ action: {ticks_with_action}  ({100.0 * ticks_with_action / max(1, total_ticks):.1f}%)")
    print(f"[inspect] unique instructions: {len(instr_count)}")
    print(f"[inspect] unique targets:      {len(target_count)}")

    if args.print_instructions:
        print("[inspect] top instructions:")
        for s, c in instr_count.most_common(20):
            print(f"  {c:4d}  {s!r}")
        print("[inspect] top targets:")
        for s, c in target_count.most_common(20):
            print(f"  {c:4d}  {s!r}")

    print(f"[inspect] sample episodes (showing up to {args.max_episodes_shown}):")
    for ep in episodes[: args.max_episodes_shown]:
        n_with_img = sum(1 for t in ep.ticks if t.image_rel)
        print(
            f"  {ep.folder}  ticks={len(ep.ticks):4d}  imgs={n_with_img:4d}  "
            f"target={ep.target_label!r}  instr={ep.instruction!r}"
        )

    if ticks_with_image == 0:
        print(
            "[inspect] WARNING: no images found. Did you run collect_data with `--enable_cameras`?"
        )
    if ticks_with_action == 0:
        print(
            "[inspect] WARNING: no policy.action_from_prev fields. Was data collected at >=1 tick?"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

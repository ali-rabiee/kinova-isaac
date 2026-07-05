"""vla_v4 — vla_v1 with a calibrated wrist (eye-in-hand) camera recorded per tick.

This is a thin profile, not a fork: it reuses vla_v1's CLI schema and run loop
and only flips the wrist camera on by default. Everything vla_v1 records stays
identical; vla_v4 sessions additionally contain:

- ``images/wrist_XXXXXX.png``            one wrist frame per logged tick
- ``image_wrist: {path}``                per tick in ticks.jsonl
- ``cameras.json``                       per episode: pinhole intrinsics for both
                                         cameras, post-DR overhead pose, and the
                                         wrist mount (hand-eye) extrinsics
- ``wrist_images``                       count in episode_summary.json

Camera definition: ``DEFAULT_WRIST_CAMERA`` in environments/reach_to_grasp_VLA/config.py.
Entry point: ``vla_lab/scripts/collect_v4.sh`` (same contract as collect_v3 + wrist).
"""

from __future__ import annotations

import argparse

from . import vla_v1


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    vla_v1.add_cli_args(parser)
    parser.set_defaults(wrist_camera=True)


def run(args) -> int:
    return vla_v1.run(args)

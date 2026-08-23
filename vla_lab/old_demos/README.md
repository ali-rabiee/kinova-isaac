# `old_demos/` — superseded artifacts

Moved here 2026-08-22 so the `vla_lab/` top level is about the live direction. Nothing was
deleted.

| | |
| --- | --- |
| `checkpoints/` | Smoke and early fine-tune checkpoints (TinyVLA `tiny_v0`/`tiny_run1`, several SmolVLA runs, `smoke_multicam`) plus their train manifests. |
| `eval_results/` | Closed-loop eval outputs from those checkpoints. |
| `results/` | Dated result directories from the act/compute/query track (2026-05 → 2026-06). |
| `datasets/` | An early LeRobot export (`lerobot_kinova_v0`). |
| `docs/` | Historical reports: the eval-flailing postmortem, the data-collection fix report, the 2026-07 reframing, the pre-pivot recommendations, the CoRL-era strategy notes, and the older agent instruction files. |
| `scripts/legacy/` | Superseded collectors (`collect.sh` at 5 Hz, `collect_v2` pick-place, `collect_temp`). Do **not** use without porting the object-respawn fix. |
| `unity_smoketest.*` | A Unity-bridge smoke test with no current consumer. |

The data-collection guide stayed in `../docs/` because it is still the reference for anything
that boots the simulator.

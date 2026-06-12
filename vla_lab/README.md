# `vla_lab/` — VLA training & evaluation for the Kinova Jaco in Isaac Sim

This package contains everything needed to go from **scripted demonstrations** in Isaac Sim
to a **language-conditioned visuomotor policy** (VLA) and back into the simulator for
closed-loop evaluation — plus the HRI-2027 *act / compute / query* allocation experiments
built on top of it.

```
collect (Isaac, scripted expert)  →  verify  →  train (TinyVLA or SmolVLA)  →  eval (Isaac, closed loop)
   collect_v3.sh                 verify_session     train.sh / train_smolvla.sh     eval.sh
```

Task: a Kinova Jaco `j2n6s300` on a table with 6 colored boxes; the policy receives one
top-down RGB image, the EE state, and an instruction like *"Pick up the red box."*, and must
reach, grasp, and lift the right box.

> **New here?** Read this file top-to-bottom, then
> [`data_collection_guide.md`](./data_collection_guide.md) before collecting any data.

---

## Quick start

```bash
conda activate riften                       # NOT isaac_env; needs numpy<2
cd ~/Desktop/Depo/Code/CORL/kinova-isaac

# 1. Collect demonstrations (repeat for more chunks; ~2 min/episode headless)
NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless

# 2. Verify the session (MANDATORY — exit 1 means do not train on it)
python -m vla_lab.verify_session logs/data_collection/session_<TS>

# 3. Train TinyVLA (edit data.data_roots in vla_lab/configs/train_tiny.yaml first)
./vla_lab/scripts/train.sh

# 4. Evaluate in Isaac Lab (closed loop, same scene/camera as collection)
./vla_lab/scripts/eval.sh --num-episodes 10 --headless
```

## 1. Environment

| Requirement | Why |
| --- | --- |
| conda env **`riften`** | Isaac Sim 5.x + Isaac Lab + torch are set up there (`isaac_env` is stale). |
| **NumPy < 2** | NumPy 2.x freezes/crashes Isaac data collection. `python -c "import numpy; print(numpy.__version__)"` |
| Isaac Lab checkout | `eval.sh` looks for `./IsaacLab/isaaclab.sh` or `~/IsaacLab/isaaclab.sh` (override with `ISAACLAB=`). |
| `pip install matplotlib` | optional, for training plots (`plot_metrics`). |
| `pip install -r vla_lab/requirements-smolvla.txt` | only for the SmolVLA path (LeRobot). |

Training (`vla_lab.train`) and all offline tools run with plain `python` — no Isaac needed.
Only collection and eval start the simulator.

## 2. Folder map

```
vla_lab/
├── README.md                    <- you are here
├── data_collection_guide.md     <- THE data-collection reference (pipeline, contract, fixes)
├── docs/                        <- historical reports (eval-flailing postmortem, design notes)
│
├── scripts/                     <- entrypoints (each wraps one command; env-var overridable)
│   ├── collect_v3.sh            <- data collection: reach/grasp/lift, 15 Hz, DR  ★
│   ├── train.sh                 <- TinyVLA training                              ★
│   ├── eval.sh                  <- closed-loop Isaac Lab evaluation              ★
│   ├── dryrun.sh                <- VRAM/latency sanity for a checkpoint (no Isaac)
│   ├── export_lerobot_dataset.sh / train_smolvla.sh / after_smolvla_train.sh
│   ├── run_tests.sh             <- 51 offline tests (allocator/calibration/human study)
│   ├── sweep_occlusion_eval.sh / fit_allocator.sh / calibration_*.sh
│   ├── human_study_pilot.sh / human_study_analyze.sh / power_analysis.sh
│   └── legacy/                  <- superseded collectors (collect.sh 5 Hz, collect_v2 pick-place,
│                                   collect_temp) — do NOT use without porting the respawn fix
│
├── dataset.py                   <- ticks.jsonl → torch Dataset (success-only filter, action chunks)
├── models.py / losses.py        <- TinyVLA (1.9 M params) + optional DINOv2 alignment
├── train.py                     <- trainer (AMP, resume, metrics.jsonl)
├── eval_isaaclab.py             <- eval loop: policy/replay backends, pre-roll, safety clamps, TTC
├── ttc.py / partial_obs.py      <- K-sample inference + occlusion axes
├── verify_session.py            <- post-collection session validator  ★ run after every collection
├── inspect_data.py              <- quick textual dump of a session
├── repair_gripper_labels.py     <- backfills gripper labels in pre-2026-06 sessions
├── dryrun.py / plot_metrics.py / stats_utils.py / checkpoint_utils.py
│
├── smolvla_bridge/              <- ticks.jsonl ↔ LeRobot dataset + SmolVLA policy wrapper
├── allocation/ calibration/     <- HRI-2027 act/compute/query allocator + Result-2 calibration
├── human_study/ baselines/ ttc_methods/  <- study runner, comparison policies
├── real_robot/                  <- Kinova bridge + safety envelope stubs for sim→real
├── tests/                       <- offline test suite (run_tests.sh)
├── configs/                     <- train_tiny.yaml, eval_isaac.yaml, train_smolvla.example.yaml, eval_real.yaml
│
├── checkpoints/ datasets/       <- training outputs / LeRobot exports   (artifacts, gitignored)
├── eval_results/ results/       <- eval + experiment outputs            (artifacts)
└── paper/                       <- CoRL LaTeX sources
```

★ = the four commands you will actually use day-to-day.

## 3. Data collection

Full reference: **[`data_collection_guide.md`](./data_collection_guide.md)** — read it before
collecting the final dataset. Short version:

```bash
# Default: 6 unique-color boxes, cycling targets, domain randomization, 15 Hz ticks
NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless

# Useful overrides (env vars):
NUM_EPISODES=120 NUM_OBJECTS=6 ./vla_lab/scripts/collect_v3.sh --headless
TARGET_SELECTION=random ./vla_lab/scripts/collect_v3.sh      # instead of cycle
DR_SEED=7 ./vla_lab/scripts/collect_v3.sh                    # reproducible randomization
USE_YCB=1 ./vla_lab/scripts/collect_v3.sh                    # YCB meshes (separate dataset!)
PLANNER=curobo_v2 ./vla_lab/scripts/collect_v3.sh            # MotionGen instead of scripted

# Watch the first run: respawn readback must CHANGE between episodes, targets must cycle.
```

Outputs land in `logs/data_collection/session_<TS>/episode_NNNN/` with `ticks.jsonl`
(15 Hz states + actions), `images/` (640×640 top-down PNGs), `instruction.json`,
`episode_summary.json` (success verdict), and `events.jsonl` (full audit trail).

**Rules that keep the data usable** (details + rationale in the guide):

1. **Verify every session** before training: `python -m vla_lab.verify_session <session_dir>`.
   It catches the historical failure modes (frozen object layout, degenerate targets, missing
   gripper labels, missing images) and refuses with exit 1.
2. **Collect in chunks** (30–60 episodes per run, headless) — long runs have been OOM-killed
   (exit 137). Each chunk is a separate session; list them all under `data.data_roots`.
3. **Never mix contracts**: sessions in one training run must share the same `--log-rate-hz`
   (15), camera config, and start pose. Sessions collected **before 2026-06-11 are not
   compatible** (camera moved, frozen-layout bug) — retire them.
4. ~120 episodes (= 20 demos per color) is a sensible floor for the 6-color task.

## 4. Inspect what was collected

```bash
# Hard pass/fail + statistics (use this one):
python -m vla_lab.verify_session logs/data_collection/session_<TS>

# Casual look at episodes/ticks/instructions:
python -m vla_lab.inspect_data --data-roots logs/data_collection/session_<TS> --print-instructions

# Old sessions (pre-2026-06) only: backfill gripper open/close labels in-place:
python -m vla_lab.repair_gripper_labels --session logs/data_collection/session_<TS> [--dry-run]
```

## 5. Training

### 5.1 TinyVLA (default, no Isaac required)

```bash
# 1. Point the config at your verified session(s):
#    vla_lab/configs/train_tiny.yaml  ->  data.data_roots: [logs/data_collection/session_<TS>, ...]
# 2. Train (checkpoints + metrics.jsonl + plots in vla_lab/checkpoints/tiny_v0/):
./vla_lab/scripts/train.sh

# Variants:
./vla_lab/scripts/train.sh --auto-resume                 # continue from last.pt
./vla_lab/scripts/train.sh --epochs 50 --batch-size 256  # quick overrides
python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml --data-roots \
    logs/data_collection/session_A logs/data_collection/session_B   # multi-session

# Sanity-check a checkpoint's latency/VRAM and action magnitudes (no Isaac):
./vla_lab/scripts/dryrun.sh --ckpt vla_lab/checkpoints/tiny_v0/last.pt --iters 50 --k 1
```

Notes:

- `data.success_only: true` (default) trains on successful demonstrations only, using
  `episode_summary.json` / `events.jsonl`.
- Action normalization stats are stored in the checkpoint; eval denormalizes automatically.
- The tokenizer is built from the sessions' instructions — if a color never appears as a
  target in your data, the model cannot understand it at eval (one more reason for
  `TARGET_SELECTION=cycle` + verify_session's distribution check).

### 5.2 SmolVLA (LeRobot fine-tune)

```bash
pip install -r vla_lab/requirements-smolvla.txt   # ideally a separate env; see docs/new_changes.md

# 1. Export ticks → LeRobot dataset (successful episodes only by default):
./vla_lab/scripts/export_lerobot_dataset.sh \
    --session-roots logs/data_collection/session_<TS> \
    --out-dir vla_lab/datasets/lerobot_kinova_v0 --fps 15 --overwrite

# 2. Fine-tune lerobot/smolvla_base on it:
DATASET_DIR=vla_lab/datasets/lerobot_kinova_v0 STEPS=20000 ./vla_lab/scripts/train_smolvla.sh

# 3. Evaluate with the same Isaac loop (backend switch in the YAML or CLI):
./vla_lab/scripts/eval.sh --policy-backend smolvla --ckpt vla_lab/checkpoints/smolvla_ft_<TS> \
    --lerobot-dataset-root vla_lab/datasets/lerobot_kinova_v0
```

## 6. Evaluation (Isaac Lab)

```bash
# Default config (vla_lab/configs/eval_isaac.yaml): 10 episodes, scene mirrors collect_v3
./vla_lab/scripts/eval.sh --num-episodes 10 --headless --ckpt vla_lab/checkpoints/tiny_v0/last.pt

# With the GUI (watch the behavior):
./vla_lab/scripts/eval.sh --num-episodes 5 --ckpt vla_lab/checkpoints/tiny_v0/last.pt

# Per-tick action diagnostics (JSONL + console magnitudes, clamp counts):
./vla_lab/scripts/eval.sh --num-episodes 3 --headless --debug-actions

# Execution-path regression test WITHOUT a policy: open-loop replay of a recorded episode.
# Expect smooth motion and ~mm-level EE tracking error vs the demo.
./vla_lab/scripts/eval.sh --replay-episode logs/data_collection/session_<TS>/episode_0000 --headless

# K-sample TTC / act-compute-query controllers: see §8.
```

Results: `vla_lab/eval_results/<run>/results.json` (per-episode success, lift heights,
latency stats).

### If eval motion ever looks erratic again

`docs/EVAL_DEBUG_REPORT.md` is the postmortem of the 2026-06 "flailing" bug. Checklist:

1. `eval.policy_rate_hz` **must equal** the collection `--log-rate-hz` (15). A mismatch
   stretches/compresses every action in time (`train_action_rate_hz` rescales + warns).
2. The eval pre-roll target `eval.start_ee_pos_b` must equal collection's
   `--start-ee-pos-b` (0.454 0.093 0.210).
3. Run the `--replay-episode` test above: if replay is smooth, the model is the problem;
   if not, the execution path is.
4. `--debug-actions` should show |Δp| ≲ 35 mm per tick and zero safety clamps.

## 7. The model contract (do not change one side only)

| Item | Value | Defined in |
| --- | --- | --- |
| Observation | one 224×224 RGB (resized from 640×640 top-down) + EE pos + gripper flag | `dataset.py`, camera in `environments/reach_to_grasp_VLA/config.py` |
| Action | `(8, 7)` chunk of per-tick deltas `[dx,dy,dz,drx,dry,drz,g]`, base frame, `g∈{-1,0,+1}` | `data_collection/core/logger.py` (`action_from_prev`) |
| Tick rate | **15 Hz** sim time | `collect_v3.sh` ≡ `eval_isaac.yaml policy_rate_hz` |
| Start pose | EE `(0.454, 0.093, 0.210)` base frame | `--start-ee-pos-b` ≡ `eval.start_ee_pos_b` |
| Scene | 6×8 cm unique-color boxes, spawn AABB (0.26,−0.34)–(0.52,0.36), ≥0.16 m apart | `collect_v3.sh` ≡ `eval_isaac.yaml` |

## 8. HRI-2027 act / compute / query experiments

Built on top of the eval loop: at each policy tick a controller decides to **act** (1 forward
pass), **compute** (K samples + consensus), or **query** the human. See `allocation/`,
`calibration/`, `human_study/`.

```bash
# Offline first — no Isaac, no robot, ~seconds:
./vla_lab/scripts/run_tests.sh                       # 51 tests must pass
./vla_lab/scripts/human_study_pilot.sh               # synthetic end-to-end study + figures
./vla_lab/scripts/power_analysis.sh                  # sample-size memo

# Robot-only calibration sweep ("Result 2", needs Isaac):
OUT_ROOT=vla_lab/results/calibration_records ./vla_lab/scripts/sweep_occlusion_eval.sh
./vla_lab/scripts/fit_allocator.sh                   # -> allocator_fit.json
./vla_lab/scripts/calibration_analyze.sh             # -> reliability/ECE/coverage figures

# Controllers inside eval (same scenes/seeds for comparisons):
./vla_lab/scripts/eval.sh --controller allocator --allocator-fit vla_lab/results/allocator_fit.json
./vla_lab/scripts/eval.sh --controller compute_gated   # or: autonomy | fixed_compute | scale | knowno | insight
```

## 9. Known gotchas

- **NumPy 2.x freezes Isaac collection** — keep `numpy<2` in `riften`.
- **`--enable_cameras` is required at collection AND eval** (the wrappers pass it). Without
  it there are no images and the dataset refuses to build.
- **Exit 137 during collection** = OOM kill → headless + smaller chunks.
- **Exit 3 during collection** = the respawn watchdog aborted the run (frozen layout
  protection). Do not bypass; see guide §5.1.
- `eval_isaaclab.py` must run under the **IsaacLab launcher** (`eval.sh` handles this);
  plain `python` can't import `isaaclab.app`.
- Floats in `ticks.jsonl` are **4-decimal strings**; parse with `vla_lab.dataset`.
- First tick of each episode has `action_from_prev = null` (no previous tick) — handled by
  the dataset.
- Sessions collected **before 2026-06-11** have the old camera framing and the frozen-layout
  bug → don't mix with new data; `tiny_v0` checkpoints predate the fixes entirely.

## 10. Document index

| Document | Content |
| --- | --- |
| [`data_collection_guide.md`](./data_collection_guide.md) | Pipeline anatomy, data/action contract, 2026-06-11 bug fixes, final-dataset procedure, troubleshooting |
| [`docs/EVAL_DEBUG_REPORT.md`](./docs/EVAL_DEBUG_REPORT.md) | Postmortem: why eval flailed and how every root cause was fixed |
| [`docs/new_changes.md`](./docs/new_changes.md) | Eval protocol design notes (Wilson CIs, occlusion axes, SmolVLA comparison) |
| [`docs/FABLE_INSTRUCTIONS.md`](./docs/FABLE_INSTRUCTIONS.md) | Historical task brief for the eval debug (context for the report) |

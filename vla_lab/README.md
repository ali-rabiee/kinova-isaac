# `vla_lab/` — VLA training & evaluation for the Kinova Jaco in Isaac Sim

This package contains everything needed to go from **scripted demonstrations** in Isaac Sim
to a **language-conditioned visuomotor policy** (VLA) and back into the simulator for
closed-loop evaluation — plus the HRI-2027 *act / compute / query* allocation experiments
built on top of it.

```
collect (Isaac, scripted expert)  →  verify  →  train (TinyVLA or SmolVLA)  →  eval (Isaac, closed loop)
   collect_v3.sh / collect_v4.sh  verify_session     train.sh / train_smolvla.sh     eval.sh
```

Task: a Kinova Jaco `j2n6s300` on a table with 6 colored boxes; the policy receives a
top-down RGB image (and, since 2026-07, optionally a calibrated **wrist camera** view),
the EE state, and an instruction like *"Pick up the red box."*, and must reach, grasp,
and lift the right box.

> **2026-07 reframing** (see [`fable_report.md`](./fable_report.md)): uncertainty is now
> decomposed by **source** — *perception* (occlusion, missing view) vs *human intent*
> (which object is meant) — and each source routes to a different remedy (compute, the
> other view, or a clarifying query). A two-way feedback channel lets the human interject
> corrections mid-episode, fused by an online reliability estimate. See §9–§10.

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
│   ├── collect_v4.sh            <- collect_v3 + calibrated WRIST camera          ★ (new data)
│   ├── train.sh                 <- TinyVLA training                              ★
│   ├── eval.sh                  <- closed-loop Isaac Lab evaluation              ★
│   ├── dryrun.sh                <- VRAM/latency sanity for a checkpoint (no Isaac)
│   ├── export_lerobot_dataset.sh / train_smolvla.sh / after_smolvla_train.sh
│   ├── run_tests.sh             <- 83 offline tests (allocator/calibration/human study/intent/feedback/multicam)
│   ├── sweep_occlusion_eval.sh / fit_allocator.sh / calibration_*.sh
│   ├── sweep_camera_sets.sh     <- overhead/wrist/both ablation (+1-cam+TTC legs)
│   ├── sweep_feedback_quality.sh<- human input quality grid (accuracy×specificity, fusion vs trust-all)
│   ├── human_study_pilot.sh / human_study_analyze.sh / power_analysis.sh
│   └── legacy/                  <- superseded collectors (collect.sh 5 Hz, collect_v2 pick-place,
│                                   collect_temp) — do NOT use without porting the respawn fix
│
├── dataset.py                   <- ticks.jsonl → torch Dataset (success-only filter, action chunks,
│                                   camera sets: overhead / wrist / both)
├── models.py / losses.py        <- TinyVLA (1.9 M params, shared-encoder multi-camera + camera
│                                   dropout) + optional DINOv2 alignment
├── train.py                     <- trainer (AMP, resume, metrics.jsonl; records camera_set +
│                                   action_rate_hz contract into checkpoints)
├── eval_isaaclab.py             <- eval loop: policy/replay backends, pre-roll, safety clamps, TTC,
│                                   camera-set ablations, intent estimator, feedback channel
├── ttc.py / partial_obs.py      <- K-sample inference + occlusion axes (per-view since 2026-07)
├── intent/                      <- intent-uncertainty estimator (counterfactual sweep) +
│                                   cross-view disagreement (vla_lab.intent)
├── feedback/                    <- two-way human channel: parser, simulated human (quality knobs),
│                                   console channel, reliability-gated fusion (vla_lab.feedback)
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
├── configs/                     <- train_tiny.yaml, train_multicam.yaml (wrist+overhead),
│                                   eval_isaac.yaml, train_smolvla.example.yaml, eval_real.yaml
│
├── checkpoints/ datasets/       <- training outputs / LeRobot exports   (artifacts, gitignored)
├── eval_results/ results/       <- eval + experiment outputs            (artifacts)
└── paper/                       <- HRI-2027 LaTeX sources (+ talk deck in presentation/)
```

★ = the four commands you will actually use day-to-day.

## 3. Data collection

Full reference: **[`data_collection_guide.md`](./data_collection_guide.md)** — read it before
collecting the final dataset. Short version:

```bash
# Default: 6 unique-color boxes, cycling targets, domain randomization, 15 Hz ticks
NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless

# WITH the wrist camera (recommended for all new data; same contract + wrist stream):
NUM_EPISODES=40 ./vla_lab/scripts/collect_v4.sh --headless

# Useful overrides (env vars, both scripts):
NUM_EPISODES=120 NUM_OBJECTS=6 ./vla_lab/scripts/collect_v4.sh --headless
TARGET_SELECTION=random ./vla_lab/scripts/collect_v4.sh      # instead of cycle
DR_SEED=7 ./vla_lab/scripts/collect_v4.sh                    # reproducible randomization
USE_YCB=1 ./vla_lab/scripts/collect_v4.sh                    # YCB meshes (separate dataset!)
PLANNER=curobo_v2 ./vla_lab/scripts/collect_v4.sh            # MotionGen instead of scripted

# Watch the first run: respawn readback must CHANGE between episodes, targets must cycle.
```

Outputs land in `logs/data_collection/session_<TS>/episode_NNNN/` with `ticks.jsonl`
(15 Hz states + actions), `images/` (640×640 top-down PNGs), `instruction.json`,
`episode_summary.json` (success verdict), and `events.jsonl` (full audit trail).
`collect_v4` sessions additionally contain `images/wrist_XXXXXX.png` (one per tick), an
`image_wrist` key per tick, and per-episode **`cameras.json`** — pinhole intrinsics for
both cameras, the post-DR overhead pose, and the wrist hand-eye mount extrinsics
(`DEFAULT_WRIST_CAMERA` in `environments/reach_to_grasp_VLA/config.py`; on the real robot
replace its offsets with your measured hand-eye calibration). The wrist mount/FOV is part
of the trained model's contract exactly like the top-down camera. `verify_session` checks
the wrist stream (every episode has it, files exist, cameras.json present).

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

# Two-camera training (collect_v4 sessions; wrist+overhead, camera dropout 0.25):
#   edit data.data_roots in vla_lab/configs/train_multicam.yaml first
CONFIG=vla_lab/configs/train_multicam.yaml ./vla_lab/scripts/train.sh

# Camera-set ablations = the same config with data.cameras edited to
#   [overhead] | [wrist] | [overhead, wrist]     (one run per set), or via a copy:
python -m vla_lab.train --config vla_lab/configs/train_multicam.yaml \
    --out-dir vla_lab/checkpoints/wrist_only_v0    # after setting cameras: [wrist]
```

The camera set and the sessions' `log_rate_hz` are recorded in each checkpoint
(`camera_set`, `action_rate_hz`); eval reads them to build the right sensors and to
check the rate contract. Old single-camera checkpoints keep loading unchanged.

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

# Camera-set ablation flags (two-camera checkpoints; see §9):
./vla_lab/scripts/eval.sh --ckpt vla_lab/checkpoints/multicam_v0/last.pt \
    --cameras overhead --headless          # wrist masked (missing-view ablation)
./vla_lab/scripts/eval.sh --cameras both --occlusion-mode bottom_strip \
    --occlusion-camera wrist --headless    # occlude ONE view (cross-view conflict)
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
| Observation | 224×224 RGB per camera (resized from 640×640) + EE pos + gripper flag | `dataset.py`, cameras in `environments/reach_to_grasp_VLA/config.py` |
| Camera set | `[overhead]` (legacy) or `[overhead, wrist]` / `[wrist]` (collect_v4 data); recorded as `camera_set` in the checkpoint | `train_*.yaml data.cameras` ≡ eval `--cameras` |
| Wrist mount | offset `(0, −0.055, −0.11)` m, rpy `(180°, 0, 0)` in the EE-link frame, FOV 87° | `WristCameraConfig`; dumped per episode in `cameras.json` |
| Action | `(8, 7)` chunk of per-tick deltas `[dx,dy,dz,drx,dry,drz,g]`, base frame, `g∈{-1,0,+1}` | `data_collection/core/logger.py` (`action_from_prev`) |
| Tick rate | **15 Hz** sim time (also stored as `action_rate_hz` in new checkpoints) | `collect_v3/v4.sh` ≡ `eval_isaac.yaml policy_rate_hz` |
| Start pose | EE `(0.454, 0.093, 0.210)` base frame | `--start-ee-pos-b` ≡ `eval.start_ee_pos_b` |
| Scene | 6×8 cm unique-color boxes, spawn AABB (0.26,−0.34)–(0.52,0.36), ≥0.16 m apart | `collect_v3/v4.sh` ≡ `eval_isaac.yaml` |

## 8. HRI-2027 act / compute / query experiments

Built on top of the eval loop: at each policy tick a controller decides to **act** (1 forward
pass), **compute** (K samples + consensus), or **query** the human. See `allocation/`,
`calibration/`, `human_study/`.

```bash
# Offline first — no Isaac, no robot, ~seconds:
./vla_lab/scripts/run_tests.sh                       # 83 tests must pass
./vla_lab/scripts/human_study_pilot.sh               # synthetic end-to-end study + figures
./vla_lab/scripts/power_analysis.sh                  # sample-size memo

# Robot-only calibration sweep ("Result 2", needs Isaac):
OUT_ROOT=vla_lab/results/calibration_records ./vla_lab/scripts/sweep_occlusion_eval.sh
./vla_lab/scripts/fit_allocator.sh                   # -> allocator_fit.json
./vla_lab/scripts/calibration_analyze.sh             # -> reliability/ECE/coverage figures

# Controllers inside eval (same scenes/seeds for comparisons):
./vla_lab/scripts/eval.sh --controller allocator --allocator-fit vla_lab/results/allocator_fit.json
./vla_lab/scripts/eval.sh --controller compute_gated   # or: autonomy | fixed_compute | scale | knowno | insight
./vla_lab/scripts/eval.sh --controller intent_allocator  # routes by uncertainty SOURCE (see §10)
```

## 9. Wrist camera & camera-set ablations (2026-07)

The premise: with two calibrated views, "is one camera + our method as good as two?"
becomes a measurable claim. Pipeline:

```bash
# 1. Collect two-camera data and verify it:
NUM_EPISODES=120 ./vla_lab/scripts/collect_v4.sh --headless
python -m vla_lab.verify_session logs/data_collection/session_<TS>

# 2. Train the two-camera policy (camera dropout teaches single-view robustness):
CONFIG=vla_lab/configs/train_multicam.yaml ./vla_lab/scripts/train.sh

# 3. Run the full ablation matrix on identical scenes/seeds:
#    both / overhead-only / wrist-only / one-camera+gated-TTC / per-view occlusion
CKPT=vla_lab/checkpoints/multicam_v0/last.pt NUM_EPISODES=50 \
    ./vla_lab/scripts/sweep_camera_sets.sh --headless
```

Mechanics: one shared vision encoder runs per view + a learned camera-ID embedding;
`model.camera_dropout: 0.25` zeroes random streams during training; at eval,
`--cameras overhead|wrist|both` masks the missing stream (`camera_present`), and
`--occlusion-camera` picks which view the occlusion mask hits. A **cross-view
disagreement** probe (overhead-only vs wrist-only predictions of the same model) joins
the allocator's feature set as a perception-uncertainty signal.
SmolVLA path: export with `--wrist` (`export_lerobot_dataset.sh` flag routes the wrist
view into `camera2`) and set `smolvla_cameras: both` in the eval YAML.

## 10. Human-intent uncertainty & real-time feedback (2026-07)

Uncertainty is decomposed by **source** and routed to the matching remedy:

- **Perception** (can't see): K-sample dispersion, occlusion, cross-view disagreement
  → more compute (best-of-K) or the other view.
- **Intent** (don't know WHICH object you mean): `vla_lab/intent` runs a counterfactual
  instruction sweep per tick (one cheap forward per candidate color) → a posterior over
  targets; its normalized entropy is the intent-uncertainty scalar → clarifying query
  (compute cannot resolve intent).

```bash
# Intent-routed controller + deliberately ambiguous instructions:
./vla_lab/scripts/eval.sh --controller intent_allocator --instruction-ambiguity half --headless

# Two-way feedback with a SIMULATED human (quality knobs: accuracy/latency/specificity/noise):
./vla_lab/scripts/eval.sh --controller intent_allocator --instruction-ambiguity half \
    --feedback-channel scripted --feedback-accuracy 0.6 --feedback-specificity color --headless

# LIVE keyboard channel (type "no, the red one", "a bit left", "stop" during rollout):
./vla_lab/scripts/eval.sh --feedback-channel console

# The input-quality study grid (accuracy × specificity, fusion vs trust-all ablation):
CKPT=vla_lab/checkpoints/multicam_v0/last.pt NUM_EPISODES=30 \
    ./vla_lab/scripts/sweep_feedback_quality.sh --headless
```

How feedback is handled (`vla_lab/feedback`): utterances are parsed into typed
refinements (target override / avoid / directional nudge / stop / resume), then **fused
by an online reliability estimate** (Beta posterior over "this human is right"):
apply, verify (a confirmation query through the same channel), or ignore.
`--no-fusion-verify` is the trust-all ablation. Robot-initiated queries and unsolicited
corrections share one human model, so `--feedback-accuracy` is the single quality
variable of the study. Per-episode feedback/intent traces land in `results_*.json`
(`feedback.events`, `intent.mean_entropy`, controller `uncertainty_type` counts).

## 11. Known gotchas

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
- **Wrist-consuming models need collect_v4 sessions** — the dataset refuses (with a clear
  error) when `cameras` includes `wrist` but ticks have no `image_wrist`. Never mix
  collect_v3 and collect_v4 sessions in one wrist-consuming training run.
- The eval `--cameras` set must be a **subset of the checkpoint's `camera_set`** — a
  single-camera checkpoint cannot consume the other view.

## 12. Document index

| Document | Content |
| --- | --- |
| [`RECOMMENDATIONS.md`](./RECOMMENDATIONS.md) | 2026-07-05 project assessment: HRI-2027 fit/verdict, 13-week critical path, go/no-go gate, prioritized implementation list |
| [`data_collection_guide.md`](./data_collection_guide.md) | Pipeline anatomy, data/action contract, 2026-06-11 bug fixes, final-dataset procedure, troubleshooting |
| [`fable_report.md`](./fable_report.md) | 2026-07 reframing report: wrist camera, intent/perception decomposition, feedback fusion — what was built, why, and the July-2026 literature map |
| [`docs/EVAL_DEBUG_REPORT.md`](./docs/EVAL_DEBUG_REPORT.md) | Postmortem: why eval flailed and how every root cause was fixed |
| [`docs/new_changes.md`](./docs/new_changes.md) | Eval protocol design notes (Wilson CIs, occlusion axes, SmolVLA comparison) — pre-wrist framing |
| [`docs/FABLE_INSTRUCTIONS.md`](./docs/FABLE_INSTRUCTIONS.md) | Historical task brief for the eval debug (context for the report) |

# `vla_lab/` — VLA-TTC starter package

This folder is the home of the **CoRL 2026 VLA-TTC** project (codename
`vla-ttc`), built on top of the existing `kinova-isaac` simulation /
data-collection stack. The full design document lives at
[`vla_ttc_engineering_spec.md`](./vla_ttc_engineering_spec.md); this
README is the practical "how do I actually run it" companion.

The folder is **fully self-contained**: every change required for the
project lives under `vla_lab/`. The rest of the repository
(`data_collection/`, `environments/`, `controllers/`, ...) was not
modified and is consumed only via standard imports.

**Paper direction (2026 sprint):** stress *partial observability* and
single-camera manipulation—test-time scaling should help most when visual
ambiguity is high. See `new_changes.md` for the full pivot notes; code
support includes occlusion ablations (`partial_obs.py`), SmolVLA TTC
(`smolvla_bridge/policy_wrapper.py`), MG-Select-style scoring
(`baselines/mg_select.py`), and uncertainty gating (`ttc_methods/`).

---

## 1. What is implemented today

The Phase-1 pipeline is end-to-end runnable: **collect → train → eval**.

| Stage | Path | What it does |
| --- | --- | --- |
| Data collection | reuses `data_collection.collect_data` (no edits needed) | Saves `vla_v1` sessions with images + ticks + per-episode language. |
| Dataset reader | `vla_lab/dataset.py` | Reads `session_*/episode_*` folders; converts stringified floats; builds tokenizer + action stats. |
| Baseline model | `vla_lab/models.py` (`TinyVLA`, ~2 M params) | Vision CNN + tiny language transformer + state MLP + cross-attention action decoder. |
| Loss(es) | `vla_lab/losses.py` | Masked action MSE + optional DINOv2 feature-alignment loss (lazy import). |
| Trainer | `vla_lab/train.py` | Single-GPU AdamW + cosine schedule + checkpointing. |
| TTC inference | `vla_lab/ttc.py` | K-noise sampling at the bottleneck + consensus scoring fallback. |
| IsaacLab eval | `vla_lab/eval_isaaclab.py` | Drops the trained policy into `reach_to_grasp_VLA` and reports success. |
| Dryrun / inspect | `vla_lab/dryrun.py`, `vla_lab/inspect_data.py` | VRAM/latency check + dataset sanity tool. |
| Plotting | `vla_lab/plot_metrics.py` | Learning curves + eval success (Wilson CI) from `metrics.jsonl` / eval JSON. |

What this gives you on day one:

- A small (~2 M parameters) **TinyVLA** baseline that trains on a single
  consumer GPU with **no required external model downloads**.
- A clean upgrade path to **SmolVLA** later: the model wrapper exposes
  `forward(image, state, lang_ids, lang_mask, noise=...)` and
  `sample_actions(...)` with the same contract used by the spec, so we
  can drop in a SmolVLA wrapper without touching the trainer / TTC /
  eval code.
- Sanity-check tooling (`dryrun`, `inspect_data`) so the spec's
  Day-3 / Day-12 / Day-14 / Day-18 hard checkpoints can be evaluated
  early.

## 2. What is intentionally stubbed / left for later

These match the spec's later phases and can be added without changing
the public APIs already shipped.

- **SmolVLA fine-tuning** — use `export_lerobot_dataset.sh`, then e.g. **`STEPS=20000 ./vla_lab/scripts/train_smolvla.sh`** or `python -m vla_lab.train_smolvla --config ...` (§5.4.1); Isaac eval can use `policy_backend: smolvla` when checkpoint paths are set.
- **Learned verifier** (spec §4.9). Phase-1 uses a simple "consensus to
  median" scorer in `vla_lab/ttc.py`. The spec's hard checkpoint at
  Day 12 says we explicitly fall back to this if verifier accuracy
  stalls, so this is a publishable baseline.
- **OOD detector / trigger** (spec §4.11/4.12). The current pipeline
  always uses the slow K-sample path; the trigger logic is a future drop-in.
- **Scene parser + inpainter** (spec §4.10). Skipped for now per the
  Day-3 VRAM-budget guidance ("drop SDXL inpainting if it doesn't fit").
- **Background augmentation** (spec §4.8). Train-time only; orthogonal
  to everything here.
- **Real-robot collection** (spec §4.1). Out of scope for the Phase-1
  sim-only loop.

## 3. Repository layout

```
vla_lab/
├── README.md                       <- you are here
├── vla_ttc_engineering_spec.md     <- the full design doc
├── __init__.py
├── _path.py                        <- sys.path fix-up for IsaacLab runs
├── configs/
│   ├── train_tiny.yaml             <- baseline training config
│   ├── train_smolvla.example.yaml  <- SmolVLA / lerobot-train YAML shim
│   └── eval_isaac.yaml             <- IsaacLab eval config
├── dataset.py                      <- KinovaSessionDataset, TinyTokenizer
├── models.py                       <- TinyVLA + ModelOutput
├── losses.py                       <- masked_action_loss, FeatureAlignmentLoss
├── ttc.py                          <- TTCPipeline (K-sample + consensus)
├── train.py                        <- TinyVLA training entrypoint
├── train_smolvla.py                <- SmolVLA: runs `lerobot-train` from YAML
├── eval_isaaclab.py                <- IsaacLab evaluation entrypoint
├── dryrun.py                       <- VRAM / latency dry-fit
├── inspect_data.py                 <- session sanity tool
├── plot_metrics.py                 <- figures for training / eval (paper-style)
└── scripts/
    ├── collect.sh                  <- **vla_v1** stable reach/grasp/lift (default data collection)
    ├── collect_v3.sh               <- same as collect.sh (--profile vla_v1); numbered alias for clarity
    ├── collect_v2.sh               <- **vla_v2** pick-and-place + bins
    ├── train.sh                    <- TinyVLA: wrapper around `vla_lab.train`
    ├── train_smolvla.sh            <- SmolVLA: `lerobot-train` (env vars + resume)
    ├── export_lerobot_dataset.sh   <- Kinova sessions → LeRobot dataset on disk
    ├── pip_install_smolvla_isaac.sh <- LeRobot into Isaac env (NumPy 1.x workaround)
    ├── after_smolvla_train.sh      <- optional TensorBoard / plot_metrics hook after SmolVLA run
    ├── eval.sh                     <- wrapper around `vla_lab.eval_isaaclab`
    ├── plot.sh                     <- wrapper around `vla_lab.plot_metrics`
    └── dryrun.sh                   <- wrapper around `vla_lab.dryrun`
```

Nothing outside `vla_lab/` was modified to add this package.

## 4. Prerequisites

The training and inspection tools have **light** dependencies:

```bash
pip install torch torchvision numpy pillow pyyaml
# For PDF/PNG figures after training (`plot_metrics`, auto-run at train end):
pip install matplotlib
```

The Isaac Lab evaluation entrypoint additionally needs the same Isaac
Sim / Isaac Lab environment that the rest of the repo already uses;
follow the top-level `README.md` to set that up.

Optional dependencies (only if you want to enable specific features):

| Feature | Install |
| --- | --- |
| DINOv2 feature alignment | `pip install transformers` |
| SmolVLA fine-tune (LeRobot) | **Dedicated env:** `pip install -r vla_lab/requirements-smolvla.txt`. **Isaac env (`riften`):** `./vla_lab/scripts/pip_install_smolvla_isaac.sh` — see [`smoll-vla-setup.md`](./smoll-vla-setup.md) |

## 5. End-to-end commands

All commands assume you run them from the **repo root**
(`kinova-isaac/`), not from inside `vla_lab/`.

### 5.1 Collect data (`vla_v1`)

**Recommendation (current setup):** Prefer **colored boxes** — default `collect_v3.sh` uses `--spawn-mode box` unless you override it. Training uses a **fixed top-down camera only** (no wrist / eye-in-hand camera yet). YCB assets are more diverse in shape and appearance; without a wrist view, policies have a harder time from a single overhead view, so **boxes are the safer default** until wrist cameras are added. YCB commands below are still documented for later or for experiments.

**Command to run (from the repo root `kinova-isaac/`, in your Isaac Lab / Isaac Sim Python env):**

```bash
# Recommended default: boxes, scripted planner, top-down images
NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh
```

**Other useful invocations (same repo root):**

```bash
# cuRobo MotionGen instead of scripted straight-line waypoints
PLANNER=curobo_v2 NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh

# More / fewer objects (default in script is 11)
NUM_OBJECTS=8 NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh

# Headless + GPU
DEVICE=cuda:0 NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh --headless

# --- YCB / USD props (optional; prefer boxes until wrist camera exists) ---

# YCB via convenience flag: adds --use-ycb, which forces spawn_mode=usd inside the profile
# (even if SPAWN_MODE=box were set on the wrapper)
USE_YCB=1 NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh

# Same outcome for assets: pass USD spawn explicitly (no --use-ycb)
SPAWN_MODE=usd NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh

# Local YCB (or any USD prop folders) instead of Nucleus default
NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh --use-ycb --objects-dataset /path/to/YCB

# Original thin wrapper (simpler defaults than collect_v3)
NUM_EPISODES=10 ./vla_lab/scripts/collect.sh
```

**`USE_YCB=1` vs `SPAWN_MODE=usd`:** For `collect_v3.sh`, both end up spawning **USD props** from the **same default YCB location** when `--objects-dataset` is empty. The difference: `--use-ycb` forces **`spawn_mode=usd` in Python** after argparse, so it **overrides** a conflicting `--spawn-mode box` on the command line. `SPAWN_MODE=usd` only sets `--spawn-mode usd`; `--use-ycb` is not set.

- Uses **scripted** straight-line waypoints + Diff IK (default `PLANNER=scripted`).
- Logs go to `logs/data_collection/session_<timestamp>/episode_####/`.
- Ensure **`--enable_cameras`** stays on for `vla_lab` training data (already in `collect_v3.sh`).

**There is no `collect_v1.sh`.** The stable profile is `vla_v1`, and it is what
`collect.sh` runs. `collect_v3.sh` adds **curriculum defaults**: more objects
(default **`NUM_OBJECTS=11`** in the script), a spawn AABB **pulled a bit toward the robot**,
`--target-selection farthest_no_repeat` (farthest reachable object, not the same as the
previous episode’s target after a **successful** lift), **straight scripted
approach** (`--approach-detour-m 0` by default). Optional detour: e.g. `--approach-detour-m 0.08 --approach-detour-safe-z-margin-m 0.04`.
For YCB props, use **`USE_YCB=1`** or **`--spawn-mode usd`** as above.

This calls `data_collection.collect_data` with the `vla_v1` profile + planner +
cameras + domain randomization. Each session is written under
`logs/data_collection/session_<TIMESTAMP>/` with one `episode_NNNN/` folder per
attempt.

```bash
# collect_v3: boxes recommended; ~11 objects by default; farthest_no_repeat target selection
NUM_EPISODES=10 ./vla_lab/scripts/collect_v3.sh

# Original wrapper — simpler defaults than collect_v3 (see vla_lab/scripts/collect.sh)
NUM_EPISODES=10 ./vla_lab/scripts/collect.sh

# Pick-and-place (`vla_v2`) — see §5.1.1
NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh
```

`collect_v3.sh` forwards extra flags; see the script for the full default list
(farthest target, spawn AABB, speeds, `NUM_OBJECTS`). Minimal mental model:

```bash
python -m data_collection.collect_data \
  --profile vla_v1 --env reach_to_grasp_VLA --control planner \
  --planner scripted --device cuda:0 --enable_cameras \
  --log-rate-hz 5 --num-episodes ${NUM_EPISODES} \
  --target-selection farthest_no_repeat --approach-detour-m 0 \
  --spawn-mode box --domain-rand --domain-rand-seed 0 \
  --logs-root logs/data_collection \
  ...
```

You can also call `data_collection.collect_data` directly with any flags you
want — see the top-level `README.md`. Just make sure `--enable_cameras` is
set; `vla_lab` requires the per-tick PNGs to train.

#### 5.1.1 Pick-and-place (`vla_v2`)

For the same **top-down-only** setup as `vla_v1`, **colored cubes remain the
recommended default** until a wrist camera exists; YCB is optional below.

`vla_v2` is a richer scene with **clutter + 3 bins** and a fully scripted
**pick-and-place** routine. Each episode, the grasp target is the object
**closest to the robot in XY** (base frame), skipping anything too close to a
bin so the arm does not fight bin geometry; clutter is also respawned with
its X range capped **in front of** the bins. You can use **colored cubes**
(default) or **YCB USD props** from Isaac Nucleus via `--spawn-mode usd`.
Motion is purely scripted (no cuRobo / MotionGen). Logging format and on-disk
layout match `vla_v1`, so the same `vla_lab.dataset` reader works without changes.

**Command to run (from the repo root, with a Python that has Isaac Lab /
`isaaclab` on the path — e.g. your usual Isaac conda env):**

```bash
NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh
```

Common variations:

```bash
# Default: 6 clutter boxes + 1 close target + 3 bins, 10 episodes
NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh

# More clutter, headless
NUM_OBSTACLE_BOXES=8 NUM_EPISODES=20 ./vla_lab/scripts/collect_v2.sh --headless

# Random bin selection per episode
BIN_SELECTION=random NUM_EPISODES=20 ./vla_lab/scripts/collect_v2.sh

# YCB objects (USD from Isaac Nucleus default path) instead of cubes
NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh --spawn-mode usd

# YCB with an explicit asset folder (optional)
NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh \
  --spawn-mode usd \
  --objects-dataset /path/to/YCB
```

Underneath, the wrapper runs `python -m data_collection.collect_data` with
`--profile vla_v2`. Equivalent explicit invocation (same defaults as the
script; add or override flags as needed):

```bash
python -m data_collection.collect_data \
  --profile vla_v2 \
  --env reach_to_grasp_VLA \
  --control planner \
  --device cuda:0 \
  --enable_cameras \
  --log-rate-hz 5 \
  --num-episodes 10 \
  --num-obstacle-boxes 6 \
  --bin-selection cycle \
  --planner-speed-mps 0.4 \
  --planner-waypoint-max-seg-m 0.01 \
  --max-steps-per-episode 10000 \
  --domain-rand \
  --domain-rand-seed 0 \
  --logs-root logs/data_collection

# Same with YCB / USD props:
python -m data_collection.collect_data \
  --profile vla_v2 \
  --env reach_to_grasp_VLA \
  --control planner \
  --device cuda:0 \
  --enable_cameras \
  --spawn-mode usd \
  --log-rate-hz 5 \
  --num-episodes 10 \
  --domain-rand \
  --logs-root logs/data_collection
```

Each episode writes the same `instruction.json`, `images/`, `ticks.jsonl`,
and `events.jsonl` artifacts as `vla_v1`, plus a `drop_result` event with the
final box pose vs. the chosen bin's footprint. See the top-level `README.md`
for the full list of `vla_v2` knobs (target / bin layout, transit clearance,
etc.).

### 5.2 Inspect what was collected

```bash
python -m vla_lab.inspect_data --data-roots logs/data_collection
python -m vla_lab.inspect_data --data-roots logs/data_collection --print-instructions
```

This walks every `session_*/episode_*` folder and prints episode count,
tick count, image coverage, and the unique instruction set. If you see
"WARNING: no images found" you forgot `--enable_cameras` during
collection.

### 5.3 Dryrun (VRAM + latency sanity)

This is the spec's Day-3 sanity script. It just builds the model and
runs forward passes; **no Isaac Lab needed**.

```bash
# Untrained model, K=4 candidates, iterate 100x
./vla_lab/scripts/dryrun.sh --iters 100

# Trained checkpoint, deployed K
./vla_lab/scripts/dryrun.sh --iters 100 --ckpt vla_lab/checkpoints/tiny_v0/last.pt

# CPU sanity run (works anywhere)
DEVICE=cpu ./vla_lab/scripts/dryrun.sh --iters 8 --warmup 2 --k 1
```

### 5.4 Train

**Quick reference (always from repo root `kinova-isaac/`):**

| Goal | Command |
| --- | --- |
| **TinyVLA** (default) | `./vla_lab/scripts/train.sh` |
| TinyVLA, custom config | `CONFIG=/path/to/config.yaml ./vla_lab/scripts/train.sh` |
| **Export** Kinova logs → LeRobot dataset | `./vla_lab/scripts/export_lerobot_dataset.sh --session-roots logs/data_collection/session_YYYYMMDD_HHMMSS --out-dir vla_lab/datasets/lerobot_kinova_v0 --overwrite` |
| **SmolVLA** (recommended) | `STEPS=20000 ./vla_lab/scripts/train_smolvla.sh` (defaults: dataset `vla_lab/datasets/lerobot_kinova_v0`, checkpoint dir under `vla_lab/checkpoints/`; override with `DATASET_DIR`, `OUT_DIR`, etc.) |
| **SmolVLA** (YAML) | `python -m vla_lab.train_smolvla --config vla_lab/configs/train_smolvla.example.yaml` |
| SmolVLA **resume** | `RESUME=true ./vla_lab/scripts/train_smolvla.sh` (same `OUT_DIR` / checkpoint dir as first run) |
| After SmolVLA train | `RUN_DIR=vla_lab/checkpoints/smolvla_kinova_ft ./vla_lab/scripts/after_smolvla_train.sh` |

TinyVLA needs PyTorch + YAML (+ `matplotlib` for plots). SmolVLA needs LeRobot: use **`pip install -r vla_lab/requirements-smolvla.txt`** in a **dedicated** env, or **`./vla_lab/scripts/pip_install_smolvla_isaac.sh`** in `riften` (NumPy 1.x vs `rerun-sdk` metadata — see [`smoll-vla-setup.md`](./smoll-vla-setup.md)). You also need a **local LeRobot dataset** (export step above). See **§5.4.1**.

---

From the **repo root** (`kinova-isaac/`), with a Python env that has PyTorch
installed (no Isaac Lab required for training).

**Default data:** `vla_lab/configs/train_tiny.yaml` uses a **single** session:

`logs/data_collection/session_20260506_232450`

(Change `data.data_roots` in that YAML if you want a different session or multiple roots.)

**Recommended — start training:**

```bash
cd /path/to/kinova-isaac
pip install torch torchvision numpy pillow pyyaml matplotlib   # matplotlib: figures after training

./vla_lab/scripts/train.sh
```

**GPU utilization:** TinyVLA is only ~2M parameters, so the GPU often waits on
**CPU data loading** (decoding thousands of PNGs). The default config enables
**larger batches** (`batch_size`), **more DataLoader workers**, **prefetch**,
**mixed precision (AMP)**, **cuDNN benchmark**, and **torchvision-based image
decode** (`data.fast_image_io`). Tune `train.batch_size` and `train.num_workers`
in `train_tiny.yaml`; if you run out of VRAM, lower `batch_size`. For very high
core-count CPUs you can try `num_workers: 12` or `16`.

Checkpoints, `metrics.jsonl`, and plots go under `vla_lab/checkpoints/tiny_v0/`
(or whatever `train.out_dir` is set to in the YAML). Override the output dir:

```bash
./vla_lab/scripts/train.sh --out-dir vla_lab/checkpoints/my_run
```

**Explicit module invocation (same as the script):**

```bash
python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml
```

**Use different data without editing the YAML:**

```bash
python -m vla_lab.train \
    --config vla_lab/configs/train_tiny.yaml \
    --data-roots logs/data_collection \
    --out-dir vla_lab/checkpoints/all_sessions
```

**Resume after interruption** (loads optimizer + scheduler + dataloader position):

```bash
python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml --auto-resume
# or: python -m vla_lab.train --config ... --resume vla_lab/checkpoints/tiny_v0/last.pt
```

**Skip automatic plotting** (e.g. no matplotlib):

```bash
python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml --no-plot-at-end
```

Output checkpoints live under `vla_lab/checkpoints/<run_name>/`:

- `last.pt`  — latest epoch boundary (full state for `--resume`)
- `best.pt`  — best validation loss (only if val split is non-empty)
- `step_*.pt` — periodic snapshots (see `train.save_every_steps` in YAML)
- `config.json` — frozen hyperparameter dump
- `metrics.jsonl` — train/val scalars for plotting

After training, figures are written to `<out_dir>/figures/` (learning curves,
LR schedule, train/val per epoch). Install `matplotlib` for this step.

Regenerate plots (optionally pass eval JSON from §5.5 for success-rate bars):

```bash
RUN_DIR=vla_lab/checkpoints/tiny_v0 ./vla_lab/scripts/plot.sh --format pdf \
  --eval-json vla_lab/eval_results/tiny_v0/results_1234567890.json
```

To enable the optional DINOv2 feature-alignment loss flip
`train.feature_alignment.enabled: true` in the YAML (or pass a YAML that
already has it on) — `transformers` must be installed and the teacher
weights cached.

#### 5.4.1 SmolVLA — LeRobot fine-tune (in-repo wrappers)

SmolVLA training is **LeRobot’s** `lerobot-train`. This repo adds:

- **`vla_lab/smolvla_bridge/convert_kinova_to_lerobot.py`** — `ticks.jsonl` + images → on-disk LeRobot dataset.
- **`vla_lab/scripts/export_lerobot_dataset.sh`** — thin wrapper around that converter.
- **`vla_lab/scripts/train_smolvla.sh`** — calls `lerobot-train` with sensible defaults and env overrides.
- **`vla_lab/train_smolvla.py`** + **`vla_lab/configs/train_smolvla.example.yaml`** — same thing, YAML-driven; supports `--dry-run`.

Official SmolVLA overview: [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla).

**1. Install (once per Python env)**

**Dedicated env (recommended for SmolVLA training):** allows NumPy 2.x if `rerun-sdk` needs it.

```bash
pip install -r vla_lab/requirements-smolvla.txt
# Linux: if evdev/pynput fails to build, see §8.5 in smoll-vla-setup.md
```

**Same env as Isaac Lab** (must keep NumPy 1.x; plain `pip -r` may hang on `resolution-too-deep` or force NumPy 2):

```bash
./vla_lab/scripts/pip_install_smolvla_isaac.sh
```

Details: [`smoll-vla-setup.md`](./smoll-vla-setup.md) (NumPy vs `rerun-sdk`, `evdev`, resolver limits).

**2. Export Kinova collection to a LeRobot dataset**

```bash
./vla_lab/scripts/export_lerobot_dataset.sh \
  --session-roots logs/data_collection/session_20260506_232450 \
  --out-dir vla_lab/datasets/lerobot_kinova_v0 \
  --overwrite
```

Point `--session-roots` at one or more `session_*` folders under `logs/data_collection/`. The default paths in `train_smolvla.example.yaml` expect **`vla_lab/datasets/lerobot_kinova_v0`** and `dataset.repo_id: kinova_isaac_vla` (a local label; must match what the converter wrote).

**3. Start training**

From repo root (the script `cd`s there automatically). **Recommended** SmolVLA fine-tune (~20k steps; override `STEPS` as needed):

```bash
STEPS=20000 ./vla_lab/scripts/train_smolvla.sh
```

Defaults use **`vla_lab/datasets/lerobot_kinova_v0`**, **`lerobot/smolvla_base`**, and a timestamped checkpoint dir under **`vla_lab/checkpoints/`**. Omitting `STEPS` falls back to the script default (**9824** steps). Override any variable via the environment, for example:

```bash
DATASET_DIR="$(pwd)/vla_lab/datasets/lerobot_kinova_v0" \
POLICY_PATH=lerobot/smolvla_base \
DATASET_REPO_ID=kinova_isaac_vla \
STEPS=20000 BATCH_SIZE=32 DEVICE=cuda \
OUT_DIR="$(pwd)/vla_lab/checkpoints/smolvla_kinova_ft" \
./vla_lab/scripts/train_smolvla.sh
```

YAML (paths in `train_smolvla.example.yaml`; edit `dataset.root`, `training.output_dir`, etc.):

```bash
python -m vla_lab.train_smolvla --config vla_lab/configs/train_smolvla.example.yaml
```

Extra `lerobot-train` flags go at the end of either invocation, e.g. `--wandb.enable=true`. See `lerobot-train --help` for your installed version.

**Training metrics and paper artifacts:** by default, `train_smolvla.sh` / `train_smolvla.py` wrap `lerobot-train` to write **`vla_lab/results/<UTC-date>/<run-name>/`**: `train_metrics.csv`, `train_metrics.jsonl`, `figures/*.pdf` (and PNG where emitted), `train_console.log`, `run_meta.json`, etc. Set **`VLA_LAB_CAPTURE_RESULTS=0`** to skip this wrapper. Full layout and field glossary: [`smoll-vla-setup.md`](./smoll-vla-setup.md) §8.6.

**4. Resume**

Use the **same** `--output_dir` as the first run. With the shell wrapper:

```bash
RESUME=true OUT_DIR=/path/to/same/checkpoint/dir ./vla_lab/scripts/train_smolvla.sh
```

Or set `training.resume: true` in the YAML and keep `training.output_dir` fixed.

**5. After training**

Checkpoints and LeRobot metadata live under `OUT_DIR` / `training.output_dir`. Optional:

```bash
RUN_DIR=vla_lab/checkpoints/smolvla_kinova_ft ./vla_lab/scripts/after_smolvla_train.sh
```

**6. Isaac eval with a SmolVLA checkpoint**

Use `vla_lab/eval_isaaclab.py` with **`policy_backend: smolvla`** (or CLI equivalent) and pass the LeRobot output directory / checkpoint path your run produced — see `vla_lab/configs/eval_isaac.yaml` and `--help` on `eval_isaaclab.py`.

### 5.5 Evaluate in Isaac Lab

Evaluation reuses the existing `reach_to_grasp_VLA` scene + Cartesian
velocity controller. The trained policy is plugged in via a custom
`PolicyInputProvider` that converts each predicted action chunk into per-
physics-step velocity commands.

```bash
# Default config: 10 episodes, K=1 (no TTC sampling)
./vla_lab/scripts/eval.sh --num-episodes 10

# Headless eval with the K=4 sample-and-verify pipeline
./vla_lab/scripts/eval.sh \
    --num-episodes 20 \
    --headless \
    --ckpt vla_lab/checkpoints/tiny_v0/last.pt

# Without the wrapper:
./IsaacLab/isaaclab.sh -p vla_lab/eval_isaaclab.py \
    --config vla_lab/configs/eval_isaac.yaml \
    --enable_cameras --device cuda:0 --num-episodes 10
```

Each run writes a `results_<unix_ts>.json` under `eval.out_dir` with:

```jsonc
{
  "num_episodes": 10,
  "num_success": 7,
  "success_rate": 0.7,
  "ckpt": "vla_lab/checkpoints/tiny_v0/last.pt",
  "config": {...},
  "results": [
    {"episode_idx": 0, "target_leaf": "Obj_01", "instruction": "...",
     "z0_target": 0.81, "z_after": 0.93, "success": true,
     "steps": 2200, "elapsed_s": 31.5},
    ...
  ]
}
```

Toggle K-sample TTC by editing `ttc.k_action_samples` in
`vla_lab/configs/eval_isaac.yaml` (1 = no TTC, 4 = the spec's default).

## 6. Action / observation contract

This is what the dataset, model, and eval all agree on. Keep this stable
when extending — it lets us swap models without touching anything else.

| Symbol | Shape | Description |
| --- | --- | --- |
| Image | `(3, H, W)` float in `[0, 1]` | Top-down camera, resized to `H = W = 224`. |
| State | `(state_dim,)` float | `[ee_x_b, ee_y_b, ee_z_b, gripper_open_flag]` (state_dim = 4). |
| Language tokens | `(max_lang_len,)` long, `(max_lang_len,)` long | `lang_ids`, `lang_mask` from `TinyTokenizer`. |
| Action | `(T, 7)` float | `T = chunk_len = 8`. Per timestep: `[dx, dy, dz, drx, dry, drz, gripper]` in base frame. `gripper ∈ {-1, 0, +1}`. |

The action is exactly what `vla_v1.py` writes as `policy.action_from_prev`
in `ticks.jsonl`, so the supervised target is always available without
custom labelling.

## 8. Known gotchas

- **`--enable_cameras` is required at collection time.** Without it
  `images/` is empty and the dataset will refuse to construct.
- **The dataset stringifies all floats** to 4 decimal places. The reader
  parses this back; any extra post-processing pipelines should do the
  same (`vla_lab.dataset._to_float_maybe`).
- **The first tick of every episode has `policy.action_from_prev = None`**
  because there is no previous tick. The dataset training-frame index
  skips frames whose chunk would only contain that tick.
- **`vla_lab/eval_isaaclab.py` must be run with the IsaacLab launcher**
  (`./IsaacLab/isaaclab.sh -p vla_lab/eval_isaaclab.py ...`). Plain
  `python` will fail because `isaaclab.app` isn't on the path.
- **Action stats live in the checkpoint.** If you run eval with a
  checkpoint that was trained with `normalize_actions: true`, the eval
  loader will automatically denormalize predicted chunks before sending
  them to the controller — no manual handling needed.

## 9. Where to look next

- The big picture and the full method are in
  [`vla_ttc_engineering_spec.md`](./vla_ttc_engineering_spec.md).
- For the on-disk format of the data we consume, see
  `data_collection/core/logger.py` (the canonical writer) and the
  `vla_v1` profile in `data_collection/profiles/vla_v1.py`.
- For the scene + camera definitions used at collection AND eval time,
  see `environments/reach_to_grasp_VLA/config.py`.

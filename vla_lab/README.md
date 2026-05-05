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

- **SmolVLA wrapper** (spec §4.4). To be added as a second concrete model
  inside `vla_lab/models.py`; trainer/TTC/eval are model-agnostic.
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
│   └── eval_isaac.yaml             <- IsaacLab eval config
├── dataset.py                      <- KinovaSessionDataset, TinyTokenizer
├── models.py                       <- TinyVLA + ModelOutput
├── losses.py                       <- masked_action_loss, FeatureAlignmentLoss
├── ttc.py                          <- TTCPipeline (K-sample + consensus)
├── train.py                        <- training entrypoint
├── eval_isaaclab.py                <- IsaacLab evaluation entrypoint
├── dryrun.py                       <- VRAM / latency dry-fit
├── inspect_data.py                 <- session sanity tool
└── scripts/
    ├── collect.sh                  <- wrapper around `data_collection.collect_data --profile vla_v1`
    ├── collect_v2.sh               <- wrapper around `data_collection.collect_data --profile vla_v2` (pick-and-place)
    ├── train.sh                    <- wrapper around `vla_lab.train`
    ├── eval.sh                     <- wrapper around `vla_lab.eval_isaaclab`
    └── dryrun.sh                   <- wrapper around `vla_lab.dryrun`
```

Nothing outside `vla_lab/` was modified to add this package.

## 4. Prerequisites

The training and inspection tools have **light** dependencies:

```bash
pip install torch torchvision numpy pillow pyyaml
```

The Isaac Lab evaluation entrypoint additionally needs the same Isaac
Sim / Isaac Lab environment that the rest of the repo already uses;
follow the top-level `README.md` to set that up.

Optional dependencies (only if you want to enable specific features):

| Feature | Install |
| --- | --- |
| DINOv2 feature alignment | `pip install transformers` |
| SmolVLA upgrade (later) | `pip install lerobot transformers peft bitsandbytes` |

## 5. End-to-end commands

All commands assume you run them from the **repo root**
(`kinova-isaac/`), not from inside `vla_lab/`.

### 5.1 Collect data

This calls the existing `data_collection.collect_data` with the recommended
`vla_v1` profile + planner + cameras + domain randomization. Each session is
written under `logs/data_collection/session_<TIMESTAMP>/` with one
`episode_NNNN/` folder per attempt.

```bash
# Recommended: uniform colored boxes + scripted planner + cameras (10 episodes)
NUM_EPISODES=10 ./vla_lab/scripts/collect.sh

# YCB objects instead of boxes
SPAWN_MODE=usd NUM_EPISODES=10 ./vla_lab/scripts/collect.sh

# Headless on a chosen GPU
DEVICE=cuda:0 NUM_EPISODES=20 ./vla_lab/scripts/collect.sh --headless
```

Underneath, the wrapper expands to:

```bash
python -m data_collection.collect_data \
  --profile vla_v1 --env reach_to_grasp_VLA --control planner \
  --planner scripted --device cuda:0 --enable_cameras \
  --log-rate-hz 5 --num-episodes ${NUM_EPISODES} \
  --spawn-mode box --domain-rand --domain-rand-seed 0 \
  --logs-root logs/data_collection \
  ...
```

You can also call `data_collection.collect_data` directly with any flags you
want — see the top-level `README.md`. Just make sure `--enable_cameras` is
set; `vla_lab` requires the per-tick PNGs to train.

#### 5.1.1 Pick-and-place (`vla_v2`)

`vla_v2` is a richer scene with **clutter + 3 bins** and a fully scripted
**pick-and-place** routine: grab the box closest to the robot, transit over
the clutter, drop into one of three colored bins. Motion is purely scripted
(no cuRobo / MotionGen). Logging format and on-disk layout are identical to
`vla_v1`, so the same `vla_lab.dataset` reader works without changes.

```bash
# Default: 6 clutter boxes + 1 close target + 3 bins, 10 episodes
NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh

# More clutter, headless
NUM_OBSTACLE_BOXES=8 NUM_EPISODES=20 ./vla_lab/scripts/collect_v2.sh --headless

# Random bin selection per episode
BIN_SELECTION=random NUM_EPISODES=20 ./vla_lab/scripts/collect_v2.sh
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

The default config trains TinyVLA on every session under
`logs/data_collection/`.

```bash
./vla_lab/scripts/train.sh

# or, explicit:
python -m vla_lab.train --config vla_lab/configs/train_tiny.yaml

# Override common knobs from the CLI:
python -m vla_lab.train \
    --config vla_lab/configs/train_tiny.yaml \
    --data-roots logs/data_collection logs/data_collection_extra \
    --epochs 50 --batch-size 64 \
    --out-dir vla_lab/checkpoints/tiny_v1
```

Output checkpoints live under `vla_lab/checkpoints/<run_name>/`:

- `last.pt`  — most recent epoch
- `best.pt`  — best validation loss (only if val split is non-empty)
- `config.json` — frozen hyperparameter dump

To enable the optional DINOv2 feature-alignment loss flip
`train.feature_alignment.enabled: true` in the YAML (or pass a YAML that
already has it on) — `transformers` must be installed and the teacher
weights cached.

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

## 7. Recommended day-by-day plan (matches the spec's schedule)

This is the practical first-week plan. It deliberately follows the
spec's hard-checkpoint structure (§5).

1. **Today (Day 0):**
   - `pip install` deps; run the Day-3 dry-fit:
     `./vla_lab/scripts/dryrun.sh --device cpu --k 4 --iters 16`.
   - Confirm the IsaacLab base demos still run (
     `./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --headless`).
2. **Day 1:** Collect a small dataset (`NUM_EPISODES=20`).
   Inspect with `python -m vla_lab.inspect_data --print-instructions`.
3. **Day 2:** Train TinyVLA for ~30 epochs on this dataset:
   `./vla_lab/scripts/train.sh`.  Watch `train_loss` come down.
4. **Day 2-3:** Run K=1 IsaacLab eval (`--num-episodes 10`). This is the
   first "does the policy do anything reasonable" check.
5. **Day 4:** Bump to ~100 episodes of collection, retrain, K=4 eval.
   Compare success rate against the K=1 baseline (this is your **first
   TTC ablation**).
6. **Day 5+:**
   - Add the SmolVLA wrapper (drop-in to `models.py`).
   - Wire in a learned verifier (`vla_lab/ttc.py` already has a hook).
   - Plug in the OOD trigger (RND) and feature alignment.

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

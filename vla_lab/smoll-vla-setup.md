# SmolVLA in `vla_lab`: integration plan

**Status (code in this repo):** The bridge is implemented under `vla_lab/smolvla_bridge/`
with `convert_kinova_to_lerobot`, `policy_wrapper` (Isaac eval), `scripts/export_lerobot_dataset.sh`,
`scripts/train_smolvla.sh`, `requirements-smolvla.txt`, and eval YAML/CLI flags (`policy_backend`,
`lerobot_dataset_root`). You still install **LeRobot** separately for export/train.

This document lists **exact changes** to keep inside **`vla_lab/`** so you can:

1. **Load** a pretrained SmolVLA (Hub / LeRobot checkpoint).
2. **Finetune** it on **`vla_v1` collection** (`session_*/episode_*`, `ticks.jsonl`, `images/`).
3. **Compare** that run fairly against **TinyVLA** training (`vla_lab/train.py`) in Isaac eval and in plots.

**Scope note:** SmolVLA itself ships in **[LeRobot](https://github.com/huggingface/lerobot)**. Staying “inside `vla_lab`” means *our* code, configs, scripts, exported datasets, and docs live under `vla_lab/`. You still `pip install lerobot[...]` (or a pinned extra) in the Python env you use for export + SmolVLA fine-tune—that dependency is expected.

---

## 1. What exists today (baseline)

| Piece | Role |
|--------|------|
| `vla_lab/dataset.py` | Reads Kinova logs → `TickRecord` / `KinovaSessionDataset` (image path, EE pose, `action_from_prev` 7-vector). |
| `vla_lab/train.py` | Trains **TinyVLA**; writes `metrics.jsonl`, `last.pt`, `best.pt`, `step_*.pt`, `config.json`, optional `figures/`. |
| `vla_lab/plot_metrics.py` | Figures from `metrics.jsonl` + optional eval `results_*.json`. |
| `vla_lab/eval_isaaclab.py` | Loads **TinyVLA** `.pt` only; `PolicyInputProvider` consumes **(T, 7)** delta chunks. |
| `vla_lab/models.py` | `TinyVLA` forward contract documented for future SmolVLA swap. |

**Gap:** LeRobot training expects a **LeRobot dataset** (parquet + metadata), not `ticks.jsonl`. **Gap:** Eval has no SmolVLA path yet.

---

## 2. Target layout (all new artifacts under `vla_lab/`)

Suggest adding (names can be adjusted, but keep the responsibilities):

```text
vla_lab/
  configs/
    train_smolvla.example.yaml    # Optional: document env vars / paths for shell wrappers
  datasets/                       # NEW: exported LeRobot-format data (gitignored)
    lerobot_kinova_v0/
      ...
  checkpoints/
    smolvla_finetune_*/           # LeRobot output_dir copies or symlinks
  figures/
    compare_tiny_vs_smolvla/      # Optional overlays from plot tooling
  scripts/
    export_lerobot_dataset.sh     # Calls converter with repo-root paths
    train_smolvla.sh              # Wraps lerobot-train; pins policy + dataset paths under vla_lab/
  smolvla_bridge/                 # NEW package (or flat modules)
    __init__.py
    convert_kinova_to_lerobot.py  # ticks.jsonl → LeRobot dataset
    action_obs_contract.py        # Document & implement key names, shapes, normalization
    lerobot_metrics.py            # Optional: parse LeRobot logs → vla_lab-style jsonl
    policy_wrapper.py             # load SmolVLA → produce (T,7) compatible with PolicyInputProvider
  requirements-smolvla.txt        # Pin lerobot + deps for the fine-tune env
  smoll-vla-setup.md              # This file
```

Add **`vla_lab/datasets/`** to `.gitignore` (large exports).

---

## 3. Phase A — Dataset converter (blocking step)

**New file:** `vla_lab/smolvla_bridge/convert_kinova_to_lerobot.py` (CLI: `--session-roots`, `--out-dir vla_lab/datasets/lerobot_kinova_v0`, `--fps`, splits).

**Reuse:** `vla_lab.dataset.discover_episodes`, `_parse_ticks` / `EpisodeRecord` logic — do **not** duplicate parsing; import from `vla_lab.dataset` or factor shared helpers if import cycles appear.

**Mapping (conceptual — verify against your installed LeRobot version + `lerobot/smolvla_base` feature keys):**

| Kinova (`ticks.jsonl`) | LeRobot frame field (typical) |
|------------------------|---------------------------------|
| `instruction.json` → `language_command` | Text / task field per episode |
| `images/image_*.png` | Camera observation column(s); match SmolVLA’s expected camera name(s) |
| `robot.ee_pose_b` | Low-dimensional state (may need pose **relative** to episode start or dataset norm — **must match** what you train with) |
| `policy.action_from_prev` (7D: Δxyz, Δrotvec×3, gripper) | Action vector; may need stacking for action horizons |

**Hard requirements:**

1. **Camera naming and resolution** must match the **pretrained** SmolVLA config (often 224-ish crop/resize; same key as base policy).
2. **Action semantics** must match: if base model was trained on **delta** vs **absolute** joints/EE, your labels must match after conversion (may require a small **adapter layer** or relabeling).
3. **Train/val split** at **episode** boundary (same as `vla_lab.dataset.split_episodes` philosophy).

**New script:** `vla_lab/scripts/export_lerobot_dataset.sh` — `cd` repo root, call `python -m vla_lab.smolvla_bridge.convert_kinova_to_lerobot ...` with default `logs/data_collection/...` inputs and `vla_lab/datasets/...` output.

---

## 4. Phase B — Fine-tune SmolVLA (training “in top shape”)

**New file:** `vla_lab/scripts/train_smolvla.sh`

- Sets `REPO_ROOT`, `DATASET_DIR="$REPO_ROOT/vla_lab/datasets/lerobot_kinova_v0"`, `OUT_DIR="$REPO_ROOT/vla_lab/checkpoints/smolvla_<run_name>"`.
- Invokes **`lerobot-train`** with:
  - `--policy.path=lerobot/smolvla_base` (or local path)
  - dataset pointing at **local** `DATASET_DIR` (LeRobot local dataset API — check current CLI for **local dir** vs `repo_id`)
  - `--output_dir` under `vla_lab/checkpoints/`
- Writes a **`run_manifest.json`** next to `output_dir` (your file) with: git hash, converter command, list of session roots, LeRobot version, full CLI string.

**Optional — metrics parity with TinyVLA:**

LeRobot’s logs may not match `metrics.jsonl` schema. Pick one:

- **(Preferred)** **`vla_lab/smolvla_bridge/lerobot_metrics.py`:** post-process LeRobot’s training logs (or parse tensorboard exports) and append **`type: train_log` / `epoch_end`** lines compatible with `plot_metrics.py`, into `vla_lab/checkpoints/.../metrics.jsonl`.
- **Or** extend **`plot_metrics.py`** with a second reader branch for LeRobot-native logs (more work, duplicated styling).

**`requirements-smolvla.txt`:** pin `lerobot[smolvla]` (or documented install) + versions known to work with your GPU stack.

**End-of-run outputs checklist (match TinyVLA quality bar):**

| Artifact | TinyVLA (`train.py`) | SmolVLA (you add) |
|----------|----------------------|-------------------|
| Frozen hyperparameters | `config.json` | Copy policy + train config from LeRobot; plus `run_manifest.json` |
| Time series | `metrics.jsonl` | Same schema via adapter **or** documented alternate |
| Curves | `figures/*.pdf` (via `plot_metrics`) | Run `plot_metrics` on adapted jsonl **or** LeRobot-native plots + copy into `figures/` |
| Checkpoints | `last.pt`, `best.pt` | LeRobot checkpoint dir; document `Policy.from_pretrained(local_dir)` in README snippet |

---

## 5. Phase C — Isaac Lab eval: load SmolVLA and feed `PolicyInputProvider`

**Edit:** `vla_lab/eval_isaaclab.py`

1. **CLI:** e.g. `--policy-backend {tiny,smolvla}`, `--ckpt` meaning:
   - `tiny`: existing `.pt` from `vla_lab.train`
   - `smolvla`: directory or Hub id for LeRobot `Policy`
2. **New module:** `vla_lab/smolvla_bridge/policy_wrapper.py`
   - `load_smolvla_policy(path_or_id, device)`
   - `predict_chunk(observation_dict) -> torch.Tensor` shape **`(T, 7)`** in the **same physical units** TinyVLA eval expects (may need denormalization using stats saved at dataset export time).
3. **Observation builder:** mirror what the converter wrote (same resize, same state vector). Share code between converter metadata (`stats.json`, `feature_keys.json`) and eval.
4. **Refactor (minimal):** extract TinyVLA forward into a small function or class implementing a common interface, e.g. `ChunkPolicy.predict(images, state_vec, lang_tokens) -> (T,7)`, and have `PolicyInputProvider` unchanged.

`PolicyInputProvider` already streams **7D** commands; the main work is **making SmolVLA’s outputs align** with that contract.

---

## 6. Phase D — Fair comparison: TinyVLA vs finetuned SmolVLA

**Eval protocol (keep identical):**

- Same `vla_lab/configs/eval_isaac.yaml`, `--num-episodes`, seeds, object loader, policy rate, TTC settings.
- Two runs:
  - `.../eval.sh --policy-backend tiny --ckpt vla_lab/checkpoints/tiny_v0/last.pt`
  - `.../eval.sh --policy-backend smolvla --ckpt vla_lab/checkpoints/smolvla_finetune_xxx/...`

**Metrics JSON:** `eval_isaaclab.py` already writes `results_*.json` with success rate and per-episode stats — ensure **`ckpt` and `policy_backend`** are recorded in that JSON for plotting legends.

**Training curves:**

- TinyVLA: `RUN_DIR=vla_lab/checkpoints/tiny_v0 ./vla_lab/scripts/plot.sh`
- SmolVLA: same once `metrics.jsonl` is adapted (§4).

**Optional new figure:** extend **`plot_metrics.py`** (or add `plot_compare_runs.py`) to overlay **two** `metrics.jsonl` files (different colors, legend from `run_manifest.json`). Keep it under `vla_lab/`.

**Optional table:** small script that prints side-by-side **final train loss**, **best val**, **Isaac success rate ± Wilson CI** (from eval JSONs) — can live in `vla_lab/scripts/summarize_comparison.sh` calling a tiny Python `-c` or module.

---

## 7. Phase E — Polish existing TinyVLA training UX (keep scripts “top shape”)

These are **already largely present**; treat as a maintenance checklist:

| Item | Location |
|------|-----------|
| Resume / periodic ckpt | `train.py` — `--resume`, `--auto-resume`, `step_*.pt` |
| Metrics | `metrics.jsonl` |
| Plots at end | `train.py` → `plot_metrics.plot_run_dir`; `--no-plot-at-end` escape hatch |
| Loader perf | `train_tiny.yaml`: `batch_size`, `num_workers`, AMP, `fast_image_io` |

**Suggested small improvements (still inside `vla_lab/`):**

1. **`train.py` exit summary:** print absolute paths to `last.pt`, `figures/`, and one-line “plot command” for copy-paste.
2. **`scripts/train.sh`:** pass through args; document `CONFIG=...` (already).
3. **`plot_metrics.py`:** optional `--title-prefix` / run name from `config.json` for multi-run clarity.
4. **Single `TRAINING.md` pointer** (optional): one paragraph in `vla_lab/README.md` linking TinyVLA + this SmolVLA doc (only if you want less duplication).

---

## 8. Validation order

1. Export dataset; spot-check N frames in LeRobot’s dataset viewer / load one batch in Python.
2. Short SmolVLA finetune (few hundred steps); loss decreases; checkpoint saves.
3. Load checkpoint in **`policy_wrapper`** without Isaac (unit smoke: random image + state → tensor shape).
4. Isaac eval **tiny** vs **smolvla** with same episode count; compare `results_*.json`.
5. Regenerate plots; archive `run_manifest.json` + CLI for reproducibility.

---

## 8.5 Linux: `evdev` / `pynput` build failure (`BUS_SDW` / `BTN_GRIPL` undeclared)

LeRobot depends on `pynput`, which builds **`evdev`** from C sources. In **conda** envs, pip sometimes uses `x86_64-conda-linux-gnu-cc` and a sysroot whose Linux input headers do not match what `evdev`’s code generator expects, so the wheel build aborts.

**Fix (pick one, then retry your LeRobot install — `pip install -r vla_lab/requirements-smolvla.txt` or `./vla_lab/scripts/pip_install_smolvla_isaac.sh`):**

1. **conda-forge binary (often easiest)**  
   `conda install -y -c conda-forge evdev`

2. **Force system GCC** (so headers match `/usr/include/linux/`)  
   `CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install -r vla_lab/requirements-smolvla.txt`

3. **Debian/Ubuntu** — install kernel UAPI headers, then reinstall  
   `sudo apt update && sudo apt install -y build-essential linux-libc-dev`

Also avoid mixing **Isaac Sim’s** `pip_prebundle` `PYTHONPATH` into this install; use a clean env for LeRobot when possible (see Risk register).

### Pip “dependency conflicts” after a successful install

If you install LeRobot into the **same** conda env as Isaac Lab (`riften`), pip may print scary `ERROR: pip's dependency resolver...` messages (e.g. `numpy 2.x` vs `isaaclab requires numpy<2`). Check the **last line**: if it says `Successfully installed ...` and `numpy` is **1.26.x**, you are often fine for Isaac. **Do not** use `requirements-smolvla.txt` alone in `riften` if you need NumPy 1.x — it may pull NumPy 2 for `rerun-sdk`. Use **`./vla_lab/scripts/pip_install_smolvla_isaac.sh`** instead, or a **dedicated** `smolvla` env with `pip install -r vla_lab/requirements-smolvla.txt`.

**Best practice:** `conda create -n smolvla python=3.11`, install PyTorch + `pip install -r vla_lab/requirements-smolvla.txt` there; keep Isaac work in `riften`.

### Pip `resolution-too-deep` / multi-minute backtracking

This usually happens with **`pip install -r vla_lab/requirements-smolvla.txt` inside the same env as Isaac Lab**: the resolver walks many versions of `rerun-sdk`, `opencv-python-headless`, `imageio`, … and may abort.

Underlying issue: **`rerun-sdk` wheels on PyPI (≥0.24) declare `numpy>=2`**, while **Isaac / `isaaclab` expect NumPy 1.x**. A single `pip` solve cannot honestly satisfy both; mixing them forces huge backtracking or failure.

**Fix (pick one):**

1. **Dedicated SmolVLA env (simplest for training):**  
   `conda create -n smolvla python=3.11 && conda activate smolvla`  
   Install PyTorch for your CUDA stack, then:  
   `pip install -r vla_lab/requirements-smolvla.txt`  
   (This path may upgrade NumPy to 2.x for `rerun-sdk` — that is OK there.)

2. **Stay in `riften` / Isaac env:** do **not** rely on a plain `pip -r` for LeRobot. Run:  
   `./vla_lab/scripts/pip_install_smolvla_isaac.sh`  
   It installs Lerobot’s dependencies explicitly, then `rerun-sdk` and `lerobot` with `--no-deps` so PyPI’s `numpy>=2` metadata on `rerun-sdk` does not force a NumPy upgrade. (You may still see `pip check` warnings; training usually works.)

3. `pip install -U "pip>=24"` can help with unrelated resolver bugs, but it does not remove the NumPy 1.x vs `rerun-sdk` metadata tension — use (1) or (2).

---

## 8.6 Training metrics, CSV, and figures (`vla_lab/results/`)

`./vla_lab/scripts/train_smolvla.sh` and `python -m vla_lab.train_smolvla` **wrap** `lerobot-train` with `python -m vla_lab.lerobot_train_capture` by default. That streams stdout, parses LeRobot’s periodic `MetricsTracker` lines, and after training writes:

| Artifact | Purpose |
|---------|---------|
| `vla_lab/results/<UTC-date>/<run_name>/train_metrics.csv` | One row per log step: `step`, `smpl`, `ep`, `epch`, `loss`, `grdn`, `lr`, `updt_s`, `data_s` |
| `train_metrics.jsonl` | Same data as JSON lines |
| `train_console.log` | Full captured training log |
| `eval_events.jsonl` / `eval_console_snippets.txt` | Lines related to mid-train **sim** eval (only if you configure `eval_freq` + env in LeRobot) |
| `eval_success.csv` | Parsed success % (best-effort) when eval logs expose it |
| `run_meta.json` | Command, git revision, paths, **metric glossary**, and notes on BC vs “accuracy” |
| `figures/*.pdf` and `*.png` | Loss, LR, gradient norm, **dataset epoch coverage** (`epch`), timing plots; optional eval success curve |

**BC / SmolVLA note:** there is no categorical **training accuracy** (no discrete correct/wrong labels). Use **loss**, **gradient norm**, **epoch coverage** (`epch`), and **sim eval success** (enable LeRobot’s eval + env) for paper-style curves.

**Disable capture** (plain `lerobot-train` only):

```bash
VLA_LAB_CAPTURE_RESULTS=0 ./vla_lab/scripts/train_smolvla.sh
```

**matplotlib** is required for figures; install if needed: `pip install matplotlib`.

---

## 9. Risk register (short)

| Risk | Mitigation |
|------|------------|
| Action / state normalization mismatch | Save `mean/std` or bounds in export dir; use in wrapper + eval. |
| Image key / resolution mismatch | Lock to `lerobot/smolvla_base` preprocess; test one forward pass in LeRobot before long train. |
| LeRobot API drift | Pin version in `requirements-smolvla.txt`; record version in `run_manifest.json`. |
| Env conflict with Isaac Lab | Use **separate conda/venv** for LeRobot fine-tune vs Isaac if needed; artifacts still under `vla_lab/`. |

---

## 10. Summary checklist

- [x] `smolvla_bridge/convert_kinova_to_lerobot.py` + `scripts/export_lerobot_dataset.sh`
- [x] `scripts/train_smolvla.sh` + `requirements-smolvla.txt` + `run_manifest.json` (written by train script)
- [x] `lerobot_metrics.py` (helpers to append TinyVLA-style `metrics.jsonl` rows)
- [x] `policy_wrapper.py` + `eval_isaaclab.py` backend flag
- [x] Eval JSON includes `policy_backend` for comparison
- [x] `scripts/summarize_comparison.sh` + `smolvla_bridge/summarize_comparison.py`
- [x] `gitignore` — root `.gitignore` already ignores `vla_lab/datasets/`, `checkpoints/`, `eval_results/`

Once these are done, you can **finetune SmolVLA on the same sessions** you use for TinyVLA and **compare** with identical Isaac eval + comparable training curves.

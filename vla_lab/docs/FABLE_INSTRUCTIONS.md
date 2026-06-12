# Fable Instructions: VLA Training + Eval Deep Audit

Use this document as the full prompt for Claude (or another agent) to debug erratic robot behavior during VLA eval.

---

You are debugging a Vision-Language-Action (VLA) manipulation pipeline in the `kinova-isaac` repo. The user runs eval and the Kinova arm immediately flails / seizes / moves erratically as soon as the policy loop starts.

Your job is to:

1. Deeply audit the full pipeline: data collection → dataset → training → eval → controller
2. Find root causes (not just symptoms) of erratic robot behavior
3. Implement fixes in the codebase where appropriate
4. Add diagnostics/logging if needed to make future debugging easier
5. Write a final markdown report at `vla_lab/EVAL_DEBUG_REPORT.md` with:
   - Executive summary
   - Root cause analysis (ranked by severity)
   - What you changed (file-by-file)
   - Verification steps the user should run
   - Recommended next steps (data, training, eval, SmolVLA path)

---

## Environment

- Repo root: `/home/kye/Desktop/Depo/Code/CORL/kinova-isaac`
- Conda env: `riften` (Isaac Sim 5.1 + Isaac Lab)
- Isaac Lab launcher: `/home/kye/IsaacLab/isaaclab.sh` (NOT `./IsaacLab/` inside repo)
- GPU: RTX 4090 Laptop

## Command that reproduces the issue

```bash
conda activate riften
cd ~/Desktop/Depo/Code/CORL/kinova-isaac
./vla_lab/scripts/eval.sh --num-episodes 10 --ckpt vla_lab/checkpoints/tiny_v0/last.pt
```

Observed behavior: Isaac Sim launches, scene loads, robot spawns, then arm flails / has violent erratic motion immediately when the policy loop begins.

Prior fix already applied: PyTorch 2.6 `torch.load(weights_only=True)` broke checkpoint loading; `vla_lab/checkpoint_utils.py` was added. Eval now loads the checkpoint but robot behavior is wrong.

---

## Project context (read first)

This is the `vla_lab/` package — a self-contained VLA research stack:

- **Data collection**: `data_collection/collect_data` with profiles `vla_v1` (reach/grasp/lift) and `vla_v2` (pick-and-place). Wrappers: `vla_lab/scripts/collect.sh`, `collect_v3.sh`, `collect_v2.sh`
- **Training (TinyVLA)**: `vla_lab/train.py` + `vla_lab/configs/train_tiny.yaml`
- **Training (SmolVLA)**: `vla_lab/train_smolvla.sh` + LeRobot export via `export_lerobot_dataset.sh`
- **Eval**: `vla_lab/eval_isaaclab.py` + `vla_lab/scripts/eval.sh` + `vla_lab/configs/eval_isaac.yaml`
- **Controller**: `controllers/cartesian_velocity/cartesian_velocity.py` (Diff-IK, translate/gripper modes)
- **Action bridge**: `PolicyInputProvider` in `eval_isaaclab.py` converts model chunks → per-physics-step commands

Full docs: `vla_lab/README.md`, design spec `vla_lab/vla_ttc_engineering_spec.md`, HRI pivot memo `vla_lab/project_pivot_VLA_HRI2027.md`.

---

## Action / observation contract (CRITICAL — verify end-to-end)

Documented in `vla_lab/README.md` §6:

| Field | Shape | Meaning |
|-------|-------|---------|
| Image | (3, 224, 224) float [0,1] | Top-down camera |
| State | (4,) | `[ee_x_b, ee_y_b, ee_z_b, gripper_open_flag]` |
| Action chunk | (T=8, 7) | Per step: `[dx, dy, dz, drx, dry, drz, gripper]` in **base frame** |
| Gripper | {-1, 0, +1} | -1=close, +1=open, 0=hold |

Training labels come from `policy.action_from_prev` in `ticks.jsonl`, computed in `data_collection/core/logger.py` as the EE delta between consecutive **logged ticks** (position delta + rotation vector delta + discrete gripper transition).

**Suspected failure mode #1**: Training action dt may not match eval execution dt.

- `collect.sh` uses `--log-rate-hz 5`
- `collect_v3.sh` uses `--log-rate-hz 15` ← 3× finer ticks, smaller per-tick deltas
- Eval defaults to `policy_rate_hz: 5` in `eval_isaac.yaml`
- `PolicyInputProvider` splits each predicted action across physics steps assuming one action = one policy interval at `policy_rate_hz`

If training data was collected at 15 Hz but eval executes at 5 Hz without rescaling, commands could be wrong magnitude/timing → flailing.

**Suspected failure mode #2**: Action normalization mismatch.

- `train_tiny.yaml` has `normalize_actions: true`
- Eval must denormalize before sending to controller (`action_stats.denormalize` in eval)
- Verify checkpoint contains valid `action_stats` and denorm is actually applied

**Suspected failure mode #3**: Train/eval scene distribution mismatch.

- Training config points to: `logs/data_collection/session_20260506_232450`
- Eval spawns only 3 objects with different spawn AABB than collect_v3 defaults (11 objects)
- Check session metadata for actual collection settings used

**Suspected failure mode #4**: Controller interprets deltas as per-physics-step pose increments.

- `CartesianVelocityJogController.step()` does `pos_des = ee_pos_b + dpos_safe` each physics step
- `PolicyInputProvider.advance()` computes `per_step = action[:6] / n_phys_per_action`
- Verify this math is consistent with how actions were logged during collection

**Suspected failure mode #5**: Rotation / mode switching.

- Controller `hold_orientation=True` in translate mode; rotation dims may behave unexpectedly
- Gripper mode switching via `current_mode_hint` — rapid mode flips could cause jitter

**Suspected failure mode #6**: Bad or undertrained model.

- TinyVLA is ~2M params; if val loss is high, model may output large random deltas
- Inspect `vla_lab/checkpoints/tiny_v0/metrics.jsonl` and training curves

**Suspected failure mode #7**: Observation bugs at eval time.

- Camera fails silently → zero image tensor → nonsense actions
- State vector construction in eval vs dataset reader mismatch
- Language instruction template at eval vs collection mismatch

---

## Files you MUST read and trace end-to-end

### Data pipeline

- `data_collection/core/logger.py` — how `action_from_prev` is computed
- `vla_lab/dataset.py` — how actions are read, normalized, chunked
- `vla_lab/inspect_data.py` — sanity check tool
- Session used for training: inspect `logs/data_collection/session_20260506_232450/` (or whatever `train_tiny.yaml` points to): `metadata.json`, sample `ticks.jsonl`, `instruction.json`

### Training

- `vla_lab/train.py`
- `vla_lab/configs/train_tiny.yaml`
- `vla_lab/models.py`, `vla_lab/losses.py`
- `vla_lab/checkpoints/tiny_v0/` — `config.json`, `metrics.jsonl`, `last.pt` keys/stats

### Eval (primary focus)

- `vla_lab/eval_isaaclab.py` — especially `PolicyInputProvider` class (~line 247) and episode loop (~line 794)
- `vla_lab/configs/eval_isaac.yaml`
- `vla_lab/scripts/eval.sh`
- `vla_lab/checkpoint_utils.py`
- `vla_lab/ttc.py` — if TTC/K-sampling affects outputs

### Controller / sim

- `controllers/cartesian_velocity/cartesian_velocity.py`
- `environments/reach_to_grasp_VLA/config.py` — camera, scene
- `data_collection/profiles/vla_v1.py` — collection behavior eval should match

### SmolVLA path (if relevant to user's other runs)

- `vla_lab/smolvla_bridge/action_obs_contract.py`
- `vla_lab/smolvla_bridge/policy_wrapper.py`
- `vla_lab/smolvla_bridge/convert_kinova_to_lerobot.py`
- User also has SmolVLA results at `vla_lab/results/2026-06-07/smolvla_ft_20260607_121915/`

---

## Diagnostic tasks (do these systematically)

### Phase 1: Offline verification (no Isaac)

1. Run `python -m vla_lab.inspect_data --data-roots logs/data_collection/session_20260506_232450`
2. Load checkpoint offline and print action_stats mean/std; compare to raw tick action magnitudes
3. Run `python -m vla_lab.dryrun --ckpt vla_lab/checkpoints/tiny_v0/last.pt --iters 10`
   - Log predicted action magnitudes (min/max/mean per dim) — are they physically plausible?
4. Write a small script or add a `--debug-actions` flag to eval that logs:
   - Raw model output (normalized)
   - Denormalized chunk
   - Per-physics-step command sent to controller
   - EE position before/after each policy tick

### Phase 2: Open-loop replay test (gold standard)

Implement or run an **open-loop replay** test:

- Load one training episode's recorded `action_from_prev` sequence from `ticks.jsonl`
- Feed those actions through `PolicyInputProvider` into the controller in sim
- **Expected**: robot should roughly reproduce scripted demo motion
- **If replay also flails**: bug is in action execution / controller / rate mismatch, NOT the model
- **If replay works but policy flails**: bug is in model outputs or observation construction

### Phase 3: Closed-loop eval with safeguards

Add temporary safety clamps for debugging:

- Max delta per physics step (e.g. 5mm translation, 2° rotation)
- Log when clamps trigger
- Compare eval with `chunk_consume: 8` vs `1` (less frequent re-querying)

### Phase 4: Align train and eval configs

Ensure these match the training session:

- `policy_rate_hz` == collection `log_rate_hz`
- `num_objects`, spawn AABB, box size
- Camera resolution / crop / domain rand off at eval unless train had it
- Instruction template selection (`language_seed`, templates in vla_v1 profile)

---

## Fixes to implement (as you find issues)

Examples of fixes you may need (don't assume — verify first):

- Resample training actions to target Hz OR scale eval actions by `log_rate_hz / policy_rate_hz`
- Fix missing/wrong denormalization path
- Align eval spawn settings with training data in `eval_isaac.yaml`
- Fix state vector (gripper flag encoding: open=1.0 vs close?)
- Fix image preprocessing (RGB order, normalization, resize)
- Add action magnitude clipping in `PolicyInputProvider` or eval config knob
- Fix first-tick garbage output (warmup with zero actions for N steps)
- Add `--replay-ticks PATH` mode to eval for regression testing
- Document required collection settings in README

Do NOT break the SmolVLA eval path while fixing TinyVLA.

---

## Verification checklist (include in report)

After fixes, the user should be able to run:

```bash
# 1. Offline action sanity
python -m vla_lab.dryrun --ckpt vla_lab/checkpoints/tiny_v0/last.pt --iters 50

# 2. Open-loop replay (if you implement it)
./vla_lab/scripts/eval.sh --replay-episode logs/data_collection/session_XXX/episode_0000 --headless

# 3. Closed-loop eval (headless first)
./vla_lab/scripts/eval.sh --num-episodes 3 --headless --ckpt vla_lab/checkpoints/tiny_v0/last.pt

# 4. Full eval
./vla_lab/scripts/eval.sh --num-episodes 10 --ckpt vla_lab/checkpoints/tiny_v0/last.pt
```

Success criteria:

- Robot moves smoothly (no seizure/flailing) even if task success rate is low
- Logged action magnitudes are physically plausible (< ~5cm per 200ms policy step)
- Open-loop replay of training ticks roughly tracks the scripted trajectory

---

## Deliverable: `vla_lab/EVAL_DEBUG_REPORT.md`

Structure the report as:

```markdown
# VLA Eval Debug Report
Date: ...
Checkpoint: vla_lab/checkpoints/tiny_v0/last.pt
Training session: ...

## Executive Summary
(2-3 sentences: what was wrong, what was fixed)

## Root Causes (ranked)
1. ...
2. ...

## Evidence
(action magnitude tables, config diffs, log snippets)

## Changes Made
| File | Change | Why |

## How to Verify
(step-by-step commands)

## Next Steps
### Immediate (before next eval)
### Data collection
### Training (TinyVLA vs SmolVLA)
### Eval / real robot
```

---

## Constraints

- Minimize scope: fix root causes, don't refactor unrelated HRI pivot code (`allocation/`, `human_study/`)
- Match existing code style and conventions
- All shell commands assume repo root `kinova-isaac/`
- Isaac eval must use `/home/kye/IsaacLab/isaaclab.sh` (eval.sh auto-detects this)
- Do not commit secrets; do not push without user asking
- If training data is insufficient/wrong, say so clearly in the report rather than over-tuning eval

---

## Additional context for the agent

Also check whether `session_20260506_232450` was collected with `collect.sh` (5 Hz) or `collect_v3.sh` (15 Hz) by reading its `metadata.json` / session config. Recent collection attempts used `collect_v2.sh` (exit 137 / OOM).

---

## Start here

1. Read `vla_lab/README.md` §5.5 and §6
2. Trace one action from `logger.py` → `dataset.py` → `train.py` → `eval_isaaclab.py` → `cartesian_velocity.py`
3. Run offline diagnostics before launching Isaac again
4. Implement open-loop replay test — this is the highest-value diagnostic
5. Fix issues, verify, write `vla_lab/EVAL_DEBUG_REPORT.md`

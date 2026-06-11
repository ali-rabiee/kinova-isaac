# VLA Eval Debug Report

Date: 2026-06-10
Checkpoint: `vla_lab/checkpoints/tiny_v0/last.pt` (trained 2026-05-07, 1.88 M params, val loss 0.058 normalized)
Training session: `logs/data_collection/session_20260506_232450` — **deleted from disk** (not in Trash either). Only `session_20260607_115038` (20 episodes, 15 Hz, collect_v3-style) survives.

## Executive Summary

The flailing is an **execution-side bug, not a model bug**: eval's per-episode settle loop stepped bare physics with no controller, so the arm was left without fresh drive targets or gravity compensation (the soft Jaco actuators sag), and the first policy steps then fought a large stale IK/hold-orientation error at `qdot = error / 4.2 ms` — a violent whip the moment the policy loop begins. Beyond that, four pipeline bugs guaranteed near-zero task success even with smooth motion: eval executed 15 Hz training actions at 5 Hz (3× too slow), the eval camera sensor silently failed so the policy always saw black images, the gripper-state logger never registered grasps (the model was trained with zero close-gripper labels), and eval's start pose / scene / language distribution didn't match training. All are fixed or mitigated in this change set; the existing 20-episode session has been repaired, and a new open-loop replay mode regression-tests the execution path (now reproducing a demo with 8 mm mean EE error).

## Root Causes (ranked)

### 1. Unpowered settle + stale hold-orientation → violent motion at policy start (the "flailing")

- `eval_isaaclab.py` settled each episode with 60 × `sim.step()` and **no `controller.step()`**. Nothing wrote drive targets or gravity-compensation efforts during that window (`robot.write_data_to_sim()` only happens inside `controller.step`).
- The Jaco arm actuators (`KINOVA_JACO2_N6S300_CFG`) have stiffness 40/15 N·m/rad and damping 1.0/0.5 — far too soft to hold the arm against gravity without the controller's explicit gravity-compensation efforts. During the settle the arm sags and picks up velocity.
- The controller's `hold_orientation` quaternion and Diff-IK state were captured **before** the settle (at `controller.reset`), so the first `controller.step` saw a large orientation error and commanded `qdot_arm = (q_des − q_arm) / physics_dt` with `physics_dt = 1/240 s` — enormous joint-velocity targets, saturating effort limits. There is no joint-velocity cap in the controller.
- Collection never hit this because `vla_v1.py` runs `controller.step()` inside its settle loop ("Keep the arm stable while the scene settles") and stabilizes ~272 steps before planning.
- The settle phase was also invisible (`render=False`), so the collapse + violent catch appears to the user exactly as "arm flails the moment the policy loop starts".

### 2. Eval camera sensor silently failed → policy always saw black images

- `CameraCfg` in this Isaac Lab version requires `spawn=None` (+ an `OffsetCfg`) to attach to an existing prim. Eval omitted `spawn`, so `Camera(...)` raised, the exception was caught, and **every eval ran the policy on all-zero images** (the brief's suspected failure mode #7, confirmed in the run logs: "could not create Camera sensor: Missing values … - spawn").
- Collection (`vla_v1.py:1311`) already used the correct `spawn=None` pattern — eval predated it.

### 3. Gripper labels never fire → model cannot learn to grasp

- `data_collection/core/logger.py` inferred gripper state as `mean(all "finger" joints) > 0.5`. The three `finger_tip` joints stay ≈ 0 even when closed, and the proximal fingers stall at ~0.3–0.5 rad when wrapped around the 8 cm box (commanded close = 1.2 rad). The mean never crossed 0.5.
- Result: in `session_20260607_115038`, **all 20 episodes grasp and lift successfully** (per `events.jsonl`: `GRIPPER_CLOSE` → `LIFT` → `grasp_result ok=true`), yet **all 2 480 ticks log `state: "open"` and `gripper_action: 0`**. A model trained on this can reach but can never close the gripper; the state vector's `gripper_open` flag is also constant (useless), and the same eval-side detector would mis-report state during eval.
- The May training session was collected with the same logger, so `tiny_v0` was almost certainly trained with zero close labels too.

### 4. Action-rate mismatch: 15 Hz training deltas executed at 5 Hz

- Training actions are EE deltas between consecutive *logged ticks* (`policy.action_from_prev`), i.e. per `1/log_rate_hz` of **sim time**.
- The training session was collected at **15 Hz** (verified: the surviving June session's per-tick delta stats match the checkpoint's `action_stats` almost exactly, and the trashed sibling sessions from the same May 6 evening switched from 5 Hz to 15 Hz at 23:21, before the 23:24 training session).
- `eval_isaac.yaml` had `policy_rate_hz: 5`: each 1/15 s delta was stretched over 1/5 s → EE moves at ⅓ of demo speed, and the closed-loop state evolution the model sees is out of distribution.

### 5. Train/eval distribution mismatch (language + scene)

- The checkpoint's tokenizer vocabulary is only 25 words: `red`, `blue`, `1`, `2` … — **no `purple`, `orange`, `yellow`, `cyan`, `3`–`6`**. The May session evidently only ever targeted *red box 1* and *blue box 2*. Eval spawned 3 boxes (red 1, blue 2, yellow 3) and asks for them cyclically, so a third of prompts contain only `<unk>` content words.
- Collection (collect_v3) spawned **6 objects** (per June session events) with spawn AABB (0.26, −0.34, 0.89)–(0.52, 0.36, 1.06) and `min_distance 0.16`, plus camera/light domain randomization; eval spawned 3 objects in a slightly different AABB with no randomization.
- Episodes ≥ 1 additionally started from wherever the previous episode left the arm (no robot reset between episodes).

### 6. Eval start pose ≠ demo start pose (~23 cm apart)

- Every demo episode's tick 0 has the EE at (0.454, 0.093, 0.210) in base frame — *before* the planner does anything (verified against `events.jsonl` timestamps), i.e. it is where collection's longer setup sequence leaves the arm. Eval's plain settle leaves the arm at ≈ (0.227, 0.094, 0.257) — a pose **no training tick ever visits**, so the policy's first observations were out of distribution, and open-loop replay carried a constant 0.233 m offset.
- Fixed pragmatically with a scripted **pre-roll** to `eval.start_ee_pos_b` (defaults to the session's tick-0 pose). Eval also now steps the sim once before `controller.reset` so the controller never captures stale pre-reset buffers (matching vla_v1's reset sequence).

### 7. Secondary issues

- First policy tick could see a black image even with a working camera (settle never rendered, so the buffer was stale at tick 0).
- `train_tiny.yaml` pointed at the deleted session — retraining would fail outright.
- No diagnostics existed to see commanded magnitudes at eval time.

## Evidence

- **Checkpoint** `action_stats` (m/tick): std = [0.0053, 0.0082, 0.0082] translation; June session per-tick deltas: std = [0.0047, 0.0067, 0.0073], |Δp| mean 9.5 mm / p95 22 mm — same policy, same 15 Hz tick basis.
- **Dryrun** of the checkpoint (random/black/real images, known/unknown words): denormalized |Δp| per action 10–18 mm, |Δrot| ≤ 0.003 rad, gripper ∈ [−0.05, 0.08]. Model outputs are physically tame in every condition — flailing could not have come from action magnitudes.
- **Gripper**: June session `events.jsonl` shows `GRIPPER_CLOSE` + `grasp_result ok=true` in 20/20 episodes; `ticks.jsonl` had 0/2480 close labels before repair, 20 close actions + 320 close-state ticks after repair.
- **Rate**: June session metadata `log_rate_hz: 15`; trashed `session_20260506_232128` (23:21 that evening) = 15 Hz, `…231135` (23:11) = 5 Hz → the 23:24 training session was 15 Hz. Eval ran `policy_rate_hz: 5`.
- **Tokenizer vocab** (full): `<pad> <unk> <bos> <eos> box the pick up and it blue 2 red 1 to lift go grab grasp reach move one for please number`.
- **Open-loop replay** (this change set): before the start-pose pre-roll, all 88 recorded actions executed smoothly (per-tick |Δp| 9–27 mm, zero clamps) but with a **constant 0.233 m offset** — i.e., the relative trajectory was reproduced near-perfectly from the wrong start pose. With the pre-roll, replay of `episode_0000` tracked the recorded EE trajectory with **mean 8.1 mm / max 16.8 mm error over 88/88 actions**, gripper close firing at the recorded tick. The execution path (provider → controller → Diff-IK → sim) is verified end-to-end.
- **Closed-loop eval after fixes** (2 episodes, headless, current checkpoint): camera attached (real images), pre-roll ~118 steps to the start pose each episode, **no flailing** — per-tick |Δp| stayed 9–34 mm over 150 policy ticks, zero safety clamps, and the arm navigated in different directions per target (EP 0 toward +y, EP 1 toward −y). Success 0/2, exactly as predicted for a model trained without gripper-close labels.

## Changes Made

| File | Change | Why |
| --- | --- | --- |
| `vla_lab/eval_isaaclab.py` | Per-episode robot reset to default state + **powered settle** (`controller.step` during settle, mirrors vla_v1) + `controller.reset` *after* settle to capture the settled hold-orientation; a few rendered hold steps before tick 0 so the camera has a fresh frame | Root cause 1 (flailing) + stale first image + episodes chaining state |
| `vla_lab/eval_isaaclab.py` | `train_action_rate_hz` support: rescales action deltas by `train_rate / policy_rate` with a loud warning when rates differ; `--policy-rate-hz` CLI override | Root cause 4 |
| `vla_lab/eval_isaaclab.py` | Per-action safety clamps in `PolicyInputProvider` (`max_action_dpos_m`, `max_action_drot_rad`), counted and reported | Turns any future runaway command into a logged event instead of a seizure |
| `vla_lab/eval_isaaclab.py` | `--debug-actions`: per-policy-tick JSONL + console log of normalized/denormalized magnitudes, EE position, gripper range, clamp count | Phase-1 diagnostics requested in the brief |
| `vla_lab/eval_isaaclab.py` | `--replay-episode <dir>`: open-loop replay of a recorded episode through `PolicyInputProvider` → controller at the session's logged rate, bypassing the policy; reports EE tracking error vs. the recorded trajectory | Phase-2 gold-standard execution test, regression tool |
| `vla_lab/eval_isaaclab.py` | Eval-side `_gripper_state()` uses proximal finger joints only, threshold 0.2 | Root cause 3 (state-vector flag consistent with fixed logger) |
| `data_collection/core/logger.py` | Gripper state detection: exclude `finger_tip` joints, threshold 0.5 → 0.2 | Root cause 3, for all future collections |
| `vla_lab/repair_gripper_labels.py` | **New tool**: backfills `gripper.state` and `action_from_prev.gripper_action` in existing sessions from `events.jsonl` GRIPPER_CLOSE/OPEN intervals (keeps `ticks.jsonl.orig` backup) | Repairs already-collected data; applied to `session_20260607_115038` (20 close labels added) |
| `vla_lab/eval_isaaclab.py` | Camera sensor: `spawn=None` + `OffsetCfg` (vla_v1 pattern), loud error if creation fails, RGBA→RGB guard | Root cause 2 |
| `vla_lab/eval_isaaclab.py` | One `sim.step()` + `robot.update()` between robot state write and `controller.reset` | Root cause 6 |
| `vla_lab/eval_isaaclab.py` + `eval_isaac.yaml` | Scripted **pre-roll** to `eval.start_ee_pos_b` (default = the training session's tick-0 EE pose (0.454, 0.093, 0.210)) before the policy/replay takes over | The demos never visit the plain settled pose — collection's setup leaves the arm ~23 cm away from eval's settle (measured by replay); the pre-roll makes eval's first observation in-distribution |
| `vla_lab/configs/eval_isaac.yaml` | `policy_rate_hz: 5 → 15`, added `train_action_rate_hz: 15`, clamps; spawn settings aligned to collect_v3 (6 objects, AABB (0.26,−0.34,0.89)–(0.52,0.36,1.06), `min_distance 0.16`) | Root causes 4 + 5 |
| `vla_lab/configs/train_tiny.yaml` | `data_roots` → surviving repaired session; notes on rate + repair tool | Old path no longer exists |
| `vla_lab/README.md` | New "Debugging erratic eval behavior" subsection (rate contract, replay, clamps, repair tool) | Documentation requested in the brief |

Not changed: `controllers/cartesian_velocity/cartesian_velocity.py` (shared with collection — behavior preserved), SmolVLA bridge (untouched; eval-loop changes are backend-agnostic and behavior-neutral when `train_action_rate_hz == policy_rate_hz`).

## How to Verify

```bash
conda activate riften
cd ~/Desktop/Depo/Code/CORL/kinova-isaac

# 1. Offline action sanity (no Isaac): magnitudes + latency
python -m vla_lab.dryrun --ckpt vla_lab/checkpoints/tiny_v0/last.pt --iters 50 --k 1
# → 1.7 ms mean latency, 1.88 M params (verified)

# 2. Gripper label repair status (idempotent; already applied)
python -m vla_lab.repair_gripper_labels --session logs/data_collection/session_20260607_115038 --dry-run

# 3. Open-loop replay (execution-path regression test; no policy involved)
./vla_lab/scripts/eval.sh --replay-episode logs/data_collection/session_20260607_115038/episode_0000 \
    --headless --debug-actions
# Expect: smooth motion, |dpos| per tick ≲ 30 mm, small EE tracking error vs the demo, gripper
# close fires at the recorded tick. Note: replay does not reproduce the episode's *object*
# layout, so judge by tracking error / smoothness, not lift success.

# 4. Closed-loop eval, headless first, then with GUI
./vla_lab/scripts/eval.sh --num-episodes 3 --headless --ckpt vla_lab/checkpoints/tiny_v0/last.pt --debug-actions
./vla_lab/scripts/eval.sh --num-episodes 10 --ckpt vla_lab/checkpoints/tiny_v0/last.pt
# Expect: smooth motion (no flailing). Task success with the *current* checkpoint will stay ≈ 0
# (it was trained without gripper labels and on a different language/scene distribution — retrain, see below).
```

Success criteria from the brief (all verified on 2026-06-10):

- Robot moves smoothly even if task success is low — verified closed-loop: 2 episodes × 1200 steps, |Δp| 9–34 mm/tick, zero clamps, no flailing.
- Logged action magnitudes physically plausible — `--debug-actions` shows ~10–34 mm per 1/15 s tick (≤ 60 mm clamp, never triggered).
- Open-loop replay tracks the scripted demo — verified: mean 8.1 mm / max 16.8 mm EE error across 88/88 actions.

## Next Steps

### Immediate (before next eval)

1. Run the replay + closed-loop verification above and confirm smooth motion.
2. **Retrain TinyVLA on the repaired session**: `./vla_lab/scripts/train.sh` (config now points at `session_20260607_115038`). Expect reach behavior + a chance at grasping; 20 episodes is still very little data.

### Data collection

- Collect a bigger session with the fixed logger (collect_v3, e.g. `NUM_EPISODES=100`). Memory pressure previously killed collect_v2 runs (exit 137) — prefer headless + `--num-episodes` in smaller chunks, multiple sessions, and list them all under `data.data_roots`.
- Fix target diversity: `--target-selection farthest_no_repeat` produced only 2 unique targets (purple 4 / orange 5) in 20 episodes — the tokenizer never sees the other colors/indices. Use a rotating/random target selection or enough episodes that every color/index appears; otherwise eval prompts contain `<unk>` words.
- Keep `--log-rate-hz` consistent across sessions, and treat it as part of the model contract (it is now recorded in the README §6 area and enforced via `eval.train_action_rate_hz`).
- Consider logging finger joint positions (`log_joint_data`) so gripper state can always be re-derived.

### Training (TinyVLA vs SmolVLA)

- TinyVLA: retrain on repaired data; watch the gripper column — with only ~1 close label per episode, consider raising `gripper_weight` (already 2.0) or oversampling ticks near the grasp.
- SmolVLA: re-export the LeRobot dataset (`export_lerobot_dataset.sh`) **after** the gripper repair, then fine-tune; the previous export inherited the broken labels. The eval-loop fixes apply to the SmolVLA path unchanged.

### Eval / real robot

- Keep `policy_rate_hz == collection log rate` (now 15) and `chunk_consume: 1`; compare `chunk_consume: 8` once motion is sane.
- The gripper-close flush in `PolicyInputProvider` holds the close command ~15 physics steps and the hold persists afterwards; if grasps slip at eval, lengthen the flush toward the demo's 60-step close + 20-step hold.
- For the real robot, the same rate contract applies; the safety clamps (`max_action_dpos_m`) map directly onto `real_robot/safety_envelope.py` limits.
- Honest caveat: with 20 demos, 2 targets, and ~1 grasp label each, expect low success rates after retraining. The execution path is now trustworthy (replay test), so further data/training investment is the bottleneck, not eval plumbing.

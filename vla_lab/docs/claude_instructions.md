# Claude Code Instructions — Fix Data Collection Quality (Success Rate + Smooth Motion)

**Priority:** P0 — blocking useful VLA training. Continue until both issues below are resolved end-to-end, verified on real collection runs, and documented.

You are working in the `kinova-isaac` repo. The user has a working eval pipeline (no more violent flailing), a fine-tuned SmolVLA checkpoint, and **bad training data**. Your job is to fix the **scripted expert** used for data collection so that:

1. **Most episodes succeed** (not ~7%).
2. **Demonstration motion is smooth and near-optimal** (not jagged / zigzaggy).

The trained policy currently makes a respectable grasp attempt but inherits the jagged trajectories from the demos. **Fix the expert first**; retraining on the same broken motion will not produce smooth behavior.

Do not stop at analysis — implement fixes, run short collection smoke tests, verify with `verify_session`, and write a short report at `vla_lab/docs/DATA_COLLECTION_FIX_REPORT.md`.

---

## Environment

| Item | Value |
|------|-------|
| Repo root | `/home/kye/Desktop/Depo/Code/CORL/kinova-isaac` |
| Conda env | `riften` (Isaac Sim 5.x + Isaac Lab; **NumPy must be &lt; 2**) |
| Isaac launcher | `~/IsaacLab/isaaclab.sh` (set `ISAACLAB=` if `eval.sh` / collection cannot find it) |
| GPU | RTX 4090 Laptop |
| Collection entrypoint | `./vla_lab/scripts/collect_v3.sh` |
| Authoritative collection docs | `vla_lab/data_collection_guide.md` |

---

## Current symptoms (user-reported, 2026-06)

### A. Catastrophically low usable data

| Session | Raw episodes | Successful | Success rate | Usable frames |
|---------|-------------|------------|--------------|---------------|
| `logs/data_collection/session_20260612_235905` | 40 | 8 | 20% | ~1.2k ticks |
| `logs/data_collection/session_20260613_012634` | 600 | 43 | **7%** | 6,306 ticks |

`verify_session` passes (layout diversity, targets, gripper labels OK) but warns success rate &lt; 70%. With `success_only: true`, **557/600 episodes are thrown away**.

Grasp-attempt histogram on the 600-ep session: `{1: 31, 2: 10, 3: 85, 4: 474}` — most episodes exhaust all 3 retries.

Failure modes observed in `events.jsonl` / `episode_summary.json`:
- Final lift check fails (`grasp_result ok=false`, `dz_above_table` ≈ 0.026 m vs 0.06 m threshold)
- `approach_stall` (arm stops making progress toward grasp point)
- `truncated: true` (hit `--max-steps-per-episode 8000`)
- Occasional `empty_close` (gripper closes but object not grasped)

### B. Jagged / zigzaggy motion (collection AND eval)

- During **data collection**, the scripted arm does not follow clean straight paths; motion is visibly zigzaggy.
- After fine-tuning **SmolVLA** on the 43 successful demos, eval no longer flails (prior eval bugs were fixed — see `vla_lab/docs/EVAL_DEBUG_REPORT.md`), but the policy makes a **decent grasp attempt** with the same **jagged, suboptimal** motion style as the demos.
- `verify_session` warns `max per-tick EE delta ~122 mm` — likely grasp-retry replan jumps, but may also indicate discontinuities in normal approach.

**Implication:** This is primarily an **expert / motion-generation** problem, not a training hyperparameter problem. Fix collection motion and success rate before collecting 100+ episodes or retraining.

---

## What already exists (do not re-litigate unless broken)

### Eval-side fixes (DONE — keep working)

Documented in `vla_lab/docs/EVAL_DEBUG_REPORT.md` and `vla_lab/docs/FABLE_INSTRUCTIONS.md`:

- Powered settle + pre-roll to canonical start pose `(0.454, 0.093, 0.210)` base frame
- `policy_rate_hz: 15` matches collection
- Camera `spawn=None` pattern
- Gripper label fix in `data_collection/core/logger.py` (proximal finger heuristic)
- Open-loop `--replay-episode` regression test

**Eval smoke test (should still pass after your changes):**

```bash
conda activate riften
cd ~/Desktop/Depo/Code/CORL/kinova-isaac

./vla_lab/scripts/eval.sh --headless \
  --replay-episode logs/data_collection/session_20260613_012634/episode_0000
```

Expect smooth replay and ~mm-level EE tracking error in console.

**Policy eval (current fine-tuned model):**

```bash
./vla_lab/scripts/eval.sh --num-episodes 3 \
  --policy-backend smolvla \
  --ckpt vla_lab/checkpoints/smolvla_kinova_ft/checkpoints/last/pretrained_model \
  --lerobot-dataset-root vla_lab/datasets/lerobot_kinova_v0
```

### Collection-side fixes already applied (2026-06-11 / 06-12)

See `vla_lab/data_collection_guide.md` §5:

- Frozen object respawn (PhysX `set_transforms` + verify readback)
- `TARGET_SELECTION=cycle` (all 6 colors)
- Pre-roll + unlogged stabilize (no sag prefix in ticks)
- Camera z = 2.2 (not 4.0)
- **2026-06-12:** Live PhysX pose for grasp targeting in `motion_generation/grasp_estimation/obb.py` (USD bbox was stale)
- **2026-06-12:** `--jog-velocity-gain 0.5` in `collect_v3.sh` to damp Diff-IK oscillation on soft Jaco actuators

The jog-gain patch **reduced shaking** but user still sees **zigzag** paths and **7% success**. Your job is to go deeper.

---

## Data / action contract (DO NOT BREAK)

Any change to these requires updating **both** collection and eval, and invalidates old sessions:

| Contract | Value | Where |
|----------|-------|-------|
| Log / action rate | **15 Hz** | `collect_v3.sh --log-rate-hz 15`; `eval_isaac.yaml policy_rate_hz: 15` |
| Action | `action_from_prev`: EE delta (pos + rotvec, base frame) + gripper {−1,0,+1} on transition tick | `data_collection/core/logger.py` |
| Start pose | `(0.454, 0.093, 0.210)` base | `collect_v3.sh` ≡ `eval.start_ee_pos_b` |
| Camera | Top-down, z=2.2, FOV 65°, 640×640 | `environments/reach_to_grasp_VLA/config.py` |
| Scene | 6 boxes, spawn AABB, min spacing 0.16 m | `collect_v3.sh` ≡ `eval_isaac.yaml` |

**Collection uses `jog_velocity_gain=0.5`; eval uses default `1.0`** so policy deltas are realized at full scale. If you change collection gain or speed, document the train/eval implication.

---

## Motion stack to audit (root-cause hunt)

Trace the full path from planner → logged actions:

```
vla_lab/scripts/collect_v3.sh
└── data_collection/profiles/vla_v1.py          # episode loop, state machine, respawn, logging
    ├── motion_generation/planners/scripted.py  # [pregrasp, grasp, lift] waypoints
    ├── motion_generation/grasp_estimation/obb.py  # grasp pose (LIVE PhysX pose fix)
    ├── motion_generation/mogen.py
    ├── controllers/input/waypoint_follower.py   # per-physics-step deltas, waypoint pop at 7 mm
    ├── controllers/cartesian_velocity/cartesian_velocity.py  # Diff-IK, jog_velocity_gain, hold_orientation
    └── data_collection/core/logger.py           # 15 Hz ticks, action_from_prev
```

### Current `collect_v3.sh` motion knobs

```
--planner-speed-mps 0.58
--jog-velocity-gain 0.5
--planner-waypoint-max-seg-m 0.022   # densify straight segments to ≤22 mm
--tolerance 0.007
--close-if-within-m 0.007
--grasp-depth -0.07
--pregrasp 0.10  --lift 0.15  (via profile defaults)
--max-grasp-attempts 3
--approach-detour-m 0
```

### Hypotheses for zigzag / jagged motion (investigate ALL)

1. **Piecewise straight-line waypoints** — pregrasp → grasp → lift are axis-aligned segments; densification to 22 mm may still produce visible corners. Consider arc blending, spline interpolation, or a single approach axis (e.g. top-down descend only after XY align).

2. **Waypoint follower pops at 7 mm tolerance** — abrupt zero-velocity stops and direction changes between segments. Read `waypoint_follower.py` `advance()` and stagnation / pop logic.

3. **Diff-IK + `hold_orientation=True` + soft Jaco actuators** — even with `jog_velocity_gain=0.5`, the EE may hunt/orbit. Try: lower gain, velocity ramping, joint-velocity caps, critically-damped tracking, or trajectory interpolation in joint space.

4. **Control rate vs log rate** — physics at 240 Hz, actions logged at 15 Hz. Logged deltas are tick-to-tick EE motion; if high-frequency oscillation exists between ticks, the 15 Hz subsample may alias into zigzag labels. Consider low-pass filtering **before** logging (without breaking the eval execution contract), or smoothing the expert command stream.

5. **Grasp retry discontinuities** — failed attempts reopen, replan, jump ~100+ mm (`verify_session` max delta warning). Retries may poison demos even when episode eventually succeeds.

6. **Approach stall / hover loops** — state machine may command small oscillating corrections when stuck. Search `approach_stall` logic in `vla_v1.py`.

7. **Planner vs controller speed mismatch** — `planner-speed-mps 0.58` with `jog_velocity_gain 0.5` → effective ~0.29 m/s. Waypoint `step_pos_m = speed * dt` may not match realized EE motion.

8. **Scripted XY-then-Z phases** — `scripted.py` `execute_scripted_motion` has multi-phase align that may insert corrective waypoints when EE dips below `safe_z` (lines ~66–75). This can cause zigzag corrections.

### Hypotheses for low success rate (investigate ALL)

1. **Grasp pose error** — OBB + live PhysX offset may still be wrong for settled boxes (orientation, top-center vs graspable point, `ee_z_offset`, `grasp-depth -0.07`).

2. **`empty_close`** — gripper closes on air; check `--close-if-within-m`, finger collision, box spacing `--min-distance 0.16`.

3. **`approach_stall`** — controller cannot reach grasp point within stall threshold; may correlate with jagged motion / IK limits.

4. **Lift detection too strict** — `--lift-success-min-dz-m 0.06` with 10 consecutive checks; object slips or slow lift fails episode despite partial grasp.

5. **Domain randomization** — camera/light jitter may make grasp harder (unlikely for scripted expert using state, but affects logged images for VLA).

6. **Clutter / spawn layout** — 6 boxes in tight AABB; some layouts physically unreachable or prone to collisions.

---

## SmolVLA-specific note (secondary)

LeRobot export uses **6D delta pose only** (no gripper in `vla_lab/smolvla_bridge/convert_kinova_to_lerobot.py`). Eval fills gripper with 0. If collection expert does not produce reliable close labels in successful episodes, SmolVLA cannot learn to close. Verify close labels in successful episodes:

```bash
python -m vla_lab.verify_session logs/data_collection/session_20260613_012634
# gripper: close-labels per episode min/max should not be 0/N
```

Address gripper in the export contract only after the expert reliably grasps.

---

## Your task — phased plan

### Phase 1: Diagnose (read-only, use existing sessions)

1. Parse `session_20260613_012634`:
   - Success vs failure breakdown by `grasp_result` reason
   - Per-successful-episode: distribution of `|delta_pos|` per tick, max, p95
   - Compare tick path curvature (angle between consecutive delta vectors) for successful vs failed episodes
   - Identify whether zigzag is in approach, retry jumps, or lift phase

2. Run **5-episode GUI collection** (no headless) with verbose logging to watch motion:

   ```bash
   NUM_EPISODES=5 ./vla_lab/scripts/collect_v3.sh
   ```

3. Optionally add **temporary** debug logging (EE pos per physics step, waypoint index, stage) — remove or gate behind a flag before final PR.

### Phase 2: Fix expert motion (smooth paths)

Goals:
- Visually straight, smooth top-down approach and lift in GUI
- Per-tick `|delta_pos|` p95 ≤ ~20 mm, no spikes &gt; 50 mm except deliberate gripper retries
- No orbiting / hunting at waypoints

Candidate fixes (evaluate empirically):
- Trajectory smoothing (minimum-jerk / spline across waypoints)
- Increase waypoint densification or reduce segment corner angle
- Tune `jog_velocity_gain`, `linear_speed_mps`, add acceleration limits
- Joint-space interpolation instead of per-step Diff-IK chasing
- Single-shot descend after XY align (avoid repeated `safe_z` corrective inserts)
- Reduce orientation-hold fighting (if EE orientation error drives zigzag)

**Keep `jog_velocity_gain` at 0.5 for collection unless you validate eval compatibility.**

### Phase 3: Fix success rate

Target: **≥ 70%** `verify_session` success rate on a 40-episode smoke session (stretch: ≥ 85%).

Candidate fixes:
- Re-tune grasp geometry (`grasp-depth`, `pregrasp`, `ee_z_offset`, `close-if-within-m`)
- Fix approach stall (increase grace, better stall detection, replan on stall)
- Improve lift confirmation (timing, hold-after-close, post-lift hold)
- Collision-aware pregrasp offset when boxes are close
- Consider `PLANNER=curobo_v2` for collision-free approach (only if smoother AND higher success)

### Phase 4: Verify + document

1. Collect `NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless`
2. `python -m vla_lab.verify_session logs/data_collection/session_<NEW_TS>`
3. Eyeball 3 successful episodes in `episode_XXXX/images/`
4. Write `vla_lab/docs/DATA_COLLECTION_FIX_REPORT.md` with:
   - Root causes (ranked)
   - Files changed
   - Before/after metrics (success rate, delta stats, qualitative motion)
   - Exact commands for user to re-collect ≥120 successful episodes
5. Update `vla_lab/data_collection_guide.md` if default knobs change.

### Phase 5: Retrain smoke test (optional but recommended)

After a verified ≥40-ep session with ≥70% success:

```bash
# Export successes only (default)
python -m vla_lab.smolvla_bridge.convert_kinova_to_lerobot \
  --session-roots logs/data_collection/session_<NEW_TS> \
  --out-dir vla_lab/datasets/lerobot_kinova_v0_v2 --fps 15 --overwrite

DATASET_DIR=vla_lab/datasets/lerobot_kinova_v0_v2 STEPS=3000 BATCH_SIZE=16 \
  OUT_DIR=vla_lab/checkpoints/smolvla_kinova_ft_v2 ./vla_lab/scripts/train_smolvla.sh

./vla_lab/scripts/eval.sh --num-episodes 5 --headless \
  --policy-backend smolvla \
  --ckpt vla_lab/checkpoints/smolvla_kinova_ft_v2/checkpoints/last/pretrained_model \
  --lerobot-dataset-root vla_lab/datasets/lerobot_kinova_v0_v2
```

Compare motion smoothness vs old checkpoint.

---

## Acceptance criteria (definition of done)

| Criterion | Target |
|-----------|--------|
| Collection success rate (40-ep smoke) | **≥ 70%** (`verify_session` no success warning) |
| Usable demos per 120 raw episodes | **≥ 84** at 70%; ideal **≥ 120** at 100% |
| Motion quality (GUI) | Smooth reach → grasp → lift, no visible orbiting or stair-step zigzag |
| Tick deltas (successful eps) | p95 ≤ 20 mm, max ≤ 50 mm except documented retry events |
| Contract | 15 Hz, start pose, camera unchanged; eval replay still passes |
| Docs | `DATA_COLLECTION_FIX_REPORT.md` written; guide updated if defaults changed |

---

## Key files (read first)

| File | Why |
|------|-----|
| `vla_lab/data_collection_guide.md` | Collection contract, known bugs, procedure |
| `vla_lab/scripts/collect_v3.sh` | Default flags including `jog_velocity_gain` |
| `data_collection/profiles/vla_v1.py` | **Main pipeline** — state machine, stall, grasp, lift, logging |
| `controllers/cartesian_velocity/cartesian_velocity.py` | Diff-IK, `jog_velocity_gain` semantics |
| `controllers/input/waypoint_follower.py` | Per-step commands, waypoint pop / stagnation |
| `motion_generation/planners/scripted.py` | Waypoint generation, multi-phase align |
| `motion_generation/grasp_estimation/obb.py` | Live PhysX grasp pose |
| `data_collection/core/logger.py` | What the policy is trained to imitate |
| `vla_lab/verify_session.py` | Post-collection QA |
| `vla_lab/docs/EVAL_DEBUG_REPORT.md` | Eval fixes already done — don't regress |
| `vla_lab/docs/FABLE_INSTRUCTIONS.md` | Prior audit template |

---

## Existing artifacts (user's machine)

| Artifact | Path |
|----------|------|
| Main collection session | `logs/data_collection/session_20260613_012634` (600 ep, 43 success) |
| Earlier session | `logs/data_collection/session_20260612_235905` (40 ep, 8 success) |
| LeRobot export | `vla_lab/datasets/lerobot_kinova_v0` (43 ep, 6306 frames, 15 fps) |
| Fine-tuned SmolVLA | `vla_lab/checkpoints/smolvla_kinova_ft/checkpoints/last/pretrained_model` |
| Training metrics | `vla_lab/results/2026-06-16/smolvla_ft_20260616_135522/` |
| Training loss at step 3000 | ≈ 0.073 (BC converged; task quality limited by demos) |

---

## Constraints

- **Do not** lower `success_only` or train on failed episodes to inflate metrics.
- **Do not** mix pre-2026-06-11 sessions with new ones.
- **Do not** break eval replay or the 15 Hz action contract.
- **Minimize scope** — prefer tuning/smoothing the existing stack over rewriting the pipeline.
- **Run commands yourself** — install deps, run 5–40 ep smoke collections, run `verify_session`.
- **No destructive git** — no force push, no hard reset.
- Only commit if the user asks.

---

## Suggested first commands

```bash
conda activate riften
cd ~/Desktop/Depo/Code/CORL/kinova-isaac
python -c "import numpy; assert numpy.__version__ < '2'"

# Baseline QA on existing session
python -m vla_lab.verify_session logs/data_collection/session_20260613_012634

# Watch motion live (GUI)
NUM_EPISODES=3 ./vla_lab/scripts/collect_v3.sh

# After fixes — headless smoke
NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless
python -m vla_lab.verify_session logs/data_collection/session_<NEW_TS>
```

---

## Deliverables checklist

- [ ] Root-cause analysis for low success rate and zigzag motion
- [ ] Code fixes in motion stack / vla_v1 profile (minimal, well-commented)
- [ ] 40-episode smoke session with ≥70% success
- [ ] `vla_lab/docs/DATA_COLLECTION_FIX_REPORT.md`
- [ ] Updated `data_collection_guide.md` if defaults changed
- [ ] Optional: retrain + eval showing smoother policy behavior

**Continue iterating until acceptance criteria are met or you document a hard blocker with evidence.**

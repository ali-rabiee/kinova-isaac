# Data Collection Guide — `collect_v3` (reach → grasp → lift)

This is the authoritative reference for collecting VLA training data with
`vla_lab/scripts/collect_v3.sh`. It documents the full pipeline, the data/action contract
that trained models depend on, the bugs that were found and fixed on **2026-06-11**, and the
exact procedure (with verification) for collecting a final dataset.

> **TL;DR — collecting the final dataset**
>
> ```bash
> conda activate riften
> cd ~/Desktop/Depo/Code/CORL/kinova-isaac
>
> # Collect in chunks (memory safety), e.g. 3 × 40 episodes:
> NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless     # run 3×
>
> # Verify EVERY session before training (non-zero exit = do not train on it):
> python -m vla_lab.verify_session logs/data_collection/session_<TS>
> ```
>
> Then list the verified sessions under `data.data_roots` in
> `vla_lab/configs/train_tiny.yaml` and train. **Do not mix in sessions collected before
> 2026-06-17** — older sessions carry the grasp bugs fixed in §5.7 (success ~7%) and, before
> 2026-06-11, the camera-framing change and frozen-layout bug.

---

## 1. What one collection run does

`collect_v3.sh` → `python -m data_collection.collect_data --profile vla_v1 --control planner`
(`data_collection/profiles/vla_v1.py`). Per episode:

1. **Reset** sim + robot to the default joint state (`_reset_sim_and_robot`), step once so
   PhysX views are registered.
2. **Respawn objects** — teleport the 6 colored boxes to fresh random poses inside the spawn
   AABB via PhysX tensor views, then **verify by reading the poses back** (see §5.1). The run
   aborts if verification fails twice in a row.
3. **Settle** (72 unlogged physics steps) so dropped boxes come to rest.
4. **Pre-roll** (unlogged): a scripted move drives the end-effector to the canonical start
   pose `(0.454, 0.093, 0.210)` in base frame — the same pose eval pre-rolls to.
5. **Domain randomization** (per episode): camera pose/yaw/FOV jitter + dome-light
   intensity/color around a fixed baseline. Sampled values are logged to `events.jsonl`.
6. **Target selection** (`--target-selection cycle`): deterministic round-robin over the
   boxes, so every color appears equally often. One target per episode, never re-selected
   mid-episode.
7. **Language instruction** generated from templates seeded by episode index
   (`"Pick up the red box."`, `"Go for box 3 and pick it up."`, …) → `instruction.json`.
8. **Stabilize** (200 physics steps): the arm is actively held at the start pose. These
   idle frames are **not logged** (no do-nothing actions in the dataset).
9. **Scripted routine** (state machine in `vla_v1.py`):
   `open gripper → plan [pregrasp, grasp] waypoints → approach (straight lines, densified to
   ≤22 mm segments at 0.58 m/s) → close (only when within 7 mm of the grasp point) → hold →
   lift 0.15 m → confirm the object rose ≥ 6 cm for 10 consecutive checks`.
   Failed grasps retry up to 3× within the episode (reopen → replan → re-approach).
10. **Episode end**: success/failure verdict written to `episode_summary.json`; logger closed.

Tick logging runs at **15 Hz of sim time** (every 16th physics step at 1/240 s) throughout
the routine; each tick saves a 640×640 PNG from the top-down camera.

### The motion stack (why the motion is smooth)

| Layer | File | Role |
| --- | --- | --- |
| Scripted planner | `motion_generation/planners/scripted.py` | 3 waypoints: pregrasp (+10 cm above object top), grasp (object top − 4 cm = box mid-height; `--ee-z-offset-m 0`), lift (+15 cm). See §5.7 for why the offset is 0. |
| OBB grasp provider | `motion_generation/grasp_estimation/obb.py` | object top-center from the world-space bounding box, recomputed at plan time |
| Waypoint follower | `controllers/input/waypoint_follower.py` | emits per-physics-step deltas capped at speed·dt ≈ 2.4 mm, pops waypoints within 7 mm |
| Cartesian controller | `controllers/cartesian_velocity/cartesian_velocity.py` | damped-least-squares Diff-IK → joint velocity targets + gravity compensation; workspace clamps; holds EE orientation |
| Gripper | `kinova/gripper.py` | position-target latch: open/close targets persist (held every step), so grips don't relax during lift |

---

## 2. The data / action contract (models break if you violate this)

These values are **baked into any model trained on the data**. Keep them identical between
collection and eval (eval reads most of them from the same code/config):

| Contract | Value | Where |
| --- | --- | --- |
| Tick / action rate | **15 Hz** (sim time) | `--log-rate-hz 15` in collect_v3.sh; `eval.policy_rate_hz: 15` in `vla_lab/configs/eval_isaac.yaml` |
| Action definition | `action_from_prev` = EE delta (pos + rotvec, base frame) between consecutive ticks; gripper ∈ {−1 close, 0 hold, +1 open} fired only on the transition tick | `data_collection/core/logger.py` |
| Episode start pose | EE at `(0.454, 0.093, 0.210)` base frame | `--start-ee-pos-b` (collection) ≡ `eval.start_ee_pos_b` (eval) |
| Camera | top-down at `(0.4, 0, 2.2)`, FOV 65°, 640×640 | `environments/reach_to_grasp_VLA/config.py` `DEFAULT_TOP_DOWN_CAMERA` (shared by collection + eval) |
| Scene | 6 boxes (8 cm, unique colors), spawn AABB (0.26, −0.34, 0.89)–(0.52, 0.36, 1.06), min spacing 0.16 m | collect_v3.sh ≡ `eval_isaac.yaml` |
| Physics step | 1/240 s | `environments/utils/physix.py` |

Consequences:

- **Never mix sessions with different `--log-rate-hz`, camera config, or start pose in one
  training run.** The training action statistics and the visual distribution must be uniform.
- At eval, `policy_rate_hz` must equal the collection log rate (the 2026-06-10 "flailing"
  postmortem item #4: 15 Hz deltas executed at 5 Hz move 3× too slow and out-of-distribution;
  see `docs/EVAL_DEBUG_REPORT.md`).
- An action chunk of 8 (training default) spans ~0.53 s of robot time.

## 3. On-disk format

```
logs/data_collection/session_YYYYMMDD_HHMMSS/
└── episode_0000/
    ├── metadata.json          # sim_dt, log_rate_hz, policy_rate_hz, robot/EE config
    ├── instruction.json       # language_command, target_prim, target_label, template meta
    ├── episode_summary.json   # success, truncated, grasp_attempts, steps, ticks, images  (since 2026-06-11)
    ├── ticks.jsonl            # one record per 1/15 s tick (see below)
    ├── events.jsonl           # episode_start/end, plan/exec/gripper actions, grasp_result,
    │                          #   domain_randomization, object_respawn{,_actual,_mismatch}, preroll_done
    └── images/image_NNNNNN.png  # 640×640 RGB top-down, one per tick (index == tick_idx)
```

A `ticks.jsonl` record (floats are stored as 4-decimal strings; `vla_lab.dataset` parses them
back):

- `robot.ee_pose_b` / `ee_pose_w` — EE pose in base/world frame
- `robot.gripper.state` — `"open"`/`"close"` (proximal finger joints > 0.2 rad heuristic)
- `robot.gripper.joint_positions` — raw finger joint angles (since 2026-06-11; lets the
  state be re-derived offline)
- `robot.joints` — arm joint positions/velocities
- `policy.action_from_prev` — **the training action**: `ee_delta_pos_b` (m),
  `ee_delta_rotvec_b` (rad), `gripper_action` ∈ {−1, 0, +1}; `None` on the first tick
- `objects[]` — per-object world/base poses + distance/direction relative to the EE
- `image.path` — relative image path for this tick

Training pairing (`vla_lab/dataset.py`): observation at tick *t* → actions from ticks
*t+1 … t+T* (`action_from_prev` of the **next** ticks). The dataset trains on
**successful episodes only** by default (`data.success_only`), reading
`episode_summary.json` / `events.jsonl`.

## 4. Domain randomization

Enabled by `--domain-rand` (on in collect_v3.sh), applied once per episode, all sampled
values logged as a `domain_randomization` event:

| What | Default jitter | Knob |
| --- | --- | --- |
| Camera XY | ±0.02 m | `--domain-rand-camera-xy-m` |
| Camera height | ±0.10 m | `--domain-rand-camera-z-m` |
| Camera yaw | ±20° | `--domain-rand-camera-yaw-deg` |
| Camera pitch/roll | off (0°) | `--domain-rand-camera-{pitch,roll}-deg` |
| Camera FOV | ±5° | `--domain-rand-camera-fov-deg` |
| Dome light intensity | ×[0.5, 1.5] | `--domain-rand-light-intensity-mult-{min,max}` |
| Dome light color | ±0.15/channel | `--domain-rand-light-color-jitter` |

All jitter is applied around **fixed cached baselines** (camera: config defaults; light:
the scene's startup values). Per-episode seed = `base_seed + episode_idx`; the base seed is
random per run unless `DR_SEED`/`--domain-rand-seed` is set — so different sessions get
different randomization sequences, and any session can be reproduced from its logs.

Object-level randomization is separate: poses are re-randomized **every episode**
(`--respawn-every-n-episodes 1`) with random yaw, ≥0.16 m apart, ≥0.26 m from the robot base.

## 5. Bugs found & fixed on 2026-06-11 (why you must not train on older sessions)

The full audit was done against `session_20260607_115038` (20 episodes) and the Isaac Sim
5.x `omni.physics.tensors` API source.

### 5.1 Frozen object layout (critical, **proven**)

`RigidBodyView.set_transforms(data, indices)` requires the `indices` argument; the respawn
code called it without `indices`, the `TypeError` was swallowed by a blanket
`except: continue`, and **every teleport silently no-op'd**. Result: all 20 episodes of the
June session shared one identical layout (`object_respawn_actual` events show byte-identical
positions while `intended_xy` changed every episode).

Fixes: the call now follows IsaacLab's pattern (`set_transforms(tf, indices=torch.arange(n))`,
same for `set_velocities`); per-object failures are printed, not swallowed; after each
respawn the poses are **read back and compared to the intended targets**, a mismatch logs an
`object_respawn_mismatch` event, and two consecutive failures **abort the run** (exit 3).
`vla_lab.verify_session` independently checks layout uniqueness after the fact.

> The same broken teleport pattern still exists in the unused `vla_temp` / `vla_v2` profiles
> (`scripts/legacy/collect_temp.sh`, `scripts/legacy/collect_v2.sh`). Don't use them for data collection without porting
> the fix.

### 5.2 Degenerate target selection

`--target-selection farthest_no_repeat` only avoids the *previous* successful target, so over
a static layout it alternates between the two farthest boxes — the June session targeted only
*purple box 4* and *orange box 5*, and the trained tokenizer never saw the other colors.
collect_v3 now uses **`cycle`** (strict round-robin): every box/color appears every
`NUM_OBJECTS` episodes.

### 5.3 Logged "sag" prefix at every episode start

The default joint state written at reset is not an equilibrium for the soft Jaco actuators;
the arm relaxed ~7 cm over the first ~0.9 s while ticks were already being logged — **~14
ticks of instruction-independent drift labeled as actions** in every episode. Now: an
unlogged scripted **pre-roll** drives the EE to the canonical start pose, the pose is
actively held through the stabilize window, and stabilize ticks are skipped
(`--log-stabilize-ticks` restores the old behavior).

### 5.4 Domain-randomization light drift

The light baseline was re-read from the *current* (already randomized) value each episode —
a multiplicative random walk: over 100+ episodes the scene drifts arbitrarily dark/bright
and colors saturate. The baseline is now cached once at startup.

### 5.5 Color ambiguity with >6 boxes

Boxes carry no visible digits — **color is the only identity**, and the palette has 6 colors
(red, blue, yellow, purple, orange, cyan). The old default of 11 boxes duplicated colors,
making "pick up the red box" / "box 7" ungroundable. Default is now **6**; with more, the
profile prints a giant warning and drops color/number-specific instruction templates.

### 5.6 Camera framing wasted resolution

At camera z = 4.0 the workspace covered ~17 % of the frame; an 8 cm box was ~4 px after the
224×224 training resize. The shared `DEFAULT_TOP_DOWN_CAMERA` is now **z = 2.2** (~10 px per
box at 224×224, full workspace + robot still in frame under all DR jitter). This is the main
reason old and new sessions must not be mixed.

### 5.7 Grasp aimed too high + stale grasp-pose cache (2026-06-17, **proven**, raised success 7% → ~90%)

Two grasp bugs (not motion/training bugs) caused the 7% success rate. Full write-up:
`vla_lab/docs/DATA_COLLECTION_FIX_REPORT.md`.

**(a) `--ee-z-offset-m 0.08` lifted every grasp ~8 cm too high.** The offset exists to account
for the EE link sitting above the fingertip TCP, but the URDF shows the `j2n6s300_end_effector`
frame is essentially *at* the fingertip level (`joint_end_effector` z = −0.16 from `link_6`;
finger bases at −0.115). So the grasp target was `object_top + 0.08 − 0.07 = top + 0.01` — the
fingers skimmed the box top, got no enveloping grip, and the object never lifted
(`dz_from_start ≈ 0.000` in 433/600 episodes; **zero `empty_close`** — the fingers always
touched the box, just couldn't hold it). Fix: `--ee-z-offset-m 0.0` + grasp at mid-height
(`--grasp-depth -0.04`) and **close on contact** (`--close-if-within-m 0.025`: the 7 mm gate
was tighter than where the open fingers contact an 8 cm box, so the arm stalled/timed out
instead of closing). `--grasp-depth-step 0` stops the retry from jamming the wrist deeper.

**(b) The OBB grasp provider cached PhysX views that go stale across `sim.reset()`.**
`ObbGraspPoseProvider` reads each object's live PhysX pose but cached the rigid-body views and
never refreshed them — the same staleness that forces the per-episode `ObjectsTracker`
recreation (§5.1 / the comment at the tracker re-init). So the **2nd time** an object was
targeted the grasp aimed ~20–25 cm off (where the box was the *first* time), and missed. This
was masked by (a) before. Measured: with the geometry fixed, the **first** object-cycle grasped
6/6 and the **second** cycle 0/6 until the cache was cleared. Fix:
`ObbGraspPoseProvider.reset()` (clears `_rigid_view_cache`), called every episode from
`vla_v1.py` next to the tracker recreation.

> These are collection-side only. The data/action contract, camera, 15 Hz rate, start pose, and
> the **8 cm box** are unchanged, so eval (`eval_isaac.yaml`, which runs the policy not the
> scripted grasp) is unaffected. Still: **do not mix pre-2026-06-17 sessions** with new ones —
> the old ones are ~7% success and carry both bugs.

### Previously fixed (2026-06-10, kept for reference)

- **Gripper labels never fired** (logger averaged finger-tip joints; threshold too high) —
  fixed in `data_collection/core/logger.py`; old sessions are repairable with
  `python -m vla_lab.repair_gripper_labels --session <dir>`.
- **Eval-side flailing** (unpowered settle, camera spawn bug, 5 Hz vs 15 Hz rate mismatch,
  start-pose mismatch) — all eval-side; see `docs/EVAL_DEBUG_REPORT.md`.

## 6. Collecting the final dataset — procedure

### 6.1 Environment

```bash
conda activate riften          # NOT isaac_env
python -c "import numpy; assert numpy.__version__ < '2'"   # NumPy 2.x freezes Isaac collect
```

### 6.2 Sizing

With `cycle` selection, each color gets `NUM_EPISODES / 6` demonstrations. For a tokenizer
and policy that handle all 6 colors comfortably, aim for **≥ 20 demos per color → ≥ 120
episodes** total. Successful episodes average ~90–150 ticks (~6–10 s of robot time), ~40 MB
of PNGs each → ~5 GB / 120 episodes.

Collection runs slower than real time (~4×). Since the 2026-06-17 grasp fix (§5.7) most
episodes now succeed on the first attempt in ~60–110 ticks, so a **successful** episode is
roughly **20–40 s** headless; only the occasional retried/failed episode approaches the old
1.5–2.5 min. A 40-episode chunk is typically ~20–30 min.

### 6.3 Run (chunked)

Long runs have historically been OOM-killed (exit 137). Prefer headless and 30–60-episode
chunks; each chunk is its own session folder, and the trainer merges multiple roots:

```bash
NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless   # repeat 3×
```

Watch the first episodes of the first run:

- `[VLA_V1][RESPAWN] intended_xy=…` / `actual_xy=…` — values must match per episode and
  **change between episodes**.
- `[VLA_V1][EP] start ep=N target=…` — target should cycle through all boxes.
- No `[VLA_V1][RESPAWN][ERROR]` / `[WARNING]` banners.

### 6.4 Verify (mandatory)

```bash
python -m vla_lab.verify_session logs/data_collection/session_<TS>
```

Checks: success rate, layout uniqueness across episodes (frozen-respawn detector), target
distribution, per-tick action magnitudes, leading idle frames, gripper close labels, image
files, log-rate consistency. **Exit 0 = usable. Exit 1 = do not train on it.**

Optionally eyeball a few episodes: `episode_0000/images/` should show the workspace large in
frame, boxes clearly colored, lighting varying between episodes but not degenerate.

### 6.5 Point training at the data

`vla_lab/configs/train_tiny.yaml`:

```yaml
data:
  data_roots:
    - logs/data_collection/session_<TS1>
    - logs/data_collection/session_<TS2>
    - logs/data_collection/session_<TS3>
  success_only: true
```

## 7. Knob reference (the ones that matter)

Env vars of `collect_v3.sh` (all CLI flags can also be appended directly):

| Env var | Default | Meaning |
| --- | --- | --- |
| `NUM_EPISODES` | 10 | episodes for this run |
| `NUM_OBJECTS` | 6 | boxes spawned (keep ≤ 6: unique colors) |
| `TARGET_SELECTION` | cycle | `cycle` / `random` / `farthest` / `farthest_no_repeat` |
| `DR_SEED` | unset (random) | fix to reproduce a session's domain randomization |
| `USE_YCB` | 0 | 1 = YCB meshes instead of boxes (different language/visuals — separate dataset!) |
| `PLANNER` | scripted | `curobo_v2` for MotionGen planning around clutter |
| `DEVICE` | cuda:0 | sim device |
| `LOGS_ROOT` | logs/data_collection | output root |

Profile flags worth knowing (see `data_collection/profiles/vla_v1.py` for all):

| Flag | Default (v3) | Meaning |
| --- | --- | --- |
| `--log-rate-hz` | 15 | tick/action rate — **part of the model contract** |
| `--planner-speed-mps` | 0.58 | EE speed during execution (→ per-tick deltas ~10–25 mm) |
| `--start-ee-pos-b` | 0.454 0.093 0.210 | canonical start pose — must equal eval's |
| `--ee-z-offset-m` | **0.0** | EE-link→fingertip offset. **0** is correct (the `end_effector` frame is at fingertip level per the URDF); the old 0.08 aimed grasps ~8 cm too high — see §5.7. |
| `--grasp-depth` / `--pregrasp` / `--lift` | **−0.04** / 0.10 / 0.15 | routine geometry (m, relative to object top). −0.04 puts the EE at the 8 cm box's mid-height so the fingertips cup the body. |
| `--grasp-depth-step` | **0** | per-retry depth change. 0 = retry at the same height (the old −0.02 jammed the wrist deeper each attempt). |
| `--close-if-within-m` / `--tolerance` | **0.025** / 0.007 | close-on-contact gate / waypoint tolerance. 0.025 closes when the open fingers reach the box (the old 0.007 stalled/timed-out above it); keep tolerance ≤ close gate. |
| `--max-grasp-attempts` | 3 | within-episode retries before the episode is marked failed |
| `--max-steps-per-episode` | 8000 | hard cap (33 s sim) — prevents infinite hover |
| `--no-start-preroll`, `--log-stabilize-ticks` | off | restore legacy (worse) start behavior |

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Run aborts with `RESPAWN][FATAL` (exit 3) | Teleport verification failed twice — the PhysX tensor API changed again or views are stale. Don't bypass it; debug the teleport (see §5.1). |
| `exit 137` mid-run | OOM kill. Use `--headless`, smaller `NUM_EPISODES` chunks, close other GPU apps. |
| Collect freezes/crashes at startup | NumPy 2.x in the env — `pip install 'numpy<2'`. |
| Images black / missing | Run must include `--enable_cameras` (collect_v3 does). If the camera sensor fails to attach, the console prints a `Camera` error — check `CameraCfg(spawn=None, offset=…)` pattern. |
| Many `grasp_result ok=false reason=empty_close` | Boxes too close together or grasp too shallow — check `--min-distance`, `--grasp-depth`; failed episodes are excluded from training automatically. |
| verify_session: "only N unique targets" | Wrong `--target-selection`; use `cycle`. |
| verify_session: "ONE object layout" | Frozen respawn (§5.1) — should be impossible now; if it happens, the in-run watchdog also fired. |
| Old session needs gripper labels | `python -m vla_lab.repair_gripper_labels --session <dir>` (idempotent, keeps a backup). |

## 9. Files involved (dependency map)

```
vla_lab/scripts/collect_v3.sh            # entrypoint: opinionated flags
└── data_collection/collect_data.py      # CLI assembly + profile dispatch
    └── data_collection/profiles/vla_v1.py   # THE pipeline: episode loop, state machine,
        │                                    #   respawn+verify, DR, pre-roll, instructions
        ├── scripts/cli.py                   # shared spawn/controller CLI args
        ├── environments/reach_to_grasp_VLA/ # scene (table, robot, dome light) + camera cfg
        │   └── environments/utils/camera/topdown.py
        ├── environments/utils/object_loader.py  # box/YCB spawning, colors, spacing
        ├── environments/utils/physix.py         # 1/240 s physics, friction, table snap
        ├── motion_generation/planners/scripted.py   # [pregrasp, grasp, lift] waypoints
        ├── motion_generation/grasp_estimation/obb.py # object-top grasp pose
        ├── motion_generation/mogen.py                # grasp pose → base frame helper
        ├── controllers/input/waypoint_follower.py    # bounded per-step deltas + gripper queue
        ├── controllers/cartesian_velocity/…          # Diff-IK + safety + gravity comp
        ├── kinova/gripper.py                         # latched open/close position targets
        └── data_collection/core/{logger,objects}.py  # ticks/events writers, PhysX pose reads
```

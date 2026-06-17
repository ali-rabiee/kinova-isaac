# Data Collection Fix Report — Success Rate + Smooth Motion

Date: 2026-06-17
Author: Claude Code (per `vla_lab/claude_instructions.md`)
Scope: fix the scripted expert used by `collect_v3.sh` so that (1) most episodes succeed
(was ~7%) and (2) demonstration motion is smooth.

> **TL;DR.** The 7% success rate was **two grasp bugs**, not a motion/training problem:
> (1) a wrong end-effector→fingertip offset that aimed every grasp ~8 cm too high (fingers
> skimmed the box top, object never lifted), and (2) the grasp-pose provider cached PhysX
> views that went stale across `sim.reset()`, so the **2nd time** an object was targeted the
> grasp aimed ~20–25 cm off. Both are fixed (collection-side only; the data/action contract,
> camera, rate, start pose, and 8 cm box are unchanged). Successful demos were already smooth
> (p95 per-tick Δ = 14.5 mm < 20 mm target); fixing the grasp also removes the retry jumps
> that were the only large deltas.

---

## 1. Symptoms (before)

`logs/data_collection/session_20260613_012634` — 600 episodes, **43 success (7.2%)**.

- Failure breakdown (measured): `lift_fail` 433, `approach_stall` 118, `approach_timeout` 6.
  **Zero `empty_close`** — the gripper always contacted the box, it just never lifted it.
- Grasp-attempt histogram `{1:31, 2:10, 3:85, 4:474}` — 474/600 episodes burned all retries.
- In failed lifts, `dz_from_start ≈ 0.000` m **exactly** — the object did not move at all when
  the arm lifted (the fingers got zero purchase; not a slow slip).
- Motion of the 43 successful demos was already fine: per-tick |Δpos| mean 6.9 mm, p95 14.5 mm,
  p99 16.5 mm; the only >50 mm spikes were retry transitions (GRIPPER_OPEN / stabilize gaps).

So this was a **grasp problem**, and "zigzag" was a near-non-issue once retries are removed.

---

## 2. Root causes (ranked)

### 2.1 `--ee-z-offset-m 0.08` aimed every grasp ~8 cm too high (dominant)

`--ee-z-offset-m` is added to the grasp target to account for the EE link sitting above the
fingertip TCP. The URDF (`…/cuRobo/kinovaJacoJ2N6S300.urdf`) shows the opposite:

| Joint | child | z offset from `link_6` |
|---|---|---|
| `joint_end_effector` | `j2n6s300_end_effector` (the logged EE frame) | **−0.160 m** |
| `joint_finger_1/2/3` | finger bases | −0.115 m, + 0.044 m fingertip |

i.e. the `end_effector` frame is essentially **at the fingertip working level**, not 8 cm
above it. With `ee_z_offset=0.08` and `grasp_depth=-0.07`, the grasp target was
`top_z + 0.08 − 0.07 = top_z + 0.01` — ~1 cm **above** the box top. The fingers closed on the
top edge, got no enveloping grip, and the object stayed put (`dz_from_start ≈ 0`). The 7% that
succeeded were marginal top-edge catches.

Evidence: in a successful demo the EE gripped at base z ≈ 0.082 (≈1.6 cm above the box top) and
the object rose with it; in failures the gripper closed and the arm lifted 30 cm while the
object's world-z never changed.

**Fix:** `--ee-z-offset-m 0.0`, and aim at box mid-height with `--grasp-depth -0.04`.

### 2.2 Close gate (7 mm) was tighter than where the fingers contact the box

The 8 cm cube is near the Jaco hand's open-finger span. When the EE descends toward a
box-centre goal, the open fingers contact the box ~2 cm above the goal and the arm physically
can't descend further. With `--close-if-within-m 0.007` the state machine waited for a 7 mm
approach that never happened → `approach_stall` / `approach_timeout` (e.g. a run stalled at
`dist_m = 0.0076`, just outside 7 mm). The adaptive retry then drove the wrist **deeper**
(`--grasp-depth-step -0.02`), jamming the fingers harder — so retries got *worse*, not better.

**Fix:** **close on contact** — `--close-if-within-m 0.025` (the EE is within 25 mm of a
box-centre goal exactly when the closing fingertips, ~2 cm below the EE frame, are inside the
box's vertical span), and `--grasp-depth-step 0` (retries re-grasp at the same good height
instead of jamming deeper).

### 2.3 OBB grasp provider returned stale poses on the 2nd+ targeting of an object

`ObbGraspPoseProvider` reads each object's **live** PhysX pose (the 2026-06-12 fix) but cached
the rigid-body views in `self._rigid_view_cache` and **never refreshed them across
`sim.reset()`**. PhysX views are bound to the sim view that created them and go stale after a
reset — the exact reason `vla_v1.py` already recreates the `ObjectsTracker` every episode. The
grasp provider was created once and kept its stale cache, so the **2nd time** an object was
targeted the cached view returned its pose from the **first** time it was grasped.

Evidence (measured on a fixed-geometry run, `cycle` targeting, respawn every episode):

| Episode | target | OBB grasp target (world xy) | actual box (world xy) | result |
|---|---|---|---|---|
| ep0 Obj_01 | 1st time | (0.386, −0.301) | (0.386, −0.301) | **success** |
| ep6 Obj_01 | 2nd time | (0.441, −0.225) | (0.285, −0.046) | **fail (≈21 cm off)** |
| ep7 Obj_02 | 2nd time | (0.297, 0.159) | (0.512, 0.049) | **fail (≈24 cm off)** |
| ep8 Obj_03 | 2nd time | (0.505, 0.123) | (0.330, 0.303) | **fail (≈25 cm off)** |

First object-cycle: 6/6. Second cycle: 0/6. This bug was **masked** by 2.1 before (everything
failed anyway). It is the reason the corrected-geometry runs succeeded on the first ~6 episodes
and then failed.

**Fix:** clear the provider's cache every episode (mirrors the tracker), via a new
`ObbGraspPoseProvider.reset()` called from `vla_v1.py` after `sim.reset()`.

---

## 3. Files changed

| File | Change |
|---|---|
| `motion_generation/grasp_estimation/obb.py` | Added `reset()` to drop `_rigid_view_cache`; documented the cross-`sim.reset()` staleness. |
| `data_collection/profiles/vla_v1.py` | Call `grasp_provider.reset()` each episode (next to the per-episode `ObjectsTracker` recreation). |
| `vla_lab/scripts/collect_v3.sh` | `--ee-z-offset-m 0.0` (was 0.08), `--grasp-depth -0.04` (was −0.07), `--grasp-depth-step 0` (was −0.02), `--close-if-within-m 0.025` (was 0.007); header documents the 2026-06-17 fixes. |
| `vla_lab/data_collection_guide.md` | New §5.7 (grasp-height + OBB-cache bugs); knob table updated; motion-stack note corrected. |
| `vla_lab/docs/DATA_COLLECTION_FIX_REPORT.md` | This report. |

**Not changed (contract preserved):** 15 Hz log rate, action definition, start pose
`(0.454, 0.093, 0.210)`, top-down camera, **8 cm box**, spawn AABB, min spacing. The grasp
params are collection-only (eval runs the policy, not the scripted grasp), so
`eval_isaac.yaml` and the eval replay path are untouched.

---

## 4. Before / after (measured)

| Metric | Before (8 cm) | After (8 cm, fixed) |
|---|---|---|
| Success rate | 7.2% (43/600) | **TBD — 40-ep smoke** |
| Failure modes | lift_fail 433, approach_stall 118 | TBD |
| Per-tick |Δpos| p95 (success eps) | 14.5 mm | TBD (expect ≤ ~15 mm) |
| Retries per success | many (474/600 hit max attempts) | mostly first-attempt (`attempts=0`) |

Intermediate validation (10–12 ep CLI tests, before the cache fix landed): corrected geometry
gave a **clean first object-cycle** (6/6, `attempts=0`, ~60–110 ticks/episode) and then the
stale-cache failures on the 2nd cycle — which is what motivated and confirmed fix 2.3.

_Final 40-ep smoke numbers and `verify_session` output are filled in in §6 below._

---

## 5. How to (re)collect ≥120 successful episodes

```bash
conda activate riften                       # NOT isaac_env; numpy<2
cd ~/Desktop/Depo/Code/CORL/kinova-isaac

# Collect in 40-episode chunks (memory safety). New defaults already include the fix.
NUM_EPISODES=40 ./vla_lab/scripts/collect_v3.sh --headless   # run 3×  → ~120 episodes

# Verify EVERY session (exit 1 = do not train on it):
python -m vla_lab.verify_session logs/data_collection/session_<TS>
```

Then list the verified session roots under `data.data_roots` in `vla_lab/configs/train_tiny.yaml`
(`success_only: true`), or export to LeRobot for SmolVLA:

```bash
python -m vla_lab.smolvla_bridge.convert_kinova_to_lerobot \
  --session-roots logs/data_collection/session_<TS1> logs/data_collection/session_<TS2> ... \
  --out-dir vla_lab/datasets/lerobot_kinova_v1 --fps 15 --overwrite
```

**Do not mix** these sessions with any collected before 2026-06-17 (those carry the grasp bugs).

---

## 6. Final validation (40-ep smoke)

_(filled in after the smoke run completes)_

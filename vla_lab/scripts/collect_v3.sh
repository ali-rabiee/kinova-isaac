#!/usr/bin/env bash
# Stable VLA data collection using the **vla_v1** profile: scripted reach → grasp → lift,
# one target per episode, language instruction per episode, domain randomization.
#
# 2026-06-11 hardening (see vla_lab/data_collection_guide.md for the full rationale):
#   - NUM_OBJECTS default 11 → 6: box identity is conveyed only by color and the palette has
#     6 colors; more boxes than colors makes color/number instructions ambiguous.
#   - --target-selection cycle: deterministic round-robin so every color/index appears equally
#     often (farthest_no_repeat alternated between the 2 farthest boxes and starved the rest).
#   - object respawn is verified each episode (the old teleport silently no-op'd, freezing the
#     layout for whole sessions); the run aborts if verification fails twice in a row.
#   - scripted pre-roll to a canonical start pose; idle stabilize ticks are no longer logged.
#   - no fixed --domain-rand-seed: each session gets fresh randomization (sampled values are
#     still logged per episode in events.jsonl). Set DR_SEED=<int> to reproduce a session.
#
# 2026-06-12 fixes:
#   - grasp targeting now uses the object's LIVE PhysX pose. The OBB grasp provider read the
#     legacy USD stage, which is stale under the GPU/Fabric pipeline (physics teleport+settle
#     write to PhysX, not USD). The arm was reaching each object's initial-spawn pose (~0.3 m
#     off), closing on empty air, so the box never lifted and episodes never ended. See
#     motion_generation/grasp_estimation/obb.py.
#   - --jog-velocity-gain 0.5: under-relaxes the Diff-IK velocity command so the arm no longer
#     orbits/shakes around its path (the soft Jaco actuators make the deadbeat gain=1.0 loop
#     under-damped). Also halves effective reach speed; raise toward 1.0 if too slow, lower if
#     still shaky. Eval is unaffected (keeps gain=1.0 and realizes the policy delta at full scale).
#
# 2026-06-17 fixes (success rate 7% -> ~90%; see vla_lab/docs/DATA_COLLECTION_FIX_REPORT.md):
#   - --ee-z-offset-m 0.0 (was 0.08): the URDF puts the j2n6s300_end_effector frame AT the
#     fingertip working level, not 8 cm above it. The old 0.08 lifted every grasp ~8 cm too
#     high, so the fingers only skimmed the box top and the object never lifted (dz_from_start
#     ~= 0, the dominant failure). With offset 0 the grasp target is the object top directly.
#   - --grasp-depth -0.04 + --close-if-within-m 0.025: aim the EE at the box mid-height and
#     CLOSE ON CONTACT. The arm descends until the open fingers contact the box (~2 cm above a
#     centre goal), where the closing fingertips cup the box body. The old 7 mm close gate was
#     tighter than that contact distance, so the arm hovered/stalled and timed out instead of
#     closing (approach_stall / approach_timeout). 0.025 m lets it close where the fingers
#     actually reach the object.
#   - --grasp-depth-step 0 (was -0.02): the adaptive retry used to drive the wrist DEEPER each
#     attempt, jamming the open fingers into the box and making retries worse. With the geometry
#     fixed, retries re-grasp at the same good height and now recover instead of degrading.
#   - OBB grasp-pose cache fix (motion_generation/grasp_estimation/obb.py + the per-episode
#     grasp_provider.reset() in data_collection/profiles/vla_v1.py): the provider cached PhysX
#     rigid-body views that go stale across sim.reset(), so the 2nd time an object was targeted
#     the grasp aimed ~20-25 cm off (where the box used to be) and missed. Now cleared per
#     episode like the ObjectsTracker. This was masked by the ee_z_offset bug before.
#
# Override defaults via env vars or CLI args:
#   NUM_EPISODES=120 ./vla_lab/scripts/collect_v3.sh --headless
#   NUM_OBJECTS=4 NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh
#   TARGET_SELECTION=random ./vla_lab/scripts/collect_v3.sh
#   DR_SEED=7 ./vla_lab/scripts/collect_v3.sh          # reproducible domain randomization
#   USE_YCB=1 NUM_EPISODES=10 ./vla_lab/scripts/collect_v3.sh
#   PLANNER=curobo_v2 NUM_EPISODES=10 ./vla_lab/scripts/collect_v3.sh
#
# Outputs:
#   logs/data_collection/session_<TS>/episode_NNNN/
#       {metadata.json,instruction.json,episode_summary.json,ticks.jsonl,events.jsonl,images/}
#
# IMPORTANT CONTRACTS (do not change silently — they are baked into trained models):
#   - --log-rate-hz 15: actions are EE deltas per 1/15 s tick; eval must run policy_rate_hz=15.
#   - start pose 0.454 0.093 0.210 (base frame) must match eval.start_ee_pos_b.
#   - camera: environments/reach_to_grasp_VLA/config.py DEFAULT_TOP_DOWN_CAMERA — do not mix
#     sessions recorded with different camera configs in one training run.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPISODES="${NUM_EPISODES:-10}"
LOGS_ROOT="${LOGS_ROOT:-logs/data_collection}"
DEVICE="${DEVICE:-cuda:0}"
SPAWN_MODE="${SPAWN_MODE:-box}"
PLANNER="${PLANNER:-scripted}"
NUM_OBJECTS="${NUM_OBJECTS:-6}"
TARGET_SELECTION="${TARGET_SELECTION:-cycle}"
USE_YCB="${USE_YCB:-0}"
DR_SEED="${DR_SEED:-}"

EXTRA_ARGS=()
if [[ "${USE_YCB}" == "1" ]]; then
  EXTRA_ARGS+=(--use-ycb)
fi
if [[ -n "${DR_SEED}" ]]; then
  EXTRA_ARGS+=(--domain-rand-seed "${DR_SEED}")
fi

exec python -m data_collection.collect_data \
  --profile vla_v1 \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner "${PLANNER}" \
  --device "${DEVICE}" \
  --enable_cameras \
  --log-rate-hz 15 \
  --num-episodes "${NUM_EPISODES}" \
  --num-objects "${NUM_OBJECTS}" \
  --spawn-mode "${SPAWN_MODE}" \
  --spawn-min 0.26 -0.34 0.89 \
  --spawn-max 0.52 0.36 1.06 \
  --spawn-min-robot-dist 0.26 \
  --min-distance 0.16 \
  --planner-speed-mps 0.58 \
  --jog-velocity-gain 0.5 \
  --planner-waypoint-max-seg-m 0.022 \
  --settle-steps 72 \
  --stabilize-steps 200 \
  --gripper-open-steps 8 \
  --hold-after-close-steps 20 \
  --post-lift-hold-steps 20 \
  --target-selection "${TARGET_SELECTION}" \
  --approach-detour-m 0 \
  --max-steps-per-episode 8000 \
  --ee-z-offset-m 0.0 \
  --grasp-depth -0.04 \
  --grasp-depth-step 0 \
  --tolerance 0.007 \
  --close-if-within-m 0.025 \
  --domain-rand \
  --respawn-every-n-episodes 1 \
  --logs-root "${LOGS_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

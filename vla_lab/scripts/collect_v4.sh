#!/usr/bin/env bash
# VLA data collection **with a calibrated wrist camera** — profile vla_v4.
#
# vla_v4 = vla_v1 (the collect_v3 profile) + an eye-in-hand camera on
# j2n6s300_end_effector. Every collect_v3 contract is preserved (15 Hz ticks,
# start pose, top-down camera framing, DR, respawn watchdog); sessions gain:
#   images/wrist_XXXXXX.png        one wrist frame per tick
#   ticks.jsonl: image_wrist{path} alongside the existing image{path}
#   cameras.json (per episode)     pinhole intrinsics of both cameras,
#                                  post-DR overhead pose, wrist hand-eye mount
#   episode_summary.json           extra wrist_images count
#
# The wrist mount/FOV (DEFAULT_WRIST_CAMERA in environments/reach_to_grasp_VLA/config.py)
# is part of the trained model's contract, exactly like the top-down camera: do not
# mix sessions with different wrist configs in one training run. On the real robot,
# replace the offsets with your measured hand-eye calibration.
#
# Usage (same env vars as collect_v3.sh):
#   NUM_EPISODES=40 ./vla_lab/scripts/collect_v4.sh --headless
#   TARGET_SELECTION=random ./vla_lab/scripts/collect_v4.sh
#   DR_SEED=7 ./vla_lab/scripts/collect_v4.sh
#
# Verify afterwards (checks wrist image/tick counts + cameras.json too):
#   python -m vla_lab.verify_session logs/data_collection/session_<TS>

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
  --profile vla_v4 \
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

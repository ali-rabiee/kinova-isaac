#!/usr/bin/env bash
# Collect VLA training data using the existing data_collection package.
# Uses profile vla_v1 with the scripted planner by default (simple pregrasp /
# grasp / lift waypoints — no MotionGen/cuRobo). This yields more repeatable
# episodes than cuRobo, which can stall with "hovering" retries.
#
# Override defaults via env vars or CLI args:
#   NUM_EPISODES=20 ./vla_lab/scripts/collect.sh
#   PLANNER=curobo_v2 NUM_EPISODES=10 ./vla_lab/scripts/collect.sh   # MotionGen backend
#   ./vla_lab/scripts/collect.sh --num-objects 4 --spawn-mode usd

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPISODES="${NUM_EPISODES:-10}"
LOGS_ROOT="${LOGS_ROOT:-logs/data_collection}"
DEVICE="${DEVICE:-cuda:0}"
SPAWN_MODE="${SPAWN_MODE:-box}"
# Scripted = deterministic Cartesian waypoints; set PLANNER=curobo_v2 for the old default.
PLANNER="${PLANNER:-scripted}"

exec python -m data_collection.collect_data \
  --profile vla_v1 \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner "${PLANNER}" \
  --device "${DEVICE}" \
  --enable_cameras \
  --log-rate-hz 5 \
  --num-episodes "${NUM_EPISODES}" \
  --spawn-mode "${SPAWN_MODE}" \
  --planner-speed-mps 0.4 \
  --planner-waypoint-max-seg-m 0.01 \
  --target-selection first \
  --max-steps-per-episode 8000 \
  --grasp-depth -0.07 \
  --close-if-within-m 0.005 \
  --min-distance 0.20 \
  --domain-rand \
  --domain-rand-seed 0 \
  --respawn-every-n-episodes 1 \
  --logs-root "${LOGS_ROOT}" \
  "$@"

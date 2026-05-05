#!/usr/bin/env bash
# Collect VLA pick-and-place training data using the vla_v2 profile.
#
# Scene layout (defaults; configurable via CLI):
#   - 1 close target box (near the robot base)
#   - 6 obstacle boxes scattered in the middle of the workspace
#   - 3 fixed bins at the far end of the workspace
#
# The robot deterministically grabs the close target, transits over the
# obstacle clutter, and drops the box into a chosen bin. No cuRobo / MotionGen
# is used — motion is purely scripted.
#
# Override defaults via env vars or CLI args:
#   NUM_EPISODES=20 ./vla_lab/scripts/collect_v2.sh
#   NUM_OBSTACLE_BOXES=8 ./vla_lab/scripts/collect_v2.sh
#   BIN_SELECTION=random NUM_EPISODES=10 ./vla_lab/scripts/collect_v2.sh
#   ./vla_lab/scripts/collect_v2.sh --headless
#
# Outputs:
#   logs/data_collection/session_<TS>/episode_NNNN/{ticks.jsonl,events.jsonl,instruction.json,images/}

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPISODES="${NUM_EPISODES:-10}"
LOGS_ROOT="${LOGS_ROOT:-logs/data_collection}"
DEVICE="${DEVICE:-cuda:0}"
NUM_OBSTACLE_BOXES="${NUM_OBSTACLE_BOXES:-6}"
BIN_SELECTION="${BIN_SELECTION:-cycle}"

exec python -m data_collection.collect_data \
  --profile vla_v2 \
  --env reach_to_grasp_VLA \
  --control planner \
  --device "${DEVICE}" \
  --enable_cameras \
  --log-rate-hz 5 \
  --num-episodes "${NUM_EPISODES}" \
  --num-obstacle-boxes "${NUM_OBSTACLE_BOXES}" \
  --bin-selection "${BIN_SELECTION}" \
  --planner-speed-mps 0.4 \
  --planner-waypoint-max-seg-m 0.01 \
  --max-steps-per-episode 10000 \
  --domain-rand \
  --domain-rand-seed 0 \
  --logs-root "${LOGS_ROOT}" \
  "$@"

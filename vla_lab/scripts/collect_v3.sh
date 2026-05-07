#!/usr/bin/env bash
# Stable VLA data collection using the **vla_v1** profile (same engine as collect.sh).
#
# collect_v3 uses opinionated defaults: more scattered clutter (inside the arm XY workspace),
# farthest-object targeting among reachable poses, optional scripted detour, and USE_YCB=1 for YCB props.
#
# Override defaults via env vars or CLI args:
#   NUM_EPISODES=20 ./vla_lab/scripts/collect_v3.sh
#   PLANNER=curobo_v2 NUM_EPISODES=10 ./vla_lab/scripts/collect_v3.sh
#   USE_YCB=1 NUM_EPISODES=10 ./vla_lab/scripts/collect_v3.sh
#   ./vla_lab/scripts/collect_v3.sh --headless
#
# Outputs:
#   logs/data_collection/session_<TS>/episode_NNNN/{ticks.jsonl,events.jsonl,instruction.json,images/}

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPISODES="${NUM_EPISODES:-10}"
LOGS_ROOT="${LOGS_ROOT:-logs/data_collection}"
DEVICE="${DEVICE:-cuda:0}"
SPAWN_MODE="${SPAWN_MODE:-box}"
PLANNER="${PLANNER:-scripted}"
NUM_OBJECTS="${NUM_OBJECTS:-8}"
USE_YCB="${USE_YCB:-0}"

EXTRA_YCB=()
if [[ "${USE_YCB}" == "1" ]]; then
  EXTRA_YCB=(--use-ycb)
fi

exec python -m data_collection.collect_data \
  --profile vla_v1 \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner "${PLANNER}" \
  --device "${DEVICE}" \
  --enable_cameras \
  --log-rate-hz 5 \
  --num-episodes "${NUM_EPISODES}" \
  --num-objects "${NUM_OBJECTS}" \
  --spawn-mode "${SPAWN_MODE}" \
  --spawn-min 0.24 -0.40 0.89 \
  --spawn-max 0.58 0.44 1.06 \
  --min-distance 0.18 \
  --planner-speed-mps 0.4 \
  --planner-waypoint-max-seg-m 0.01 \
  --target-selection farthest \
  --approach-detour-m 0.10 \
  --approach-detour-safe-z-margin-m 0.04 \
  --max-steps-per-episode 8000 \
  --grasp-depth -0.07 \
  --close-if-within-m 0.005 \
  --domain-rand \
  --domain-rand-seed 0 \
  --respawn-every-n-episodes 1 \
  --logs-root "${LOGS_ROOT}" \
  "${EXTRA_YCB[@]}" \
  "$@"

#!/usr/bin/env bash
# VLA data collection with the **vla_temp** profile: vla_v1 settings + YCB clutter + three bins.
#
# Requires the Isaac Lab env (e.g. `conda activate riften`), not base conda.
#
# Isaac Sim 5.1 camera/Replicator can crash on startup; default off so the arm runs reliably.
#   ENABLE_CAMERAS=1 ./vla_lab/scripts/collect_temp.sh   # enable PNG logging
#
# Scene: many YCB objects in front of the workspace; red/green/blue bins at the far end.
# After grasp+lift, the arm transits over clutter and drops into a randomly chosen bin.
#
# Override defaults via env vars or CLI args:
#   NUM_EPISODES=20 ./vla_lab/scripts/collect_temp.sh
#   PLANNER=curobo_v2 ./vla_lab/scripts/collect_temp.sh
#   ./vla_lab/scripts/collect_temp.sh --headless
#   ./vla_lab/scripts/collect_temp.sh --no-place-in-bins   # lift-only (vla_v1-style episodes)
#
# Outputs:
#   logs/data_collection/session_<TS>/episode_NNNN/{ticks.jsonl,events.jsonl,instruction.json,images/}

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_EPISODES="${NUM_EPISODES:-10}"
LOGS_ROOT="${LOGS_ROOT:-logs/data_collection}"
DEVICE="${DEVICE:-cuda:0}"
PLANNER="${PLANNER:-scripted}"
NUM_OBJECTS="${NUM_OBJECTS:-10}"
HEADLESS="${HEADLESS:-0}"

EXTRA_HEADLESS=()
if [[ "${HEADLESS}" == "1" ]]; then
  EXTRA_HEADLESS=(--headless)
fi

CAMERA_ARGS=()
if [[ "${ENABLE_CAMERAS:-0}" == "1" ]]; then
  CAMERA_ARGS=(--enable_cameras)
fi

exec python -m data_collection.collect_data \
  --profile vla_temp \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner "${PLANNER}" \
  --device "${DEVICE}" \
  "${CAMERA_ARGS[@]}" \
  "${EXTRA_HEADLESS[@]}" \
  --log-rate-hz 15 \
  --num-episodes "${NUM_EPISODES}" \
  --num-objects "${NUM_OBJECTS}" \
  --use-ycb \
  --spawn-mode usd \
  --spawn-min 0.26 -0.34 0.89 \
  --spawn-max 0.42 0.36 1.06 \
  --spawn-min-robot-dist 0.26 \
  --spawn-bin-clearance-m 0.32 \
  --min-distance 0.16 \
  --planner-speed-mps 0.58 \
  --planner-waypoint-max-seg-m 0.022 \
  --settle-steps 72 \
  --stabilize-steps 40 \
  --workspace-max-x 0.88 \
  --bin-center-x 0.78 \
  --bin-footprint 0.14 0.14 \
  --bin-y-spacing 0.22 \
  --bin-wall-height 0.05 \
  --gripper-open-steps 8 \
  --hold-after-close-steps 20 \
  --post-lift-hold-steps 20 \
  --target-selection farthest_no_repeat \
  --approach-detour-m 0 \
  --max-steps-per-episode 10000 \
  --grasp-depth -0.07 \
  --tolerance 0.007 \
  --close-if-within-m 0.007 \
  --domain-rand \
  --domain-rand-seed 0 \
  --respawn-every-n-episodes 1 \
  --bin-selection random \
  --logs-root "${LOGS_ROOT}" \
  "$@"

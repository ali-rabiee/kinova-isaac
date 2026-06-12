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
  --planner-waypoint-max-seg-m 0.022 \
  --settle-steps 72 \
  --stabilize-steps 200 \
  --gripper-open-steps 8 \
  --hold-after-close-steps 20 \
  --post-lift-hold-steps 20 \
  --target-selection "${TARGET_SELECTION}" \
  --approach-detour-m 0 \
  --max-steps-per-episode 8000 \
  --grasp-depth -0.07 \
  --tolerance 0.007 \
  --close-if-within-m 0.007 \
  --domain-rand \
  --respawn-every-n-episodes 1 \
  --logs-root "${LOGS_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

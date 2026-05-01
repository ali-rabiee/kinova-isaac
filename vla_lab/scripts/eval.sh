#!/usr/bin/env bash
# Evaluate a trained TinyVLA inside Isaac Lab using the existing
# `reach_to_grasp_VLA` environment.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
DEVICE="${DEVICE:-cuda:0}"
ISAACLAB="${ISAACLAB:-./IsaacLab/isaaclab.sh}"

if [[ ! -x "${ISAACLAB}" ]]; then
  echo "[eval] ${ISAACLAB} not found or not executable. Set ISAACLAB= to your isaaclab.sh path." >&2
  exit 1
fi

exec "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --enable_cameras \
  "$@"

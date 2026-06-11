#!/usr/bin/env bash
# Evaluate a trained TinyVLA inside Isaac Lab using the existing
# `reach_to_grasp_VLA` environment.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${ISAACLAB:-}" ]]; then
  for candidate in "./IsaacLab/isaaclab.sh" "${HOME}/IsaacLab/isaaclab.sh"; do
    if [[ -x "${candidate}" ]]; then
      ISAACLAB="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ISAACLAB:-}" || ! -x "${ISAACLAB}" ]]; then
  echo "[eval] isaaclab.sh not found. Set ISAACLAB= to your isaaclab.sh path (e.g. ${HOME}/IsaacLab/isaaclab.sh)." >&2
  exit 1
fi

exec "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --enable_cameras \
  "$@"

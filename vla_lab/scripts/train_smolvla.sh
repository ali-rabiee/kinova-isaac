#!/usr/bin/env bash
# Fine-tune SmolVLA on a local LeRobot dataset under vla_lab/ (requires `lerobot` install).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/vla_lab/datasets/lerobot_kinova_v0}"
RUN_NAME="${RUN_NAME:-smolvla_ft_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/vla_lab/checkpoints/${RUN_NAME}}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
DATASET_REPO_ID="${DATASET_REPO_ID:-kinova_isaac_vla}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

GIT_REV=""
if command -v git >/dev/null 2>&1; then
  GIT_REV="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
fi
LR_VER=""
if command -v python3 >/dev/null 2>&1; then
  LR_VER="$(python3 -c 'import importlib.util;print(importlib.util.find_spec("lerobot") is not None)' 2>/dev/null || true)"
fi

MANIFEST="${OUT_DIR}/run_manifest.json"
{
  echo "{"
  echo "  \"created\": \"$(date -Iseconds)\","
  echo "  \"git_rev\": \"${GIT_REV}\","
  echo "  \"python_has_lerobot_module\": \"${LR_VER}\","
  echo "  \"policy_path\": \"${POLICY_PATH}\","
  echo "  \"dataset_repo_id\": \"${DATASET_REPO_ID}\","
  echo "  \"dataset_root\": \"${DATASET_DIR}\","
  echo "  \"output_dir\": \"${OUT_DIR}\","
  echo "  \"cmd\": \"lerobot-train with dataset.root=${DATASET_DIR}\""
  echo "}"
} > "${MANIFEST}"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "[train_smolvla] lerobot-train not found. Install: pip install -r vla_lab/requirements-smolvla.txt" >&2
  exit 1
fi

exec lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_DIR}" \
  --policy.device="${DEVICE}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --output_dir="${OUT_DIR}" \
  --job_name="${RUN_NAME}" \
  "$@"

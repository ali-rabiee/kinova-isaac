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
# LeRobot defaults push_to_hub=true; without policy.repo_id training validation fails.
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_REPO_ID="${DATASET_REPO_ID:-kinova_isaac_vla}"
STEPS="${STEPS:-9824}"
# ~4 passes over ~78k-frame export at batch 32; override for other data or full training (e.g. STEPS=20000).
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1000}"

# Do not mkdir OUT_DIR here: lerobot-train requires --output_dir to be absent when resume=false
# (it errors if the directory already exists). Stash run metadata beside checkpoints instead.
MANIFEST_STASH="${MANIFEST_STASH:-${REPO_ROOT}/vla_lab/checkpoints/_train_manifests}"
mkdir -p "${MANIFEST_STASH}"

if [ -e "${OUT_DIR}" ] && [ "${RESUME:-false}" != "true" ]; then
  echo "[train_smolvla] ERROR: output dir already exists and RESUME is not true:" >&2
  echo "  ${OUT_DIR}" >&2
  echo "Pick a new run:  OUT_DIR=... ./vla_lab/scripts/train_smolvla.sh" >&2
  echo "Or resume:       RESUME=true OUT_DIR=... ./vla_lab/scripts/train_smolvla.sh" >&2
  echo "Or remove:       rm -rf \"${OUT_DIR}\"" >&2
  exit 1
fi

GIT_REV=""
if command -v git >/dev/null 2>&1; then
  GIT_REV="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
fi
LR_VER=""
if command -v python3 >/dev/null 2>&1; then
  LR_VER="$(python3 -c 'import importlib.util;print(importlib.util.find_spec("lerobot") is not None)' 2>/dev/null || true)"
fi

MANIFEST="${MANIFEST_STASH}/${RUN_NAME}.json"
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

export VLA_LAB_RUN_NAME="${RUN_NAME}"

SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-0}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
RESUME_ARGS=()
if [ "${RESUME:-false}" = "true" ]; then
  RESUME_ARGS=(--resume=true)
fi

# One flag list for both launch paths (capture wrapper vs bare lerobot-train).
TRAIN_ARGS=(
  --policy.path="${POLICY_PATH}"
  --policy.push_to_hub="${PUSH_TO_HUB}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_DIR}"
  --policy.device="${DEVICE}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --seed="${SEED}"
  --output_dir="${OUT_DIR}"
  --job_name="${RUN_NAME}"
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --eval_freq="${EVAL_FREQ}"
  "${RESUME_ARGS[@]}"
  "$@"
)

if [ "${VLA_LAB_CAPTURE_RESULTS:-1}" = "0" ]; then
  exec lerobot-train "${TRAIN_ARGS[@]}"
fi

exec python -m vla_lab.lerobot_train_capture -- "${TRAIN_ARGS[@]}"

#!/usr/bin/env bash
# After SmolVLA fine-tuning: paper-style figures + eval reminders.
#
# LeRobot writes checkpoints and (often) TensorBoard events under RUN_DIR — not
# the TinyVLA `metrics.jsonl` format. Use one of:
#   A) tensorboard --logdir "${RUN_DIR}"
#   B) Weights & Biases if you trained with wandb flags
#   C) This repo's plot_metrics for *eval* JSON (success + Wilson CIs):
#        RUN_DIR=/tmp EVAL_JSONS="eval1.json eval2.json" ./vla_lab/scripts/after_smolvla_train.sh
#
# Usage:
#   RUN_DIR=vla_lab/checkpoints/smolvla_kinova_ft ./vla_lab/scripts/after_smolvla_train.sh
#
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-}"
if [ -z "${RUN_DIR}" ]; then
  echo "Set RUN_DIR to your lerobot --output_dir (e.g. vla_lab/checkpoints/smolvla_kinova_ft)" >&2
  exit 1
fi

if [ ! -d "${RUN_DIR}" ]; then
  echo "RUN_DIR not found: ${RUN_DIR}" >&2
  exit 1
fi

FIG_DIR="${FIG_DIR:-${RUN_DIR}/figures_paper}"
mkdir -p "${FIG_DIR}"

echo "[after_smolvla] checkpoint / run directory: ${RUN_DIR}"
echo "[after_smolvla] suggest: tensorboard --logdir ${RUN_DIR}  (or inspect checkpoints/ inside)"

if [ -f "${RUN_DIR}/metrics.jsonl" ]; then
  echo "[after_smolvla] found metrics.jsonl → generating learning curves into ${FIG_DIR}"
  python -m vla_lab.plot_metrics --run-dir "${RUN_DIR}" --figures-dir "${FIG_DIR}" --format pdf
else
  echo "[after_smolvla] no TinyVLA-style metrics.jsonl (expected for raw LeRobot-only runs)."
fi

EVAL_ARGS=()
if [ -n "${EVAL_JSONS:-}" ]; then
  for j in ${EVAL_JSONS}; do
    EVAL_ARGS+=(--eval-json "${j}")
  done
  echo "[after_smolvla] plotting with eval JSON: ${EVAL_JSONS}"
  python -m vla_lab.plot_metrics --run-dir "${RUN_DIR}" --figures-dir "${FIG_DIR}" --format pdf "${EVAL_ARGS[@]}"
fi

echo "[after_smolvla] done. Figures (if any): ${FIG_DIR}"

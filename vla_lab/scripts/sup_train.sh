#!/usr/bin/env bash
# ★ Train one Carryover-Aware VLA cell.
#
#   ./vla_lab/scripts/sup_train.sh                                   # tiny / token
#   MODEL=smolvla CONTEXT=text ./vla_lab/scripts/sup_train.sh
#   MODEL=qwen25vl-3b CONTEXT=text BATCH=1 ACCUM=16 ./vla_lab/scripts/sup_train.sh
#   FRAMES=vla_lab/results/physics/frames ./vla_lab/scripts/sup_train.sh   # real Isaac images
#
# The trainer runs a VRAM preflight before the first optimiser step and prints peak memory.
# Splits are BY SUPERVISOR, never by sample.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"

EXTRA=()
[[ -n "${FRAMES:-}" ]] && EXTRA+=(--frames "${FRAMES}")
[[ -n "${ADAPT:-}"  ]] && EXTRA+=(--adapt "${ADAPT}")

exec python -m vla_lab.training.train \
  --model "${MODEL:-tiny}" --context "${CONTEXT:-token}" \
  --epochs "${EPOCHS:-25}" --supervisors "${SUPERVISORS:-80}" \
  --batch "${BATCH:-32}" --accum "${ACCUM:-1}" --seed "${SEED:-1}" \
  --workers "${WORKERS:-0}" "${EXTRA[@]}" "$@"

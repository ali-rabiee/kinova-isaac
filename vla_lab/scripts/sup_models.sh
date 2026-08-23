#!/usr/bin/env bash
# ★ The architecture comparison table: every (backbone × context mode) cell.
#
#   ./vla_lab/scripts/sup_models.sh                                     # tiny, 3 context modes
#   MODELS="tiny smolvla" CONTEXTS="none text token film" ./vla_lab/scripts/sup_models.sh
#
# Cells the registry declares impossible (a verbalised context on a backbone with no language
# model) are skipped and reported as skipped, never as failures.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
EXTRA=()
[[ -n "${LR:-}" ]] && EXTRA+=(--lr "${LR}")
[[ -n "${FRAMES:-}" ]] && EXTRA+=(--frames "${FRAMES}")

exec python -m vla_lab.training.sweep_models \
  --models ${MODELS:-tiny} --contexts ${CONTEXTS:-none token film} \
  --epochs "${EPOCHS:-25}" --supervisors "${SUPERVISORS:-80}" \
  --batch "${BATCH:-32}" --accum "${ACCUM:-1}" --seed "${SEED:-1}" \
  --out "${OUT:-vla_lab/results/models}" "${EXTRA[@]}" "$@"

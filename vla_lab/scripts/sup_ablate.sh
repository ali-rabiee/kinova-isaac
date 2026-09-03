#!/usr/bin/env bash
# ★ Objective ablations: which loss term produces the de-biasing?
#
#   ./vla_lab/scripts/sup_ablate.sh
#   MODEL=smolvla CONTEXT=text FRAMES=vla_lab/results/physics/frames/topdown ./vla_lab/scripts/sup_ablate.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
EXTRA=()
[[ -n "${FRAMES:-}" ]] && EXTRA+=(--frames "${FRAMES}")
exec python -m vla_lab.training.sweep_ablations \
  --model "${MODEL:-tiny}" --context "${CONTEXT:-film}" \
  --epochs "${EPOCHS:-25}" --supervisors "${SUPERVISORS:-80}" \
  --batch "${BATCH:-32}" --accum "${ACCUM:-1}" --seeds ${SEEDS:-1 2 3 4 5} --skip-existing \
  --out "${OUT:-vla_lab/results/ablations}" "${EXTRA[@]}" "$@"

#!/usr/bin/env bash
# ★ The flip diagnostic (P1-4): every headline contrast as a function of the assumed curve.
#
#   ./vla_lab/scripts/sup_flip.sh                      # w sweep and m* sweep, N=80 per cell
#   W_VALUES="0.3 0.5 1.0" ./vla_lab/scripts/sup_flip.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N="${SUPERVISORS:-80}"; SEED="${SEED:-20260822}"; JOBS="${JOBS:-7}"
python -m vla_lab.supervisory.flip run --param w --values ${W_VALUES:-0.17 0.3 0.5 0.81 1.2 2.0 3.12} \
  --supervisors "${N}" --seed "${SEED}" --jobs "${JOBS}" --out vla_lab/results/flip_w
python -m vla_lab.supervisory.flip run --param mstar --values ${MSTAR_VALUES:-3.5 3.9 4.5 5.2 6.0 7.0 8.5} \
  --supervisors "${N}" --seed "${SEED}" --jobs "${JOBS}" --out vla_lab/results/flip_mstar

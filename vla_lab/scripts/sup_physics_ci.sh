#!/usr/bin/env bash
# ★ The primary study under the bootstrap interval of the measured physics (P0-2).
#
#   ./vla_lab/scripts/sup_physics_ci.sh        # lower and upper draws of w, N=80 each, in parallel
#
# ``physics_lower.json`` / ``physics_upper.json`` are written by the fit (``measure.py fit --bootstrap``)
# next to ``physics.json``. Each run is stamped with its draw and refuses to be pooled.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N="${SUPERVISORS:-80}"; SEED="${SEED:-20260822}"
for q in lower upper; do
  python -m vla_lab.supervisory.run_study --supervisors "${N}" --seed "${SEED}" --physics-quantile "${q}" \
    --out "vla_lab/results/tier1_physics_${q}" --quiet > "vla_lab/results/tier1_physics_${q}.log" 2>&1 &
done
wait
python -m vla_lab.supervisory.physics_ci

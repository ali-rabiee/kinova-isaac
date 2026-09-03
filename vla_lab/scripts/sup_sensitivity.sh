#!/usr/bin/env bash
# ★ The sensitivity sweep (R10) plus the placebo control for the dose-tracking result.
#
#   ./vla_lab/scripts/sup_sensitivity.sh            # all cells, N=48 each, in parallel, then aggregate
#   JOBS=4 SUPERVISORS=48 ./vla_lab/scripts/sup_sensitivity.sh
#
# Cells: the dose ladder, the flat belief prior, the alternating coaching regime, and three
# PLACEBO cells that vary parameters the belief module has no access to (the supervisor's lapse
# rate; answer latency). The carryover-aware policy's counter-proposal rate is claimed to track
# the dose; it must not track the placebo.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N="${SUPERVISORS:-48}"; SEED="${SEED:-20260822}"; ROOT="${OUT:-vla_lab/results/sensitivity}"; JOBS="${JOBS:-8}"
mkdir -p "${ROOT}"
run() { # name, args...
  local name="$1"; shift
  python -m vla_lab.supervisory.run_study --supervisors "${N}" --seed "${SEED}" --out "${ROOT}/${name}" --quiet "$@" \
    > "${ROOT}/${name}.log" 2>&1 && echo "  [ok] ${name}" || echo "  [FAIL] ${name} (see ${ROOT}/${name}.log)"
}
export -f run; export N SEED ROOT
cat <<CELLS | xargs -P "${JOBS}" -I{} bash -c 'run {}'
dose_weak --dose weak
dose_moderate --dose moderate
dose_strong --dose strong
prior_flat --population-prior none
regime_alternating --coach-regime alternating
placebo_lapse_low --population lapse_range=0.0,0.02
placebo_lapse_high --population lapse_range=0.15,0.25
placebo_latency_slow --population latency_range_s=6.0,12.0
CELLS
python -m vla_lab.supervisory.sensitivity "${ROOT}"

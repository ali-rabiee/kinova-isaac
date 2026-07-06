#!/usr/bin/env bash
# Turn calibration logs into the Result-2 figures + summary JSON (memo §4.2 / §4.3).
#
# Produces, under OUT_DIR: reliability_diagram, ece_vs_occlusion, agreement_vs_correctness
# (the decoupling/inversion finding), and coverage_vs_occlusion (conformal under shift), plus
# calibration_summary.json. Pure NumPy + matplotlib; no Isaac needed.
#
#   CALIB='vla_lab/calibration_runs/*/*/records.jsonl' ./vla_lab/scripts/calibration_analyze.sh
#   FORMAT=png ./vla_lab/scripts/calibration_analyze.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CALIB="${CALIB:-vla_lab/calibration_runs/*/*/records.jsonl}"
OUT_DIR="${OUT_DIR:-vla_lab/results/calibration}"
FORMAT="${FORMAT:-pdf}"   # pdf | png

exec python -m vla_lab.calibration.analyze \
  --calib "${CALIB}" \
  --out-dir "${OUT_DIR}" \
  --format "${FORMAT}" \
  "$@"

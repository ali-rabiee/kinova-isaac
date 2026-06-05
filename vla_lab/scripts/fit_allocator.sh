#!/usr/bin/env bash
# Fit the act/compute/query allocator from calibration logs -> allocator_fit.json.
#
# This is the "training" step for the query branch (memo C3): it fits the irreducibility
# detector + the conformal escalation rule from a robot-only calibration sweep (produced by
# calibration_eval.sh). No Isaac / torch needed — NumPy only.
#
#   # default: glob every records.jsonl under vla_lab/calibration_runs/
#   ./vla_lab/scripts/fit_allocator.sh
#
#   # explicit inputs/outputs:
#   CALIB='vla_lab/calibration_runs/run_a/*/records.jsonl' \
#   OUT=vla_lab/fits/allocator_fit.json \
#   CONFORMAL=mondrian ALPHA=0.1 LABEL=auto \
#     ./vla_lab/scripts/fit_allocator.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CALIB="${CALIB:-vla_lab/calibration_runs/*/*/records.jsonl}"
OUT="${OUT:-vla_lab/fits/allocator_fit.json}"
REPORT="${REPORT:-vla_lab/fits/allocator_fit.report.json}"
CONFORMAL="${CONFORMAL:-mondrian}"   # mondrian | split | none
ALPHA="${ALPHA:-0.1}"
LABEL="${LABEL:-auto}"               # auto | identifiable | confidently_wrong | occlusion

# NB: CALIB is a glob expanded *inside* fit_allocator (glob.glob), so pass it as one arg.
exec python -m vla_lab.fit_allocator \
  --calib "${CALIB}" \
  --out "${OUT}" \
  --report "${REPORT}" \
  --conformal "${CONFORMAL}" \
  --alpha "${ALPHA}" \
  --label "${LABEL}" \
  "$@"

#!/usr/bin/env bash
# Robot-only calibration sweep (the empirical heart of the pivot, memo §4.2 "Result 2").
#
# Runs the Isaac eval under an act/compute/query controller across occlusion levels, emitting
# per-step calibration records (dispersion + occlusion + correctness) for `fit_allocator` and
# `calibration.analyze`. No humans needed — this de-risks the schedule.
#
# Requires Isaac Lab (set ISAACLAB= to your isaaclab.sh). Mirrors sweep_occlusion_eval.sh.
#
#   OUT_ROOT=vla_lab/calibration_runs/run_a NUM_EPISODES=50 \
#     ./vla_lab/scripts/calibration_eval.sh
#
#   # with a fitted allocator (unlocks the query branch) and the SmolVLA backend:
#   CONTROLLER=allocator FIT=vla_lab/fits/allocator_fit.json \
#     ./vla_lab/scripts/calibration_eval.sh --policy-backend smolvla

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-vla_lab/calibration_runs/$(date +%Y%m%d_%H%M%S)}"
NUM_EPISODES="${NUM_EPISODES:-50}"
CONTROLLER="${CONTROLLER:-compute_gated}"   # autonomy|fixed_compute|compute_gated|scale|knowno|insight|allocator
FIT="${FIT:-}"                              # optional allocator_fit.json (unlocks query/INSIGHT/KnowNo)
EXTRA_FLAGS=( "$@" )

if [[ -z "${ISAACLAB:-}" ]]; then
  for candidate in "./IsaacLab/isaaclab.sh" "${HOME}/IsaacLab/isaaclab.sh"; do
    if [[ -x "${candidate}" ]]; then
      ISAACLAB="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ISAACLAB:-}" || ! -x "${ISAACLAB}" ]]; then
  echo "[calibration_eval] isaaclab.sh not found. Set ISAACLAB= to your isaaclab.sh path." >&2
  exit 1
fi

run_one() {
  local mode="$1" frac="$2" label="$3"
  local odir="${OUT_ROOT}/${label}"
  mkdir -p "${odir}"
  echo "[calibration_eval] controller=${CONTROLLER} mode=${mode} fraction=${frac} -> ${odir}"
  local fit_args=()
  [[ -n "${FIT}" ]] && fit_args=(--allocator-fit "${FIT}")
  "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --enable_cameras \
    --num-episodes "${NUM_EPISODES}" \
    --out-dir "${odir}" \
    --occlusion-mode "${mode}" \
    --occlusion-fraction "${frac}" \
    --observability-mode "${label}" \
    --controller "${CONTROLLER}" \
    --emit-calibration "${odir}/records.jsonl" \
    "${fit_args[@]}" \
    "${EXTRA_FLAGS[@]}"
}

mkdir -p "${OUT_ROOT}"
run_one "none" "0.0" "clean"
for rho in 0.15 0.25 0.35 0.50; do
  run_one "bottom_strip" "${rho}" "bottom_strip_${rho}"
done

echo "[calibration_eval] done. Calibration records under ${OUT_ROOT}/*/records.jsonl"
echo "[calibration_eval] next:"
echo "    CALIB='${OUT_ROOT}/*/records.jsonl' ./vla_lab/scripts/fit_allocator.sh"
echo "    CALIB='${OUT_ROOT}/*/records.jsonl' ./vla_lab/scripts/calibration_analyze.sh"

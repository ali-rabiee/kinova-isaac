#!/usr/bin/env bash
# Graded occlusion sweep (reviewer W1): run eval_isaaclab for several modes/fractions.
# Override CONFIG, DEVICE, ISAACLAB, EXTRA_FLAGS as needed.
#
# Example:
#   OUT_ROOT=vla_lab/eval_sweeps/run_a NUM_EPISODES=50 ./vla_lab/scripts/sweep_occlusion_eval.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
DEVICE="${DEVICE:-cuda:0}"
ISAACLAB="${ISAACLAB:-./IsaacLab/isaaclab.sh}"
OUT_ROOT="${OUT_ROOT:-vla_lab/eval_sweeps/occlusion_sweep}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EXTRA_FLAGS=( "$@" )

if [[ ! -x "${ISAACLAB}" ]]; then
  echo "[sweep_occlusion] ${ISAACLAB} not found or not executable." >&2
  exit 1
fi

run_one() {
  local mode="$1"
  local frac="$2"
  local label="$3"
  local odir="${OUT_ROOT}/${label}"
  mkdir -p "${odir}"
  echo "[sweep_occlusion] mode=${mode} fraction=${frac} -> ${odir}"
  "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --enable_cameras \
    --num-episodes "${NUM_EPISODES}" \
    --out-dir "${odir}" \
    --occlusion-mode "${mode}" \
    --occlusion-fraction "${frac}" \
    --observability-mode "${label}" \
    "${EXTRA_FLAGS[@]}"
}

mkdir -p "${OUT_ROOT}"

# Clean baseline
run_one "none" "0.0" "clean"

# bottom_strip fractions (paper Appendix F style grid)
for rho in 0.15 0.25 0.35 0.50; do
  run_one "bottom_strip" "${rho}" "bottom_strip_${rho}"
done

# Robustness masks (single fraction; edit FRAC_ROBUST if needed)
FRAC_ROBUST="${FRAC_ROBUST:-0.35}"
run_one "center_box" "${FRAC_ROBUST}" "center_box_${FRAC_ROBUST}"
run_one "random_patch" "${FRAC_ROBUST}" "random_patch_${FRAC_ROBUST}"

echo "[sweep_occlusion] done. Results under ${OUT_ROOT}"

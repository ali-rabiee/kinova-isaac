#!/usr/bin/env bash
# τ sweep for twin / noisy_pair gating (reviewer W4/W5 frontier).
# Requires ttc.gating and ttc.k_action_samples > 1 in CONFIG (or pass EXTRA_FLAGS).
#
# Example:
#   NUM_EPISODES=30 ./vla_lab/scripts/sweep_gating_threshold.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
DEVICE="${DEVICE:-cuda:0}"
ISAACLAB="${ISAACLAB:-./IsaacLab/isaaclab.sh}"
OUT_ROOT="${OUT_ROOT:-vla_lab/eval_sweeps/tau_sweep}"
NUM_EPISODES="${NUM_EPISODES:-50}"
# Space-separated thresholds
TAU_VALUES="${TAU_VALUES:-0.01 0.015 0.02 0.03 0.05 0.08}"
EXTRA_FLAGS=( "$@" )

if [[ ! -x "${ISAACLAB}" ]]; then
  echo "[sweep_tau] ${ISAACLAB} not found or not executable." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

for tau in ${TAU_VALUES}; do
  label="tau_$(echo "${tau}" | tr '.' '_')"
  odir="${OUT_ROOT}/${label}"
  mkdir -p "${odir}"
  echo "[sweep_tau] gating_threshold=${tau} -> ${odir}"
  "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --enable_cameras \
    --num-episodes "${NUM_EPISODES}" \
    --out-dir "${odir}" \
    --gating-threshold "${tau}" \
    "${EXTRA_FLAGS[@]}"
done

echo "[sweep_tau] done. Summarize with vla_lab.plot_metrics or jq on results_*.json ttc_aggregate"

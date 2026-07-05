#!/usr/bin/env bash
# Camera-set ablation sweep (2026-07 wrist camera): for a TWO-camera checkpoint
# (train_multicam.yaml), measure how much each view is worth and whether
# "one camera + test-time compute" recovers two-camera performance.
#
# Legs (all on identical scenes/seeds):
#   both_clean          overhead+wrist            (upper anchor)
#   overhead_only       wrist masked              (missing-view ablation)
#   wrist_only          overhead masked
#   overhead_only_gated wrist masked + compute_gated TTC   <- the claim leg
#   wrist_only_gated    overhead masked + compute_gated TTC
#   both_occ_overhead   both views, occlusion on the overhead view
#   both_occ_wrist      both views, occlusion on the wrist view
#
# Usage:
#   CKPT=vla_lab/checkpoints/multicam_v0/last.pt NUM_EPISODES=50 \
#     ./vla_lab/scripts/sweep_camera_sets.sh
# Extra eval flags pass through:  ... ./vla_lab/scripts/sweep_camera_sets.sh --headless

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
CKPT="${CKPT:-vla_lab/checkpoints/multicam_v0/last.pt}"
DEVICE="${DEVICE:-cuda:0}"
ISAACLAB="${ISAACLAB:-./IsaacLab/isaaclab.sh}"
OUT_ROOT="${OUT_ROOT:-vla_lab/eval_sweeps/camera_sets}"
NUM_EPISODES="${NUM_EPISODES:-50}"
OCC_MODE="${OCC_MODE:-bottom_strip}"
OCC_FRAC="${OCC_FRAC:-0.35}"
GATED_K="${GATED_K:-4}"
EXTRA_FLAGS=( "$@" )

if [[ ! -x "${ISAACLAB}" ]] && [[ -x "${HOME}/IsaacLab/isaaclab.sh" ]]; then
  ISAACLAB="${HOME}/IsaacLab/isaaclab.sh"
fi
if [[ ! -x "${ISAACLAB}" ]]; then
  echo "[sweep_cameras] ${ISAACLAB} not found or not executable." >&2
  exit 1
fi

run_one() {
  local cameras="$1"; local occ_cam="$2"; local occ_mode="$3"; local label="$4"; shift 4
  local odir="${OUT_ROOT}/${label}"
  mkdir -p "${odir}"
  echo "[sweep_cameras] cameras=${cameras} occ=${occ_mode}@${occ_cam} -> ${odir}"
  "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --enable_cameras \
    --num-episodes "${NUM_EPISODES}" \
    --out-dir "${odir}" \
    --cameras "${cameras}" \
    --occlusion-mode "${occ_mode}" \
    --occlusion-fraction "${OCC_FRAC}" \
    --occlusion-camera "${occ_cam}" \
    --observability-mode "${label}" \
    "$@" \
    "${EXTRA_FLAGS[@]}"
}

mkdir -p "${OUT_ROOT}"

# Value of each view (clean scenes)
run_one both     overhead none "both_clean"
run_one overhead overhead none "overhead_only"
run_one wrist    overhead none "wrist_only"

# One camera + test-time compute ("our approach" legs)
run_one overhead overhead none "overhead_only_gated" --controller compute_gated
run_one wrist    overhead none "wrist_only_gated"    --controller compute_gated

# Per-view occlusion with both cameras (cross-view conflict)
run_one both overhead "${OCC_MODE}" "both_occ_overhead_${OCC_FRAC}"
run_one both wrist    "${OCC_MODE}" "both_occ_wrist_${OCC_FRAC}"

echo "[sweep_cameras] done. Results under ${OUT_ROOT} (compare success_rate across legs;"
echo "  the claim: overhead_only_gated / wrist_only_gated ≈ both_clean, > overhead_only/wrist_only)"

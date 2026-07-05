#!/usr/bin/env bash
# Human-input-QUALITY sweep (2026-07 intent axis): how does policy performance
# depend on the accuracy/specificity/latency of real-time human feedback, and
# how much does quality-aware fusion recover when the human is unreliable?
#
# Design: instructions are made ambiguous (default: half of episodes) so human
# input MATTERS; a scripted simulated human interjects corrections; the
# intent_allocator controller can also query. Grid:
#   accuracy   x  specificity            (+ fixed latency)
#   and for each accuracy: a trust-all ablation (--no-fusion-verify)
# Compare `success_rate` and feedback/query counts across cells; the claim:
# fusion ≈ trust-all when accuracy is high, fusion >> trust-all when low.
#
# Usage:
#   CKPT=vla_lab/checkpoints/multicam_v0/last.pt NUM_EPISODES=30 \
#     ./vla_lab/scripts/sweep_feedback_quality.sh --headless

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/eval_isaac.yaml}"
CKPT="${CKPT:-vla_lab/checkpoints/multicam_v0/last.pt}"
DEVICE="${DEVICE:-cuda:0}"
ISAACLAB="${ISAACLAB:-./IsaacLab/isaaclab.sh}"
OUT_ROOT="${OUT_ROOT:-vla_lab/eval_sweeps/feedback_quality}"
NUM_EPISODES="${NUM_EPISODES:-30}"
AMBIGUITY="${AMBIGUITY:-half}"                 # none|half|full
CONTROLLER="${CONTROLLER:-intent_allocator}"   # robot-initiated query arm
ACCURACIES="${ACCURACIES:-1.0 0.85 0.6 0.3}"
SPECIFICITIES="${SPECIFICITIES:-color directional vague}"
LATENCY_TICKS="${LATENCY_TICKS:-3}"
FB_SEED="${FB_SEED:-0}"
EXTRA_FLAGS=( "$@" )

if [[ ! -x "${ISAACLAB}" ]] && [[ -x "${HOME}/IsaacLab/isaaclab.sh" ]]; then
  ISAACLAB="${HOME}/IsaacLab/isaaclab.sh"
fi
if [[ ! -x "${ISAACLAB}" ]]; then
  echo "[sweep_feedback] ${ISAACLAB} not found or not executable." >&2
  exit 1
fi

run_one() {
  local acc="$1"; local spec="$2"; local label="$3"; shift 3
  local odir="${OUT_ROOT}/${label}"
  mkdir -p "${odir}"
  echo "[sweep_feedback] accuracy=${acc} specificity=${spec} -> ${odir}"
  "${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --enable_cameras \
    --num-episodes "${NUM_EPISODES}" \
    --out-dir "${odir}" \
    --controller "${CONTROLLER}" \
    --instruction-ambiguity "${AMBIGUITY}" \
    --feedback-channel scripted \
    --feedback-accuracy "${acc}" \
    --feedback-specificity "${spec}" \
    --feedback-latency-ticks "${LATENCY_TICKS}" \
    --feedback-seed "${FB_SEED}" \
    --observability-mode "${label}" \
    "$@" \
    "${EXTRA_FLAGS[@]}"
}

mkdir -p "${OUT_ROOT}"

# Reference: no human at all (queries answered by nobody -> autonomy-style floor)
odir="${OUT_ROOT}/no_feedback"
mkdir -p "${odir}"
"${ISAACLAB}" -p vla_lab/eval_isaaclab.py \
  --config "${CONFIG}" --ckpt "${CKPT}" --device "${DEVICE}" --enable_cameras \
  --num-episodes "${NUM_EPISODES}" --out-dir "${odir}" \
  --controller compute_gated --instruction-ambiguity "${AMBIGUITY}" \
  --observability-mode no_feedback "${EXTRA_FLAGS[@]}"

# Quality grid (fusion ON)
for acc in ${ACCURACIES}; do
  for spec in ${SPECIFICITIES}; do
    run_one "${acc}" "${spec}" "acc${acc}_${spec}"
  done
done

# Trust-all ablation (fusion verification OFF) across accuracies, color specificity
for acc in ${ACCURACIES}; do
  run_one "${acc}" color "acc${acc}_color_trustall" --no-fusion-verify
done

echo "[sweep_feedback] done. Results under ${OUT_ROOT}"
echo "  plot success_rate vs accuracy per specificity; overlay *_trustall to show fusion's cover."

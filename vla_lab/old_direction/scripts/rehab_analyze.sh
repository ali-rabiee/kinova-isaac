#!/usr/bin/env bash
# Phase 0 outcomes, tables, and every paper figure, from one command (W17).
#
# Reports test-retest reliability of the reference map FIRST (it bounds how much of the measured
# estimation error is irreducible drift), then the primary outcomes, the ablation decomposition,
# the budget manipulation check, and the per-person carryover heterogeneity.
#
#   ./vla_lab/old_direction/scripts/rehab_analyze.sh
#   SESSION_ROOT=logs/rehab FIGURES_DIR=vla_lab/paper/figures ./vla_lab/old_direction/scripts/rehab_analyze.sh
#   OFFPOLICY=1 ./vla_lab/old_direction/scripts/rehab_analyze.sh   # + the model-based secondary analysis (§12.6)
#
# Figures are named rehab_*.<fmt>, so pointing FIGURES_DIR at the paper's figures/ directory
# writes them straight into the paper without colliding with the VLA track's figures.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SESSION_ROOT="${SESSION_ROOT:-logs/rehab}"
OUT_DIR="${OUT_DIR:-vla_lab/old_direction/results/rehab_phase0}"
FIGURES_DIR="${FIGURES_DIR:-${OUT_DIR}}"
FORMAT="${FORMAT:-pdf}"

EXTRA=()
[[ -n "${OFFPOLICY:-}" ]] && EXTRA+=(--offpolicy)
[[ -n "${GROUND_TRUTH:-}" ]] && EXTRA+=(--ground-truth "${GROUND_TRUTH}")

exec python -m vla_lab.old_direction.rehab.analyze \
  --session-root "${SESSION_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --figures-dir "${FIGURES_DIR}" \
  --format "${FORMAT}" \
  "${EXTRA[@]}" "$@"

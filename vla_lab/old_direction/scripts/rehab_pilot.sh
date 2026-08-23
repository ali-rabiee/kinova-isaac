#!/usr/bin/env bash
# ★ Synthetic Phase 0 study: end-to-end, no robot, no participants, seconds.
#
# Runs the REAL session code path (protocol -> schedulers -> observers -> safety -> event-locked
# logging -> gate -> analysis) with a generative participant behind the observer seam and the
# null apparatus behind the apparatus seam. This is the rehearsal that makes every Tier-A work
# item verifiable before hardware or IRB approval exists (rehab.md §6).
#
# It is a rehearsal, NOT evidence about people: every number follows from the population prior
# in vla_lab/rehab/sim_participant.py, which the M4 lab pilot is supposed to replace.
#
#   ./vla_lab/old_direction/scripts/rehab_pilot.sh
#   PARTICIPANTS=24 ALL_CONDITIONS=1 ./vla_lab/old_direction/scripts/rehab_pilot.sh
#   MISDETECT=0.15 ./vla_lab/old_direction/scripts/rehab_pilot.sh      # W8 stress test: bad online labels

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PARTICIPANTS="${PARTICIPANTS:-8}"
SEED="${SEED:-20260816}"
LOG_ROOT="${LOG_ROOT:-logs/rehab_sim}"
OUT_DIR="${OUT_DIR:-vla_lab/old_direction/results/rehab_phase0}"
FORMAT="${FORMAT:-pdf}"
CONFIG="${CONFIG:-vla_lab/old_direction/configs/rehab_sim_pilot.yaml}"

EXTRA=()
[[ -n "${ALL_CONDITIONS:-}" ]] && EXTRA+=(--all-conditions)
[[ -n "${CONDITIONS:-}" ]] && EXTRA+=(--conditions "${CONDITIONS}")
[[ -n "${MISDETECT:-}" ]] && EXTRA+=(--misdetect-rate "${MISDETECT}")

exec python -m vla_lab.old_direction.rehab.run_pilot \
  --participants "${PARTICIPANTS}" \
  --seed "${SEED}" \
  --log-root "${LOG_ROOT}" \
  --config "${CONFIG}" \
  --analyze \
  --out-dir "${OUT_DIR}" \
  --format "${FORMAT}" \
  "${EXTRA[@]}" "$@"

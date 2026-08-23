#!/usr/bin/env bash
# Monte-Carlo power / sample-size memo for the Phase 0 primary contrast (W16).
#
# Runs the WHOLE pipeline over synthetic populations and reports the fraction of simulated
# studies in which the paired contrast is detected — not a closed-form approximation of a test
# the study does not run. The analytic paired-t N is printed as a cross-check.
#
#   ./vla_lab/old_direction/scripts/rehab_power.sh
#   N=8,12,16,24,32 SIMS=100 ./vla_lab/old_direction/scripts/rehab_power.sh
#   EFFECT_SCALE=1.5 ./vla_lab/old_direction/scripts/rehab_power.sh     # a stronger COACH manipulation (§12.7)
#
# WARNING: every number inherits the population prior in sim_participant.py. Re-run this after
# the M4 pilot with the pilot's fitted (lambda, beta, g) and say in the memo which numbers came
# from where — that is W16's actual "done when".

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

N="${N:-8,12,16,24}"
SIMS="${SIMS:-40}"
OUT="${OUT:-vla_lab/old_direction/results/rehab_phase0/power_memo.json}"

EXTRA=()
[[ -n "${EFFECT_SCALE:-}" ]] && EXTRA+=(--effect-scale "${EFFECT_SCALE}")
[[ -n "${TRIALS_PER_BLOCK:-}" ]] && EXTRA+=(--trials-per-block "${TRIALS_PER_BLOCK}")
[[ -n "${FINE:-}" ]] && EXTRA+=(--fine)

exec python -m vla_lab.old_direction.rehab.power --n "${N}" --sims "${SIMS}" --out "${OUT}" "${EXTRA[@]}" "$@"

#!/usr/bin/env bash
# ★ The Tier-1 synthetic study: whole cohort, every condition, seconds to minutes.
#
# Runs the REAL session code path (protocol → schedulers → grounding → event-locked records →
# estimators → analysis) with the surrogate apparatus behind the apparatus seam and generative
# supervisors behind the human seam. This is the rehearsal that makes every claim verifiable
# before a simulator run, a checkpoint, or an IRB protocol exists.
#
# It is a rehearsal, NOT evidence about people: every number follows from the population prior
# in vla_lab/supervisory/supervisor.py.
#
#   ./vla_lab/scripts/sup_study.sh                          # N=80, all conditions, + figures
#   SUPERVISORS=24 ./vla_lab/scripts/sup_study.sh
#   PHYSICS=vla_lab/results/physics/physics.json ./vla_lab/scripts/sup_study.sh
#   REGIME=alternating ./vla_lab/scripts/sup_study.sh       # the identification sensitivity arm
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"

SUPERVISORS="${SUPERVISORS:-80}"
SEED="${SEED:-20260822}"
OUT="${OUT:-vla_lab/results/tier1}"

EXTRA=()
[[ -n "${PHYSICS:-}" ]] && EXTRA+=(--physics "${PHYSICS}")
[[ -n "${REGIME:-}"  ]] && EXTRA+=(--coach-regime "${REGIME}")
[[ -n "${DOSE:-}"    ]] && EXTRA+=(--dose "${DOSE}")
[[ -n "${LOG_ROOT:-}" ]] && EXTRA+=(--log-root "${LOG_ROOT}")

exec python -m vla_lab.supervisory.run_study \
  --supervisors "${SUPERVISORS}" --seed "${SEED}" --out "${OUT}" --analyze "${EXTRA[@]}" "$@"

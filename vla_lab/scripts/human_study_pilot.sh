#!/usr/bin/env bash
# Offline human-study pilot: synthetic robot + human -> study log -> analysis + figures.
#
# Exercises the whole study pipeline (generate -> run -> analyze -> figures) with NO robot and
# NO human subjects (memo §10 July "pilot to debug the protocol"). Pure Python + NumPy +
# matplotlib. To run for real, swap SimEpisodeRunner/SimQuestionnaireProvider for the Kinova
# loop + study front-end inside session.py — the rest is unchanged.
#
#   PARTICIPANTS=12 DESIGN=within ./vla_lab/scripts/human_study_pilot.sh
#   PARTICIPANTS=24 DESIGN=between FORMAT=png ./vla_lab/scripts/human_study_pilot.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PARTICIPANTS="${PARTICIPANTS:-12}"
DESIGN="${DESIGN:-within}"          # within | between
OUT="${OUT:-vla_lab/study_runs/pilot/study.jsonl}"
OUT_DIR="${OUT_DIR:-vla_lab/results/human_study}"
FORMAT="${FORMAT:-pdf}"             # pdf | png

exec python -m vla_lab.human_study.run_pilot \
  --participants "${PARTICIPANTS}" \
  --design "${DESIGN}" \
  --out "${OUT}" \
  --analyze \
  --out-dir "${OUT_DIR}" \
  --format "${FORMAT}" \
  "$@"

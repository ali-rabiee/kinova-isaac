#!/usr/bin/env bash
# Analyze a study log (pilot or real) -> per-condition reliance / trust / workload tables,
# paired McNemar vs. the reference condition, and the over-trust-trap figures (memo §5.4).
#
# Works on the JSONL written by run_pilot.sh or by a real-robot StudyLogger run. No Isaac.
#
#   LOG=vla_lab/study_runs/study_2026.jsonl ./vla_lab/scripts/human_study_analyze.sh
#   REFERENCE=allocator_type FORMAT=png ./vla_lab/scripts/human_study_analyze.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LOG="${LOG:-vla_lab/study_runs/pilot/study.jsonl}"
OUT_DIR="${OUT_DIR:-vla_lab/results/human_study}"
REFERENCE="${REFERENCE:-allocator_type}"   # condition to McNemar everything against
FORMAT="${FORMAT:-pdf}"

exec python -m vla_lab.human_study.analyze \
  --log "${LOG}" \
  --out-dir "${OUT_DIR}" \
  --reference "${REFERENCE}" \
  --format "${FORMAT}" \
  "$@"

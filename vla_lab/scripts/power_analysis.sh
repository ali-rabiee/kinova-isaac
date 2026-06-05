#!/usr/bin/env bash
# Power / sample-size analysis for the human study (memo §10 July milestone, §11).
#
# Thin pass-through to the power CLI. Pure standard library (no scipy). Three modes:
#
#   # 1) from a pilot log (estimate effect, recommend N):
#   ./vla_lab/scripts/power_analysis.sh --from-log vla_lab/study_runs/pilot/study.jsonl \
#       --design paired --cond-a allocator_type --cond-b autonomy
#
#   # 2) a continuous paired DV (trust / TLX) via Cohen's d:
#   ./vla_lab/scripts/power_analysis.sh --d 0.5
#
#   # 3) explicit proportions (e.g. appropriate-reliance rates):
#   ./vla_lab/scripts/power_analysis.sh --p1 0.88 --p2 0.56

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

exec python -m vla_lab.human_study.power "$@"

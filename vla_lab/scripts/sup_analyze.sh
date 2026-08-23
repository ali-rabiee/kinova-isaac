#!/usr/bin/env bash
# ★ Regenerate every figure from a finished study directory (no re-running).
#   ./vla_lab/scripts/sup_analyze.sh vla_lab/results/tier1
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
exec python -m vla_lab.supervisory.analyze "${1:-vla_lab/results/tier1}" "${@:2}"

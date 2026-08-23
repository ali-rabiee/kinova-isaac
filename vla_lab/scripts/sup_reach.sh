#!/usr/bin/env bash
# ★ Measure the arm's reachable envelope, then set the scene geometry from it.
#
#   ./vla_lab/scripts/sup_reach.sh --dry-run
#   ./vla_lab/scripts/sup_reach.sh --headless
#
# Run this BEFORE the margin sweep. The sweep is expensive and a scene placed inside the arm's
# inner workspace fails in a way that reads like a controller problem.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
exec python -m vla_lab.supervisory.run_reach --out "${OUT:-vla_lab/results/reach}" "$@"

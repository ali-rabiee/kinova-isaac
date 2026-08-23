#!/usr/bin/env bash
# ★ Audit training runs from what they wrote to disk. No re-running.
#
# Prints, per run: provenance (image source, adaptation actually applied, split disjointness,
# peak VRAM, prompt truncation), final metrics, and any flag a reader must be told about.
#
#   ./vla_lab/scripts/sup_audit.sh vla_lab/results/models
#   ./vla_lab/scripts/sup_audit.sh vla_lab/results/models --figures
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
exec python -m vla_lab.training.audit "${1:-vla_lab/results/models}" --figures "${@:2}"

#!/usr/bin/env bash
# Print a one-line comparison from two eval JSON files (success rate + policy_backend if present).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 results_a.json results_b.json" >&2
  exit 1
fi

exec python3 -m vla_lab.smolvla_bridge.summarize_comparison "$@"

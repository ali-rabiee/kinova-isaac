#!/usr/bin/env bash
# Paper-style figures from metrics.jsonl and optional eval JSON outputs.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-vla_lab/checkpoints/tiny_v0}"

exec python -m vla_lab.plot_metrics --run-dir "${RUN_DIR}" "$@"

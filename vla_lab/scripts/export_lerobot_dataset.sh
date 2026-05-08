#!/usr/bin/env bash
# Export Kinova `vla_v1` logs to a LeRobot dataset (see `vla_lab/smoll-vla-setup.md`).
#
# Example:
#   ./vla_lab/scripts/export_lerobot_dataset.sh \\
#     --session-roots logs/data_collection/session_YYYYMMDD_HHMMSS \\
#     --out-dir vla_lab/datasets/lerobot_kinova_v0 --overwrite

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m vla_lab.smolvla_bridge.convert_kinova_to_lerobot "$@"

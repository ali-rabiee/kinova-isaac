#!/usr/bin/env bash
# ★ The session gate. Run after every session; a session that fails it is not poolable.
#
#   ./vla_lab/scripts/sup_verify.sh logs/supervisory/S000/carryover_aware
#   ./vla_lab/scripts/sup_verify.sh --root logs/supervisory --pool
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
exec python -m vla_lab.supervisory.verify_session "$@"

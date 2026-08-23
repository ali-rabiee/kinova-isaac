#!/usr/bin/env bash
# ★ The Phase 0 session gate. Run after EVERY session, before any analysis.
#
# Exit 1 means do not analyze the session as-is. Refuses on contract or prompt drift, missing or
# ambiguous arm selections above threshold, classifier-vs-coder kappa below threshold, thin
# crossover-band coverage, budget mismatch across compared conditions, unexplained clock jumps,
# un-annotated safety halts, or a missing handedness inventory (rehab.md §6/W15).
#
# Partial sessions — the participant stopped, which is their right — are ACCEPTED as partial.
#
#   ./vla_lab/old_direction/scripts/rehab_verify.sh logs/rehab/participant_P001/session_20260901_101500
#   ./vla_lab/old_direction/scripts/rehab_verify.sh --root logs/rehab --pool     # + poolability across sessions

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

exec python -m vla_lab.old_direction.rehab.verify_session "$@"

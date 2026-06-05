#!/usr/bin/env bash
# Run the offline HRI-pivot test suite (allocation / calibration / human_study).
#
# Pure Python + NumPy: no Isaac, no torch, no LeRobot, no pytest required. This is the fast
# CI gate that validates the act/compute/query allocator, the Result-2 calibration metrics,
# the fit_allocator round-trip, and the human-study pipeline.
#
#   ./vla_lab/scripts/run_tests.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

exec python -m vla_lab.tests.run_tests "$@"

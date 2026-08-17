#!/usr/bin/env bash
# Camera + table + participant-frame calibration (W13).
#
# The participant frame is re-solved for EVERY participant: the midline defines the crossover
# band and therefore the informative region of the estimand (rehab.md §9). The output's midline
# uncertainty is reported, and a session refuses to start if it exceeds tolerance.
#
#   ./vla_lab/scripts/rehab_calibrate.sh --points rig_points.json --participant P001
#
# See vla_lab/rehab/run_calibrate.py for the input JSON schema. Producing those points (camera
# capture + marker detection) is an environment dependency, not repo code (rehab.md §8).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

exec python -m vla_lab.rehab.run_calibrate "$@"

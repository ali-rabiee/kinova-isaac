#!/usr/bin/env bash
# ★ Run ONE real participant session.
#
#   ./vla_lab/old_direction/scripts/rehab_session.sh --participant P001 --participant-idx 1 \
#       --calibration logs/rehab/calibration_P001.json
#
# Before you run this on a person:
#   1. IRB approval in hand, consent taken (rehab.md §11)
#   2. ./vla_lab/old_direction/scripts/rehab_twin_dryrun.sh passes (M2)
#   3. every safety interlock demonstrated and recorded (M3, W12)
#   4. the Gen2 driver bridge is up:  ls -l ${GEN2_SOCKET:-/tmp/kinova_gen2_bridge.sock}
#   5. ./vla_lab/old_direction/scripts/rehab_calibrate.sh has been run FOR THIS PARTICIPANT
#
# The handedness inventory is administered first and is required — it defines "nonpreferred
# arm", the label the estimand is expressed in. Mixed handedness is an exclusion.
#
# APPARATUS=twin rehearses the whole flow in the simulator; APPARATUS=null rehearses it with no
# robot at all (useful for practising the keyed observer).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/old_direction/configs/rehab_phase0.yaml}"
APPARATUS="${APPARATUS:-real}"
OBSERVER="${OBSERVER:-both}"
GEN2_SOCKET="${GEN2_SOCKET:-/tmp/kinova_gen2_bridge.sock}"
LOG_ROOT="${LOG_ROOT:-logs/rehab}"

exec python -m vla_lab.old_direction.rehab.run_session \
  --config "${CONFIG}" \
  --apparatus "${APPARATUS}" \
  --observer "${OBSERVER}" \
  --gen2-socket "${GEN2_SOCKET}" \
  --log-root "${LOG_ROOT}" \
  "$@"

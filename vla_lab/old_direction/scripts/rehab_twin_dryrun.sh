#!/usr/bin/env bash
# ★ Isaac digital-twin dry-run of the apparatus (W10, milestone M2).
#
# Validates geometry, reachability, presentation trajectories, and wrist-camera framing BEFORE a
# person is anywhere near the arm. Passes when 100% of contract targets are reachable, zero
# trajectory intersections with the seated-participant proxy, and one wrist render per target.
#
# Needs Isaac Lab (conda activate riften). GEOMETRY_ONLY=1 runs the clearance half with no
# simulator at all — useful in CI and on a laptop.
#
#   ./vla_lab/old_direction/scripts/rehab_twin_dryrun.sh
#   GUI=1 ./vla_lab/old_direction/scripts/rehab_twin_dryrun.sh
#   GEOMETRY_ONLY=1 ./vla_lab/old_direction/scripts/rehab_twin_dryrun.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/old_direction/configs/rehab_twin.yaml}"
OUT_DIR="${OUT_DIR:-vla_lab/old_direction/results/rehab_phase0/twin}"

EXTRA=()
[[ -n "${GUI:-}" ]] && EXTRA+=(--gui)
[[ -n "${GEOMETRY_ONLY:-}" ]] && EXTRA+=(--geometry-only)
[[ -n "${NO_RENDER:-}" ]] && EXTRA+=(--no-render)

exec python environments/bilateral_choice/demo.py \
  --config "${CONFIG}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA[@]}" "$@"

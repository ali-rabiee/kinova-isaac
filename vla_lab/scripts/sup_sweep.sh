#!/usr/bin/env bash
# ★ The Isaac margin sweep: measure the scene physics the whole study is defined in terms of.
#
# Runs the scripted experts across clearance gaps and fits the success/duration curves. Until
# this has run, ScenePhysics.source == "prior" and every figure says so.
#
#   ./vla_lab/scripts/sup_sweep.sh --dry-run                # geometry check, no simulator
#   ./vla_lab/scripts/sup_sweep.sh --headless --fit         # the real sweep (~1-2 h)
#   REPEATS=20 ./vla_lab/scripts/sup_sweep.sh --headless --fit
#
# Needs the `riften` conda env (Isaac Sim 5.x + Isaac Lab, numpy<2).
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
exec python -m vla_lab.supervisory.run_sweep \
  --repeats "${REPEATS:-12}" --out "${OUT:-vla_lab/results/physics}" "$@"

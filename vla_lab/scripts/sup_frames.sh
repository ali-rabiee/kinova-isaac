#!/usr/bin/env bash
# ★ Re-render the scene atlas. Run this after ANY change to the scene geometry.
#
#   ./vla_lab/scripts/sup_frames.sh --headless
#
# The frames are both the training images and the paper's pictures of the task, so a geometry
# change that is not followed by a re-render silently decouples the two.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
exec python -m vla_lab.supervisory.run_frames --out "${OUT:-vla_lab/results/physics/frames}" "$@"

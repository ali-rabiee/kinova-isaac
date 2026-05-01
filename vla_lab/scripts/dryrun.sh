#!/usr/bin/env bash
# Dry-fit (VRAM + latency) for the TinyVLA inference pipeline.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
K="${K:-4}"

exec python -m vla_lab.dryrun --device "${DEVICE}" --k "${K}" "$@"

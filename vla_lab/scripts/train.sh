#!/usr/bin/env bash
# Train TinyVLA on collected data.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-vla_lab/configs/train_tiny.yaml}"

exec python -m vla_lab.train --config "${CONFIG}" "$@"

#!/usr/bin/env bash
# Minimal PyTorch GPU smoketest (random data + 10-neuron MLP).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"

exec python -m vla_lab.unity_smoketest --device "${DEVICE}" "$@"

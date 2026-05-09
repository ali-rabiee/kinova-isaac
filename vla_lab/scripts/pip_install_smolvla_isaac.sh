#!/usr/bin/env bash
# Install SmolVLA deps into an **Isaac** conda env where NumPy must stay \< 2.
#
# PyPI wheels for rerun-sdk ≥0.24 declare `numpy>=2`; Isaac / isaaclab expect
# NumPy 1.x. We satisfy Lerobot at runtime by installing `rerun-sdk` with
# `--no-deps` after NumPy 1.x is in place, then `lerobot` with `--no-deps`.
#
# Usage (repo root):
#   ./vla_lab/scripts/pip_install_smolvla_isaac.sh
#
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LEROBOT_VER="${LEROBOT_VER:-0.4.4}"
RERUN_VER="${RERUN_VER:-0.26.2}"

echo "[pip_install_smolvla_isaac] pinning lerobot==${LEROBOT_VER}, rerun-sdk==${RERUN_VER} (--no-deps for the latter)"

pip install \
  'numpy>=1.22,<2.0' \
  'num2words>=0.5.14,<0.6' \
  pillow \
  'matplotlib>=3.8,<4.0' \
  'datasets==4.1.1' \
  'diffusers==0.35.2' \
  'huggingface-hub[cli,hf-transfer]==0.35.3' \
  'accelerate>=1.10.0,<2.0.0' \
  'setuptools>=71.0.0,<81.0.0' \
  'cmake>=3.29.0.1,<4.2.0' \
  'einops>=0.8.0,<0.9.0' \
  'opencv-python-headless==4.11.0.86' \
  'av>=15.0.0,<16.0.0' \
  'jsonlines>=4.0.0,<5.0.0' \
  'packaging>=24.2,<26.0' \
  'pynput>=1.7.7,<1.9.0' \
  'pyserial>=3.5,<4.0' \
  'wandb>=0.24.0,<0.25.0' \
  'torchcodec>=0.2.1,<0.11.0' \
  'draccus==0.10.0' \
  'gymnasium==1.2.1' \
  'deepdiff>=7.0.1,<9.0.0' \
  'imageio[ffmpeg]==2.37.0' \
  'termcolor>=2.4.0,<4.0.0' \
  'transformers>=4.57.1,<5.0.0'

pip install --no-deps "rerun-sdk==${RERUN_VER}"
pip install --no-deps "lerobot[smolvla]==${LEROBOT_VER}"

echo "[pip_install_smolvla_isaac] done. Verify:  python -c 'import numpy, lerobot; print(numpy.__version__, lerobot.__version__)'"

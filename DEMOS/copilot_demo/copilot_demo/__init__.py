"""Grasp copilot IsaacSim demo package."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    """Ensure the sibling grasp-copilot repo is importable (for data_generator, llm, etc.)."""
    root = Path(__file__).resolve().parents[2]  # kinova-isaac/
    gp_root = root.parent / "grasp-copilot"
    if gp_root.exists():
        if str(gp_root) not in sys.path:
            sys.path.insert(0, str(gp_root))


_bootstrap_paths()

__all__ = [
    "quantize",
    "extractor",
    "backends",
    "executor",
    "ui_omni",
]

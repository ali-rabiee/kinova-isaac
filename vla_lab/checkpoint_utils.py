"""Checkpoint I/O helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Union

import torch

PathLike = Union[str, Path]


def torch_load_checkpoint(path: PathLike, map_location: Any = None) -> Dict[str, Any]:
    """Load a local TinyVLA checkpoint (.pt).

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
    numpy objects stored in our training checkpoints. Local checkpoints are trusted.
    """
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)

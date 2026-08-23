"""Backbone implementations. Heavy ones import lazily so the package stays light."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

from .base import Backbone, BackboneOutput


def _accepted(cls, kw: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Split ``kw`` into what ``cls.__init__`` takes and what it does not.

    Backbone options travel with a checkpoint's manifest, and they are not the same set for every
    backbone: the from-scratch model needs the vocabulary size the trainer derived from its
    corpus, and a pretrained VLA has no such parameter. Reloading a SmolVLA checkpoint through
    the generic path therefore handed it ``vocab_size`` and died on a ``TypeError`` --- an
    unhelpful failure for something the caller could not reasonably have known.

    Dropping the surplus is right, but doing it silently is not: the returned tuple lets the
    caller record what was ignored, so a genuinely misspelled option is visible instead of
    quietly having no effect.
    """
    sig = inspect.signature(cls.__init__)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kw), ()
    names = {p for p in sig.parameters if p != "self"}
    keep = {k: v for k, v in kw.items() if k in names}
    return keep, tuple(sorted(set(kw) - names))


def build_backbone(kind: str, **kw: Any) -> Backbone:
    """Instantiate a backbone, ignoring options it does not accept (and saying which)."""
    k = str(kind)
    if k == "tiny":
        from .tiny import TinyBackbone

        cls, extra = TinyBackbone, {}
    elif k == "smolvla":
        from .smolvla import SmolVLABackbone

        cls, extra = SmolVLABackbone, {}
    elif k.startswith("qwen"):
        from .qwen import QwenVLBackbone

        cls, extra = QwenVLBackbone, {"model_id": kw.pop("model_id", None) or _QWEN_IDS[k]}
    else:
        raise KeyError(f"unknown backbone {kind!r}")

    keep, dropped = _accepted(cls, {**kw, **extra})
    backbone = cls(**keep)
    backbone.ignored_options = dropped          # type: ignore[attr-defined]
    return backbone


_QWEN_IDS = {
    "qwen2vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen25vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}

__all__ = ["Backbone", "BackboneOutput", "build_backbone"]

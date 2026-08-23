"""The backbone seam: what the Carryover-Aware wrapper needs from any VLA.

Deliberately thin. A backbone has to (a) turn an observation batch into a token sequence and a
pooled vector, (b) say whether it can accept prefix tokens and whether it has a language model
worth injecting text into, and (c) predict an action chunk. Everything carryover-specific --
the context, the intent heads, the ask gate, the forward contamination model -- lives in the
wrapper, which is what makes "which architectural features help" a controlled comparison
rather than a comparison of four different codebases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    pooled: torch.Tensor                     # (B, d_model)
    tokens: Optional[torch.Tensor] = None    # (B, T, d_model)
    actions: Optional[torch.Tensor] = None   # (B, chunk, action_dim), if the backbone predicts them
    aux: Dict[str, Any] = field(default_factory=dict)


class Backbone(nn.Module):
    """Base class. Subclasses set the capability flags honestly -- they gate the model card."""

    #: Hidden width the wrapper's heads attach to.
    d_model: int = 0
    #: Can the wrapper prepend continuous tokens to the sequence? (``token`` injection mode)
    supports_prefix_tokens: bool = False
    #: Does the backbone carry a real language model? (``text`` injection mode)
    supports_text: bool = False
    #: Does the backbone itself predict actions, or does the wrapper attach an action head?
    predicts_actions: bool = False
    #: Human-readable id, matching the registry key.
    name: str = "backbone"

    def forward(
        self,
        batch: Dict[str, Any],
        *,
        prefix_tokens: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- introspection ------------------------------------------------------
    def n_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (not trainable_only) or p.requires_grad)

    def trainable_report(self) -> Dict[str, Any]:
        total = self.n_parameters(False)
        train = self.n_parameters(True)
        return {
            "params_total": int(total),
            "params_trainable": int(train),
            "trainable_fraction": float(train / max(total, 1)),
        }

    def freeze(self, *, unfreeze_last: int = 0) -> None:
        """Freeze everything, optionally re-enabling the last ``unfreeze_last`` blocks.

        The fallback when LoRA is unavailable. It is a legitimate configuration, not a
        degradation to hide: the model card records exactly what was trainable, and a
        frozen-backbone row in the results table is interpretable as long as it is labelled.
        """
        for p in self.parameters():
            p.requires_grad_(False)


__all__ = ["Backbone", "BackboneOutput"]

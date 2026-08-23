"""Torch side of the carryover-context injection. Imported only when torch is present."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .context import CONTEXT_DIM, CONTEXT_FILM, CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN


class ContextEncoder(nn.Module):
    """Turns the numeric context vector into whichever injection the mode asks for.

    ``text`` and ``none`` produce nothing here: ``text`` is injected upstream, into the prompt
    string, before the backbone tokenises anything, and ``none`` is the ablation. Keeping all
    four modes behind one module means the training script and the model card never have to
    branch on mode, which is what keeps the comparison apples-to-apples.
    """

    def __init__(self, mode: str, *, d_model: int, n_tokens: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.mode = str(mode)
        self.d_model = int(d_model)
        self.n_tokens = int(n_tokens)
        if self.mode == CONTEXT_TOKEN:
            self.net = nn.Sequential(
                nn.Linear(CONTEXT_DIM, hidden), nn.GELU(), nn.Linear(hidden, self.n_tokens * self.d_model)
            )
        elif self.mode == CONTEXT_FILM:
            self.net = nn.Sequential(nn.Linear(CONTEXT_DIM, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.d_model))
            # Start as the identity so an untrained FiLM head does not scramble a pretrained
            # backbone's features on step zero.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = None

    @property
    def emits_tokens(self) -> bool:
        return self.mode == CONTEXT_TOKEN

    @property
    def emits_film(self) -> bool:
        return self.mode == CONTEXT_FILM

    def tokens(self, ctx: torch.Tensor) -> Optional[torch.Tensor]:
        """(B, CONTEXT_DIM) -> (B, n_tokens, d_model), or ``None`` outside ``token`` mode."""
        if self.mode != CONTEXT_TOKEN:
            return None
        return self.net(ctx).view(ctx.size(0), self.n_tokens, self.d_model)

    def film(self, ctx: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """(B, CONTEXT_DIM) -> (gamma, beta), or ``None`` outside ``film`` mode."""
        if self.mode != CONTEXT_FILM:
            return None
        gb = self.net(ctx)
        gamma, beta = gb.chunk(2, dim=-1)
        return 1.0 + gamma, beta

    def apply_film(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gb = self.film(ctx)
        if gb is None:
            return h
        gamma, beta = gb
        while gamma.dim() < h.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return h * gamma + beta


__all__ = ["ContextEncoder"]

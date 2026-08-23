r"""The two heads that make a policy carryover-aware, plus the forward contamination model.

**IntentHead.** Two logits, from one shared trunk:

``said``
    What the supervisor's utterance means -- plain grounding. Any VLA needs this to act at all.
``unprompted``
    What they would have said with no recent coaching. This is the de-biased estimate, and it
    is the actual object of the paper.

Splitting them is what makes the claim testable. A model that has learned nothing about
compliance will simply set ``unprompted = said``; the gap between the two heads is the model's
own estimate of how much of the instruction is echo, and it can be scored directly against the
supervisor's uncoached preference.

**The forward model is the load-bearing piece.** Rather than only supervising ``unprompted``
against a label -- which needs an uncontaminated reference the model will not have at
deployment -- the training objective also pushes the model's *own* de-biased belief back
through the known contamination model and requires it to reproduce what was actually said:

.. math::
    \Pr[\text{said}=A] \;=\; \sigma\big(\operatorname{logit}\Pr[\text{unprompted}=A]
                                        + \rho\,\hat\beta\,\kappa\big)

This is self-supervised, it needs no reference block, and it is what makes the de-biasing
identifiable from ordinary interaction: the model is not free to invent an unprompted belief,
because that belief has to explain the observed utterance under the residue the belief module
reports. :class:`ForwardContamination` implements it, with ``beta`` either supplied by the
belief module or learned per-model.

**AskGateHead.** One logit: is this an occasion to re-open the option the coaching closed? A
counter-proposal costs the supervisor time and attention, so this head is trained against
whether countering would actually have changed the answer -- not against "is kappa large",
which any thresholding rule can already do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentHead(nn.Module):
    """Grounding + de-biasing over a binary strategy axis."""

    def __init__(self, d_model: int, *, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.said = nn.Linear(hidden, 2)
        # The de-biased head is parameterised as a *correction* to the grounded one and
        # initialised at zero, so an untrained model starts by believing exactly what it was
        # told. That is the right null: the burden is on the model to earn any departure from
        # the instruction, and "unprompted == said" is precisely the Memoryless baseline.
        self.delta = nn.Linear(hidden, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(h)
        said = self.said(z)
        return said, said + self.delta(z)


class AskGateHead(nn.Module):
    """One logit: would re-opening the alternative change what this supervisor says?"""

    def __init__(self, d_model: int, *, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class ForwardContamination(nn.Module):
    r"""``logit P[said=A] = logit P[unprompted=A] + rho * beta * kappa``.

    ``beta`` is the supervisor's compliance sensitivity. Two sources, both supported:

    ``supplied``
        The belief module's posterior mean, passed in per sample. This is what a deployed
        system does -- the estimate already exists and is calibrated.
    ``learned``
        A single global scalar, fitted with the model. Useful as an ablation: if a model with
        a learned global ``beta`` matches one given the per-person estimate, then personalising
        the compliance sensitivity was not worth its machinery, and that is worth knowing.
    """

    def __init__(self, *, learn_beta: bool = False, init_beta: float = 1.0) -> None:
        super().__init__()
        self.learn_beta = bool(learn_beta)
        self.log_beta = nn.Parameter(torch.tensor(float(init_beta)).clamp_min(1e-3).log())

    def beta(self, supplied: Optional[torch.Tensor]) -> torch.Tensor:
        if self.learn_beta or supplied is None:
            return self.log_beta.exp()
        return supplied

    def forward(
        self,
        unprompted_logits: torch.Tensor,
        kappa: torch.Tensor,
        *,
        beta: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicted ``said`` logits implied by the model's own de-biased belief."""
        b = self.beta(beta)
        r = torch.ones_like(kappa) if rho is None else rho
        shift = (r * b * kappa).view(-1, 1)
        # Binary logit difference: add the shift to class A, leave class B.
        return unprompted_logits + torch.cat([shift, torch.zeros_like(shift)], dim=-1)


__all__ = ["IntentHead", "AskGateHead", "ForwardContamination"]

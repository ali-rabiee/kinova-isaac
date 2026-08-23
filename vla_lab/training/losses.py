r"""The training objective.

.. math::
    \mathcal{L} = \underbrace{\lambda_a \mathcal{L}_{\text{act}}}_{\text{can it do the task}}
    + \underbrace{\lambda_s \mathcal{L}_{\text{said}}}_{\text{grounding}}
    + \underbrace{\lambda_f \mathcal{L}_{\text{fwd}}}_{\text{consistency}}
    + \underbrace{\lambda_u \mathcal{L}_{\text{unprompted}}}_{\text{reference, where it exists}}
    + \underbrace{\lambda_c \mathcal{L}_{\text{anti-copy}}}_{\text{the compliance penalty}}
    + \underbrace{\lambda_g \mathcal{L}_{\text{ask}}}_{\text{when to re-open the option}}

Each term, and why it is not redundant with the others:

``L_said``
    Cross-entropy on the grounded reading of the utterance. Every VLA needs this; it is not the
    contribution, but omitting it lets the model satisfy the other terms with a de-biased
    belief that is untethered to what was actually said.

``L_fwd`` -- **the load-bearing term**
    The model's own de-biased belief, pushed forward through the contamination model, must
    reproduce the observed instruction:
    ``sigma(logit P[unprompted=A] + rho*beta*kappa) ~ said``. This is self-supervised: it needs
    no uncontaminated reference, so it is available from ordinary interaction and therefore
    available after deployment. It is also what makes de-biasing *identifiable* -- the model
    cannot invent an unprompted belief, because that belief has to explain the utterance under
    the residue the belief module reports.

``L_unprompted``
    Cross-entropy against a reference label, used only on samples that have one (a no-coach
    reference block, or -- in simulation -- the supervisor's true preference). Masked, because
    most samples do not have one, and pretending otherwise is how a model learns to expect a
    supervision signal it will never see again.

``L_anti-copy`` -- the compliance penalty
    A hinge that charges the model for agreeing with the instruction *in proportion to how
    contaminated that instruction is*: when ``|kappa|`` is large and the instruction points the
    same way the coaching did, matching it without any de-biasing costs something. This is the
    penalty the project proposal asks for, written so that it does exactly one thing -- it is
    zero when the coaching and the instruction disagree, zero when there was no coaching, and
    it never pushes the belief *past* neutral. A model can always satisfy it by having actually
    learned the person's preference; it cannot satisfy it by copying.

``L_ask``
    Binary cross-entropy on whether a counter-proposal would have flipped the answer. Trained
    against the counterfactual rather than against ``|kappa| > threshold``, because the latter
    is a rule the belief module already implements and a head that learned it would add nothing.

Every weight is ablatable, and the paper reports the objective with ``L_fwd`` and
``L_anti-copy`` removed, because a de-biasing claim whose mechanism has not been ablated is an
assertion about a loss function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class CarryoverLossConfig:
    w_action: float = 1.0
    w_said: float = 1.0
    w_forward: float = 1.0
    w_unprompted: float = 1.0
    w_anti_copy: float = 0.5
    w_ask: float = 0.3
    #: Contamination below this (logits) is treated as clean, so the anti-copy term does not
    #: fire on noise. Matches the scheduler's ``clean_tau``.
    anti_copy_tau: float = 0.15
    #: Huber delta for the action loss; robust to the occasional bad demonstration chunk.
    action_huber_delta: float = 1.0
    label_smoothing: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CarryoverLossOutput:
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]

    def item_parts(self) -> Dict[str, float]:
        return {k: float(v.detach()) for k, v in self.parts.items()}


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean() if x.numel() else x.new_zeros(())
    m = mask.to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def carryover_loss(
    out,
    batch: Dict[str, Any],
    cfg: Optional[CarryoverLossConfig] = None,
) -> CarryoverLossOutput:
    """Compute the objective. ``out`` is a
    :class:`~vla_lab.policy.carryover_vla.CarryoverVLAOutput`.

    Expected batch keys (all optional -- each term is masked on its own availability, so one
    dataloader can mix manipulation-only and dialogue-only samples):

    ``action``, ``action_mask``            (B, T, A), (B,)
    ``said``, ``said_mask``                (B,) long in {0,1}, (B,)
    ``unprompted``, ``unprompted_mask``    (B,) long in {0,1}, (B,)
    ``kappa``, ``rho``, ``beta``           (B,)
    ``coach_direction``                    (B,) in {-1, 0, +1}
    ``ask_label``, ``ask_mask``            (B,), (B,)
    """
    cfg = cfg or CarryoverLossConfig()
    parts: Dict[str, torch.Tensor] = {}
    device = out.said.device
    zero = torch.zeros((), device=device)

    # -- action ------------------------------------------------------------
    if out.actions is not None and batch.get("action") is not None:
        tgt = batch["action"].to(out.actions.dtype)
        per = F.huber_loss(out.actions, tgt, delta=float(cfg.action_huber_delta), reduction="none")
        per = per.flatten(1).mean(1)
        parts["action"] = _masked_mean(per, batch.get("action_mask"))
    else:
        parts["action"] = zero

    # -- grounding ---------------------------------------------------------
    if batch.get("said") is not None:
        per = F.cross_entropy(out.said, batch["said"].long(), reduction="none",
                              label_smoothing=float(cfg.label_smoothing))
        parts["said"] = _masked_mean(per, batch.get("said_mask"))
    else:
        parts["said"] = zero

    # -- forward consistency: the de-biased belief must explain what was said ----
    if batch.get("said") is not None:
        per = F.cross_entropy(out.said_from_unprompted, batch["said"].long(), reduction="none",
                              label_smoothing=float(cfg.label_smoothing))
        parts["forward"] = _masked_mean(per, batch.get("said_mask"))
    else:
        parts["forward"] = zero

    # -- reference supervision, where a reference exists --------------------
    if batch.get("unprompted") is not None:
        per = F.cross_entropy(out.unprompted, batch["unprompted"].long(), reduction="none",
                              label_smoothing=float(cfg.label_smoothing))
        parts["unprompted"] = _masked_mean(per, batch.get("unprompted_mask"))
    else:
        parts["unprompted"] = zero

    # -- the compliance penalty --------------------------------------------
    kappa = batch.get("kappa")
    if kappa is not None and batch.get("said") is not None:
        k = kappa.to(out.said.dtype)
        said = batch["said"].long()
        # +1 when the instruction points the way the coaching pushed, -1 when it opposes it.
        said_sign = torch.where(said == 0, torch.ones_like(k), -torch.ones_like(k))  # class 0 == strategy A
        agrees = (torch.sign(k) * said_sign > 0).to(k.dtype)
        strength = (k.abs() - float(cfg.anti_copy_tau)).clamp_min(0.0)
        # How far the model moved its belief away from the instruction, in the direction that
        # undoes the coaching. Clamped at zero from below (no credit for agreeing harder) and
        # at `strength` from above (no credit for overshooting past what the residue can explain).
        moved = (-torch.sign(k) * out.debias_gap()).clamp(min=0.0)
        hinge = (strength - moved).clamp_min(0.0)
        parts["anti_copy"] = _masked_mean(agrees * hinge, batch.get("said_mask"))
    else:
        parts["anti_copy"] = zero

    # -- ask gate ----------------------------------------------------------
    if batch.get("ask_label") is not None:
        per = F.binary_cross_entropy_with_logits(out.ask, batch["ask_label"].to(out.ask.dtype), reduction="none")
        parts["ask"] = _masked_mean(per, batch.get("ask_mask"))
    else:
        parts["ask"] = zero

    total = (
        cfg.w_action * parts["action"]
        + cfg.w_said * parts["said"]
        + cfg.w_forward * parts["forward"]
        + cfg.w_unprompted * parts["unprompted"]
        + cfg.w_anti_copy * parts["anti_copy"]
        + cfg.w_ask * parts["ask"]
    )
    return CarryoverLossOutput(total=total, parts=parts)


__all__ = ["CarryoverLossConfig", "CarryoverLossOutput", "carryover_loss"]

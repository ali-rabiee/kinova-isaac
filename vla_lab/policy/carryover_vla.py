r"""The Carryover-Aware VLA wrapper.

One class, any backbone. It adds the three carryover-specific pieces and nothing else, so that
the model table compares *architectural features* rather than four unrelated implementations:

.. code-block:: text

    observation ---------------------------\
    instruction (+ verbalised context) ------> backbone --> pooled, tokens, [actions]
    carryover context ---> encoder ---------/       |
                             |  (token: prefix)     |
                             |  (film:  modulate) --+--> IntentHead   -> said, unprompted
                                                    +--> AskGateHead  -> counter-propose?
                                                    +--> ActionHead   -> action chunk

The output is deliberately richer than an action chunk, because the thing being studied is a
*belief*, not a trajectory:

``said``
    The grounded reading of the supervisor's utterance.
``unprompted``
    What the model thinks they would have said cold. Initialised as an exact copy of ``said``,
    so an untrained model is precisely the Memoryless baseline and every departure has to be
    learned.
``ask``
    Whether to spend a counter-proposal.
``said_from_unprompted``
    ``unprompted`` pushed forward through the contamination model. The self-supervised
    consistency signal that makes de-biasing identifiable without a reference block.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .backbones import Backbone, build_backbone
from .context import CONTEXT_DIM, CONTEXT_FILM, CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN, CarryoverContext
from .heads import AskGateHead, ForwardContamination, IntentHead


@dataclass
class CarryoverVLAConfig:
    model_key: str = "tiny"
    backbone: str = "tiny"
    context_mode: str = CONTEXT_TOKEN
    adapt: str = "full"
    n_context_tokens: int = 4
    head_hidden: int = 256
    dropout: float = 0.1
    action_dim: int = 7
    chunk_len: int = 8
    learn_beta: bool = False
    #: LoRA settings, used only when ``adapt == "lora"``.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    #: Blocks to unfreeze when ``adapt == "frozen"`` and LoRA is unavailable.
    unfreeze_last: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CarryoverVLAConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CarryoverVLAOutput:
    said: torch.Tensor                      # (B, 2)
    unprompted: torch.Tensor                # (B, 2)
    ask: torch.Tensor                       # (B,)
    said_from_unprompted: torch.Tensor      # (B, 2)
    actions: Optional[torch.Tensor] = None  # (B, chunk, action_dim)
    pooled: Optional[torch.Tensor] = None

    def debias_gap(self) -> torch.Tensor:
        """How far the model's de-biased belief departs from what it was told, in logits.

        Zero means the model is behaving as a Memoryless VLA on this sample. Reported as a
        diagnostic because a model can score well on the de-biasing label while leaving this at
        zero everywhere and simply being right about the population -- and that would not be
        de-biasing.
        """
        d_said = self.said[:, 0] - self.said[:, 1]
        d_un = self.unprompted[:, 0] - self.unprompted[:, 1]
        return d_un - d_said


class CarryoverVLA(nn.Module):
    def __init__(self, cfg: CarryoverVLAConfig, **backbone_kw: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone: Backbone = build_backbone(cfg.backbone, **backbone_kw)
        d = int(self.backbone.d_model)

        if cfg.context_mode == CONTEXT_TOKEN and not self.backbone.supports_prefix_tokens:
            raise ValueError(f"backbone {cfg.backbone!r} cannot take prefix tokens")
        if cfg.context_mode == CONTEXT_TEXT and not self.backbone.supports_text:
            raise ValueError(f"backbone {cfg.backbone!r} has no language model to inject text into")

        from ._torch_context import ContextEncoder

        self.context = ContextEncoder(cfg.context_mode, d_model=d, n_tokens=cfg.n_context_tokens)
        self.intent = IntentHead(d, hidden=cfg.head_hidden, dropout=cfg.dropout)
        self.ask_gate = AskGateHead(d, hidden=max(64, cfg.head_hidden // 2), dropout=cfg.dropout)
        self.forward_model = ForwardContamination(learn_beta=bool(cfg.learn_beta))
        self.action_head: Optional[nn.Module] = None
        if not self.backbone.predicts_actions:
            self.action_head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, cfg.head_hidden),
                nn.GELU(),
                nn.Linear(cfg.head_hidden, cfg.chunk_len * cfg.action_dim),
            )
        self._apply_adaptation()

    # -- adaptation ---------------------------------------------------------
    def _apply_adaptation(self) -> None:
        """Full fine-tune, LoRA, or frozen-with-heads.

        The LoRA path degrades **loudly**: if ``peft`` is missing the backbone is frozen
        instead and :attr:`adapt_report` says so, rather than silently full-fine-tuning a 3B
        model onto a 16 GB card and dying at step one. Which path ran is recorded in the run
        manifest and printed in the model table, because "frozen backbone" and "LoRA" are
        different experiments and a reader must be able to tell them apart.
        """
        self.adapt_report: Dict[str, Any] = {"requested": self.cfg.adapt, "applied": self.cfg.adapt}
        if self.cfg.adapt == "full":
            return
        if self.cfg.adapt == "lora":
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore

                targets = getattr(self.backbone, "lora_target_modules", None)
                lcfg = LoraConfig(
                    r=int(self.cfg.lora_r),
                    lora_alpha=int(self.cfg.lora_alpha),
                    lora_dropout=float(self.cfg.lora_dropout),
                    bias="none",
                    target_modules=targets or ["q_proj", "k_proj", "v_proj", "o_proj"],
                )
                inner = getattr(self.backbone, "lm", None) or self.backbone
                wrapped = get_peft_model(inner, lcfg)
                if getattr(self.backbone, "lm", None) is not None:
                    self.backbone.lm = wrapped  # type: ignore[attr-defined]
                else:
                    self.backbone = wrapped  # type: ignore[assignment]
                self.adapt_report["lora_targets"] = list(lcfg.target_modules)
                return
            except ImportError:
                self.adapt_report.update(
                    {"applied": "frozen", "reason": "peft not installed; `pip install peft` to enable LoRA"}
                )
            except Exception as exc:  # pragma: no cover - backbone-specific target mismatch
                self.adapt_report.update({"applied": "frozen", "reason": f"LoRA setup failed: {exc}"})
        self.backbone.freeze(unfreeze_last=int(self.cfg.unfreeze_last))

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        batch: Dict[str, Any],
        *,
        context: Optional[torch.Tensor] = None,
        kappa: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> CarryoverVLAOutput:
        """``context`` is (B, CONTEXT_DIM) from :meth:`CarryoverContext.features`.

        In ``text`` mode the context has already been folded into ``batch["instruction"]``
        upstream (see :func:`prepare_batch`), so ``context`` is used only for the numeric modes.
        """
        b = _batch_size(batch)
        device = _batch_device(batch)
        ctx = context if context is not None else torch.zeros(b, CONTEXT_DIM, device=device)
        kap = kappa if kappa is not None else torch.zeros(b, device=device)

        prefix = self.context.tokens(ctx) if self.context.emits_tokens else None
        out = self.backbone(batch, prefix_tokens=prefix)
        h = self.context.apply_film(out.pooled, ctx) if self.context.emits_film else out.pooled

        said, unprompted = self.intent(h)
        ask = self.ask_gate(h)
        said_hat = self.forward_model(unprompted, kap, beta=beta, rho=rho)

        actions = out.actions
        if actions is None and self.action_head is not None:
            actions = self.action_head(h).view(b, self.cfg.chunk_len, self.cfg.action_dim)

        return CarryoverVLAOutput(
            said=said, unprompted=unprompted, ask=ask, said_from_unprompted=said_hat,
            actions=actions, pooled=h,
        )

    # -- introspection ------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_key": self.cfg.model_key,
            "backbone": self.cfg.backbone,
            "context_mode": self.cfg.context_mode,
            "adapt": self.adapt_report,
            "params_total": int(total),
            "params_trainable": int(train),
            "trainable_fraction": float(train / max(total, 1)),
            "config": self.cfg.to_dict(),
        }


def prepare_instruction(instruction: str, ctx: CarryoverContext, mode: str,
                        axis_labels=("CLEAR_FIRST", "DIRECT"), *, compact: bool = False) -> str:
    """Fold the carryover context into the prompt, for the ``text`` injection mode only.

    Kept as a free function so the training script, the evaluation loop, and the closed-loop
    Isaac runner all build the prompt exactly the same way. Prompt drift between training and
    deployment is the classic silent failure of text-conditioned policies.
    """
    if mode != CONTEXT_TEXT:
        return instruction
    return f"{ctx.to_text(axis_labels=axis_labels, compact=compact)}\nSupervisor: {instruction}"


def _batch_size(batch: Dict[str, Any]) -> int:
    for k in ("image", "image_wrist", "state", "lang_ids", "pixel_values", "input_ids"):
        v = batch.get(k)
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
    raise ValueError("cannot infer batch size")


def _batch_device(batch: Dict[str, Any]) -> torch.device:
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


__all__ = ["CarryoverVLAConfig", "CarryoverVLAOutput", "CarryoverVLA", "prepare_instruction"]

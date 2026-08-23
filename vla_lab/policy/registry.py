"""The model roster, and the card that turns it into the paper's architecture table.

Each entry declares the **architectural features** the paper is actually comparing, so the
results table can be read as an experiment rather than a leaderboard. The columns are chosen
to be the plausible causes of a difference in de-biasing ability:

``pretrained``
    ``none`` / ``vla`` (pretrained on robot data) / ``vlm`` (pretrained on web image-text).
``language``
    Does the model carry a language model, or only a token embedding? This gates the ``text``
    context mode -- and it is the feature the central hypothesis is about, because a prior over
    persuasion, compliance and hedging is a *language* prior.
``action_head``
    ``regress`` / ``flow`` / ``token``. Included because it is the axis VLA papers usually
    compare on, and the paper needs to show that it is *not* the axis that matters here.
``context_modes``
    Which injection mechanisms the backbone can physically accept.
``adapt``
    How the backbone is adapted: full fine-tune, LoRA, or frozen with trainable heads.

A model may appear in the table more than once under different context modes; that is the
point, since context mode is the independent variable and backbone is the blocking factor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .context import CONTEXT_FILM, CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN

PRETRAIN_NONE = "none"
PRETRAIN_VLA = "vla"
PRETRAIN_VLM = "vlm"

ADAPT_FULL = "full"
ADAPT_LORA = "lora"
ADAPT_FROZEN = "frozen"


@dataclass
class ModelCard:
    key: str
    backbone: str
    display: str
    params: str
    pretrained: str
    language: bool
    action_head: str
    context_modes: Tuple[str, ...]
    default_adapt: str
    #: Approximate peak VRAM for training at the default settings, in GB. Advisory: the
    #: trainer measures the real figure and records it in the run manifest.
    vram_gb_hint: float = 2.0
    #: Tokens this backbone will accept for the instruction. Decides whether the verbalised
    #: carryover context can be given in full or must be compacted. It is not a detail: SmolVLA
    #: budgets 48 tokens and truncates from the LEFT, so a 77-token prompt loses precisely the
    #: informative half of the context and the run reads as "verbalised context does not work".
    instruction_budget: int = 48
    #: Which verbalisation fits that budget alongside the instruction itself.
    context_style: str = "compact"
    hf_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["context_modes"] = list(self.context_modes)
        return d


MODEL_CARDS: Dict[str, ModelCard] = {
    "tiny": ModelCard(
        key="tiny",
        backbone="tiny",
        display="TinyVLA-2M (from scratch)",
        params="~2.0M",
        pretrained=PRETRAIN_NONE,
        language=False,
        action_head="regress",
        context_modes=(CONTEXT_NONE, CONTEXT_TOKEN, CONTEXT_FILM),
        default_adapt=ADAPT_FULL,
        vram_gb_hint=1.5,
        instruction_budget=32,
        context_style="compact",
        notes="The floor. No pretraining, byte-level token embedding, no language model. "
              "Present to separate 'has a context input' from 'has a prior about persuasion'.",
    ),
    "smolvla": ModelCard(
        key="smolvla",
        backbone="smolvla",
        display="SmolVLA-450M",
        params="~450M",
        pretrained=PRETRAIN_VLA,
        language=True,
        action_head="flow",
        context_modes=(CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN, CONTEXT_FILM),
        default_adapt=ADAPT_FULL,
        vram_gb_hint=9.0,
        instruction_budget=48,
        context_style="compact",
        hf_id="lerobot/smolvla_base",
        notes="Pretrained on robot demonstrations with a flow-matching action expert. The "
              "closest thing in the roster to a purpose-built VLA.",
    ),
    "qwen2vl-2b": ModelCard(
        key="qwen2vl-2b",
        backbone="qwen2vl-2b",
        display="Qwen2-VL-2B + action head",
        params="~2.2B",
        pretrained=PRETRAIN_VLM,
        language=True,
        action_head="regress",
        context_modes=(CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN, CONTEXT_FILM),
        default_adapt=ADAPT_LORA,
        vram_gb_hint=11.0,
        instruction_budget=512,
        context_style="rich",
        hf_id="Qwen/Qwen2-VL-2B-Instruct",
        notes="A web-pretrained VLM adapted to actions. Strong language prior, no robot "
              "pretraining -- the complement of SmolVLA, which is why both are in the table.",
    ),
    "qwen25vl-3b": ModelCard(
        key="qwen25vl-3b",
        backbone="qwen25vl-3b",
        display="Qwen2.5-VL-3B + action head",
        params="~3.8B",
        pretrained=PRETRAIN_VLM,
        language=True,
        action_head="regress",
        context_modes=(CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN, CONTEXT_FILM),
        default_adapt=ADAPT_LORA,
        vram_gb_hint=14.0,
        instruction_budget=512,
        context_style="rich",
        hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
        notes="The largest model that trains on a 16 GB card with LoRA, bf16 and gradient "
              "checkpointing. Tests whether the language prior keeps paying past 2B.",
    ),
}


def available_models() -> List[str]:
    return list(MODEL_CARDS)


def describe_models(keys: Optional[Sequence[str]] = None) -> str:
    """The architecture table, as text. Same content as the paper's table."""
    ks = list(keys) if keys else available_models()
    head = (
        f"{'model':30s}{'params':>9}{'pretrain':>10}{'lang':>6}{'action':>9}"
        f"{'adapt':>8}{'VRAM~':>7}{'lang tok':>7}  context modes"
    )
    lines = [head, "-" * (len(head) + 20)]
    for k in ks:
        c = MODEL_CARDS[k]
        lines.append(
            f"{c.display:30s}{c.params:>9}{c.pretrained:>10}{'yes' if c.language else 'no':>6}"
            f"{c.action_head:>9}{c.default_adapt:>8}{c.vram_gb_hint:>6.0f}G"
            f"{c.instruction_budget:>7}  {', '.join(c.context_modes)}"
        )
    return "\n".join(lines)


def build_model(
    key: str,
    *,
    context_mode: str = CONTEXT_TOKEN,
    adapt: Optional[str] = None,
    **kw: Any,
):
    """Instantiate one roster entry as a :class:`~vla_lab.policy.carryover_vla.CarryoverVLA`."""
    from .carryover_vla import CarryoverVLA, CarryoverVLAConfig

    card = MODEL_CARDS[str(key)]
    if context_mode not in card.context_modes:
        raise ValueError(
            f"{card.display} cannot take context mode {context_mode!r}; it supports {card.context_modes}. "
            f"(A backbone with no language model cannot be given a verbalised context.)"
        )
    cfg = CarryoverVLAConfig(
        model_key=card.key,
        backbone=card.backbone,
        context_mode=str(context_mode),
        adapt=str(adapt or card.default_adapt),
        **{k: v for k, v in kw.items() if k in CarryoverVLAConfig.__dataclass_fields__},
    )
    # Anything not consumed by the wrapper config is a backbone option (e.g. the vocabulary
    # size the trainer just derived from the corpus, or a HF model id).
    backbone_kw = {k: v for k, v in kw.items() if k not in CarryoverVLAConfig.__dataclass_fields__}
    return CarryoverVLA(cfg, **backbone_kw)


__all__ = [
    "PRETRAIN_NONE",
    "PRETRAIN_VLA",
    "PRETRAIN_VLM",
    "ADAPT_FULL",
    "ADAPT_LORA",
    "ADAPT_FROZEN",
    "ModelCard",
    "MODEL_CARDS",
    "available_models",
    "describe_models",
    "build_model",
]

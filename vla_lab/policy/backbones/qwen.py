"""Qwen2-VL / Qwen2.5-VL backbone with a regression action head.

The web-pretrained VLM arm of the roster. Where SmolVLA brings a prior about *robot* data,
Qwen brings a prior about *language* -- including, the central hypothesis says, about
persuasion, agreement, hedging and deference, which is what de-biasing an instruction requires.
Having both in the table is what lets the paper attribute a difference to the kind of
pretraining rather than to scale.

**Fitting on 16 GB.** The defaults are chosen to train, not to be fast: bf16, gradient
checkpointing, a frozen vision tower, LoRA on the attention projections, and batch size 1 with
accumulation. The vision tower is frozen for a reason beyond memory -- the visual difference
between a 4 cm and a 7 cm gap is a fine-grained judgement the pretrained tower already makes
better than a few hundred fine-tuning steps would teach it.

Images are passed through the processor at a **capped pixel budget**: Qwen's dynamic-resolution
tokenizer will happily turn two camera views into thousands of visual tokens, which is where
out-of-memory failures actually come from on this card.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .base import Backbone, BackboneOutput

DEFAULT_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenVLBackbone(Backbone):
    name = "qwen"
    supports_prefix_tokens = True
    supports_text = True
    predicts_actions = False
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ID,
        max_pixels: int = 200_704,   # 256 * 28 * 28 -> ~256 visual tokens per view
        min_pixels: int = 50_176,
        dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
        freeze_vision: bool = True,
        attn_implementation: Optional[str] = "sdpa",
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoProcessor  # type: ignore

        self.model_id = str(model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=int(min_pixels), max_pixels=int(max_pixels)
        )
        cls = _model_class(self.model_id)
        self.lm = cls.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, str(dtype)),
            attn_implementation=attn_implementation,
        )
        if gradient_checkpointing:
            self.lm.gradient_checkpointing_enable()
            self.lm.config.use_cache = False
        if freeze_vision:
            for name, p in self.lm.named_parameters():
                if "visual" in name or "vision" in name:
                    p.requires_grad_(False)
        cfg = AutoConfig.from_pretrained(self.model_id)
        self.d_model = int(getattr(getattr(cfg, "text_config", cfg), "hidden_size", 2048))

    def freeze(self, *, unfreeze_last: int = 0) -> None:
        for p in self.lm.parameters():
            p.requires_grad_(False)
        if unfreeze_last > 0:
            layers = _decoder_layers(self.lm)
            for m in layers[-int(unfreeze_last):]:
                for p in m.parameters():
                    p.requires_grad_(True)

    def _encode(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Turn ``prompt`` strings and ``image`` tensors into Qwen's own inputs.

        This used to be documented as the collator's job and was never actually implemented
        there, so every Qwen cell died on ``embedding(): argument 'indices' must be Tensor, not
        NoneType`` -- the model was handed pixel values and no token ids at all. Doing it here
        costs image processing on the training thread, which at batch size 1 with LoRA is not the
        bottleneck, and it keeps the collator backbone-agnostic like every other path.

        The instruction goes through the chat template because that is the format the instruction
        tuning used; feeding a bare string to an instruction-tuned VLM asks it to do the task in a
        format it never saw, which is a silent handicap rather than an error.
        """
        import numpy as np
        from PIL import Image

        prompts: Sequence[str] = batch.get("prompt") or [""] * _batch_len(batch)
        imgs = batch.get("image")
        pil: List[Image.Image] = []
        if isinstance(imgs, torch.Tensor):
            arr = imgs.detach().float().clamp(0.0, 1.0).cpu().numpy()
            for a in arr:
                pil.append(Image.fromarray((np.transpose(a, (1, 2, 0)) * 255).astype("uint8")))

        texts: List[str] = []
        for p in prompts:
            content: List[Dict[str, Any]] = ([{"type": "image"}] if pil else []) + [{"type": "text", "text": str(p)}]
            texts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
        enc = self.processor(text=texts, images=pil or None, return_tensors="pt", padding=True)
        p0 = next(self.lm.parameters())
        dev, mdl_dtype = p0.device, p0.dtype
        out: Dict[str, torch.Tensor] = {}
        for k, v in enc.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.is_floating_point():
                # Pixel values arrive float32 from the processor; the model runs in bf16, and a
                # dtype mismatch here is a runtime error deep inside the vision tower.
                out[k] = v.to(dev, dtype=mdl_dtype)
            else:
                out[k] = v.to(dev)
        return out

    def forward(self, batch: Dict[str, Any], *, prefix_tokens: Optional[torch.Tensor] = None) -> BackboneOutput:
        """``batch`` carries ``prompt`` strings and ``image`` tensors; Qwen's own inputs are built here."""
        model_kwargs = {
            k: v
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos")
            and isinstance(v, torch.Tensor)
        }
        if "input_ids" not in model_kwargs:
            model_kwargs = self._encode(batch)
        if prefix_tokens is not None:
            embed = self.lm.get_input_embeddings()
            ids = model_kwargs.pop("input_ids")
            inputs_embeds = embed(ids)
            prefix = prefix_tokens.to(inputs_embeds.dtype)
            model_kwargs["inputs_embeds"] = torch.cat([prefix, inputs_embeds], dim=1)
            am = model_kwargs.get("attention_mask")
            if am is not None:
                pad = torch.ones(prefix.shape[:2], dtype=am.dtype, device=am.device)
                model_kwargs["attention_mask"] = torch.cat([pad, am], dim=1)

        out = self.lm(**model_kwargs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        mask = model_kwargs.get("attention_mask")
        if mask is not None:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            pooled = h.mean(1)
        return BackboneOutput(pooled=pooled.float(), tokens=h, actions=None)


def _batch_len(batch: Dict[str, Any]) -> int:
    for k in ("image", "state", "lang_ids"):
        v = batch.get(k)
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
    p = batch.get("prompt")
    return len(p) if isinstance(p, (list, tuple)) else 1


def _model_class(model_id: str):
    import transformers  # type: ignore

    lowered = str(model_id).lower()
    for name in (
        "Qwen2_5_VLForConditionalGeneration" if "2.5" in lowered or "2_5" in lowered else "Qwen2VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        f"transformers {transformers.__version__} has no Qwen-VL model class; upgrade transformers"
    )


def _decoder_layers(model: nn.Module) -> List[nn.Module]:
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        cur: Any = model
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None:
            return list(cur)
    return []


__all__ = ["QwenVLBackbone", "DEFAULT_ID"]

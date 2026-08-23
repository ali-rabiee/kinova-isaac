"""From-scratch convolutional + small-transformer backbone.

The floor of the model roster: no pretraining of any kind, ~2M parameters. Its job in the
paper is to answer "how much of the de-biasing is the pretrained prior, and how much is just
having a context input at all?" If a randomly-initialised 2M-parameter model can de-bias as
well as a 3B pretrained VLM given the same context vector, the architectural claim collapses --
and that is a result worth being able to get.

Reuses the encoders from :mod:`vla_lab.models`, which are already trained and evaluated
elsewhere in this repository, rather than re-implementing them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...models import StateEncoder, TinyVLA, TinyVLAConfig
from .base import Backbone, BackboneOutput


class TinyBackbone(Backbone):
    name = "tiny"
    supports_prefix_tokens = True
    supports_text = False  # a 256-symbol byte tokenizer is not a language model
    predicts_actions = True

    def __init__(self, cfg: Optional[TinyVLAConfig] = None, **overrides: Any) -> None:
        """``overrides`` are :class:`TinyVLAConfig` fields, so the trainer can pass the
        vocabulary size it just built from the corpus without constructing the config itself."""
        super().__init__()
        base = (cfg or TinyVLAConfig()).to_dict()
        known = set(TinyVLAConfig.__dataclass_fields__)
        base.update({k: v for k, v in overrides.items() if k in known})
        self.cfg = TinyVLAConfig.from_dict(base)
        self.net = TinyVLA(self.cfg)
        self.d_model = int(self.cfg.embed_dim)

    def forward(self, batch: Dict[str, Any], *, prefix_tokens: Optional[torch.Tensor] = None) -> BackboneOutput:
        img_tok = self.net._camera_streams(
            batch.get("image"), batch.get("image_wrist"), batch.get("camera_present")
        )
        lang_ids, lang_mask = batch["lang_ids"], batch["lang_mask"]
        lang_tok = self.net.language(lang_ids, lang_mask)
        st_tok = self.net.state_enc(batch["state"])
        memory, key_pad = self.net._assemble_memory(img_tok, lang_tok, st_tok, lang_mask)
        if prefix_tokens is not None:
            memory = torch.cat([prefix_tokens, memory], dim=1)
            key_pad = torch.cat(
                [torch.zeros(prefix_tokens.shape[:2], dtype=torch.bool, device=memory.device), key_pad], dim=1
            )
        actions = self.net.decoder(memory, key_pad)
        pooled = memory.masked_fill(key_pad.unsqueeze(-1), 0.0).sum(1) / (
            (~key_pad).sum(1, keepdim=True).clamp_min(1)
        )
        return BackboneOutput(pooled=pooled, tokens=memory, actions=actions)


__all__ = ["TinyBackbone"]

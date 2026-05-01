"""TinyVLA: a small, self-contained VLA baseline.

Design goals for this initial model:
- Pure PyTorch, no external model downloads required (works offline).
- Small enough to fine-tune on a single consumer GPU (~5-10M params).
- Same input/output contract as the planned SmolVLA wrapper, so we can
  swap in SmolVLA later without touching downstream code:
      forward(image, state, lang_ids, lang_mask, noise=None)
        -> ModelOutput(actions=[B, T, 7], features=[B, F, D])

The architecture intentionally mirrors the structure described in the spec
(see vla_ttc_engineering_spec.md sections 4.4 and 4.6):
- A vision encoder produces a sequence of patch tokens (the natural place
  for the future DINOv2 feature-alignment loss to tap).
- A language encoder produces token features with attention mask.
- A state encoder turns the proprioception vector into one token.
- All tokens are projected to a common dimension, concatenated, then an
  action decoder produces a chunk of T actions via cross-attention from
  T learnable action queries.
- Optional Gaussian noise can be injected at the bottleneck, which is the
  knob the TTC pipeline uses to draw K candidate action chunks per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyVLAConfig:
    image_size: int = 224
    in_channels: int = 3
    vis_channels: Tuple[int, int, int, int] = (32, 64, 128, 192)
    vis_strides: Tuple[int, int, int, int] = (2, 2, 2, 2)
    embed_dim: int = 192
    lang_layers: int = 2
    lang_heads: int = 4
    state_dim: int = 4
    decoder_layers: int = 2
    decoder_heads: int = 4
    chunk_len: int = 8
    action_dim: int = 7
    vocab_size: int = 256
    pad_id: int = 0
    max_lang_len: int = 24
    dropout: float = 0.05
    noise_std: float = 0.0  # train-time bottleneck noise; set >0 to encourage diversity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "vis_channels": list(self.vis_channels),
            "vis_strides": list(self.vis_strides),
            "embed_dim": self.embed_dim,
            "lang_layers": self.lang_layers,
            "lang_heads": self.lang_heads,
            "state_dim": self.state_dim,
            "decoder_layers": self.decoder_layers,
            "decoder_heads": self.decoder_heads,
            "chunk_len": self.chunk_len,
            "action_dim": self.action_dim,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "max_lang_len": self.max_lang_len,
            "dropout": self.dropout,
            "noise_std": self.noise_std,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TinyVLAConfig":
        return cls(
            image_size=int(d.get("image_size", 224)),
            in_channels=int(d.get("in_channels", 3)),
            vis_channels=tuple(d.get("vis_channels", (32, 64, 128, 192))),
            vis_strides=tuple(d.get("vis_strides", (2, 2, 2, 2))),
            embed_dim=int(d.get("embed_dim", 192)),
            lang_layers=int(d.get("lang_layers", 2)),
            lang_heads=int(d.get("lang_heads", 4)),
            state_dim=int(d.get("state_dim", 4)),
            decoder_layers=int(d.get("decoder_layers", 2)),
            decoder_heads=int(d.get("decoder_heads", 4)),
            chunk_len=int(d.get("chunk_len", 8)),
            action_dim=int(d.get("action_dim", 7)),
            vocab_size=int(d.get("vocab_size", 256)),
            pad_id=int(d.get("pad_id", 0)),
            max_lang_len=int(d.get("max_lang_len", 24)),
            dropout=float(d.get("dropout", 0.05)),
            noise_std=float(d.get("noise_std", 0.0)),
        )


@dataclass
class ModelOutput:
    actions: torch.Tensor          # (B, T, action_dim)
    features: torch.Tensor         # (B, F, embed_dim) - vision tokens (post-projector)
    bottleneck: torch.Tensor       # (B, n_tokens, embed_dim) - fused token sequence


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class VisionEncoder(nn.Module):
    """Small CNN that returns a (B, N_tokens, embed_dim) sequence of features.

    We deliberately produce a *sequence* (h*w tokens) rather than a single
    pooled vector so the future feature-alignment loss can compare per-token
    cosine similarity against DINOv2 patch tokens (see losses.py).
    """

    def __init__(self, cfg: TinyVLAConfig) -> None:
        super().__init__()
        chs = list(cfg.vis_channels)
        strides = list(cfg.vis_strides)
        assert len(chs) == len(strides), "vis_channels and vis_strides must align"

        layers = []
        prev = cfg.in_channels
        for c, s in zip(chs, strides):
            layers.append(_ConvBlock(prev, c, stride=s))
            prev = c
        self.stem = nn.Sequential(*layers)
        self.proj = nn.Conv2d(prev, cfg.embed_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.stem(image)
        feat = self.proj(feat)
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2).contiguous()  # (B, h*w, C)


class _SinPosEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for short sequences."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-(torch.log(torch.tensor(10000.0)) / d_model))
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L]


class LanguageEncoder(nn.Module):
    def __init__(self, cfg: TinyVLAConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_id)
        self.pos = _SinPosEmbedding(cfg.embed_dim, max_len=max(cfg.max_lang_len, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.lang_heads,
            dim_feedforward=cfg.embed_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.lang_layers)

    def forward(self, lang_ids: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(lang_ids)  # (B, L, D)
        x = self.pos(x)
        # Convert attention mask (1 = real) to key_padding_mask (True = pad).
        kp = lang_mask == 0
        return self.encoder(x, src_key_padding_mask=kp)  # (B, L, D)


class StateEncoder(nn.Module):
    def __init__(self, cfg: TinyVLAConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).unsqueeze(1)  # (B, 1, D)


class ActionDecoder(nn.Module):
    """T learnable action queries that cross-attend to fused tokens."""

    def __init__(self, cfg: TinyVLAConfig) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(cfg.chunk_len, cfg.embed_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.embed_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.decoder_layers)
        self.head = nn.Linear(cfg.embed_dim, cfg.action_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = memory.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1).contiguous()  # (B, T, D)
        out = self.decoder(q, memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.head(out)


class TinyVLA(nn.Module):
    """End-to-end small VLA: vision + language + state -> action chunk."""

    def __init__(self, cfg: TinyVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.language = LanguageEncoder(cfg)
        self.state_enc = StateEncoder(cfg)
        self.token_norm = nn.LayerNorm(cfg.embed_dim)
        self.decoder = ActionDecoder(cfg)

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (not trainable_only) or p.requires_grad
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lang_ids: torch.Tensor,
        lang_mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        noise_std: Optional[float] = None,
    ) -> ModelOutput:
        """Standard forward pass.

        - `noise`: optional (B, n_tokens, D) tensor added to the fused
          token sequence (the bottleneck) to draw stochastic candidates.
        - `noise_std`: convenience override for the per-call sampling std
          (used for K-sampling at inference). Ignored if `noise` is given.
        """

        img_tok = self.vision(image)                        # (B, Nv, D)
        lang_tok = self.language(lang_ids, lang_mask)       # (B, L, D)
        st_tok = self.state_enc(state)                      # (B, 1, D)

        memory = torch.cat([img_tok, lang_tok, st_tok], dim=1)
        memory = self.token_norm(memory)

        # Bottleneck noise injection.
        std = float(noise_std) if noise_std is not None else float(self.cfg.noise_std)
        if noise is None and self.training and std > 0:
            noise = torch.randn_like(memory) * std
        if noise is None and (not self.training) and std > 0:
            noise = torch.randn_like(memory) * std
        if noise is not None:
            memory = memory + noise

        # Build memory key-padding mask: vision/state are always real, language uses lang_mask.
        b, l = lang_mask.shape
        nv = img_tok.size(1)
        ns = st_tok.size(1)
        full_mask = torch.zeros(b, nv + l + ns, dtype=torch.bool, device=memory.device)
        full_mask[:, nv : nv + l] = lang_mask == 0  # True where language token is pad

        actions = self.decoder(memory, memory_key_padding_mask=full_mask)
        return ModelOutput(actions=actions, features=img_tok, bottleneck=memory)

    def sample_actions(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lang_ids: torch.Tensor,
        lang_mask: torch.Tensor,
        k: int = 1,
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """Return K candidate action chunks of shape (K, B, T, A).

        The model is run K times with independent Gaussian noise injected
        at the fused-token bottleneck. K=1 is deterministic (no noise).
        """

        if k <= 1:
            out = self.forward(image, state, lang_ids, lang_mask, noise_std=0.0)
            return out.actions.unsqueeze(0)
        outs = []
        for _ in range(int(k)):
            out = self.forward(image, state, lang_ids, lang_mask, noise_std=noise_std)
            outs.append(out.actions)
        return torch.stack(outs, dim=0)


def build_model(cfg_dict: Dict[str, Any]) -> Tuple[TinyVLA, TinyVLAConfig]:
    cfg = TinyVLAConfig.from_dict(cfg_dict)
    return TinyVLA(cfg), cfg

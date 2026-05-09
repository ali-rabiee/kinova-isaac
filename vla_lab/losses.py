"""Loss functions used by `vla_lab`.

The default training loss is masked MSE on the action chunk (the gripper
column is included; with values clipped to [-1, +1] this is a fine fit
for the discrete -1/0/+1 used in the dataset).

The optional `FeatureAlignmentLoss` is the D3 hybrid component from the
spec: encourage the student's vision-token features to stay aligned (via
cosine similarity) with the patch-token features of a frozen DINOv2
teacher. The loss is *optional* and DINOv2 is imported lazily so the
training loop works with no internet / no transformers install.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Action loss
# ---------------------------------------------------------------------------


def masked_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gripper_weight: float = 1.0,
) -> torch.Tensor:
    """Per-element MSE loss masked by `mask` (B, T).

    - `pred`, `target`: (B, T, A)
    - `mask`: (B, T) with 1.0 where the timestep is valid

    The gripper column (index 6) is up-weighted by `gripper_weight` when set
    >1 because it carries discrete -1/0/+1 information that can otherwise be
    drowned out by the continuous delta-pose dims.
    """

    if pred.shape != target.shape:
        raise ValueError(f"masked_action_loss: shape mismatch {pred.shape} vs {target.shape}")
    sq = (pred - target) ** 2  # (B, T, A)
    if gripper_weight != 1.0 and pred.size(-1) >= 7:
        w = torch.ones(pred.size(-1), device=pred.device, dtype=pred.dtype)
        w[6] = gripper_weight
        sq = sq * w
    per_step = sq.mean(dim=-1)  # (B, T)
    msum = mask.sum().clamp_min(1.0)
    return (per_step * mask).sum() / msum


# ---------------------------------------------------------------------------
# Optional DINOv2 feature alignment
# ---------------------------------------------------------------------------


@dataclass
class FeatureAlignmentConfig:
    teacher_name: str = "facebook/dinov2-base"
    teacher_dim: int = 768
    image_size: int = 224
    pool_to_tokens: int = 64
    enabled: bool = False  # off by default (CoRL sprint: keep the mainline on action-only loss)


class FeatureAlignmentLoss(nn.Module):
    """Cosine-distance loss between projected student features and frozen
    DINOv2 teacher patch tokens.

    Both feature maps are pooled (via adaptive avg-pool over the spatial
    grid) to a fixed token count `pool_to_tokens` so any student image
    encoder shape is supported. DINOv2 expects ImageNet normalisation,
    which we apply internally.

    The teacher is loaded *lazily* on first forward pass; if the user has
    no `transformers` install or no internet access, the loss returns 0
    without crashing the trainer.
    """

    IMNET_MEAN = (0.485, 0.456, 0.406)
    IMNET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: FeatureAlignmentConfig, student_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(student_dim, cfg.teacher_dim)
        self._teacher: Optional[nn.Module] = None
        self._teacher_failed: bool = False

    @torch.no_grad()
    def _ensure_teacher(self) -> None:
        if self._teacher is not None or self._teacher_failed:
            return
        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            print(f"[FeatureAlignmentLoss] disabled (transformers missing): {exc}")
            self._teacher_failed = True
            return
        try:
            teacher = AutoModel.from_pretrained(self.cfg.teacher_name)
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
            self._teacher = teacher
        except Exception as exc:  # pragma: no cover
            print(f"[FeatureAlignmentLoss] disabled (could not load {self.cfg.teacher_name}): {exc}")
            self._teacher_failed = True

    @torch.no_grad()
    def _normalize_imnet(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.IMNET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.IMNET_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    @torch.no_grad()
    def _teacher_tokens(self, image: torch.Tensor) -> torch.Tensor:
        assert self._teacher is not None
        out = self._teacher(pixel_values=self._normalize_imnet(image))
        # DINOv2 returns last_hidden_state with first token = CLS.
        return out.last_hidden_state[:, 1:, :]  # (B, N, D_t)

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor, target_n: int) -> torch.Tensor:
        b, n, d = tokens.shape
        if n == target_n:
            return tokens
        side = int(round(math.sqrt(n)))
        target_side = int(round(math.sqrt(target_n)))
        if side * side == n and target_side * target_side == target_n:
            x = tokens.transpose(1, 2).reshape(b, d, side, side)
            x = F.adaptive_avg_pool2d(x, (target_side, target_side))
            return x.flatten(2).transpose(1, 2).contiguous()
        # Fallback: 1D adaptive pool
        x = tokens.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, target_n)
        return x.transpose(1, 2).contiguous()

    def forward(self, student_features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return image.new_zeros(())
        self._ensure_teacher()
        if self._teacher_failed or self._teacher is None:
            return image.new_zeros(())

        student_proj = self.proj(student_features)  # (B, Ns, D_t)
        student_proj = self._pool_tokens(student_proj, self.cfg.pool_to_tokens)
        teacher_tok = self._teacher_tokens(image)
        teacher_tok = self._pool_tokens(teacher_tok, self.cfg.pool_to_tokens)
        cos = F.cosine_similarity(student_proj, teacher_tok, dim=-1)
        return 1.0 - cos.mean()

"""Concrete :class:`ComputeBranch` adapters over TinyVLA and SmolVLA (torch).

These reuse the existing "compute" machinery — :func:`vla_lab.ttc.select_from_candidates`
and the model sampling APIs — and expose the three primitives the allocator needs:
``act`` (one deterministic forward), ``probe`` (twin/noisy-pair dispersion + act-candidate
+ features), and ``compute`` (best-of-K). Chunks are returned in **physical** units (the
contract :class:`vla_lab.eval_isaaclab.PolicyInputProvider` expects).

Imported only by the Isaac glue / real-robot loop, never by ``allocation.__init__`` — that
keeps the allocator core importable without torch.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from ..ttc import select_from_candidates
from .uncertainty import ComputeResult, ProbeResult, chunk_dispersion, twin_dispersion


def _features(dispersion: float, obs: Dict[str, Any], chunk: Any) -> Dict[str, float]:
    feats: Dict[str, float] = {"dispersion": float(dispersion)}
    if isinstance(obs, dict) and obs.get("occlusion_fraction") is not None:
        feats["occlusion_fraction"] = float(obs["occlusion_fraction"])
    # Uncertainty-source features (2026-07 intent/perception decomposition): the
    # eval loop precomputes these per tick (see vla_lab.intent) and drops them
    # into obs so fitted allocators and the intent-aware controller can use them.
    for key in ("intent_entropy", "intent_top2_margin", "cross_view_disagreement", "instruction_ambiguous"):
        if isinstance(obs, dict) and obs.get(key) is not None:
            try:
                feats[key] = float(obs[key])
            except (TypeError, ValueError):
                pass
    try:
        c = chunk.detach().float()
        feats["chunk_l2"] = float(c.pow(2).mean().sqrt().item())
    except Exception:
        pass
    return feats


class TTCComputeBranch:
    """Adapter over a trained :class:`vla_lab.models.TinyVLA`.

    ``obs`` is a dict with ``image`` (3,H,W float in [0,1], already occluded by the eval
    loop), ``state`` (state_dim,), and optional ``occlusion_fraction``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device,
        action_stats=None,
        noise_std: float = 0.1,
        gating_noise_std: float = 0.05,
        selection: str = "naive_consensus",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.action_stats = action_stats
        self.noise_std = float(noise_std)
        self.gating_noise_std = float(gating_noise_std)
        self.selection = str(selection)

    def _encode(self, instruction: str):
        ids, attn = self.tokenizer.encode(str(instruction))
        return ids.to(self.device), attn.to(self.device)

    def _tensors(self, obs: Dict[str, Any]):
        img = obs["image"]
        st = obs["state"]
        if img.dim() == 4:
            img = img[0]
        return img.to(self.device), st.to(self.device)

    def _extra_model_kwargs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Optional wrist stream + camera-set mask, batched for B=1 forwards."""
        kw: Dict[str, Any] = {}
        w = obs.get("image_wrist") if isinstance(obs, dict) else None
        if w is not None:
            if w.dim() == 4:
                w = w[0]
            kw["image_wrist"] = w.to(self.device).unsqueeze(0)
        cp = obs.get("camera_present") if isinstance(obs, dict) else None
        if cp is not None:
            kw["camera_present"] = cp.to(self.device).view(1, -1)
        return kw

    def _denorm(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.action_stats is not None:
            return self.action_stats.denormalize(chunk.detach().cpu()).to(self.device)
        return chunk

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], instruction: str) -> ComputeResult:
        t0 = time.time()
        img, st = self._tensors(obs)
        ids, attn = self._encode(instruction)
        kw = self._extra_model_kwargs(obs)
        out = self.model.forward(img.unsqueeze(0), st.unsqueeze(0), ids.unsqueeze(0), attn.unsqueeze(0), noise_std=0.0, **kw)
        chunk = out.actions.squeeze(0)
        return ComputeResult(chunk=self._denorm(chunk), dispersion=0.0, k_used=1, forward_ms=(time.time() - t0) * 1000.0)

    @torch.no_grad()
    def probe(self, obs: Dict[str, Any], instruction: str) -> ProbeResult:
        t0 = time.time()
        img, st = self._tensors(obs)
        ids, attn = self._encode(instruction)
        kw = self._extra_model_kwargs(obs)
        a0 = self.model.forward(img.unsqueeze(0), st.unsqueeze(0), ids.unsqueeze(0), attn.unsqueeze(0), noise_std=0.0, **kw).actions.squeeze(0)
        a1 = self.model.forward(img.unsqueeze(0), st.unsqueeze(0), ids.unsqueeze(0), attn.unsqueeze(0), noise_std=self.gating_noise_std, **kw).actions.squeeze(0)
        disp = twin_dispersion(a0, a1)
        return ProbeResult(
            dispersion=disp, act_chunk=self._denorm(a0), features=_features(disp, obs, a0), forward_ms=(time.time() - t0) * 1000.0,
        )

    @torch.no_grad()
    def compute(self, obs: Dict[str, Any], instruction: str, k: int) -> ComputeResult:
        t0 = time.time()
        img, st = self._tensors(obs)
        ids, attn = self._encode(instruction)
        kw = self._extra_model_kwargs(obs)
        cands = self.model.sample_actions(
            img.unsqueeze(0), st.unsqueeze(0), ids.unsqueeze(0), attn.unsqueeze(0), k=int(k), noise_std=self.noise_std, **kw
        ).squeeze(1)
        best, scores = select_from_candidates(cands, self.selection)
        disp = chunk_dispersion(cands)
        return ComputeResult(
            chunk=self._denorm(best), dispersion=disp, k_used=int(cands.size(0)), candidates=cands, scores=scores,
            forward_ms=(time.time() - t0) * 1000.0,
        )


class SmolVLAComputeBranch:
    """Adapter over :class:`vla_lab.smolvla_bridge.policy_wrapper.SmolVLAIsaacPolicy`.

    ``obs`` is a dict with ``image`` (3,H,W float in [0,1]), ``state`` (6,), and optional
    ``occlusion_fraction``.
    """

    def __init__(self, smol_policy, *, selection: str = "naive_consensus", base_seed: int = 2048) -> None:
        self.policy = smol_policy
        self.selection = str(selection)
        self.base_seed = int(base_seed)

    def _seed(self, s: int) -> None:
        try:
            torch.manual_seed(int(s))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(s))
        except Exception:
            pass
        try:
            self.policy.reset()
        except Exception:
            pass

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], instruction: str) -> ComputeResult:
        t0 = time.time()
        self._seed(0)
        chunk = self.policy.predict_chunk_phys(rgb_chw_float=obs["image"], state6=obs["state"], instruction=str(instruction))
        return ComputeResult(chunk=chunk, dispersion=0.0, k_used=1, forward_ms=(time.time() - t0) * 1000.0)

    @torch.no_grad()
    def probe(self, obs: Dict[str, Any], instruction: str) -> ProbeResult:
        t0 = time.time()
        self._seed(0)
        a0 = self.policy.predict_chunk_phys(rgb_chw_float=obs["image"], state6=obs["state"], instruction=str(instruction))
        self._seed(1)
        a1 = self.policy.predict_chunk_phys(rgb_chw_float=obs["image"], state6=obs["state"], instruction=str(instruction))
        disp = twin_dispersion(a0, a1)
        return ProbeResult(dispersion=disp, act_chunk=a0, features=_features(disp, obs, a0), forward_ms=(time.time() - t0) * 1000.0)

    @torch.no_grad()
    def compute(self, obs: Dict[str, Any], instruction: str, k: int) -> ComputeResult:
        t0 = time.time()
        cands = self.policy.predict_chunk_phys_k(
            rgb_chw_float=obs["image"], state6=obs["state"], instruction=str(instruction), k=int(k), base_seed=self.base_seed,
        )
        best, scores = select_from_candidates(cands, self.selection)
        disp = chunk_dispersion(cands)
        return ComputeResult(chunk=best, dispersion=disp, k_used=int(cands.shape[0]), candidates=cands, scores=scores, forward_ms=(time.time() - t0) * 1000.0)


def build_compute_branch(policy_backend: str, **kwargs):
    """Convenience factory used by the Isaac glue."""

    b = str(policy_backend).lower().strip()
    if b == "tiny":
        return TTCComputeBranch(**kwargs)
    if b == "smolvla":
        return SmolVLAComputeBranch(**kwargs)
    raise ValueError(f"unknown policy_backend {policy_backend!r}")

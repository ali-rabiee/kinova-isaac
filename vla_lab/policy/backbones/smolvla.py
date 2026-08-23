"""SmolVLA-450M backbone, through LeRobot.

The purpose-built VLA in the roster: pretrained on robot demonstrations, with a flow-matching
action expert and a small language model (SmolLM2) fronting a SigLIP vision tower.

Two things this wrapper has to get right, and both are contract issues rather than plumbing:

**Feature extraction.** The carryover heads attach to the *language model's* pooled hidden
state, not to the action expert's. That is deliberate. The de-biasing question -- "is this
person telling me what they think, or repeating me?" -- is a language-and-context question, and
attaching the heads downstream of the action expert would ask a module trained to emit
trajectories to answer it.

**Normalisation.** LeRobot policies carry their own input/output normalisation buffers, fitted
to the dataset they were trained on. The wrapper never bypasses them: images and state go in
through the policy's own preprocessing, so a checkpoint's contract is preserved. Bypassing it
is the standard way to get a policy that trains to a low loss and then does nothing sensible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import Backbone, BackboneOutput

DEFAULT_ID = "lerobot/smolvla_base"


class SmolVLABackbone(Backbone):
    name = "smolvla"
    supports_prefix_tokens = True
    supports_text = True
    predicts_actions = False
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ID,
        chunk_len: int = 8,
        action_dim: int = 7,
        freeze_vision: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
        except Exception:  # pragma: no cover - older LeRobot layout
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore

        self.policy = SmolVLAPolicy.from_pretrained(model_id)
        self.model_id = model_id
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.d_model = int(self._infer_width())
        if freeze_vision:
            self._freeze_vision()

    # -- introspection ------------------------------------------------------
    def _infer_width(self) -> int:
        """Width of the tensor :meth:`forward` actually returns.

        This must be the **VLM's** hidden size (960 in ``smolvla_base``), not the action
        expert's (720). ``embed_prefix`` emits the fused prefix in VLM width and the expert
        projects down from it afterwards, so a width read off the expert type-checks everywhere
        and then fails inside the first LayerNorm. The most reliable single source is the
        projection the policy itself uses to lift proprioception into that space.
        """
        proj = getattr(getattr(self.policy, "model", None), "state_proj", None)
        out_features = getattr(proj, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            return int(out_features)
        vwe = getattr(getattr(self.policy, "model", None), "vlm_with_expert", None)
        get_vlm = getattr(vwe, "get_vlm_model", None)
        vlm = get_vlm() if callable(get_vlm) else None
        cfg = getattr(vlm, "config", None)
        for holder in (getattr(cfg, "text_config", None), cfg):
            v = getattr(holder, "hidden_size", None) if holder is not None else None
            if isinstance(v, int) and v > 0:
                return int(v)
        raise RuntimeError("could not infer the SmolVLA prefix width")

    def _language_model(self) -> nn.Module:
        node: Any = self.policy
        for path in (("model", "vlm_with_expert", "lm_expert"), ("model", "vlm_with_expert", "vlm"), ("model",)):
            cur = node
            ok = True
            for part in path:
                cur = getattr(cur, part, None)
                if cur is None:
                    ok = False
                    break
            if ok and isinstance(cur, nn.Module):
                return cur
        return self.policy

    @property
    def lm(self) -> nn.Module:
        return self._language_model()

    def _freeze_vision(self) -> None:
        for name, p in self.policy.named_parameters():
            if any(tag in name for tag in ("vision", "siglip", "image_encoder", "vision_tower")):
                p.requires_grad_(False)

    def freeze(self, *, unfreeze_last: int = 0) -> None:
        for p in self.policy.parameters():
            p.requires_grad_(False)
        if unfreeze_last > 0:
            blocks = [m for n, m in self.policy.named_modules() if n.endswith(tuple(f"layers.{i}" for i in range(64)))]
            for m in blocks[-int(unfreeze_last):]:
                for p in m.parameters():
                    p.requires_grad_(True)

    # -- batch adaptation ---------------------------------------------------
    def to_lerobot_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Rename this package's batch into the key names the checkpoint declares.

        LeRobot policies address their inputs by dataset feature name
        (``observation.images.*``, ``observation.state``, ``observation.language.tokens``), and
        which names exist is a property of the checkpoint rather than of the class. So the
        mapping is read from ``config.input_features`` at run time instead of hard-coded, and a
        rename upstream fails loudly here rather than silently feeding the policy an empty
        observation --- which is exactly what a missing instruction looks like from the outside:
        a model that trains and does not learn.

        The single top-down image is broadcast to every declared camera slot. That is a real
        limitation and worth naming: this task has one view, the checkpoint expects several, and
        duplicating is the honest stand-in until the wrist and front views from the Isaac sweep
        are wired in.
        """
        cfg = getattr(self.policy, "config", None)
        feats = dict(getattr(cfg, "input_features", {}) or {})
        out: Dict[str, Any] = {}

        img = batch.get("image")
        n = int(img.shape[0]) if img is not None else 1
        if img is not None:
            for k in [k for k in feats if "image" in k]:
                shape = getattr(feats[k], "shape", None)
                x = img
                if shape and len(shape) == 3 and tuple(x.shape[-2:]) != tuple(shape[-2:]):
                    x = torch.nn.functional.interpolate(
                        x, size=(int(shape[-2]), int(shape[-1])), mode="bilinear", align_corners=False
                    )
                out[k] = x

        state_key = next((k for k in feats if k.endswith("state")), "observation.state")
        st = batch.get("state")
        if st is not None:
            want = getattr(feats.get(state_key), "shape", None)
            if want and int(want[-1]) != int(st.shape[-1]):
                pad = torch.zeros(st.shape[0], int(want[-1]), device=st.device, dtype=st.dtype)
                width = min(int(want[-1]), int(st.shape[-1]))
                pad[:, :width] = st[:, :width]
                st = pad
            out[state_key] = st

        prompts = batch.get("task") or batch.get("prompt")
        if isinstance(prompts, str):
            prompts = [prompts] * n
        if not prompts:
            raise ValueError(
                "no instruction in the batch: SmolVLA addresses language through `task`/`prompt`, "
                "and a batch without it trains the policy on empty strings -- which presents as "
                "slow learning rather than as a bug"
            )
        tok = self._tokenizer()
        enc = tok(
            list(prompts)[:n], padding="max_length", truncation=True,
            max_length=int(getattr(self.policy.config, "tokenizer_max_length", 48) or 48),
            return_tensors="pt",
        )
        device = img.device if img is not None else torch.device("cpu")
        out["observation.language.tokens"] = enc["input_ids"].to(device)
        out["observation.language.attention_mask"] = enc["attention_mask"].to(device)
        return out

    def _tokenizer(self):
        if getattr(self, "_tok", None) is None:
            from transformers import AutoProcessor  # type: ignore

            name = getattr(self.policy.config, "vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            proc = AutoProcessor.from_pretrained(name)
            self._tok = getattr(proc, "tokenizer", proc)
        return self._tok

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Dict[str, Any], *, prefix_tokens: Optional[torch.Tensor] = None) -> BackboneOutput:
        """Encode through the policy's own **prefix embedding**, differentiably.

        The obvious route -- call ``select_action`` and hook the language model -- does not work
        and fails quietly: ``select_action`` is decorated ``@torch.no_grad()``, so the captured
        activation is detached, no gradient reaches either the backbone or the heads, and
        training runs to completion having learned nothing. Instead this calls
        ``VLAFlowMatching.embed_prefix`` directly, which is the fused image + language + state
        representation the action expert itself consumes, and is exactly where a question about
        *what the supervisor meant* should be asked.
        """
        lb = self.to_lerobot_batch(batch)
        images, img_masks = self.policy.prepare_images(lb)
        state = self.policy.prepare_state(lb)
        lang_tokens = lb["observation.language.tokens"]
        lang_masks = lb["observation.language.attention_mask"]

        embs, pad_masks, _att = self.policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        h = embs
        # Pool the **instruction segment**, not the whole prefix. ``embed_prefix`` lays out
        # image tokens first, then the language tokens, then one state token; with three camera
        # slots the image tokens outnumber the 48 language ones by an order of magnitude, and a
        # mean over the whole sequence dilutes the utterance to the point where the intent head
        # sits at chance. Everything the heads are asked about -- what was said, what they would
        # have said -- lives in the language and state positions.
        n_tail = int(lang_tokens.shape[1]) + 1
        if prefix_tokens is not None:
            tail = torch.cat([prefix_tokens.to(h.dtype), h[:, -n_tail:]], dim=1)
            tail_mask = torch.cat(
                [torch.ones(prefix_tokens.shape[:2], dtype=pad_masks.dtype, device=pad_masks.device),
                 pad_masks[:, -n_tail:]], dim=1
            )
        else:
            tail, tail_mask = h[:, -n_tail:], pad_masks[:, -n_tail:]
        m = tail_mask.unsqueeze(-1).to(tail.dtype)
        pooled = (tail * m).sum(1) / m.sum(1).clamp_min(1.0)
        return BackboneOutput(pooled=pooled.float(), tokens=h, actions=None)


def _bs(batch: Dict[str, Any]) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return 1


def _dev(batch: Dict[str, Any]) -> torch.device:
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


__all__ = ["SmolVLABackbone", "DEFAULT_ID"]

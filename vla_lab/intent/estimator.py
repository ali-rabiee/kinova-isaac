"""Counterfactual intent estimator + cross-view perception disagreement.

Both signals are cheap on TinyVLA (a handful of extra deterministic forwards of
a ~2 M-parameter model per policy tick) and are designed to be *decomposable*:

- ``IntentEstimator.estimate`` returns a posterior over candidate targets and
  its normalized entropy — high entropy = the policy's behavior is not pinned
  down by the instruction (intent uncertainty).
- ``cross_view_disagreement`` compares the action the policy would take seeing
  only the overhead view vs only the wrist view — high disagreement = the two
  views tell conflicting stories (perception uncertainty), which K-sample
  dispersion under a single view cannot expose.

No Isaac dependency; unit-tested offline in vla_lab/tests/test_intent.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def lexical_target_candidates(instruction: str, known_labels: Sequence[str]) -> List[str]:
    """Which of ``known_labels`` (e.g. box colors) the instruction names, by token match."""

    toks = set()
    cur: List[str] = []
    for ch in str(instruction).lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                toks.add("".join(cur))
                cur = []
    if cur:
        toks.add("".join(cur))
    return [lbl for lbl in known_labels if str(lbl).lower() in toks]


@dataclass
class IntentEstimate:
    """Posterior over candidate targets + summary scalars."""

    posterior: Dict[str, float]
    entropy: float  # normalized to [0, 1] (1 = uniform over candidates)
    top1: str
    top2_margin: float  # p(top1) - p(top2); low margin = ambiguous
    instruction_ambiguous: bool  # lexical: instruction does not name exactly one candidate
    lexical_candidates: List[str] = field(default_factory=list)

    def features(self) -> Dict[str, float]:
        """Flat features for the allocator / calibration records."""
        return {
            "intent_entropy": float(self.entropy),
            "intent_top2_margin": float(self.top2_margin),
            "instruction_ambiguous": 1.0 if self.instruction_ambiguous else 0.0,
        }


class IntentEstimator:
    """Estimate WHICH candidate target the instruction commits the policy to.

    Method (per call):
    1. Lexical grounding: does the instruction name exactly one candidate label?
       If yes, the posterior is anchored on it with ``lexical_prior_weight``.
    2. Counterfactual sweep: run one deterministic forward under the actual
       instruction and one under a canonical instruction per candidate
       ("Pick up the {label} box."). The behavioral posterior is a softmax over
       negative RMS distances between the actual chunk and each counterfactual
       chunk — if the policy behaves as if told "the red box", intent is red.
    3. Geometric grounding (optional): the direction the predicted chunk moves
       the EE is compared against the direction to each candidate object.

    The combined posterior's normalized entropy is the intent-uncertainty
    scalar consumed by the allocator (``intent_entropy`` feature).
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device,
        template: str = "Pick up the {label} box.",
        temperature: float = 0.1,
        lexical_prior_weight: float = 0.75,
        geometric_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.template = str(template)
        self.temperature = float(temperature)
        self.lexical_prior_weight = float(lexical_prior_weight)
        self.geometric_weight = float(geometric_weight)
        self._cf_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if text not in self._cf_cache:
            if len(self._cf_cache) >= 512:  # mid-episode refinements vary the key set
                self._cf_cache.clear()
            ids, attn = self.tokenizer.encode(text)
            self._cf_cache[text] = (ids.to(self.device), attn.to(self.device))
        return self._cf_cache[text]

    @torch.no_grad()
    def _chunk(self, model_inputs: Dict[str, Any], ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        out = self.model.forward(
            model_inputs.get("image"),
            model_inputs["state"],
            ids.unsqueeze(0),
            attn.unsqueeze(0),
            noise_std=0.0,
            image_wrist=model_inputs.get("image_wrist"),
            camera_present=model_inputs.get("camera_present"),
        )
        return out.actions.squeeze(0)  # (T, A)

    @torch.no_grad()
    def _counterfactual_chunks(self, model_inputs: Dict[str, Any], texts: Sequence[str]) -> torch.Tensor:
        """Action chunks for all `texts` over one observation, shape (len(texts), T, A).

        Uses the model's batched `counterfactual_actions` when available (one
        vision encode + one batched decode for the whole sweep); falls back to
        sequential per-text forwards for wrapped/mock policies that only
        expose `forward`.
        """

        enc = [self._encode(t) for t in texts]
        fn = getattr(self.model, "counterfactual_actions", None)
        if callable(fn):
            ids = torch.stack([e[0] for e in enc])
            attn = torch.stack([e[1] for e in enc])
            return fn(
                image=model_inputs.get("image"),
                state=model_inputs["state"],
                lang_ids=ids,
                lang_mask=attn,
                image_wrist=model_inputs.get("image_wrist"),
                camera_present=model_inputs.get("camera_present"),
            )
        return torch.stack([self._chunk(model_inputs, ids_t, attn_t) for ids_t, attn_t in enc])

    @torch.no_grad()
    def estimate(
        self,
        *,
        instruction: str,
        candidate_labels: Sequence[str],
        image: Optional[torch.Tensor] = None,
        image_wrist: Optional[torch.Tensor] = None,
        state: torch.Tensor,
        camera_present: Optional[torch.Tensor] = None,
        object_pos_b: Optional[Dict[str, Sequence[float]]] = None,
        ee_pos_b: Optional[Sequence[float]] = None,
    ) -> IntentEstimate:
        labels = [str(c) for c in candidate_labels]
        if not labels:
            return IntentEstimate(posterior={}, entropy=0.0, top1="", top2_margin=1.0,
                                  instruction_ambiguous=True, lexical_candidates=[])

        def _batchify(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            return x.unsqueeze(0) if x.dim() in (1, 3) else x

        inputs = {
            "image": _batchify(image),
            "image_wrist": _batchify(image_wrist),
            "state": _batchify(state),
            "camera_present": _batchify(camera_present),
        }

        lexical = lexical_target_candidates(instruction, labels)
        ambiguous = len(lexical) != 1

        # Behavioral: actual chunk vs counterfactual per-candidate chunks — one
        # batched sweep (see `_counterfactual_chunks`). Distances are normalized
        # by their mean so the posterior sharpness is SCALE-FREE: a policy whose
        # counterfactuals barely differ (undertrained, or a genuinely ambiguous
        # instruction) yields a near-uniform posterior (high entropy) regardless
        # of the model's raw action magnitudes.
        texts = [str(instruction)] + [self.template.format(label=lbl) for lbl in labels]
        chunks = self._counterfactual_chunks(inputs, texts)  # (1+N, T, A)
        chunk0 = chunks[0]
        rms_t = (chunks[1:] - chunk0.unsqueeze(0)).pow(2).mean(dim=(1, 2)).sqrt().to("cpu", torch.float32)
        scale = rms_t.mean().clamp_min(1e-9)
        logits = -(rms_t / scale) / max(1e-6, self.temperature)

        # Geometric: does the predicted displacement head toward the candidate?
        if object_pos_b and ee_pos_b is not None and self.geometric_weight > 0:
            dp = chunk0[:, :3].sum(dim=0)  # cumulative EE displacement of the chunk
            dp_n = (dp / (dp.norm() + 1e-9)).cpu()
            geo: List[float] = []
            for lbl in labels:
                p = object_pos_b.get(lbl)
                if p is None:
                    geo.append(0.0)
                    continue
                to_obj = torch.tensor(
                    [float(p[k]) - float(ee_pos_b[k]) for k in range(3)], dtype=torch.float32
                )
                to_obj = to_obj / (to_obj.norm() + 1e-9)
                geo.append(float(torch.dot(dp_n, to_obj).item()))
            logits = logits + self.geometric_weight * torch.tensor(geo) / max(1e-6, self.temperature)

        post = torch.softmax(logits, dim=0)

        # Lexical anchor: a uniquely named target concentrates the posterior.
        if not ambiguous:
            onehot = torch.tensor([1.0 if l == lexical[0] else 0.0 for l in labels])
            post = self.lexical_prior_weight * onehot + (1.0 - self.lexical_prior_weight) * post
            post = post / post.sum()

        p = post.clamp_min(1e-9)
        ent = float(-(p * p.log()).sum().item())
        ent_norm = ent / math.log(len(labels)) if len(labels) > 1 else 0.0
        order = torch.argsort(post, descending=True)
        top1 = labels[int(order[0])]
        margin = float(post[order[0]] - (post[order[1]] if len(labels) > 1 else 0.0))

        return IntentEstimate(
            posterior={l: float(v) for l, v in zip(labels, post.tolist())},
            entropy=float(min(1.0, max(0.0, ent_norm))),
            top1=top1,
            top2_margin=margin,
            instruction_ambiguous=ambiguous,
            lexical_candidates=lexical,
        )


@torch.no_grad()
def cross_view_disagreement(
    model,
    *,
    image: Optional[torch.Tensor],
    image_wrist: Optional[torch.Tensor],
    state: torch.Tensor,
    lang_ids: torch.Tensor,
    lang_mask: torch.Tensor,
) -> Optional[float]:
    """RMS distance between overhead-only and wrist-only action predictions.

    Only defined for two-camera models with both frames available; returns None
    otherwise. High disagreement = the views tell conflicting stories about the
    scene — a perception-uncertainty signal invisible to single-view dispersion.
    """

    cams = tuple(getattr(getattr(model, "cfg", None), "cameras", ()) or ())
    if "overhead" not in cams or "wrist" not in cams:
        return None
    if image is None or image_wrist is None:
        return None

    def _b(x: torch.Tensor, d: int) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == d else x

    image = _b(image, 3)
    image_wrist = _b(image_wrist, 3)
    state = _b(state, 1)
    lang_ids = _b(lang_ids, 1)
    lang_mask = _b(lang_mask, 1)
    if image.size(0) != 1:
        raise ValueError("cross_view_disagreement expects a single observation (batch 1)")
    # One batched forward: row 0 sees only the overhead stream, row 1 only the
    # wrist stream (identical pixels, per-row camera_present masks).
    masks = torch.tensor(
        [
            [1.0 if c == "overhead" else 0.0 for c in cams],
            [1.0 if c == "wrist" else 0.0 for c in cams],
        ],
        device=image.device,
    )
    acts = model.forward(
        image.expand(2, -1, -1, -1),
        state.expand(2, -1),
        lang_ids.expand(2, -1),
        lang_mask.expand(2, -1),
        noise_std=0.0,
        image_wrist=image_wrist.expand(2, -1, -1, -1),
        camera_present=masks,
    ).actions
    return float((acts[0] - acts[1]).pow(2).mean().sqrt().item())

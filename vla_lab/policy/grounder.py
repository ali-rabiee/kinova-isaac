"""Put a trained Carryover-Aware VLA into the session loop as the grounding channel.

Everything up to here evaluates the policy *offline*, on held-out dialogues. That measures
whether the heads learned something; it does not measure what happens when the model is the
thing the robot actually listens to. This closes that gap: :class:`PolicyGrounder` satisfies the
same ``Grounder`` protocol the lexical reference channel does, so the identical session runner,
schedulers and estimators can be driven by a checkpoint instead.

Two readings are available and they answer different questions:

``said``
    Ground the utterance the way any VLA would. Compared against the lexical channel this is a
    pure grounding-quality measurement: does the model read the sentence correctly?
``unprompted``
    Report what the model believes the supervisor would have said uncoached. This is the
    de-biasing head *acting* -- the robot answering its own question rather than the
    supervisor's -- and it is the setting in which the estimand is built from the model's
    beliefs rather than from what was said.

The second is deliberately not the default. A system that silently substitutes its own guess for
a person's instruction is doing something a deployment has to opt into, and the paper reports it
as a separate condition rather than folding it into the main table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..supervisory import STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED
from ..supervisory.narration import ground as lexical_ground
from ..supervisory.strategies import get_axis
from .context import CarryoverContext
from .carryover_vla import prepare_instruction

def _infer_state_dim(model: Any, default: int = 4) -> int:
    """Width of the proprioceptive vector this backbone expects.

    Not a config field -- each backbone decides for itself -- so it is read off the first module
    that consumes it. Getting this wrong is a shape error at the first call, not a silent one,
    but the default keeps backbones that ignore state entirely working.
    """
    import torch.nn as nn

    for name, mod in model.named_modules():
        if name.endswith(("state_enc", "state_proj")) and isinstance(mod, nn.Linear):
            return int(mod.in_features)
        if name.endswith(("state_enc", "state_proj")):
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    return int(sub.in_features)
    return int(default)


READ_SAID = "said"
READ_UNPROMPTED = "unprompted"


class PolicyGrounder:
    """Grounds utterances with a trained checkpoint, through the same protocol as the lexical one.

    ``min_confidence`` is not decoration. The lexical channel refuses to guess -- an utterance
    that matches both strategies or neither resolves to *unresolved* and contributes nothing --
    and a learned channel that always emits a class would look better simply by guessing on the
    cases the reference channel declines. Below the threshold this abstains too, so the two
    channels are compared on the same terms.
    """

    name = "policy"

    def __init__(
        self,
        checkpoint: Path,
        *,
        axis: str = "plan",
        read: str = READ_SAID,
        min_confidence: float = 0.60,
        device: Optional[str] = None,
        atlas: Optional[Any] = None,
        context_style: Optional[str] = None,
    ) -> None:
        import torch

        from ..checkpoint_utils import torch_load_checkpoint
        from ..dataset import TinyTokenizer
        from .carryover_vla import CarryoverVLA, CarryoverVLAConfig
        from .registry import build_model

        ckpt = torch_load_checkpoint(Path(checkpoint), map_location="cpu")
        cfg_d: Dict[str, Any] = dict(ckpt.get("config", {}))
        manifest: Dict[str, Any] = dict(ckpt.get("manifest", {}) or {})

        # Rebuild through the registry, not through ``CarryoverVLA(cfg)`` directly. Backbone
        # options that are not wrapper-config fields -- the vocabulary size the trainer derived
        # from its own corpus, above all -- live outside the config, and a model rebuilt without
        # them has a differently shaped embedding table. ``load_state_dict`` raises on the shape
        # mismatch rather than loading something wrong, but only if we get here; silently
        # defaulting the vocabulary would be the worse failure.
        vocab = int((manifest.get("data") or {}).get("vocab_size") or 0)
        extra: Dict[str, Any] = {"vocab_size": vocab} if vocab else {}
        passthrough = {k: v for k, v in cfg_d.items()
                       if k not in ("model_key", "backbone", "context_mode", "adapt")}
        key = str((manifest.get("card") or {}).get("key") or cfg_d.get("model_key", "tiny"))
        try:
            self.model = build_model(key, context_mode=str(cfg_d.get("context_mode", "token")),
                                     adapt=str(cfg_d.get("adapt", "full")), **passthrough, **extra)
        except (KeyError, ValueError):
            self.model = CarryoverVLA(CarryoverVLAConfig.from_dict(cfg_d), **extra)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        # Frozen/LoRA runs legitimately save a subset, so a few missing keys are normal; a
        # missing *head* is not, and would leave the grounder reading a randomly initialised
        # classifier while looking perfectly healthy.
        bad = [k for k in missing if k.startswith(("said_head", "unprompted_head", "ask_head"))]
        if bad:
            raise RuntimeError(f"checkpoint {checkpoint} is missing trained heads: {bad[:6]}")
        self.missing_keys, self.unexpected_keys = list(missing), list(unexpected)
        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        tok = ckpt.get("tokenizer")
        self.tokenizer = TinyTokenizer(**tok) if isinstance(tok, dict) else TinyTokenizer()
        self.axis = get_axis(axis)
        self.read = str(read)
        self.min_confidence = float(min_confidence)
        self.atlas = atlas
        self.context_mode = str(manifest.get("context_mode", cfg_d.get("context_mode", "token")))
        self.context_style = str(context_style or manifest.get("context_style", "compact"))
        self.name = f"policy[{self.read}]"
        self._rng = np.random.default_rng(0)
        #: Filled by the session runner before each call, when it has one.
        self.context: CarryoverContext = CarryoverContext.empty()
        self._state_dim = _infer_state_dim(self.model)
        self.stats = {"calls": 0, "abstained": 0, "agree_lexical": 0}
        #: Per-scene call and abstention counts, keyed by ``|c|`` band. The docstring's warning
        #: -- a model that abstains exactly in the crossover band starves the estimand while
        #: looking healthy on an average abstention rate -- is only a warning if it is measured.
        self._by_band: Dict[str, List[int]] = {"band": [0, 0], "flank": [0, 0]}
        #: Class probabilities from the most recent call, for auditing and for tests that need
        #: to confirm the injected context actually reached the network.
        self.last_probs: Tuple[float, float] = (0.0, 0.0)

    def set_context(self, context: CarryoverContext) -> None:
        """The belief module hands the grounder the same context the model trained on."""
        self.context = context

    def ground(self, utterance: str, scene: Any) -> str:
        import torch

        self.stats["calls"] += 1
        ctx = self.context
        prompt = prepare_instruction(
            str(utterance), ctx, self.context_mode,
            axis_labels=(self.axis.label_a, self.axis.label_b),
            compact=(self.context_style == "compact"),
        )
        ids, mask = self.tokenizer.encode(prompt)
        if self.atlas is not None:
            img, _src = self.atlas.image(int(scene.scene_id), rng=self._rng)
        else:
            img = np.zeros((3, 224, 224), dtype=np.float32)
        batch = {
            "image": torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0).to(self.device),
            "state": torch.zeros(1, self._state_dim, device=self.device),
            "lang_ids": torch.as_tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device),
            "lang_mask": torch.as_tensor(mask, dtype=torch.long).unsqueeze(0).to(self.device),
            "prompt": [prompt],
        }
        with torch.no_grad():
            out = self.model(
                batch,
                context=torch.tensor([ctx.features()], dtype=torch.float32, device=self.device),
                kappa=torch.tensor([float(ctx.kappa)], device=self.device),
            )
        logits = out.said if self.read == READ_SAID else out.unprompted
        probs = torch.softmax(logits, dim=-1)[0]
        self.last_probs = (float(probs[0]), float(probs[1]))
        conf, idx = float(probs.max()), int(probs.argmax())
        label = STRATEGY_A if idx == 0 else STRATEGY_B
        band = "band" if abs(float(getattr(scene, "c", 0.0))) <= 1.5 else "flank"
        self._by_band[band][0] += 1
        if conf < self.min_confidence:
            self.stats["abstained"] += 1
            self._by_band[band][1] += 1
            return STRATEGY_UNRESOLVED
        self.stats["agree_lexical"] += int(label == lexical_ground(utterance, self.axis))
        return label

    def report(self) -> Dict[str, Any]:
        n = max(int(self.stats["calls"]), 1)
        return {
            "grounder": self.name,
            "read": self.read,
            "calls": int(self.stats["calls"]),
            "abstain_rate": self.stats["abstained"] / n,
            "agreement_with_lexical": self.stats["agree_lexical"] / max(n - self.stats["abstained"], 1),
            "abstain_rate_band": self._by_band["band"][1] / max(self._by_band["band"][0], 1),
            "abstain_rate_flank": self._by_band["flank"][1] / max(self._by_band["flank"][0], 1),
            "min_confidence": self.min_confidence,
            "context_mode": self.context_mode,
            "context_style": self.context_style,
        }


__all__ = ["PolicyGrounder", "READ_SAID", "READ_UNPROMPTED"]

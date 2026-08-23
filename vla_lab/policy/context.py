r"""The carryover context, and the four ways of injecting it.

This is the paper's architectural independent variable. Every model in the roster sees the
*same* information about the robot's recent behaviour; they differ only in **where that
information enters the network**, and the comparison is designed so that difference is the
only thing that varies.

The context itself is a small, fully-specified record::

    CarryoverContext(
        recent = [(strategy, slots_ago, narrated), ...],   # what the robot just demonstrated
        kappa  = 0.83,      # signed residue estimate from the belief module (A positive)
        kappa_sd = 0.31,    # how sure the belief module is
        lambda_hat = 0.62,  # this supervisor's estimated decay
        slots_since = 2,
    )

Four injection modes:

``none``
    The context is dropped. This is the **Memoryless VLA** -- the ablation, and the model of
    a system that trusts whatever it is told.
``text``
    The context is verbalised into the language prompt: *"Context: in the last 3 episodes I
    demonstrated CLEAR_FIRST and said why. Compliance risk high (kappa 0.83 +/- 0.31, decay
    0.62)."* This is the literal proposal, and it is the mode that can only work if the
    backbone has a language model worth the name -- which is exactly what makes it a useful
    probe of the architecture.
``token``
    The context is embedded by a small MLP into ``n_tokens`` continuous vectors and prepended
    to the backbone's token sequence. Available to any transformer backbone, no language
    ability required.
``film``
    The context modulates the action/intent heads through FiLM (feature-wise affine
    modulation). The cheapest mode, available to backbones with no token interface at all --
    including the from-scratch convolutional baseline.

Why all four rather than the best one: the hypothesis under test is that *language-space*
injection is what lets a model reason about its own influence, because that is where the
pretrained prior about persuasion, compliance, and hedging lives. If ``token`` and ``film``
match ``text`` on a large backbone, that hypothesis is wrong and the paper says so.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

CONTEXT_NONE = "none"
CONTEXT_TEXT = "text"
CONTEXT_TOKEN = "token"
CONTEXT_FILM = "film"
CONTEXT_MODES = (CONTEXT_NONE, CONTEXT_TEXT, CONTEXT_TOKEN, CONTEXT_FILM)

#: Length of the numeric feature vector produced by :meth:`CarryoverContext.features`.
CONTEXT_DIM = 10


@dataclass
class CarryoverContext:
    """What the robot knows about the residue it has just deposited in the person watching.

    Every field is something the belief module
    (:mod:`vla_lab.supervisory.carryover`) actually produces at run time, so a model trained on
    these features can be driven by the real system rather than by an oracle. ``kappa`` is
    signed with the package convention: positive means the robot has been demonstrating the
    cautious strategy A.
    """

    kappa: float = 0.0
    kappa_sd: float = 0.0
    lambda_hat: float = 0.5
    beta_g_hat: float = 0.0
    slots_since_coach: int = 99
    #: ``(strategy, slots_ago, narrated_rationale)``, most recent first.
    recent: Tuple[Tuple[str, int, bool], ...] = ()
    #: Optional scene coordinate, so a model can learn that residue matters most near the
    #: crossover. Not required -- a real deployment may not know ``c``.
    scene_c: Optional[float] = None

    # -- numeric view -------------------------------------------------------
    def features(self) -> List[float]:
        """Fixed-width numeric encoding for the ``token`` and ``film`` modes."""
        last = self.recent[0] if self.recent else None
        n_a = sum(1 for s, _, _ in self.recent if s == "A")
        n_b = sum(1 for s, _, _ in self.recent if s == "B")
        n = max(len(self.recent), 1)
        return [
            float(self.kappa),
            float(self.kappa_sd),
            float(self.lambda_hat),
            float(self.beta_g_hat),
            math.exp(-float(self.slots_since_coach) / 4.0),
            1.0 if (last and last[0] == "A") else (-1.0 if last else 0.0),
            float(n_a - n_b) / n,
            float(min(len(self.recent), 8)) / 8.0,
            1.0 if (last and last[2]) else 0.0,
            float(self.scene_c) if self.scene_c is not None else 0.0,
        ]

    # -- language view ------------------------------------------------------
    def to_text(self, *, axis_labels: Tuple[str, str] = ("CLEAR_FIRST", "DIRECT"),
                compact: bool = False) -> str:
        """Verbalise the context for the ``text`` injection mode.

        Deliberately reads as something one agent would tell another about its own recent
        behaviour, not as a feature dump: the whole point of language-space injection is to let
        a pretrained model bring its prior about influence and compliance to bear, and a comma-
        separated list of floats gives it nothing to bring.
        """
        if not self.recent:
            return ("Note: no recent demo." if compact else
                    "Context: I have not demonstrated any strategy recently, so the supervisor's "
                    "answer is unprompted.")
        label = {"A": axis_labels[0], "B": axis_labels[1]}
        n = len(self.recent)
        last_strategy = self.recent[0][0]
        narrated = sum(1 for _, _, nr in self.recent if nr)
        same = all(s == last_strategy for s, _, _ in self.recent)
        risk = "high" if abs(self.kappa) > 0.8 else ("moderate" if abs(self.kappa) > 0.3 else "low")
        if compact:
            # ~20 tokens instead of ~70. Necessary, not cosmetic: SmolVLA's instruction budget
            # is 48 tokens and its tokenizer truncates from the LEFT, so the rich form silently
            # deletes exactly the informative half -- which strategy, how many times -- and
            # leaves the model a context stub that is worse than no context at all.
            return (f"Note: I showed {label.get(last_strategy, last_strategy)} {n}x, "
                    f"{self.slots_since_coach} ago; echo risk {risk} (k={self.kappa:+.2f}).")
        parts = [
            f"Context: in the last {n} episode{'s' if n != 1 else ''} I demonstrated "
            f"{label.get(last_strategy, last_strategy)}"
            + (" every time" if same and n > 1 else " most recently")
            + (f" and explained why {narrated} time{'s' if narrated != 1 else ''}" if narrated else "")
            + "."
        ]
        parts.append(
            f"That was {self.slots_since_coach} interaction"
            f"{'s' if self.slots_since_coach != 1 else ''} ago."
        )
        parts.append(
            f"Risk that the supervisor is echoing me rather than stating a preference: {risk} "
            f"(residue {self.kappa:+.2f} +/- {self.kappa_sd:.2f}, decay {self.lambda_hat:.2f})."
        )
        return " ".join(parts)

    def text_variants(self, *, axis_labels: Tuple[str, str] = ("CLEAR_FIRST", "DIRECT")) -> Dict[str, str]:
        """Both verbalisations, for the audit that reports which one a run used."""
        return {"rich": self.to_text(axis_labels=axis_labels),
                "compact": self.to_text(axis_labels=axis_labels, compact=True)}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recent"] = [list(r) for r in self.recent]
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CarryoverContext":
        d = dict(d or {})
        if "recent" in d and d["recent"] is not None:
            d["recent"] = tuple((str(a), int(b), bool(c)) for a, b, c in d["recent"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def empty(cls) -> "CarryoverContext":
        return cls()


def build_context_encoder(mode: str, *, d_model: int, n_tokens: int = 4, hidden: int = 128):
    """Torch module turning :meth:`CarryoverContext.features` into the chosen injection.

    Imported lazily so that the whole package -- including the study runner and the analysis --
    stays importable on a machine with no torch.
    """
    from ._torch_context import ContextEncoder  # noqa: WPS433

    return ContextEncoder(mode, d_model=d_model, n_tokens=n_tokens, hidden=hidden)


__all__ = [
    "CONTEXT_NONE",
    "CONTEXT_TEXT",
    "CONTEXT_TOKEN",
    "CONTEXT_FILM",
    "CONTEXT_MODES",
    "CONTEXT_DIM",
    "CarryoverContext",
    "build_context_encoder",
]

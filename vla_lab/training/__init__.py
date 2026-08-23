"""Training the Carryover-Aware VLA: the losses, the data, and the trainer.

Three supervision signals, and the middle one is the novel part:

``action``
    Standard behaviour cloning on the scripted experts' chunks, so the policy can actually
    execute both strategies. Without this the intent head is grounding instructions the robot
    cannot carry out, and the regret numbers mean nothing.
``intent``
    Two logits per sample -- what the supervisor *said*, and what they would have said
    uncoached. The second is supervised three ways: against a reference label where one exists,
    against the forward contamination model (self-supervised, needs no reference), and against
    an anti-copy term that penalises agreeing with a heavily-contaminated instruction without
    having checked.
``ask``
    Whether to spend a counter-proposal, trained against whether one would actually have
    changed the answer.

See :mod:`vla_lab.training.losses` for the exact objective and why each term is there.
"""

from __future__ import annotations

from .losses import CarryoverLossConfig, CarryoverLossOutput, carryover_loss

__all__ = ["CarryoverLossConfig", "CarryoverLossOutput", "carryover_loss"]

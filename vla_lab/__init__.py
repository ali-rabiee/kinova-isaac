"""vla_lab: carryover-aware supervisory control for the Kinova Jaco in Isaac Sim.

**The question.** A robot that demonstrates and narrates a manipulation strategy teaches the
person supervising it what to say. When it then asks *"how should I approach this one?"* on an
ambiguous scene, the instruction it gets back is a mixture of that person's own preference and
the residue of the robot's own coaching. This package estimates the unprompted preference,
models the residue, and decides what to do about it.

Live modules:

- ``vla_lab.supervisory``   the study: estimand, carryover model, schedulers, session, gate, analysis
- ``vla_lab.policy``        the Carryover-Aware VLA: context injection, intent heads, backbone registry
- ``vla_lab.training``      losses, dialogue data, trainer, the architecture sweep
- ``vla_lab.dataset``       ticks.jsonl -> torch Dataset (carried through from the VLA track)
- ``vla_lab.models``        TinyVLA, used as the from-scratch backbone in the model roster
- ``vla_lab.eval_isaaclab`` closed-loop Isaac evaluation (carried through)
- ``vla_lab.allocation`` / ``feedback`` / ``intent`` / ``calibration`` / ``human_study``
                            inference and interaction utilities carried through the pivot
- ``vla_lab.smolvla_bridge`` ticks.jsonl <-> LeRobot, SmolVLA policy wrapper

Archived, intact and still runnable, not maintained:

- ``vla_lab.old_direction``  the arm-choice rehabilitation submission and its ``rehab`` package
- ``vla_lab/old_demos/``     superseded checkpoints, results, docs, and legacy scripts

Start at ``vla_lab/README.md``; run ``./vla_lab/scripts/sup_study.sh``.
"""

__version__ = "0.1.0"

# `old_direction/` — the arm-choice rehabilitation submission

Archived 2026-08-22, intact and still runnable. Nothing here was deleted or rewritten.

**The paper.** *"When Is the Robot's Next Measurement Trustworthy? Carryover-Aware Scheduling
of Neutral Arm-Choice Assessment"* — Phase 0 of a post-stroke adaptive-robotics programme, in
which a Kinova Gen2 presents standardized reach targets and the participant reaches with their
own arm, and the question is when a neutral assessment is least contaminated by the robot's own
previous coaching. Source in [`paper/K_Farhadi_Paper_HRI_2027/`](./paper/), with its own
`legacy_vla_uncertainty/` archive one level deeper.

**The code.** [`rehab/`](./rehab/) is the complete Phase 0 implementation — workspace and
contract, the latent carryover model, three π\*(ℓ) estimators, COACH/WAIT/ASSESS schedulers,
apparatus backends, the observation stack, protocol and session runner, the session gate, power
analysis, and the analysis that produces every figure.

```bash
python -m vla_lab.tests.run_tests --archived-only     # 162/162
./vla_lab/old_direction/scripts/rehab_pilot.sh        # a full synthetic study, seconds
```

**Its relationship to the live direction.** The carryover mathematics in
`vla_lab/supervisory/` is a descendant of `rehab/carryover.py` and `rehab/estimand.py`. Three
things changed and each is documented at the line that changed it: κ became **signed** (so the
design can counterbalance the direction of the robot's influence), a **counter-proposal
attenuation** ρ was added (the fourth action has no analogue here), and the scene coordinate is
derived from a **task-value model** rather than from workspace geometry (which is what makes
decision regret measurable). The arm-choice-specific parts — the bilateral workspace, the
arm-choice observers, the human-proximate safety envelope — have no counterpart in the live
direction, which involves no physical contact.

`environments/bilateral_choice/` at the repository root is this track's Isaac digital twin. It
is kept runnable and repointed at `vla_lab.old_direction.rehab`, and is not maintained.

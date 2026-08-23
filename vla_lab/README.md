# `vla_lab/` — Carryover-Aware Supervisory Control

**A robot that demonstrates a strategy teaches the person watching it what to say.** When it
then asks *"how should I approach this one?"* on a genuinely ambiguous scene, the answer it
gets is a mixture of what that person actually prefers and what the robot just spent three
episodes showing them. This package treats that mixture as the object of study: it estimates
the supervisor's **unprompted strategy-preference map**, models the residue of the robot's own
coaching as a latent decaying state, and lets the robot decide when — and whether — to re-open
the option its own coaching closed.

Target venue: **HRI 2027**. Paper source: [`paper/`](./paper/).

---

## Quick start

```bash
conda activate riften                     # Isaac Sim 5.x + Isaac Lab + torch; numpy<2

# 1. The whole study, synthetic, no robot. Seconds to a few minutes.
./vla_lab/scripts/sup_study.sh            # -> vla_lab/results/tier1/{table.txt,fig_*.pdf}

# 2. Train one Carryover-Aware VLA cell.
./vla_lab/scripts/sup_train.sh            # tiny / token, ~2 min on a laptop 4090

# 3. The architecture comparison table.
./vla_lab/scripts/sup_models.sh           # -> vla_lab/results/models_isaac/table.txt

# 4. Put a trained checkpoint in the loop as the robot's ear.
./vla_lab/scripts/sup_deployed.sh         # -> vla_lab/results/deployed/table.txt

# 5. Audit what a run actually produced — provenance flags, curves, no re-running.
./vla_lab/scripts/sup_audit.sh vla_lab/results/models_isaac --figures

# 6. The offline gate (238 tests, no Isaac, no torch needed for most).
./vla_lab/scripts/run_tests.sh
```

Everything above runs with **no simulator**. The Isaac steps are separate, and the **order
matters** — each one invalidates what the next depends on:

```bash
./vla_lab/scripts/sup_reach.sh --headless          # 1. where the arm can actually reach
./vla_lab/scripts/sup_sweep.sh --headless --fit    # 2. measure the scene physics (~30 min)
./vla_lab/scripts/sup_frames.sh --headless         # 3. re-render the atlas on the new physics
#    ... then retrain (3 above), because the scene ids now map to different clearance gaps.
```

`build_scene_grid()` picks the fitted physics up from `vla_lab/results/physics/physics.json`
automatically, so a measurement propagates to the study, the atlas, the training data and the
closed-loop evaluation at once.

**Building the paper:**

```bash
./vla_lab/scripts/build_paper.sh          # syncs every figure and table, then builds
```

Never build it another way. The script regenerates the manuscript's figures and its *measured*
tables from the result files, lints the source for the two failure modes that compile cleanly and
read wrong, and refuses to finish with a dangling reference.

---

## The idea in one page

**The estimand.** For supervisor *p*, `π*_p(c) = Pr[p instructs the cautious strategy | scene at
c, no recent coaching]`. The coordinate `c` is **not** a raw distance: it is the signed
task-value margin between the two strategies, in units of the transition width, so `c > 0` means
the cautious strategy is objectively better here, `c < 0` means the efficient one is, and `c = 0`
is the crossover where the two are worth the same and the supervisor's answer is genuinely a
preference rather than a correct answer. Defining it that way buys three things: a map that is
monotone and saturating (so the information concentrates in a band), a coordinate that means the
same thing across strategy axes, and an **objective regret** — executing the wrong strategy costs
real task value, so compliance bias is a performance loss and not only a measurement artifact.

**The contamination.**

```
Pr[y_t = A | c_t, κ_t] = σ( logit π*_p(c_t) + ρ_t · β_p · κ_t )
κ_{t+1}                = λ_p^Δt · ( κ_t + g_p · s_t · d_t · 1[a_t = COACH] )
```

`κ` is **signed** (`d_t = ±1`: the robot can coach either strategy), `β_p` is the person's
compliance sensitivity, `λ_p` their decay rate, and `ρ_t ∈ [0,1]` the **counter-proposal
attenuation** — how much of the residue's grip a re-opened option removes.

**Four actions**, one per interaction slot:

| | what the robot does |
| --- | --- |
| `COACH` | executes a strategy on an *unambiguous* scene and narrates it. The manipulation; protocol-fixed. |
| `PROBE` | presents an ambiguous scene, asks the neutral query, executes what it is told. |
| `WAIT` | a strategy-neutral filler. Costs budget and wall-clock; decays the residue. |
| `COUNTER` | a probe **plus** naming the option the robot did *not* just demonstrate. The active de-biasing action. |

**Six conditions and three ablations** (`supervisory/scheduler/`): B0 no-coach reference,
B1 **Memoryless VLA**, B2 fixed washout, B3 random/static, B4 always-counter, B5 **carryover-aware**,
plus B5 with each of its three switches off.

---

## What the Tier-1 study says

`vla_lab/results/tier1/table.txt`, N = 80 synthetic supervisors, paired, all conditions on an
identical budget, under the **measured** scene physics. Read the test–retest floor (crossover
MAE **0.1016**) first: it bounds everything below it, and the study audit prints that reminder
itself.

- **Ignoring your own coaching is expensive, and it is the only thing that clears the floor.**
  B1 Memoryless is +0.0573 crossover-MAE [+0.0461, +0.0686] worse than the fixed washout and
  +0.0616 [+0.0535, +0.0698] worse than always asking, with +0.0043 [+0.0029, +0.0058] more
  decision regret. It wins **16 %** of paired comparisons. That gap is about four times the
  entire spread among the other seven conditions.
- **The penalty scales with the person.** Stratified by true compliance strength `βg`, B1's
  excess error over the washout grows **+0.044 → +0.046 → +0.082** across terciles. Nothing in
  any condition observes `βg`, so that dose–response runs through the contamination term.
- **Among the remedies, nothing separates.** Their whole spread is 0.0155 against a floor of
  0.1016; every paired interval against the washout contains zero. We report the ordering and
  decline to interpret it.
- **λ is poorly identified from one session**: identified in **44 %** of supervisors against
  **86 %** for compliance strength. Schedule personalisation therefore runs largely on its prior
  — and the schedule-only ablation is worst in the high-compliance tercile, where it would
  matter most.
- **A finding that did not survive re-measuring the physics.** Under the *prior* value model,
  always-counter beat the washout by −0.0148 [−0.0248, −0.0046] and "asking beats waiting" was a
  headline. Under the measured model, same policies and seeds, it is −0.0043 [−0.0171, +0.0089].
  Nothing about the methods changed; the difficulty curve they are scored against did.
- The study audit raises exactly two flags on this run: the λ identifiability, and that the
  non-memoryless spread is inside the floor.

### The closed-loop evaluation

`vla_lab/results/deployed{,_smolvla}/table.txt`. Dropping a trained checkpoint into the session as
the grounding channel, on both backbones with a full set of injection modes.

- **Reading the utterance (`said`) is free accuracy.** Every checkpoint matches its lexical
  reference to within 0.003 and resolves nearly all the hedged answers the keyword grounder
  refuses (1.1–1.4 → 0.0–0.6 ungrounded per block). This replicates on both backbones.
- **Letting the model answer (`unprompted`) helps on the pretrained backbone.** `token` and `film`
  reach 0.1042 / 0.1038 against the reference's 0.1105, alignment 0.900 → 0.939 / 0.953, and
  deployment regret cut four-fold (0.0027 → 0.0009 / 0.0006).
- **And breaks one cell on the from-scratch one.** `film` — best in its row on *both* offline
  metrics — is the worst channel anywhere: 0.1500 against a 0.1187 reference.
- **Offline metrics do not determine deployment.** Across the seven `unprompted` cells, offline
  gain orders deployed error only weakly (Spearman ρ = −0.46, n = 7) and in-band abstention barely
  at all (ρ = −0.14). Moderate abstention (≤5% in band) is fine and sometimes better; the one cell
  above 11% is the one that fails. Watch it as a threshold, not as a ranking.

### The architecture comparison

`vla_lab/results/models_isaac{,_smolvla,_qwen}/table.txt`. All cells train on **rendered Isaac
frames** from the study's own scene, so the comparison is between architectures under matched
perception.

- **The from-scratch backbone's context-blind cell lands on zero** (gap∼κ = **−0.006**), which is
  what it must do — its encoder cannot see κ — and is the cleanest confirmation that the metric
  measures what it claims to. Token injection takes it to +0.052, FiLM to +0.108.
- **Scale changes what "context-blind" means.** SmolVLA with no context injection still reaches
  +0.151, presumably from image and wording correlates of the residue. On a large backbone,
  injection roughly *doubles* an ability the model already has in part — a weaker claim than the
  from-scratch numbers alone would support, and we report it.
- **`film` gives the best gain on both backbones** (+0.135, +0.137), and the ordering
  none < token < film replicates.
- **On SmolVLA, `text` achieves the highest residue-tracking of any cell (+0.273) and the lowest
  gain (+0.082)** — tracking the residue and using it well are different abilities, and this is
  where they come apart. Verbalising costs prompt: 43 context tokens against ~7 for the
  instruction.

### The objective ablations

`vla_lab/results/ablations/table.txt`, on the best from-scratch cell. **Removing the reference
supervision collapses the gain to +0.003 while sending gap∼κ to +0.353 — the highest anywhere in
this project.** The model learns to move its belief *in step with* the residue and nothing about
*where to move it*. That is a direct warning about the headline metric: a high gap∼κ is necessary
and manifestly not sufficient, which is why both columns are always reported. Removing forward
consistency *improves* the gain (+0.031) while reducing residue tracking — the self-supervised
term costs a little fit to buy mechanism, and we say so rather than implying every term earns its
place.

Regenerate: `./vla_lab/scripts/sup_study.sh`. Figures land next to each result and reach the
manuscript through `./vla_lab/scripts/build_paper.sh`.

---

## Two tiers, and why

A scheduling study needs thousands of sessions; a closed-loop Isaac episode costs a minute.
Running the whole grid in the simulator is not a patience problem, it is a design mistake — it
re-measures the same execution outcomes over and over.

- **Tier 1** (`supervisory/apparatus/surrogate.py`) samples execution outcomes from success and
  duration curves **measured once** from an Isaac sweep. The scheduling, belief, and estimation
  code paths are the real ones.
- **Tier 2** (`supervisory/apparatus/isaac.py`) runs complete sessions end-to-end in Isaac with
  the policy in the loop.
- The **fidelity check** (`fidelity_report`) compares them and is reported *before* any Tier-1
  result. Tier 1 is a variance-reduction device whose validity rests entirely on that check.

> **Status of the scene driver (2026-08-23): working.** The margin sweep runs end to end and the
> scene physics is **measured** — 216 rollouts, crossover 8.5 cm, transition width 3.12 cm,
> `ScenePhysics.source == "measured"`. Getting there took ten distinct defects, seven of which
> presented with the identical symptom (*"every rollout times out on an early waypoint"*). They
> are recorded in the paper (§ *Making the instrument work*) and at the line of code that fixes
> each one. The four worth knowing before touching this scene:
>
> 1. **The controller's rotate mode is TOOL-frame.** `drot = quat_apply(ee_quat_b, drot)`, so a
>    base-frame axis is applied twice. The wrist alignment burned 768 steps and left downwardness
>    at −0.997 (worse than it started). Converting to the tool frame gives **+0.874** in one pass.
> 2. **`hold_after_orient` must be `True`** — and the measurement that said otherwise was taken
>    while (1) was active, i.e. it was holding an *inverted* wrist. Without the hold, the tool
>    drifts back to +0.17 downwardness during the first translate.
> 3. **A waypoint that only changes the gripper is done when the gripper moves.** Requiring
>    position convergence made the follower jog a *loaded* gripper for 16 000 steps and log a
>    timeout on a grasp that had succeeded.
> 4. **The start pose was silently favouring one strategy.** `start_ee_pos_b` was configured and
>    never applied, so every rollout began folded against the base; clear-first's opening move
>    took 409 steps and direct's diverged and timed out — making the direct strategy fail at
>    *every* gap. Pre-rolling to the collection contract's pose: **73 steps**.
>
> Geometry is now checked against a **measured** reachable envelope by a unit test
> (`sup_reach.sh` produces the map: `x` in [0.22, 0.66] m, 55/60 poses reached at grasp height).
>
> **One correction worth keeping.** The probe's first version jogged back to the home pose
> between points instead of resetting the episode, and that drifts — residual home error grew
> 1.8 cm → 6.5 cm → 14–18 cm. The artefact was *structured*: the row along `y = 0` came back
> marginal along its whole length while the rows either side looked clean, which had a ready
> mechanical story. We believed it and moved the scene corridor to `y = -0.10`. With a hard reset
> per point that row is fully reachable. The corridor stays where it is because the measured
> physics was fitted there, not because `y = 0` is a problem.

---

## The model roster

`vla_lab/policy/` wraps **any** VLA backbone with the same three additions, so the paper's table
compares *architectural features* and not four unrelated codebases:

| model | pretrain | language model | action head | context modes |
| --- | --- | --- | --- | --- |
| TinyVLA-2M (from scratch) | none | no | regression | none, token, film |
| SmolVLA-450M | robot (VLA) | yes | flow matching | none, text, token, film |
| Qwen2-VL-2B + head | web (VLM) | yes | regression | none, text, token, film |
| Qwen2.5-VL-3B + head | web (VLM) | yes | regression | none, text, token, film |

The independent variable is **where the carryover context enters**:

- `none` — dropped. This *is* the Memoryless VLA.
- `text` — verbalised into the prompt (*"in the last 3 episodes I demonstrated CLEAR_FIRST…"*).
  Requires a real language model, which is what makes it a probe of the architecture.
- `token` — a learned embedding prepended to the token sequence.
- `film` — feature-wise modulation of the heads. Available even to a backbone with no token interface.

The wrapper adds an **intent head** (two logits: what they *said*, and what they would have said
*unprompted* — initialised so an untrained model is exactly the Memoryless baseline), an **ask
gate**, and a **forward contamination model** that pushes the de-biased belief back through the
residue and requires it to reproduce the observed utterance. That last term is self-supervised,
so it needs no reference block and survives deployment.

LoRA is used for the Qwen cells and **degrades loudly**: without `peft` the backbone is frozen
instead and the manifest says so, because "LoRA" and "frozen backbone" are different experiments.

Four integration facts worth knowing before you touch `policy/backbones/smolvla.py`, each of
which cost a debugging cycle because it fails *quietly*:

- The instruction must reach the policy as `task`/`prompt`. A batch without it trains on empty
  strings and presents as slow learning, not as an error. The adapter now raises instead.
- Features must come from `VLAFlowMatching.embed_prefix`, **not** from a hook on
  `select_action` — that method is `@torch.no_grad()`, so the captured activation is detached
  and training completes having learned nothing.
- Pool the **instruction segment**, not the whole prefix. With three camera slots the image
  tokens outnumber the 48 language tokens by an order of magnitude, and a full-sequence mean
  leaves the intent head at chance. With tail pooling it reaches `acc_said = 1.00`.
- **Respect the instruction budget.** SmolVLA accepts 48 tokens and its tokenizer truncates from
  the **left**, so a 77-token verbalised context deletes exactly the informative half and keeps
  the instruction. That produced a run in which `text` injection scored *worst* of four modes
  (gap∼κ = −0.02) and looked like a clean falsification of the paper's central hypothesis. With
  the compact verbalisation inside budget the same cell reaches **+0.22**. The trainer now runs
  a prompt-budget audit before the first step and refuses to leave it unreported.

---

## Folder map

```
vla_lab/
├── supervisory/          ★ the study: estimand, carryover model, schedulers, session, gate, analysis
│   ├── scenes.py           SceneSpec/SceneGrid, the ambiguity coordinate c, the task-value model
│   ├── carryover.py        signed κ, the grid posterior over (λ, β, g), the population prior
│   ├── estimand.py         π*(c), three estimators, error/calibration/regret metrics
│   ├── supervisor.py       the generative supervisor  ← read its docstring before citing any number
│   ├── narration.py        what the robot says, and the conservative grounder
│   ├── scheduler/          B0–B5 + B5's three ablations
│   ├── apparatus/          surrogate | Isaac | the fidelity check | the physics sweep
│   ├── protocol.py         blocks, counterbalancing, the matched budget
│   ├── session.py          the one session runner   verify_session.py  the gate
│   ├── run_study.py        ★ Tier-1 study     run_sweep.py  ★ the Isaac margin sweep
│   └── analyze.py          every figure in the paper
├── policy/               ★ the Carryover-Aware VLA: context modes, heads, backbone registry
├── training/             ★ losses, dialogue data, trainer, the architecture sweep
├── paper/                ★ the HRI 2027 manuscript
├── results/                tier1/ · models/ · physics/
├── old_direction/          the previous (arm-choice rehabilitation) submission + its code, intact
├── old_demos/              superseded checkpoints, results, docs, legacy scripts
└── …                       dataset.py, models.py, eval_isaaclab.py, allocation/, feedback/,
                            intent/, calibration/, human_study/, smolvla_bridge/  (carried through)
```

`environments/supervisory_fetch/` holds the Isaac scene, the two scripted experts, and the
geometry that realises a given clearance gap.

---

## Reading the numbers honestly

Four boundaries are enforced in code and repeated in the paper:

1. **The simulated supervisor is not evidence about people.** A de-biasing method evaluated
   against a simulator whose bias we injected is being tested for whether it can invert a
   process we wrote down. That is how any estimator is validated, and a method that cannot
   recover a known ground truth will not recover an unknown one — but it is not evidence that
   human supervisors exhibit compliance carryover. Only participants can supply that.
2. **Prior versus measured.** `ScenePhysics.source` says which, and the analysis prints it next
   to any figure that depends on it. It currently reads `measured` — and that mattered: the
   headline "asking beats waiting" was significant under the prior and is not under the
   measurement. A simulated study's task-difficulty curve is a modelling choice that can decide
   a significance test, so measure it and plot it (`fig_physics.pdf`).
3. **Schematic versus rendered.** With no Isaac frames the trainer draws schematic scenes and
   prints a warning; those runs validate the pipeline and never enter the headline table. Every
   reported cell trains on rendered frames, and the audit refuses to be quiet about a run that
   does not.
4. **Offline versus deployed.** A learned intent head's held-out accuracy does not determine what
   it does in the loop (Spearman ρ = −0.46 over seven cells) — see `sup_deployed.sh`. Nothing here
   reports one as evidence for the other, and the in-band abstention column is a red flag rather
   than a ranking: ≤5% is fine, the one cell above 11% is the one that fails.

---

## The previous direction

The arm-choice rehabilitation submission (*"When Is the Robot's Next Measurement Trustworthy?"*)
and its complete `rehab/` implementation are intact in [`old_direction/`](./old_direction/),
including its own 162-test suite:

```bash
python -m vla_lab.tests.run_tests --archived-only     # 162/162
./vla_lab/old_direction/scripts/rehab_pilot.sh        # still runs
```

Nothing was deleted. The carryover mathematics here is a descendant of that work — signed κ, a
counter-proposal attenuation, and a value-defined coordinate are the three things that changed.

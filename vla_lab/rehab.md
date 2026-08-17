# `rehab.md` — Rehabilitation pivot: what `vla_lab/` needs for **Phase 0**

*Written 2026-08-16. Companion to the vision deck
(`paper/Post_Stroke_Adaptive_Robotics_Deck.pptx`) and its slide-by-slide script
(`paper/HRI_2027_Presentation_Script_v2.pdf`, rev. 2, 27 slides + 10 bibliography slides).*

This document specifies **everything that must be added to or changed in `vla_lab/`** so the
repository can run **Phase 0** of the post-stroke adaptive-robotics program:

> **Phase 0 (HRI 2027).** With **healthy adults** and the **Kinova Gen2 (JACO 2, `j2n6s300`)**,
> under a **matched interaction budget**, can a **personalized, carryover-aware
> COACH / WAIT / ASSESS policy** estimate an individual's **unprompted arm-choice map** more
> accurately, and with better-calibrated uncertainty, than naive assessment schedules?

It is an *additive* plan. Per the pivot decision, **the existing VLA / act–compute–query track
stays in the repository, maintained and documented as it is today.** Phase 0 lands beside it as
a new `vla_lab/rehab/` package plus a small number of edits to shared files. Nothing in
`models.py`, `train.py`, `dataset.py`, `eval_isaaclab.py`, `intent/`, `feedback/`, or
`smolvla_bridge/` is deleted or rewritten by this plan.

**Scope decisions already made** (recorded here so later readers do not re-litigate them):

| Decision | Choice |
| --- | --- |
| Robot role in Phase 0 | **Apparatus + coach (BARTR-style).** The Kinova *presents* standardized reach targets across a bilateral tabletop workspace and delivers scripted prompts. **The participant reaches with their own left or right arm.** The robot does not manipulate the target. |
| Platform | **Real Kinova Gen2**, with **Isaac Sim retained as a digital twin** for apparatus dry-runs, reachability/geometry checks, safety-envelope testing, and synthetic-participant pilots. |
| Existing VLA stack | **Kept and maintained.** Phase 0 is a parallel track inside the same package. |
| Paper | New comprehensive single-column draft; the current VLA paper is archived (see `paper/`). |

---

## Table of contents

1. [Phase 0, precisely](#1-phase-0-precisely)
2. [The pivot in one table: what actually changes](#2-the-pivot-in-one-table-what-actually-changes)
3. [Coexistence policy](#3-coexistence-policy)
4. [Target architecture: the new `vla_lab/rehab/` package](#4-target-architecture-the-new-vla_labrehab-package)
5. [Verdict on every existing `vla_lab/` module](#5-verdict-on-every-existing-vla_lab-module)
6. [Work items W1–W18 (the build)](#6-work-items-w1w18-the-build)
7. [Edits to existing files](#7-edits-to-existing-files)
8. [Changes outside `vla_lab/`](#8-changes-outside-vla_lab)
9. [The Phase 0 contract](#9-the-phase-0-contract)
10. [Data model and on-disk schema](#10-data-model-and-on-disk-schema)
11. [Safety, ethics, IRB](#11-safety-ethics-irb)
12. [Open design decisions (with recommendations)](#12-open-design-decisions-with-recommendations)
13. [Milestones and critical path](#13-milestones-and-critical-path)
14. [Risks](#14-risks)
15. [What Phase 0 explicitly does *not* need](#15-what-phase-0-explicitly-does-not-need)

---

## 1. Phase 0, precisely

Everything downstream depends on stating the science exactly, so this section is normative.
The implementation sections refer back to it by symbol.

### 1.1 The estimand

Let $L$ be a fixed set of **standardized target locations** on a tabletop, spanning the
participant's bilateral reaching workspace (left of midline through right of midline). For a
participant $p$, define

$$\pi^\*_p(\ell) \;=\; \Pr\big[\text{participant selects the \emph{nonpreferred} arm} \;\big|\; \text{target at } \ell,\ \text{no recent robot prompt}\big],\qquad \ell\in L .$$

The vector $\pi^\*_p = (\pi^\*_p(\ell))_{\ell\in L}$ is the **unprompted arm-choice map**. It is
the Phase 0 estimand. It is the healthy-adult analogue of the BART / BARTR *arm-choice
workspace* (Han et al. 2013; Dennler et al. 2023), with "nonpreferred" defined per participant
by a handedness inventory rather than by lesion side.

Two properties of $\pi^\*$ drive the whole design:

- **It is not uniform.** Far to the nonpreferred side, $\pi^\*\to 1$; far to the preferred
  side, $\pi^\*\to 0$. Nearly all the *information* — and nearly all the between-person
  variance — lives in the **crossover band** near the midline where $\pi^\*\approx 0.5$.
  Target placement must concentrate there (§9).
- **It is a probability, not a label.** Each presentation yields one Bernoulli draw. Estimating
  $\pi^\*$ to useful precision requires repeated presentations at the same or nearby targets,
  which is what makes the interaction budget scarce and the scheduling question real.

### 1.2 The contamination mechanism (the phenomenon under study)

The robot's own prior COACH actions bias the very quantity it is trying to measure
(deck slide 12; script slide 12). Model this with a **latent carryover state** $\kappa_t\ge 0$:

$$\Pr[a_t=\text{nonpref}\mid \ell_t,\kappa_t] \;=\; \sigma\!\big(\operatorname{logit}\pi^\*(\ell_t) \;+\; \beta_p\,\kappa_t\big),$$

$$\kappa_{t+1} \;=\; \lambda_p^{\,\Delta_t}\,\kappa_t \;+\; g_p\cdot\mathbb{1}[a_t=\text{COACH}],$$

where $\Delta_t$ is elapsed time (or intervening trials — see §12.3), $\lambda_p\in(0,1)$ is the
per-participant decay, $g_p$ the prompt gain, and $\beta_p$ the sensitivity. $(\lambda_p,\beta_p,g_p)$
are **person-specific and unknown**; that is the whole point. A fixed washout is a bet on a
population-level $\lambda$; the proposed policy estimates $\lambda_p$ online.

Additional terms the model must at least *admit*, even if Phase 0 does not identify them all
(they are the candidate confounds the paper must name): success/failure history at $\ell$,
fatigue accumulating over the session, habituation to the probe itself, and demand
characteristics from being observed.

### 1.3 The decision problem

At trial $t$ the scheduler observes the history $h_t$ (all previous targets, actions, selections,
outcomes, timestamps) and emits an action:

| Action | Effect |
| --- | --- |
| **ASSESS** | Present target $\ell_t$ with **no cue**. Observe one arm selection. Consumes budget. Sample is clean iff $\kappa_t\approx 0$. |
| **COACH** | Present target $\ell_t$ **with the scripted prompt** to use the nonpreferred arm. Observe selection. Consumes budget. Sets $\kappa \mathrel{+}= g_p$. |
| **WAIT** | Idle / neutral filler for a fixed dwell. Consumes budget (and wall-clock). Decays $\kappa$. |

The **budget is matched across conditions**: identical total trials $T$, identical number of
COACH events $C$, identical target sequence (see §12.1). Conditions differ **only in where the
ASSESS probes are placed relative to the COACH events**, and in whether the estimator corrects
for residual $\kappa$.

Objective: minimize the error of the final estimate $\hat\pi^\*$, with well-calibrated
uncertainty.

### 1.4 Conditions

From deck slide 22 ("BASELINES"), made concrete:

| # | Condition | Rule |
| --- | --- | --- |
| B0 | **No-prompt reference** | Never COACH. All trials ASSESS. Defines the reference map $\tilde\pi^\*$ (§12.2) and doubles as the cleanest possible baseline. |
| B1 | **Immediate assessment** | ASSESS on the trial immediately following each COACH. Maximum contamination. |
| B2 | **Fixed washout** | ASSESS only after a fixed interval $w$ (trials or seconds) since the last COACH; WAIT to fill. $w$ is a population constant. |
| B3 | **Random / static schedule** | ASSESS placed at random (or at a fixed pre-specified pattern) independent of history. |
| B4 | **Carryover-aware personalized policy** *(proposed)* | Online posterior over $(\lambda_p,\beta_p,g_p)$; schedule ASSESS when expected contamination is low **or** when the model can correct for it, and report $\hat\pi^\*$ marginalized over the carryover posterior. |

B4's advantage is expected to come from two distinguishable sources, and the analysis must
separate them: **(i) better probe placement in time**, and **(ii) the ability to use
mildly-contaminated observations by de-biasing them**. An ablation that gives B4 the estimator
but a fixed schedule (and vice versa) isolates the two.

### 1.5 Outcomes

| Tier | Outcome | Operationalization |
| --- | --- | --- |
| **Primary** | Estimation error of $\hat\pi^\*$ | Mean absolute error and Brier score of $\hat\pi^\*$ against the reference map $\tilde\pi^\*$, averaged over $L$ and weighted toward the crossover band. |
| **Primary** | Calibration of $\hat\pi^\*$'s uncertainty | Credible-interval coverage at nominal levels; ECE over binned predicted probabilities. Reuses `vla_lab/calibration/metrics.py`. |
| Secondary | Task success | Fraction of presentations where the reach was completed cleanly (target touched/grasped, no drop, no re-attempt). |
| Secondary | Interaction budget actually spent | Trials, COACH events, WAIT dwell, wall-clock. Matched by design → report as a manipulation check. |
| Secondary | Waiting cost | Total idle time attributable to WAIT. |
| Secondary | Participant burden | NASA-TLX (already in `human_study/instruments.py`) + a short session-burden/fatigue item set. |
| Exploratory | Per-person carryover parameters | Posterior over $(\lambda_p,\beta_p,g_p)$; between-person heterogeneity is a headline figure, not a nuisance. |

### 1.6 Claim boundary (must appear verbatim in code docs and paper)

> Phase 0 is a **healthy-adult physical-HRI measurement and scheduling** result. It is **not**
> evidence of stroke nonuse, rehabilitation efficacy, clinical construct validity, or recovery.
> "Nonpreferred arm" is a handedness-defined label, not a paretic limb.

---

## 2. The pivot in one table: what actually changes

| Axis | Current VLA track | **Phase 0 rehab track** |
| --- | --- | --- |
| Who acts | The robot (reach → grasp → lift) | **The human** (reaches to a presented target with one arm) |
| Robot's job | Execute a policy | **Present** targets, **prompt**, **observe**, **schedule** |
| Learned policy | TinyVLA / SmolVLA over pixels + language → EE deltas | **None on the critical path.** A statistical carryover model + a scheduling policy |
| Uncertainty being routed | Perception vs. human intent | **Prompt carryover** (deck slide 13, row 1) — a different row of the same table |
| Action set | act / compute / query | **COACH / WAIT / ASSESS** |
| Primary measurement | Task success rate | **Estimation error + calibration of a per-person probability map** |
| Human's role | Instructor / corrector of the robot | **The subject of the measurement** |
| Platform | Isaac Sim (real robot = stubs) | **Real Kinova Gen2**, Isaac as digital twin |
| Episode unit | Episode = one pick attempt, ~15 Hz tick log | **Trial** = one target presentation + one arm selection; event-locked, not fixed-rate |
| Safety regime | Robot alone in a sim scene | **Human limbs inside the robot's workspace** → interlocks, e-stop, motion/reach interleaving |
| Ethics | None required | **IRB review required** (long lead item) |

The single biggest architectural consequence: **the fixed-rate, image-and-action `ticks.jsonl`
data model is not the Phase 0 data model.** Phase 0's unit of analysis is a *trial record*
(target, prompt, selection, outcome, latency, history pointer). Fixed-rate logging is still
useful — as raw evidence and for offline video coding — but it is secondary. §10 specifies the
new schema.

---

## 3. Coexistence policy

Because the VLA track stays maintained, these rules keep the two from entangling:

1. **All Phase 0 code lives under `vla_lab/rehab/`** and imports *upward* from shared utilities
   (`vla_lab/calibration/`, `vla_lab/stats_utils.py`, `vla_lab/human_study/instruments.py`).
   Nothing in the VLA track may import from `vla_lab/rehab/`.
2. **Shared modules are extended, never repurposed.** Where a Phase 0 need is close to an
   existing abstraction (the allocator's three-way decision, the Beta-reliability fusion, the
   simulated human), Phase 0 gets its **own** implementation in `rehab/` that *borrows the
   design*, rather than adding rehab branches to `allocation/allocator.py` or
   `feedback/sim_human.py`. Those files keep serving the VLA paper unchanged.
3. **Separate config namespace**: `vla_lab/configs/rehab_*.yaml`.
4. **Separate script namespace**: `vla_lab/scripts/rehab_*.sh`.
5. **Separate log root**: `logs/rehab/participant_<PID>/session_<TS>/` — never mixed into
   `logs/data_collection/`.
6. **One test gate**: `vla_lab/tests/run_tests.py` gains the `test_rehab_*` modules so a single
   `./vla_lab/scripts/run_tests.sh` still covers the whole repository.
7. **Documentation**: `README.md` gains a "Two tracks" preamble and a §13 pointing at this
   document; `rehab.md` is the Phase 0 entry point, `README.md` stays the VLA entry point.

---

## 4. Target architecture: the new `vla_lab/rehab/` package

```
vla_lab/rehab/                          ★ NEW — everything Phase 0
├── __init__.py
├── README.md                     Phase 0 quick start (run order, one screen)
│
├── workspace.py                  Bilateral target geometry. TargetGrid, participant-midline
│                                 frame, robot↔table↔participant transforms, reachability
│                                 filter, crossover-band densification, target IDs.
├── contract.py                   The Phase 0 contract as code (§9): geometry, timing, budget,
│                                 prompt wording hash, apparatus version. Stamped into every
│                                 session; a session whose contract hash differs is not poolable.
│
├── apparatus/
│   ├── __init__.py
│   ├── base.py                   Apparatus protocol: present(target_id) → moves EE and settles;
│   │                             prompt(kind); go_signal(); home(); halt(); state().
│   ├── kinova_gen2.py            REAL JACO 2 backend (kinova-ros / JACO SDK). Blocking moves,
│   │                             settle detection, e-stop, joint/velocity limits, fault codes.
│   ├── isaac_apparatus.py        Digital-twin backend (Isaac Lab) — same protocol, same targets.
│   └── null.py                   No-robot backend for offline/synthetic pilots.
│
├── prompts.py                    COACH content library: exact wording, TTS/audio assets, optional
│                                 gesture spec, and a stable hash so wording drift is detectable.
│
├── observation/
│   ├── __init__.py
│   ├── base.py                   ArmChoiceObserver protocol → ArmSelection{arm, t_ms, confidence}
│   ├── vision.py                 Online left/right hand classifier from wrist + front camera
│   ├── keyed.py                  Experimenter keypress observer (backup + real-time gold standard)
│   ├── coding.py                 Offline video-coding ingest; inter-rater and human-vs-classifier
│   │                             agreement (Cohen's κ), disagreement export
│   └── calibration.py            Physical camera calibration + table/participant frame solve
│
├── trial.py                      Trial / TrialResult dataclasses; the per-trial phase machine
│                                 (PRESENT → SETTLE → GO → REACH → SELECT → RETURN → LOG).
├── logging.py                    Phase 0 session writer: trials.jsonl, events.jsonl, media/,
│                                 contract.json, participant.json. Event-locked timestamps.
├── verify_session.py             Phase 0 session gate (analogue of vla_lab/verify_session.py):
│                                 refuses sessions with missing selections, contract drift,
│                                 degenerate target coverage, classifier-vs-coder disagreement
│                                 over threshold, or safety-halt events.
│
├── carryover.py                  The latent carryover model (§1.2): likelihood, priors,
│                                 sequential posterior over (λ, β, g), predictive contamination.
├── estimand.py                   π*(ℓ) estimators: pooled-Bernoulli, GP/logistic-spatial over the
│                                 workspace, carryover-corrected variant; posterior intervals;
│                                 error + calibration metrics against a reference map.
│
├── scheduler/
│   ├── __init__.py
│   ├── base.py                   Scheduler protocol: decide(history) → (COACH|WAIT|ASSESS, ℓ)
│   ├── baselines.py              B0 no_prompt, B1 immediate, B2 fixed_washout, B3 random/static
│   └── carryover_aware.py        B4 proposed policy + its two ablations (schedule-only,
│                                 estimator-only)
│
├── sim_participant.py            Generative participant: latent π* drawn from a population prior,
│                                 (λ, β, g), fatigue, lapse rate, reach-time model, misdetection.
│                                 The Phase 0 analogue of feedback/sim_human.py, written fresh.
│
├── protocol.py                   Block structure, condition assignment, counterbalancing,
│                                 reference-first / retest-last layout (§12.2), randomization seeds.
├── session.py                    Session runner: wires apparatus + observer + scheduler +
│                                 logging + safety; runs real, twin, or synthetic sessions
│                                 through the same code path.
├── safety.py                     Human-proximate envelope: motion/reach mutual exclusion,
│                                 speed/force caps, dwell watchdog, e-stop plumbing, halt reasons.
│
├── analyze.py                    Primary/secondary outcome computation, per-condition tables,
│                                 heterogeneity plots, figures for the paper.
└── power.py                      Phase-0 power/sample size from pilot estimates (paired difference
                                  in estimation error), driven by sim_participant Monte Carlo.
```

New configs, scripts, tests:

```
vla_lab/configs/rehab_phase0.yaml         The contract + protocol for the real study
vla_lab/configs/rehab_sim_pilot.yaml      Synthetic-participant pilot settings
vla_lab/configs/rehab_twin.yaml           Isaac digital-twin dry-run settings

vla_lab/scripts/rehab_pilot.sh            End-to-end synthetic study, no robot, seconds  ★
vla_lab/scripts/rehab_twin_dryrun.sh      Apparatus dry-run in Isaac (geometry/reachability) ★
vla_lab/scripts/rehab_calibrate.sh        Camera + table/participant frame calibration
vla_lab/scripts/rehab_session.sh          Run one real participant session               ★
vla_lab/scripts/rehab_verify.sh           Post-session gate                              ★
vla_lab/scripts/rehab_analyze.sh          Outcomes + figures
vla_lab/scripts/rehab_power.sh            Sample-size memo from pilot data

vla_lab/tests/test_rehab_workspace.py
vla_lab/tests/test_rehab_carryover.py
vla_lab/tests/test_rehab_estimand.py
vla_lab/tests/test_rehab_scheduler.py
vla_lab/tests/test_rehab_sim_participant.py
vla_lab/tests/test_rehab_protocol.py
vla_lab/tests/test_rehab_safety.py
vla_lab/tests/test_rehab_logging.py
```

★ = the five commands that will be used day-to-day.

---

## 5. Verdict on every existing `vla_lab/` module

Legend — **SHARE**: Phase 0 imports it unchanged. **PATTERN**: Phase 0 writes its own version
borrowing the design (no edits to the original). **EXTEND**: the existing file gains additive
Phase-0 content. **VLA-ONLY**: untouched, stays for the VLA track.

| Module | Verdict | Note |
| --- | --- | --- |
| `calibration/metrics.py` | **SHARE** | `reliability_bins`, `expected_calibration_error`, `auroc`, `coverage_vs_occlusion` apply directly to $\hat\pi^\*$ calibration. Generalize `coverage_vs_occlusion`'s bucket axis or add a thin wrapper. |
| `calibration/records.py` | **PATTERN** | `CalibrationRecord` is VLA-shaped (dispersion, occlusion). Phase 0 gets its own record type in `rehab/estimand.py`. |
| `stats_utils.py` | **SHARE** | `wilson_ci`, `mcnemar_test` used as-is for secondary outcomes. |
| `human_study/instruments.py` | **EXTEND** | Reuse NASA-TLX scoring verbatim. **Add**: Edinburgh Handedness Inventory (defines "nonpreferred arm" — required, not optional) and a short session-burden/fatigue scale. Trust/MDMT stay for the VLA study. |
| `human_study/protocol.py` | **PATTERN** | Latin-square counterbalancing logic is directly reusable in spirit; Phase 0's IVs are different (schedule condition, block position), so `rehab/protocol.py` is its own file. Factor the balanced-Latin-square helper into a shared function if it stays identical. |
| `human_study/session.py` | **PATTERN** | Same shape (runner + logger + questionnaire provider), different trial semantics. |
| `human_study/analyze.py`, `reliance.py` | **VLA-ONLY** | Reliance/over-trust quadrants are not Phase 0 outcomes. |
| `human_study/power.py` | **SHARE + EXTEND** | `sample_size_paired`, `power_paired_t`, `normal_ppf` reused. `rehab/power.py` adds the Monte-Carlo path over `sim_participant`. |
| `allocation/allocator.py` | **PATTERN** | The `Controller.decide()` → `Decision` shape is exactly the right interface for COACH/WAIT/ASSESS, and `allocation/conformal.py` is the right idea for calibrated intervals. Copy the *design*; do not add rehab branches here. |
| `allocation/value_of_information.py` | **PATTERN** | The VOI machinery maps cleanly onto "which target, when" information gain — the closest existing analogue to B4's target/timing choice. |
| `allocation/transparency.py` | **VLA-ONLY** (revisit in Phase 3) | Typed-uncertainty messaging is a later-phase action. |
| `feedback/*` | **VLA-ONLY** | Phase 0 has no mid-execution human corrections. `sim_human.py` is the *pattern* for `sim_participant.py`. |
| `intent/estimator.py` | **VLA-ONLY** | Counterfactual instruction sweep over colors has no Phase 0 role. |
| `models.py`, `losses.py`, `train.py`, `dataset.py` | **VLA-ONLY** | No learned visuomotor policy on the Phase 0 critical path. The vision *encoder* may later be reused for the arm-choice classifier — treat that as optional (§12.5). |
| `eval_isaaclab.py` | **VLA-ONLY** | Phase 0's Isaac use is a *different* env and a different loop (`rehab/apparatus/isaac_apparatus.py`). Do not extend this 84 KB file. |
| `ttc.py`, `ttc_methods/`, `partial_obs.py` | **VLA-ONLY** | — |
| `smolvla_bridge/` | **VLA-ONLY** | — |
| `real_robot/kinova_bridge.py` | **PATTERN** (new Gen2 file; original left in place) | ⚠️ The current stub is written for **Gen3 / Kortex** (`arm_namespace: /my_gen3`, "Kinova Gen 3 class naming"), and its interface is *policy-chunk stepping* — the wrong shape for Phase 0. Phase 0 needs blocking, settle-verified Cartesian/joint moves on a **Gen2 (JACO 2)**, which uses `kinova-ros` (`j2n6s300_driver`) or the JACO SDK. Leave the existing file for the VLA track; write `rehab/apparatus/kinova_gen2.py` fresh. |
| `real_robot/safety_envelope.py` | **PATTERN + EXTEND** | `WorkspaceAABB` and the velocity clamps are a reasonable seed, but Phase 0 needs *human-proximate* interlocks (mutual exclusion between robot motion and participant reach, dwell watchdog, e-stop state machine) that this file does not model. Build `rehab/safety.py`; keep this one for the VLA track. |
| `verify_session.py` | **PATTERN** | The "hard gate before you analyze anything" discipline is the most valuable thing in the repo. Phase 0 gets `rehab/verify_session.py` with its own failure modes (§6, W12). |
| `plot_metrics.py`, `checkpoint_utils.py`, `dryrun.py`, `inspect_data.py`, `repair_gripper_labels.py` | **VLA-ONLY** | — |
| `tests/run_tests.py` | **EXTEND** | Add the eight `test_rehab_*` modules to `MODULES`. |

---

## 6. Work items W1–W18 (the build)

Each item lists **why**, **where**, **depends on**, and **done when**. Ordered so that everything
before W11 can be built and validated with **no robot and no participants**.

### Tier A — offline foundations (no robot, no humans)

#### W1 · Workspace geometry and target set
- **Why.** Every downstream quantity is indexed by target location; the estimand's information
  lives in the crossover band, so target placement is a scientific decision, not a layout detail.
- **Where.** `rehab/workspace.py`, `rehab/contract.py`, `configs/rehab_phase0.yaml`.
- **What.** A `TargetGrid` in a **participant-centred frame** (origin at the participant's
  sternum projection on the table, +x forward, +y to the participant's left). Transform chain
  participant ↔ table ↔ robot base. Parameters: lateral extent, number of lateral bins, reach
  distances, densification factor in the crossover band, minimum inter-target spacing.
  Reachability filter against the JACO 2 workspace (validated in the twin, W10) *and* against a
  human reach model so no target requires trunk displacement (trunk compensation would confound
  arm choice).
- **Depends on.** —
- **Done when.** `test_rehab_workspace.py` passes: frames round-trip; every emitted target is
  robot-reachable and human-reachable; crossover densification is monotone in the parameter;
  target IDs are stable across runs given the same contract hash.

#### W2 · Trial model, phase machine, and event-locked logging
- **Why.** "Event-locked targets, prompts, selections, outcomes" (deck slide 22) is a hard
  requirement: the carryover model is a function of *elapsed time since prompt*, so timestamp
  fidelity is a first-class correctness property, not logging hygiene.
- **Where.** `rehab/trial.py`, `rehab/logging.py`.
- **What.** `Trial{trial_idx, block_idx, condition, action, target_id, prompt_id, t_present_ms,
  t_settled_ms, t_go_ms}` and `TrialResult{arm, t_select_ms, reach_time_ms, success, observer,
  confidence, halted, halt_reason}`. A strict phase machine with per-phase timeouts. A session
  writer producing `trials.jsonl`, `events.jsonl`, `contract.json`, `participant.json`, `media/`.
  All timestamps from one monotonic clock, with the wall-clock offset recorded once.
- **Depends on.** W1.
- **Done when.** `test_rehab_logging.py` passes: phase machine rejects out-of-order transitions;
  every trial record round-trips; clock offsets recorded; a truncated session file is detected
  rather than silently parsed.

#### W3 · Carryover model
- **Why.** It *is* the mechanism claim.
- **Where.** `rehab/carryover.py`.
- **What.** Likelihood of §1.2; priors over $(\lambda_p,\beta_p,g_p)$; **sequential** posterior
  update (particle filter or grid — the parameter space is 3-D, so a grid is sufficient and
  deterministic, which matters for a real-time scheduler); `predict_contamination(history, t)`
  returning expected bias and its uncertainty. Must support both time-based and trial-based
  decay parameterizations (§12.3) behind one flag.
- **Depends on.** W2.
- **Done when.** `test_rehab_carryover.py` passes: parameter recovery on data generated from the
  model (posterior mean within tolerance, coverage of credible intervals at nominal rate);
  posterior concentrates monotonically with more data; degenerate cases ($g=0$, $\beta=0$) are
  identified as such rather than producing spurious confidence.

#### W4 · Estimand and estimators
- **Why.** The primary outcome is defined here.
- **Where.** `rehab/estimand.py`.
- **What.** Three estimators, all returning a posterior over $\pi^\*(\ell)$ for every $\ell\in L$:
  (a) **pooled** per-target Beta-Bernoulli, ignoring carryover; (b) **spatial** — logistic/GP
  over workspace coordinates so nearby targets share strength (essential: budget per target is
  tiny); (c) **carryover-corrected** — (b) with the observation likelihood conditioned on
  $\kappa_t$ from W3, marginalized over the carryover posterior. Plus error metrics (MAE, Brier,
  crossover-weighted variants) and calibration metrics delegating to `calibration/metrics.py`.
- **Depends on.** W3.
- **Done when.** `test_rehab_estimand.py` passes: on synthetic clean data all three agree; on
  synthetic contaminated data (c) is unbiased while (a)/(b) are biased in the predicted
  direction; interval coverage is at nominal level; metrics match hand-computed values on a
  small fixture.

#### W5 · Schedulers (B0–B4 + ablations)
- **Why.** These are the compared conditions.
- **Where.** `rehab/scheduler/`.
- **What.** One `Scheduler` protocol; five implementations plus B4's two ablations
  (schedule-only, estimator-only). B4 selects the action that maximizes expected information
  about $\pi^\*$ per unit budget, using the W3 posterior — the VOI structure in
  `allocation/value_of_information.py` is the model to follow. **All conditions must consume the
  same target sequence** (§12.1); the scheduler chooses the *action*, and target selection is
  supplied by the protocol unless the exploratory target-personalization arm is enabled.
- **Depends on.** W3, W4.
- **Done when.** `test_rehab_scheduler.py` passes: each baseline reproduces its defining rule
  exactly; budget accounting is identical across conditions given the same protocol; B4 is
  deterministic given a seed; B4 degenerates to B2 when the carryover posterior is forced to a
  point mass at the population $\lambda$.

#### W6 · Simulated participant
- **Why.** It makes W1–W5 and the whole analysis pipeline testable end-to-end before any
  hardware or IRB approval exists, and it is the substrate for the power analysis.
- **Where.** `rehab/sim_participant.py`.
- **What.** Draw $\pi^\*_p$ from a population prior over crossover location/steepness; draw
  $(\lambda_p,\beta_p,g_p)$; simulate selections, reach times, occasional lapses, monotone
  fatigue, and **observer misdetection** at a configurable rate (so the pipeline is stress-tested
  against imperfect arm-choice detection, which is a real risk — W8).
- **Depends on.** W3.
- **Done when.** `test_rehab_sim_participant.py` passes: sampled maps are monotone in lateral
  coordinate; carryover shows the expected post-COACH elevation and decay; seeding is
  reproducible; misdetection rate is honoured.

#### W7 · Protocol, counterbalancing, and the reference/retest layout
- **Why.** The reference map is what "estimation error" is measured against; getting its
  position in the session wrong invalidates the primary outcome.
- **Where.** `rehab/protocol.py`.
- **What.** Block layout implementing **reference-first / retest-last** (§12.2): an uncontaminated
  no-prompt reference block at session start, the compared schedule blocks in counterbalanced
  order with enforced inter-block washout, and a terminal no-prompt retest block that yields both
  a test–retest reliability estimate for $\pi^\*$ and a residual-contamination check. Balanced
  Latin square for block order; explicit seeds; assignment file written before the session begins
  (so the analysis plan can be preregistered against it).
- **Depends on.** W1, W5.
- **Done when.** `test_rehab_protocol.py` passes: budget and COACH count identical across
  conditions; Latin square balanced across the planned N; the same participant ID + seed always
  produces the same assignment; reference and retest blocks contain zero COACH actions.

#### W8 · Arm-choice observation stack
- **Why.** This is the **highest-risk new sensing component**: the scheduler consumes the
  detection *online*, so misdetection propagates into $\hat\kappa$ and into every decision, and
  the outcome depends on it too.
- **Where.** `rehab/observation/`.
- **What.** Three observers behind one protocol: **vision** (online left/right classification of
  the reaching hand from the wrist camera and a front camera), **keyed** (experimenter keypress
  — always running, as ground truth and as fallback), and **coding** (offline frame-accurate
  video coding, the gold standard). `coding.py` computes classifier-vs-coder and coder-vs-coder
  agreement (Cohen's κ) and exports disagreements for review. Any trial whose online and coded
  labels disagree must be flagged, not silently overwritten.
- **Depends on.** W2.
- **Done when.** Offline: agreement machinery validated on fixtures. Online: on pilot video,
  vision-vs-keyed κ ≥ 0.9 with detection latency below the trial's SELECT-phase budget. **If the
  target κ is not reached, the fallback is to run the study keyed-only and treat vision as an
  exploratory contribution** — decide this at the pilot, not later.

#### W9 · Prompt library
- **Why.** COACH content must be *identical* across conditions and stable across participants,
  or the manipulation is confounded.
- **Where.** `rehab/prompts.py`.
- **What.** Fixed wording, rendered once to audio (or a fixed TTS voice + version pin), optional
  gesture spec, and a content hash written into `contract.json`. A "neutral filler" utterance for
  WAIT that carries no arm information.
- **Depends on.** —
- **Done when.** Wording hash changes iff wording changes; a session recorded under a different
  hash is refused by `rehab/verify_session.py` from pooling.

#### W10 · Isaac digital twin of the apparatus
- **Why.** Validate geometry, reachability, presentation trajectories, settle behaviour, and the
  safety envelope *before* a person is near the arm. This is the retained value of the Isaac stack.
- **Where.** `rehab/apparatus/isaac_apparatus.py`; new env at `environments/bilateral_choice/`
  (§8).
- **What.** Table + JACO 2 at the study mounting pose + a seated-participant proxy volume + the
  target grid; the same `Apparatus` protocol as the real backend; reachability sweep over all
  targets; collision check of every presentation trajectory against the participant proxy;
  render of the wrist-camera view at each target (to verify the participant's approaching hand
  will actually be in frame — a real risk with the current wrist mount, W11).
- **Depends on.** W1.
- **Done when.** `rehab_twin_dryrun.sh` reports 100 % of contract targets reachable, zero
  trajectory intersections with the participant proxy, and a wrist-view render per target.

### Tier B — hardware (robot, no participants yet)

#### W11 · Real Kinova Gen2 apparatus backend
- **Why.** Nothing in `real_robot/` is usable for this: wrong generation, wrong interface shape.
- **Where.** `rehab/apparatus/kinova_gen2.py`.
- **What.** Driver integration (`kinova-ros` `j2n6s300_driver` on ROS 1, or the JACO SDK via its
  Python binding — pick one and pin it, §12.4). Blocking `present(target_id)` with verified
  settle (position tolerance held for a dwell), `home()`, `halt()`, e-stop, torque/current
  read-back, fault-code surfacing, and a heartbeat. **Camera re-aim:** the current
  `WristCameraConfig` mount (offset `(0, −0.055, −0.11)` m, rpy `(180°, 0, 0)`, FOV 87°) points
  along the *grasp approach* axis; verify in the twin (W10) and on hardware that it frames the
  participant's approaching hand at every target, and record the revised hand-eye calibration in
  `contract.json`.
- **Depends on.** W10.
- **Done when.** All contract targets presented and settled on hardware, 200 consecutive
  presentations with zero faults, measured settle time and its variance recorded, e-stop verified
  from both participant and experimenter positions.

#### W12 · Safety envelope and interlocks
- **Why.** A human puts their hands where the robot moves. This gates IRB and everything after it.
- **Where.** `rehab/safety.py`.
- **What.** **Mutual exclusion between robot motion and participant reach** — the arm moves,
  *stops*, and only then is the GO signal issued; a reach detected during motion triggers an
  immediate halt. Speed and acceleration caps well below the JACO 2 defaults; torque/current
  threshold halt; dwell watchdog (no motion command may exceed a maximum duration); dual e-stop;
  padded end-effector target puck; a halt-reason taxonomy written to `events.jsonl`. Workspace
  AABB constrained so the arm's *body*, not just the EE, stays clear of reach paths.
- **Depends on.** W11.
- **Done when.** `test_rehab_safety.py` passes on the state machine; on hardware, every interlock
  is demonstrated and the demonstration is recorded for the IRB packet.

#### W13 · Physical calibration
- **Where.** `rehab/observation/calibration.py`, `scripts/rehab_calibrate.sh`.
- **What.** Camera intrinsics/extrinsics for the front camera; wrist hand-eye; table plane; and
  the **participant-frame solve** (chair position, sternum reference, midline) — repeated per
  participant, because the midline defines the crossover band and therefore the estimand's
  informative region. Written into `contract.json` and `participant.json`.
- **Depends on.** W11.
- **Done when.** Re-running calibration on a fixed rig reproduces target positions within a
  stated tolerance; per-participant midline is recorded and its uncertainty reported.

### Tier C — study

#### W14 · Session runner
- **Where.** `rehab/session.py`, `scripts/rehab_session.sh`, `scripts/rehab_pilot.sh`.
- **What.** One code path over apparatus backend ∈ {real, twin, null} and observer ∈ {vision,
  keyed, both, simulated}. Pause/resume, participant-initiated stop, mid-session safety halt with
  clean resume, block boundaries with rest, and questionnaire administration points.
- **Depends on.** W2, W5, W7, W8, W11, W12.
- **Done when.** `rehab_pilot.sh` runs a full synthetic study end-to-end offline in seconds; the
  same command with `--apparatus real` runs a real session; an induced halt mid-block leaves a
  session that `rehab/verify_session.py` accepts as *partial* rather than corrupt.

#### W15 · Phase 0 session gate
- **Where.** `rehab/verify_session.py`, `scripts/rehab_verify.sh`.
- **What.** Exit-1 refusal on: contract-hash drift; prompt-wording drift; missing or ambiguous
  arm selections above threshold; classifier-vs-coder κ below threshold; target coverage below
  the minimum per crossover bin; COACH-count mismatch across conditions; unexplained clock jumps;
  any un-annotated safety halt; missing handedness inventory.
- **Depends on.** W14.
- **Done when.** Each failure mode has a fixture that triggers it and a passing session that
  does not.

#### W16 · Power analysis and preregistration
- **Where.** `rehab/power.py`, `scripts/rehab_power.sh`, plus a preregistration document.
- **What.** Monte-Carlo power over `sim_participant` populations for the primary paired contrast
  (B4 vs. the strongest baseline, on crossover-weighted MAE), swept over plausible effect sizes
  and over the pilot's measured carryover magnitude. Output: N, trials per participant, and the
  minimum detectable effect. The preregistration fixes the estimand, the reference-map
  definition, the primary/secondary outcomes, the analysis model, exclusion rules, and the
  stopping rule **before** the first non-pilot participant.
- **Depends on.** W6, and pilot data for realistic parameters.
- **Done when.** A power memo exists with its assumptions traced to pilot measurements, and the
  preregistration is filed.

#### W17 · Analysis and figures
- **Where.** `rehab/analyze.py`, `scripts/rehab_analyze.sh`.
- **What.** Primary contrast (mixed model over participants, or paired test per §12.6);
  calibration curves; per-person carryover posteriors (the heterogeneity figure); budget
  manipulation check; burden; test–retest reliability from the reference/retest pair; and the
  ablation decomposition of B4's advantage. All figures written as PDFs into the paper's
  `figures/` directory by the same script that computes the numbers.
- **Depends on.** W4, W15.
- **Done when.** Running it on the synthetic pilot produces every paper figure with no manual steps.

#### W18 · Documentation and test-gate integration
- **Where.** `rehab/README.md`, `README.md`, `tests/run_tests.py`, `pyproject.toml`.
- **Done when.** `./vla_lab/scripts/run_tests.sh` runs both tracks' suites green, and a new
  reader can get from `README.md` to a working synthetic Phase 0 pilot without asking anyone.

---

## 7. Edits to existing files

Deliberately small — the coexistence policy (§3) keeps Phase 0 out of VLA-track internals.

| File | Change |
| --- | --- |
| `vla_lab/README.md` | Add a "Two tracks" preamble at the top (VLA / act-compute-query **and** rehab Phase 0) with a pointer to `rehab.md`; add `rehab.md` and `rehab/` to the folder map (§2) and the document index (§12). Do not restructure the rest. |
| `vla_lab/tests/run_tests.py` | Append the eight `test_rehab_*` modules to `MODULES`. |
| `vla_lab/human_study/instruments.py` | **Add** Edinburgh Handedness Inventory scoring and a session-burden/fatigue scale. Existing NASA-TLX / Jian / MDMT code untouched. |
| `vla_lab/human_study/protocol.py` | Optional: factor the balanced-Latin-square helper into a module-level function so `rehab/protocol.py` can import it instead of duplicating. No behaviour change. |
| `vla_lab/calibration/metrics.py` | Optional: generalize `coverage_vs_occlusion`'s bucketing axis (or add `coverage_vs_bucket`) so Phase 0 can bucket by carryover level instead of occlusion. Additive. |
| `pyproject.toml` (repo root) | Add `vla_lab.rehab`, `vla_lab.rehab.apparatus`, `vla_lab.rehab.observation`, `vla_lab.rehab.scheduler`, and `environments.bilateral_choice` to `[tool.setuptools] packages`. |
| `.gitignore` | Ignore `logs/rehab/**/media/` (video is large and identifiable — see §11) while keeping `trials.jsonl` / `events.jsonl` tracked-able. |
| `vla_lab/RECOMMENDATIONS.md` | Add a dated note at the top recording the 2026-08 pivot and pointing to `rehab.md`; the July analysis stays as the VLA-track record. |

**Explicitly *not* edited:** `eval_isaaclab.py`, `allocation/allocator.py`, `feedback/*`,
`intent/*`, `models.py`, `train.py`, `dataset.py`, `smolvla_bridge/*`, `real_robot/*`.

---

## 8. Changes outside `vla_lab/`

Phase 0 is not fully containable inside `vla_lab/`; these are the out-of-package additions.

| Location | Change |
| --- | --- |
| `environments/bilateral_choice/` (**new**) | Isaac Lab scene for the twin: table, JACO 2 at the study mounting pose, seated-participant proxy volume, target-grid markers, front + wrist cameras. Mirrors the structure of `environments/reach_to_grasp_VLA/` (`config.py` + `demo.py` + `utils.py`). Reuses `environments/utils/camera/wrist.py` and `topdown.py`. |
| `environments/reach_to_grasp_VLA/config.py` | **Untouched.** The twin gets its own `WristCameraConfig`-shaped class with the re-aimed Phase 0 mount, so the VLA contract is not disturbed. |
| `controllers/` | Reusable as-is for the twin (`cartesian_velocity`, `safety.py`, `input/waypoint_follower.py`). The real Gen2 backend does **not** go through these — it talks to the vendor driver. |
| `motion_generation/planners/` | Optional for the twin if presentation trajectories need planning around the participant proxy; `scripted.py` is likely sufficient for a fixed target grid. |
| `data_collection/` | **Untouched.** Phase 0 does not use `ticks.jsonl`, the episode runner, or the profile registry. |
| ROS / driver workspace (out of tree) | `kinova-ros` (or JACO SDK) install, plus the camera driver. Document the exact versions in `rehab/README.md`; this is an environment dependency, not repo code. |
| IRB / institutional | Protocol, consent form, recruitment materials, data-management plan. Not in the repo (§11), but referenced from `rehab/README.md`. |

---

## 9. The Phase 0 contract

The analogue of `README.md` §7 ("the model contract"). Anything in this table that changes
between participants makes their sessions **non-poolable**; `rehab/contract.py` hashes it and
`rehab/verify_session.py` enforces it.

| Item | Value | Defined in |
| --- | --- | --- |
| Robot | Kinova **Gen2 / JACO 2 `j2n6s300`**, 6-DoF, 3-finger | `rehab/apparatus/kinova_gen2.py` |
| Robot mounting pose | Fixed relative to the table; recorded as a transform | `configs/rehab_phase0.yaml` |
| Participant frame | Origin = sternum projection on table plane; +x forward, +y to participant's left. **Solved per participant** | `rehab/observation/calibration.py` |
| Target set $L$ | Fixed lateral × depth grid in the participant frame, densified in the crossover band; stable integer target IDs | `rehab/workspace.py` |
| Presentation | EE moves to target, **stops**, settle verified, GO signal issued | `rehab/apparatus/base.py` |
| Trial timing | Fixed SETTLE dwell, fixed GO→timeout window, fixed inter-trial interval | `configs/rehab_phase0.yaml` |
| COACH content | Exact wording + audio asset, content-hashed | `rehab/prompts.py` |
| Action set | `{COACH, WAIT, ASSESS}` — nothing else in Phase 0 | `rehab/scheduler/base.py` |
| Budget | Total trials $T$ and COACH count $C$, **identical across conditions** | `rehab/protocol.py` |
| Block layout | Reference-first / conditions counterbalanced / retest-last | `rehab/protocol.py` |
| Nonpreferred arm | Defined by Edinburgh Handedness Inventory, recorded before any trial | `human_study/instruments.py` |
| Arm-choice label | Online observer + offline coded gold standard; both stored | `rehab/observation/` |
| Cameras | Wrist (re-aimed) + front; intrinsics/extrinsics per session | `rehab/observation/calibration.py` |
| Clock | One monotonic source; wall-clock offset recorded once per session | `rehab/logging.py` |

---

## 10. Data model and on-disk schema

```
logs/rehab/participant_<PID>/session_<TS>/
├── contract.json          hashed Phase 0 contract (§9) + apparatus/driver versions + git commit
├── participant.json       PID (pseudonymous), handedness inventory + score, demographics as
│                          approved by IRB, participant-frame calibration, condition assignment
├── protocol.json          block layout, seeds, target sequence — written BEFORE the first trial
├── trials.jsonl           one record per trial (the unit of analysis)
├── events.jsonl           full audit trail: phase transitions, halts, prompts, pauses, faults
├── observers.jsonl        per-trial labels from every observer, kept separately (never merged)
└── media/                 wrist + front video segments per trial (gitignored; see §11)
```

**`trials.jsonl` record** (one line per trial, floats fixed-precision as in the VLA logger):

```jsonc
{
  "trial_idx": 42, "block_idx": 2, "condition": "carryover_aware",
  "action": "ASSESS",                       // COACH | WAIT | ASSESS
  "target_id": 17, "target_xy_participant_m": [0.34, -0.02],
  "prompt_id": null, "prompt_hash": "…",
  "t_present_ms": 1712, "t_settled_ms": 2480, "t_go_ms": 2500,
  "selection": {"arm": "nonpreferred", "t_ms": 3210, "observer": "vision", "confidence": 0.97},
  "reach_time_ms": 710, "success": true,
  "kappa_prior_mean": 0.31,                 // scheduler's belief BEFORE this trial (for audit)
  "since_last_coach_ms": 48200, "coach_count_so_far": 6,
  "halted": false, "halt_reason": null
}
```

Design rules:
- **`observers.jsonl` is never collapsed into `trials.jsonl`.** The online label is what the
  scheduler acted on; the coded label is what the analysis uses. Both must remain recoverable,
  and their disagreement is a reported quantity.
- **`protocol.json` is written before trial 1** so the preregistered analysis can be checked
  against the realized assignment.
- **`kappa_prior_mean` is logged** so the scheduler's decisions can be audited post hoc — an
  adaptive policy whose reasoning is not recoverable is not reviewable.

---

## 11. Safety, ethics, IRB

**This is the longest-lead item in the entire plan. Start it before writing code (§13).**

- **Review level.** Healthy adults reaching into the workspace of a powered 6-DoF arm is very
  unlikely to be exempt. Budget for expedited or full review, with a safety appendix documenting
  W12's interlocks and the recorded demonstration of each.
- **Screening and consent.** Exclude upper-limb injury, pain, or conditions affecting reaching;
  screen for anything that would make repeated reaching unsafe. Consent must cover video
  recording of the participant's arms and torso.
- **Video is identifiable data.** `media/` holds recordings of people. Gitignore it, store it
  under the IRB-approved data-management plan, and keep the *coded labels* (not the video) as the
  shared analysis artifact. Frame-level coding should be doable on segments cropped to the reach
  region.
- **Pseudonymous IDs.** `participant.json` carries a study ID; the linking file lives outside the
  repository.
- **Right to stop.** The participant can end a trial, a block, or the session at any time; this
  must be a *logged event*, not a silent abort, and `rehab/verify_session.py` must accept the
  resulting partial session.
- **Physical setup.** Padded target puck on the EE; chair positioned so the participant cannot
  reach the arm's body; e-stop within the participant's reach **and** the experimenter's; the
  experimenter present for every trial.
- **Fatigue is both an ethics and a validity issue.** Cap session length, mandate rest at block
  boundaries, and log fatigue ratings — fatigue drifts arm choice and would masquerade as
  carryover.
- **Paper disclosures.** ACM/HRI requires an ethics statement; the claim boundary (§1.6) belongs
  there too. The venue's LLM-use disclosure policy applies to the paper as well.

---

## 12. Open design decisions (with recommendations)

These are genuine forks. Each carries a recommendation so the build can proceed; each should be
settled explicitly (and, for the ones that touch the primary outcome, before preregistration).

### 12.1 Does B4 also choose *where* to probe, or only *when*?
The script is explicit that the comparison "must hold the interaction budget and task
opportunities as constant as possible while varying the assessment schedule."
**Recommendation:** the primary contrast varies **only the timing/action**; every condition
consumes an identical target sequence. Target-selection personalization becomes a clearly-labelled
**exploratory arm**, because giving B4 both advantages while the baselines get neither confounds
the mechanism claim.

### 12.2 What is the reference map $\tilde\pi^\*$ measured against?
$\pi^\*$ is latent; the primary outcome needs an operational reference.
**Recommendation:** **reference-first / retest-last.** A no-prompt block at session start defines
$\tilde\pi^\*$ (uncontaminated by construction). A second no-prompt block at session end, after
enforced washout, provides (a) a test–retest reliability estimate that bounds how much of the
measured "estimation error" is irreducible drift, and (b) a residual-contamination check. Report
both; a study that cannot show test–retest stability of $\tilde\pi^\*$ cannot interpret its
primary outcome. The cost — reference-block position is confounded with session position — is
acceptable because the reference is a *measurement*, not a compared condition, and the terminal
retest quantifies the confound.

### 12.3 Does carryover decay in **time** or in **intervening trials**?
They are different mechanisms (memory decay vs. interference) and they imply different WAIT
semantics — waiting idle is cheap in trials but expensive in wall-clock.
**Recommendation:** parameterize both (W3 supports a flag), fit the pilot with both, preregister
whichever the pilot supports, and report the comparison. This is a genuinely interesting
secondary result, not just a modelling nuisance.

### 12.4 `kinova-ros` (ROS 1) or the JACO SDK directly?
`kinova-ros` gives a maintained driver, MoveIt, and a well-trodden path, at the cost of a ROS 1
dependency on a machine currently running Isaac Sim in a conda env (`riften`).
**Recommendation:** run the driver **in a separate process/container** and talk to it over a
narrow IPC surface, so `vla_lab/rehab` never has to import ROS. This keeps the Isaac environment
(and the `numpy<2` pin the VLA track depends on) untouched. If the ROS install proves hostile,
the JACO SDK's Python binding is the fallback — the `Apparatus` protocol makes this swap local
to one file.

### 12.5 Is the arm-choice classifier learned, or geometric?
A learned hand-side classifier is attractive but needs labelled data, and the whole study depends
on it.
**Recommendation:** start **geometric/heuristic** (which side of the participant midline the
approaching hand enters from, via the front camera, with the wrist camera as a confirming view).
It is auditable, needs no training data, and its failure modes are legible. Escalate to a learned
classifier only if the pilot shows the heuristic cannot hit κ ≥ 0.9 (W8). The VLA vision encoder
is *not* the right starting point — it was trained to locate colored boxes, not human hands.

### 12.6 Within-subject with prospective blocks, or a single dense session evaluated off-policy?
Running every schedule prospectively within one participant is the strongest evidence but is
expensive in session time and risks cross-block contamination.
**Recommendation:** **hybrid.** Run the reference block, B4, and the *strongest* baseline
prospectively (that is the confirmatory contrast); evaluate the full baseline set **off-policy**
on each participant's fitted carryover model as a clearly-labelled secondary analysis. Say plainly
in the paper that off-policy results are model-based. Which baseline is "strongest" should be
chosen from the synthetic study (W6) and fixed in the preregistration.

### 12.7 Is the manipulation strong enough in healthy adults?
Healthy adults have less room to move than stroke survivors, and a purely verbal prompt may
produce carryover too small to schedule around. This is the **single biggest threat to the
study's viability**, and the pilot must answer it before the full N is recruited.
**Recommendation:** (a) concentrate targets in the crossover band where choice is genuinely
near-chance and therefore movable; (b) make COACH stronger than a bare verbal cue — the
literature the deck cites shows arm choice is **effort-sensitive** (Nguyen et al. 2023), so a
task-difficulty/effort manipulation plus the prompt is both better-motivated and more likely to
produce measurable carryover; (c) make "is there a detectable, decaying carryover effect at all?"
the **pilot's go/no-go criterion**. If there is none, Phase 0's scheduling question is
unanswerable as posed and the design must change before, not after, data collection.

---

## 13. Milestones and critical path

Ordered by dependency, with the two long-lead items started immediately and in parallel with
coding.

| # | Milestone | Gate |
| --- | --- | --- |
| M0 | **IRB protocol drafted and submitted**; **Gen2 driver bring-up started** | Both begin in week 1, before Tier A is finished — they are the critical path, not the code. |
| M1 | Tier A complete (W1–W9): synthetic Phase 0 runs end-to-end offline | `rehab_pilot.sh` green; all `test_rehab_*` pass |
| M2 | Twin dry-run (W10) | 100 % target reachability, zero participant-proxy collisions, wrist view verified per target |
| M3 | Hardware apparatus + safety (W11–W13) | 200 fault-free presentations; every interlock demonstrated and recorded |
| M4 | **Lab pilot (N≈4–6, lab members)** | §12.7 go/no-go: a measurable, decaying carryover effect exists; observer κ ≥ 0.9; session length tolerable |
| M5 | Power memo + preregistration (W16) | Filed before the first non-pilot participant |
| M6 | Data collection | Every session passes `rehab_verify.sh` |
| M7 | Analysis + figures (W17) | `rehab_analyze.sh` regenerates every paper figure with no manual steps |

**Do not** wait for M1 to start M0. IRB turnaround and vendor-driver bring-up are the two things
that cannot be compressed by working harder on Python.

---

## 14. Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Carryover too small to measure in healthy adults (§12.7) | **Highest — kills the study** | Crossover-band targeting + effort-based COACH; make it the pilot's explicit go/no-go |
| IRB delay | **High — blocks everything after M3** | Start week 1; the safety appendix (W12) is the long pole inside it |
| Online arm-choice misdetection corrupts the scheduler's beliefs | **High** | Keyed observer always running; `sim_participant` stress-tests the pipeline at realistic misdetection rates (W6); pre-registered κ threshold with a keyed-only fallback |
| Gen2 driver integration harder than expected | Medium | Out-of-process driver + narrow IPC (§12.4); `Apparatus` protocol localizes the swap; twin keeps development unblocked |
| Session too long → fatigue confounds carryover | Medium | Cap trials from the power analysis, not from ambition; mandated rest; log fatigue and model it |
| Reference map itself unstable | Medium | Reference-first/retest-last gives a direct test–retest estimate; report it before the primary outcome |
| Two tracks in one package drift or entangle | Low–Medium | The coexistence rules in §3, enforced by the single test gate |
| Wrist camera does not frame the reaching hand at all targets | Low–Medium | Verified in the twin (W10) before hardware; front camera is the primary view, wrist is confirming |

---

## 15. What Phase 0 explicitly does *not* need

Stating this prevents scope creep back toward the VLA project — and prevents the paper from
claiming machinery it does not use.

- **No learned visuomotor policy.** No TinyVLA, no SmolVLA, no action chunks, no
  `ticks.jsonl`, no `success_only` filtering, no camera-set ablations on the Phase 0 path.
- **No object manipulation by the robot.** The robot presents; it does not grasp.
- **No clinical measures.** No FMA-UE, ARAT, WMFT, MAL, or AAUT in Phase 0 — those are Phase 2
  anchors, and administering them here would over-claim.
- **No patients.** Healthy adults only.
- **No stroke, nonuse, efficacy, or recovery claims.** See §1.6.
- **No transparency/typed-uncertainty messaging, no trust or reliance outcomes.** Those belong to
  the VLA study (which keeps them) and to later phases.
- **The remaining five actions of the long-term action library** (PRACTICE, ADAPT, SUPPORT
  AGENCY, TOLERATE, ESCALATE) are **out of scope by design** — deck slide 18 is explicit that
  Phase 0 studies only COACH / WAIT / ASSESS and that the rest must be earned in later phases.

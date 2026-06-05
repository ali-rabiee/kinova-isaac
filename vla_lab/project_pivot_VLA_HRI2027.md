# Project Pivot Memo
### From "Observability-Gated Test-Time Scaling for SmolVLA" → a Compute-vs-Query Allocation Account of VLA Policies in Human–Robot Teams

> **Purpose.** A working memo for how the current project needs to change, why, and what to build instead, so the resulting paper is a defensible HRI contribution rather than an incremental inference trick.

| | |
|---|---|
| **Target venue** | IEEE/ACM HRI **2027** (Santa Clara, CA; conference Mar 8–12, 2027) |
| **Binding deadline** | Full-paper deadline officially TBD; based on HRI's fixed pattern (HRI 2026 = abstract Sep 22 / paper Sep 30, 2025), plan for **~late September 2026**. Abstract + author list typically due ~1 week before the paper. |
| **Time remaining** | ~3.5 months from June 2026. |
| **Hardware** | Real Kinova JACO Gen2 (available). |
| **Capability** | Comprehensive human-subjects study (available). |
| **Status of pivot** | Committed to a genuine HRI pivot + deeper theoretical contribution. |

---

## 0. TL;DR — the pivot in three sentences

The current paper is a competent but incremental inference-optimization result, evaluated in simulation, with no human in it — wrong topic for HRI and a crowded one for robot-learning venues. **Pivot to a single, more durable claim:** *self-generated uncertainty can tell a policy how to spend compute, but it cannot tell the policy when compute is useless — and the regime where compute is useless is exactly where an external information source (a human, or a new sensor) is required.* Build this as a three-way allocation framework (act / compute / query), prove the small limit result that makes it non-obvious, and ground it in a controlled human-subjects study on the real JACO where **visual observability is the manipulated variable** and the headline outcome is *appropriate reliance / calibrated trust*, not just success rate.

**Before → After**

| | Current framing | Pivoted framing |
|---|---|---|
| Object of study | "Does test-time compute help VLAs?" | "When should a policy compute vs. acquire information, under partial observability — and what does that do to a human teammate?" |
| Decision | Binary: act (K=1) vs. compute (best-of-N), gated | Trichotomy: **act / compute / query** |
| Core asset | A gating heuristic (twin-probe disagreement) | A decision-theoretic limit + a calibrated escalation rule |
| Evidence | Sim-only, N=50, two observability points | Real JACO + human study, observability as a graded IV |
| Headline metric | Lift success rate | Appropriate reliance, trust calibration, workload (+ success) |
| Venue fit | CoRL/ICRA/IROS | HRI (genuinely) |

---

## 1. Where the current paper stands (honest diagnosis)

### 1.1 What is genuinely good — keep it
- **The framing instinct:** treating "does extra compute help?" as a property of the *deployment* (observability) rather than the *method* is a real and underexplored angle. Keep this DNA.
- **The gating result is statistically real:** gated TTC beats fixed best-of-8 (paired McNemar p ≈ 0.031) at roughly half the latency. This survives as the "compute" branch of the new framework.
- **The instrumented pipeline:** Isaac Lab → LeRobot → SmolVLA fine-tune, the occlusion injector, the swappable selectors, and the McNemar/Wilson/bootstrap reporting. This is reusable infrastructure and a credibility asset.
- **The failure-mode observation:** "under heavy occlusion the robot reaches the wrong box because the distinguishing color is the part the mask hides — no resampling fixes it, since the information is absent rather than uncertain." **This single sentence is the seed of the entire pivot.** It is currently buried in §5.4; it should become the thesis.

### 1.2 What makes it incremental / what is weak
- **Adaptive test-time compute for VLAs is crowded** (your own refs: MG-Select, RoboMonkey, RoVer, SCALE). A disagreement-gated budget is an engineering increment on top of that line.
- **The headline Q1 claim is not statistically significant** — by your own admission the per-condition Wilson intervals overlap (N=50). "Occlusion roughly doubles the scaling benefit" is a point estimate, not a result.
- **The "doubling" leans on a denominator effect:** absolute gains are +6 pts clean vs +9 pts occluded (≈1.5×). The 2× only appears in *relative* terms because the occluded baseline is lower.
- **Sim-only, one task, one real model.** TinyVLA is explicitly a wiring check. Generality is thin.
- **The numbers may be from a non-final run** ("refresh against the final run before camera-ready").

### 1.3 Why it does not fit HRI as written
HRI's scope is human–robot *interaction* — robotics plus HCI, human factors, psychology, and social/behavioral science. The current paper contains **no human, no interaction, no human-factors measurement.** As written it is a robot-learning / ML-systems paper and would most likely be desk-returned or rejected at HRI for being out of scope, regardless of execution quality. **The venue mismatch is a larger threat to acceptance than any methods weakness.**

---

## 2. The competitive landscape (why the *obvious* pivot is not enough)

The obvious pivot — "VLA detects its own uncertainty and asks a human for help" — is **already a live, somewhat crowded subarea.** If we ship that as the headline, an informed reviewer rejects it on novelty. We must position against and step beyond the following:

| Work | What it does | What it leaves open (our wedge) |
|---|---|---|
| **KnowNo** (Ren et al., CoRL 2023) | Conformal prediction over **discrete LLM-planner options**; robot asks for help when the prediction set has >1 option; guarantees on success while minimizing help. Ambiguity is mostly *instruction* ambiguity. | Binary (act/ask), discrete options not continuous VLA actions, no **compute** axis, no controlled **observability** variable, real-robot demos but **not a controlled human-subjects study**. |
| **INSIGHT** (Karli, Shangguan, Fitzgerald, 2025) | Learns **help triggers** from token-level uncertainty in a VLA (entropy, log-prob, Dirichlet aleatoric/epistemic), trains classifiers; "first systematic eval of uncertainty-based introspection in VLAs." | A CS/ML introspection eval — **no human study**, no compute-vs-query allocation, no observability-as-IV, no value-of-information/computation theory. This is the closest threat; engage it head-on. |
| **SCALE** (your ref [10], 2026 preprint) | Autonomous **self-uncertainty-conditioned** adaptive perception + execution, single forward pass, no verifier. | **Never queries a human** — stays autonomous. Our limit result is precisely the statement that autonomy cannot suffice in the information-absent regime. |
| **OneTwoVLA** | Proactive **human clarification** before acting. | No compute axis, no controlled observability study, no calibration theory. |
| **MG-Select / RoboMonkey / RoVer / self-consistency** (your refs) | Verifier or verifier-free **selection among samples**, fixed or verifier-gated budget. | All are the "compute" branch only; none gate on observability or include querying. We *subsume* them as one branch. |

**Conclusion:** novelty cannot be "asks for help." Novelty must be **(i) the act/compute/query trichotomy as one allocation problem, (ii) observability as the controlled independent variable, (iii) the limit result on test-time compute, and (iv) a real human-subjects study about trust/reliance.** Those four together are not covered by any single work above.

---

## 3. The reframed contribution

### 3.1 The one-sentence thesis (memorize this; make it true)
> Self-generated uncertainty tells a policy *how* to spend compute, but it cannot tell the policy *when compute is useless* — and the regime where compute is useless is exactly where external information (a human, or a new sensor) is required.

This generalizes beyond VLAs, beyond manipulation, beyond this task. It is the kind of clean, falsifiable statement that ages well.

### 3.2 From a binary to a trichotomy
At each query the controller chooses among:
- **act** — execute a single forward (K=1); cost ≈ 0 extra.
- **compute** — draw K samples and select (your existing best-of-N / twin-probe / selector machinery); cost ∝ K (latency).
- **query** — acquire external information: **ask the human** ("which box?" / "target is occluded, confirm"), or actively move the sensor; cost = latency + human workload + interruption.

The research question becomes: *as a function of visual observability and the costs of each action, when does each branch dominate?* — answered with a decision-theoretic value-of-computation (VoC) vs. value-of-information (VoI) account.

### 3.3 What is novel relative to the field
- Prior ask-for-help work is **binary** (act/ask) or has no human (act/think). We **unify all three**.
- We make **observability the manipulated variable**, not a fixed background condition.
- We give a **limit result** (§4.1) explaining *why* a third branch is necessary, not just useful.
- We run a **controlled human study** measuring trust/reliance, which the ML-side competitors do not.

---

## 4. The theoretical core (keep it minimal — HRI punishes decorative math)

Two small, honest results. The goal is **one page + one proposition + one calibration figure**, not a treatise. HRI reviewers are skeptical of gratuitous formalism; over-theorizing will hurt.

### 4.1 Result 1 — a limit on test-time compute (a sufficiency / data-processing argument)
**Setup.** Let latent state `z` determine the optimal action `a* = a*(z)`. The policy sees a masked observation `õ = m(o)`. Test-time scaling draws `K` samples `{Â^(k)} ~ π(·|õ)` and a selector `s` returns one.

**Claim (informal).** Conditioned on `õ`, the samples are generated using only `õ` and RNG, so they are conditionally independent of `z` given `õ`. By the data-processing inequality, **no statistic of the samples carries information about `z` beyond what `õ` already carries.** Therefore, if `a*` depends on a component of `z` that the mask removed from `õ`, **no selector over samples can recover it** — the value of computation for that component is zero. Compute can only resolve ambiguity that `π(·|õ)` actually represents *and* the selector can identify (reducible / epistemic-within-the-policy uncertainty), **not information that is absent from `õ`**.

**Honest caveat (state it, don't hide it).** This assumes the selector uses only the samples + `õ`. A selector that injects *outside* information — a learned prior over likely targets, temporal history, a verifier trained on extra data — can do better. But that outside information *is itself acquired information*, which proves the broader point: improvement in this regime requires information from outside `π(·|õ)`. This is the rigorous version of your "the information is absent rather than uncertain" note. It is nearly a tautology once stated correctly — and that is fine; clean "here is exactly when method X cannot help" results are precisely what get cited.

### 4.2 Result 2 — the calibration corollary (the over-trust trap; your HRI hook)
**Claim.** Self-agreement signals (twin-probe disagreement, sample variance/entropy, leave-one-out KL) measure the *dispersion* of `π(·|õ)`. Calibration concerns the *gap between confidence and correctness*. These are different quantities. A mask can collapse `π(·|õ)` onto a single confident **wrong** mode — the "confidently wrong" failure — so dispersion is low (high agreement) while error is high. Hence self-agreement is **not** a reliable proxy for correctness under masking; in the worst case it is **anti-correlated** with error precisely in the regime where querying is most valuable.

**Falsifiable predictions you can run on the JACO (this is the empirical heart):**
1. Calibration error (ECE) / reliability diagrams **degrade monotonically** with occlusion fraction.
2. The agreement→correctness mapping **decouples or inverts** under heavy occlusion.
3. A query rule built only on self-agreement **under-queries exactly the high-error masked cases.**

Prediction (2) — a single figure showing self-agreement decoupling from correctness as the camera is occluded — is a strong, fundamental result on its own and is **robot-only** (no humans needed to produce it), which de-risks the schedule.

### 4.3 The detector and the conformal-under-shift twist
The principled "am I in the useless-compute regime?" detector is **conformal prediction** (KnowNo's tool), which gives finite-sample coverage *if the calibration set is exchangeable with deployment*. **Occlusion is a deliberate covariate shift**, so vanilla split-conformal coverage **degrades** under exactly our manipulation. Options: mask-stratified / weighted conformal (coverage only approximate under shift), or accept the human as the backstop. **Foreground this** — measure coverage vs. occlusion fraction; the degradation is itself a finding and pre-empts the obvious reviewer objection. It also motivates the human: when the calibration guarantee fails, the team needs an external check, not more samples.

---

## 5. The human study (the heart of the HRI paper)

### 5.1 Platform & task
Real Kinova JACO Gen2, single overhead RGB camera, the SmolVLA policy reach–grasp–lift task. Apply occlusion to the **real** camera feed (and consider **physical** occluders too, for ecological validity — see Open Decisions).

### 5.2 Design — independent variables
- **IV1 — Observability:** clean → graded occlusion (reuse your `bottom_strip(ρ)` dial: ρ ∈ {0.15, 0.25, 0.35, 0.50}, plus center/random patches).
- **IV2 — Controller / interaction condition:**
  1. **Autonomy** (K=1, never asks).
  2. **Fixed compute** (best-of-N, never asks).
  3. **Compute-gated** (your current method — computes on ambiguous steps, never asks).
  4. **Compute-or-query allocator** (proposed — computes on reducible ambiguity, asks on irreducible).
  5. **Allocator + uncertainty-*type* transparency** (the robot signals *which* uncertainty it is in: "thinking…" vs "I can't see — which one?").

Condition 5 vs 4 isolates the HRI question of whether communicating *type* (not just magnitude) of uncertainty changes human behavior. Counterbalance order; consider within- vs between-subjects per the power analysis (Open Decisions).

### 5.3 Measures — dependent variables
- **Objective:** task success / throughput, time-to-completion, **query rate & timing**, **human intervention rate**, human idle/interruption time.
- **Subjective:** **trust** (validated scale — e.g., Jian et al. 2000, or the MDMT), **workload** (NASA-TLX), perceived predictability/transparency.
- **The key construct — appropriate reliance:** does the human intervene on the *confidently-wrong* cases (avoiding over-trust) while *not* over-intervening on correct autonomous steps (avoiding under-trust)? Measure both error types.

### 5.4 The headline HRI result to target
**The over-trust trap:** the robot is most dangerous to its human partner precisely when it is confidently wrong, because the human has no internal-signal reason to intervene. Target finding: *a controller that distinguishes "I should think" from "I must ask you" — and communicates which — improves appropriate reliance and reduces over-trust failures*, on top of improving success/trust/workload vs. the autonomous, fixed-compute, and ask-for-help baselines. That is a durable, measurable HRI contribution.

### 5.5 Baselines (must be in the paper)
KnowNo-style conformal ask-for-help (binary), SCALE-style autonomous adaptive compute, INSIGHT-style learned help trigger, and your own fixed/gated TTC. The story: prior work is binary on one axis; we unify the axes and measure the human consequences.

---

## 6. What carries over vs. what is new

**Carries over (reuse, don't rebuild):**
- SmolVLA backbone + fine-tuning recipe; flow-matching stochasticity as the sampling substrate.
- The occlusion injector (`partial_obs.py`) — now applied to the real feed.
- The selector/controller code (`ttc.py`), twin-probe machinery → becomes the "compute" branch.
- Wilson/McNemar/bootstrap reporting; paired-seed protocol.

**Net new (the actual work):**
- The act/compute/query allocator + the conformal escalation rule.
- The two theory statements (§4) + the calibration/coverage experiments (robot-only).
- The **human study**: protocol, IRB, interaction/transparency design, trust & workload instrumentation, analysis.
- Real-robot integration of the whole loop on the JACO.

---

## 7. Draft contribution statements (HRI framing)

- **C1 (conceptual).** We reframe inference-time decision-making for VLA policies as a three-way allocation among **acting, computing, and querying**, governed by visual observability, with a value-of-computation vs. value-of-information account of when each is optimal.
- **C2 (theory).** We show that self-generated uncertainty **cannot detect the regime where computation is useless** — when occlusion makes the optimal action unidentifiable from the current view — and characterize this as a **calibration failure** (agreement decouples from correctness), motivating an external information source.
- **C3 (method).** A calibrated **compute-or-query controller** for any stochastic VLA: extra forwards on reducible ambiguity, escalation to a human (or active sensing) on irreducible ambiguity, with a conformal detector whose coverage we characterize under occlusion-induced shift.
- **C4 (HRI empirical).** A controlled human-subjects study on a real Kinova JACO Gen2 showing that observability governs the value of each action, that communicating uncertainty **type** improves **appropriate reliance** and reduces over-trust failures, and that the allocator improves success, trust calibration, and workload over autonomous, fixed-compute, and ask-for-help baselines.

---

## 8. Related-work positioning (the differentiation paragraph to write)

One tight paragraph that says, in order: prior selection/verification work spends a fixed or verifier-gated compute budget but never decides to *stop computing and acquire information* (MG-Select, RoboMonkey, RoVer, self-consistency); adaptive-compute work gates on self-uncertainty but stays autonomous (SCALE); ask-for-help work escalates to a human but is **binary** (act/ask), uses discrete planner options or instruction ambiguity, and is not studied as a controlled human-subjects experiment under graded visual observability (KnowNo, OneTwoVLA, INSIGHT). **We unify acting, computing, and querying into one observability-governed allocation problem, prove why the query branch is necessary (not merely useful), and measure the human consequences.** — Engaging INSIGHT and KnowNo *explicitly* here is non-optional.

---

## 9. Risks, scope, and prioritization (honest)

- **Scope is the #1 risk.** Real-robot + comprehensive human study + new theory by September is a lot. The failure mode is **three things done adequately instead of one done excellently.** Prioritize ruthlessly: *the human study is the paper; the theory is one page; the hardware is scoped to what the study needs* (you are **not** also writing a sim-to-real generalization paper).
- **IRB is the critical-path long pole.** Approval can take 4–8+ weeks. **Start the IRB protocol now**, in parallel with everything else, or the September deadline is dead on arrival.
- **Crowding.** If reviewers see "VLA asks for help" without the trichotomy + observability-IV + human study + limit, it's a reject. Lead with the wedge, not the help-asking.
- **Conformal coverage degrades under shift.** Treat as a measured phenomenon and a motivation for the human backstop; do not claim guarantees you don't have.
- **The pivot de-risks the old weak claim.** The contribution no longer rests on "occlusion doubles the gain" (the non-significant Q1). If real-robot stochasticity doesn't reproduce that pattern, the paper still stands on the limit + calibration + human study. Good.
- **Study confounds:** learning/carryover effects (counterbalance; consider between-subjects), realism of digital masks on a real feed (consider physical occluders), Wizard-of-Oz vs. genuinely autonomous querying (decide and disclose), single task / single arm (acknowledge as a generality limit).
- **"Timeless."** Drop the word internally. Almost no accepted paper is timeless; chasing it produces over-reaching work. The achievable durable thing is the one clean claim above, well-supported. Make that sentence true.

---

## 10. Rough timeline to the deadline (~late September 2026)

> Aggressive but feasible *only if IRB starts immediately and scope is held.*

- **June (now):** Lock the thesis + the two theory statements. Write the related-work positioning paragraph. **Submit the IRB protocol.** Reproduce the existing sim results cleanly. Port the occlusion injector to the real JACO feed; get the policy running end-to-end on hardware.
- **July:** Real-robot loop solid. Run the **robot-only calibration experiments** (reliability diagrams / ECE vs occlusion; agreement-vs-correctness; conformal coverage vs occlusion) → produces the Result-2 figure and validates the limit. Pilot the human study (2–3 participants) to debug the protocol; run a **power analysis** from pilot effect sizes to set N.
- **August:** Run the **full human study**. Collect and clean data.
- **Early September:** Analysis, figures, writing. Draft complete ~2 weeks before the deadline.
- **Mid–late September:** Internal review, polish, submit (remember the abstract/author-list deadline ~1 week before the paper).
- **Fallback:** If the full study slips, an HRI **Late-Breaking Report** (later deadline, lower bar) using the calibration results + a pilot study is a real Plan B — but the full-paper September date is the target.

---

## 11. Open decisions (resolve early)

- **Within- vs between-subjects?** Within gives statistical power but risks learning/carryover — counterbalance carefully, or go between-subjects if the budget allows.
- **Digital masks vs physical occluders** on the real feed? (Control vs ecological validity — possibly both.)
- **Trust instrument:** Jian et al. (2000) automation-trust scale vs. the Multi-Dimensional Measure of Trust (MDMT)?
- **"Query" = human only, or also active sensing** (move the camera)? Active sensing is a clean *second* external information source — a strong condition or a tidy "future work."
- **Querying mechanism:** genuinely autonomous escalation vs. Wizard-of-Oz for the study? Decide and disclose.
- **N (participants):** set by a power analysis on pilot effect sizes (HRI studies are commonly ~16–40).
- **Backbone:** keep SmolVLA (open, flow-matching, already integrated). Don't switch.

---

## 12. Key references to engage

> Confirm exact metadata before citing; arXiv IDs / venues below are starting points.

**Directly competing / parent work (must cite and differentiate):**
- Ren et al., *Robots That Ask For Help: Uncertainty Alignment for LLM Planners* (KnowNo), CoRL 2023 — arXiv:2307.01928.
- Karli, Shangguan, Fitzgerald, *INSIGHT: Inference-time Sequence Introspection for Generating Help Triggers in VLAs*, 2025 — arXiv:2510.01389.
- *SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for VLAs* (your ref [10]) — arXiv:2602.04208.
- *OneTwoVLA* (active human clarification) — see the VLA survey below for pointer.
- Jang et al., *Verifier-free Test-time Sampling for VLAs* (MG-Select, your ref [2]); RoboMonkey; RoVer; Wang et al., *Self-Consistency* (your ref [1]).

**Theory you're building on:**
- Howard, *Information Value Theory* (1966) — value of information.
- Russell & Wefald, *Principles of Metareasoning* (1991) — value of computation / bounded rationality.
- Kendall & Gal, *What Uncertainties Do We Need…* (2017) — aleatoric vs. epistemic.
- Vovk/Gammerman/Shafer; Angelopoulos & Bates (2021 tutorial) — conformal prediction (and weighted/shift-robust variants).
- Bajcsy, *Active Perception* (1988) — the active-sensing branch.

**HRI methods / instruments:**
- Hart & Staveland, *NASA-TLX* (1988) — workload.
- Jian, Bisantz & Drury (2000) — trust in automation scale (or MDMT, Ullman & Malle).
- Adjustable/sliding autonomy & mixed-initiative HRI literature (Goodrich & Schultz survey; Sheridan levels of automation) — for the autonomy-handoff framing.

**Landscape / survey:**
- *An Anatomy of Vision-Language-Action Models* (2025) — arXiv:2512.11362 — for situating VLAs, active clarification, and adaptive "how much to think" directions.

---

*Bottom line: keep the pipeline and the gating result, throw out the "test-time compute doubles under occlusion" headline, and rebuild around one claim — compute cannot manufacture missing information, so a calibrated team must know when to think versus when to ask — proven minimally and demonstrated on a real robot with real people.*

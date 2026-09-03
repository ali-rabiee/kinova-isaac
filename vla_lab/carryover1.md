# Implementation Brief — Carryover-Aware VLA Project

**Audience:** an implementing agent with write access to the `vla_lab` package, the Isaac apparatus, and the paper source.
**Status of the project:** early-stage. The current manuscript is an internal full-disclosure report, not a submission draft. A human study with IRB approval is planned and not yet run.
**Purpose of this brief:** close the statistical holes in the current results, run the experiments that decide whether the paper's architectural and mechanism claims survive, retire one proposed component the evidence argues against, and prepare the instrument for human participants.

---

## 0. Ground rules

Read these before starting. They constrain every task below.

1. **Do not weaken the reporting standards.** The test–retest floor, the pre-committed evidence standard in §3.8, the refusal to interpret differences inside the floor, and the defect log in §5.3 are the strongest features of this work. Every change below *raises* the standard; none relaxes it.
2. **Do not delete negative results.** If a new experiment overturns a claim, replace the claim, keep the record of what it replaced, and say what changed. That pattern is already established in §6.3 and should be reused.
3. **Locate before you edit.** File paths in this brief are inferred from the manuscript's Appendix E and may not match the repository. Find the real modules first; do not create parallel implementations.
4. **Every number in the paper must be generated, never transcribed.** Table 4 already follows this (generated from run manifests). Extend the same discipline to every new table and figure produced here.
5. **Manifest everything.** Any new sweep writes its own manifest with seed, config hash, git SHA, and the contract hash, in the same format the existing runs use.
6. **Regressions are failures.** The existing offline test suite must still pass at the end. Add tests for each new invariant.

---

## 1. P0 — Statistical holes that must close first

These are the changes most likely to alter what the paper can claim. Do them before any new modelling work.

### P0-1. Seed variance for every model-level table

**Problem.** Tables 4, 5 and 7 report single-run point estimates with no run-to-run variability. The paper's entire epistemic posture is "is this difference inside the noise floor?" — applied rigorously to the supervisor study and not at all to the model comparisons. A Brier-gain difference of +0.135 vs +0.113 (Table 4) and an objective-ablation delta of +0.0314 (Table 5) are currently uninterpretable.

**Implement.**
- Add a `--seeds` argument to the architecture sweep and the objective sweep (`sup_models.sh`, `sup_ablate.sh` per Appendix E). Default to **5 seeds**; accept 3 as a minimum if wall-clock forces it.
- Seed must control: model init, data shuffling, the dialogue-generation RNG, and the supervisor draw for the training cohort. Confirm all four are actually seeded — check for any global RNG use that escapes the seed.
- For each cell, report **mean ± seed SD** and a **seed-level noise floor** for both headline metrics: the Brier de-biasing gain, and the `gap~κ` rank correlation. Compute the floor the same way the test–retest floor is computed — as the spread attributable to nothing but re-running.
- Add the seed floor as a horizontal band to Figure 10 and as a footnote row in Tables 4 and 5, mirroring the `test–retest floor` row in Table 3.

**Expected outcome.** Several orderings will not survive. In particular, verify whether `none < token < film` on the from-scratch and SmolVLA backbones is stable across seeds, and whether the Qwen row's flat gain (spread 0.005) is inside the seed floor — if so, the paper should state that explicitly rather than describing it as saturation.

**Paper changes.** Rewrite §6.6's mechanism-ordering paragraph to report only the orderings that clear the seed floor. Add the seed floor to §6.7's opening. If the objective ablation deltas fall inside the floor, say so and drop the causal language about which terms are load-bearing.

**Acceptance criteria.** No table cell in the paper reports a model-level metric without an accompanying dispersion estimate. No ordering claim survives that is not larger than its own floor.

---

### P0-2. Uncertainty in the task-value model, propagated

**Problem.** `m* = 8.5 cm` and `w = 3.12 cm` are treated as constants throughout. They are estimates from 216 rollouts at 12 reps per gap. Every scene coordinate `c = (m* − m)/w`, every MAE×, every band weight, and every regret number inherits their sampling error, and none of it is reported. This is the same class of error §6.3 warns about — an unexamined physics assumption deciding a contrast — applied to the measured physics rather than the assumed one.

**Implement.**
- Bootstrap the 216 rollouts (resample within (strategy, gap) cells, B ≥ 2000). For each replicate, refit the success and duration curves, recompute `V_A`, `V_B`, and derive `m*` and `w`.
- Report the bootstrap distribution of `m*` and `w` with percentile CIs. Write both to `physics_report.json`.
- Add a `--physics-quantile {lower, point, upper}` option to the study runner that rebuilds the scene grid and the band weights under the corresponding physics draw.
- Re-run the primary study (§6.3) under the lower and upper bounds of `w`. Report the memoryless contrast and the always-counter-vs-washout contrast under all three.

**Expected outcome.** The memoryless contrast should be robust (it is ~4× the floor). The remedy contrasts are already inside the floor and should stay there. The value of this task is that it converts §6.3's finding from an anecdote into a bounded statement: *here is how much of the headline is decided by physics estimation error*.

**Paper changes.** Add a paragraph to §5.4 reporting the CIs on `m*` and `w`. Add a row or panel to §6.3 showing the primary contrast under the physics CI bounds. Amend §9's limitation bullet on the scene physics to cite the measured interval rather than gesturing at "a different controller would move the crossover."

**Acceptance criteria.** `physics_report.json` contains bootstrap CIs for `m*` and `w`. The primary table is reproducible under all three physics quantiles by a single flag.

---

### P0-3. Replace the two-parameter logistic fit

**Problem.** §5.4 admits a worst-cell disagreement of 0.42 between fit and data (fit puts clear-first at 0.59 against a measured 2/12 at a 0 cm gap). A coordinate system is being defined through a curve that is wrong by that much in the tails. The in-band disagreement of 0.205 is also not small.

**Implement.**
- Refit success curves with a **lapse/floor-parameterised psychometric** (add asymptote parameters at both ends) or an **isotonic regression with bootstrap bands**. Fit both; report the comparison.
- Collect **additional rollouts at the tight end**, where the curve is steepest and the current fit is worst: gaps 0–4 cm, at least 24 reps per (strategy, gap) rather than 12.
- Re-derive `m*` and `w` from the improved fit. Re-run P0-2's bootstrap on top of it.
- Report the worst in-band and out-of-band residual for the new fit alongside the old one.

**Paper changes.** Update Figure 5, the fit description in §5.4, and every downstream number. Keep the old fit's residuals in the text as the "what we had before" comparison — this is a genuine improvement worth documenting, not an embarrassment to hide.

**Acceptance criteria.** Worst in-band fit residual below 0.10. Tail behaviour no longer contradicted by the data at any measured gap.

---

### P0-4. Remove or restate every underpowered correlation

**Problem.** §6.12 and the abstract report Spearman `ρ = −0.46` (n = 7) and `ρ = −0.14` (n = 7) as findings. With seven points and one extreme outlier, neither is analysable. The §6.12 prose already hedges; the abstract does not.

**Implement.**
- Search the manuscript and analysis code for every correlation, rank correlation, or trend line computed on fewer than ~15 units. Enumerate them in a checklist.
- For each: either (a) increase n so it is analysable, or (b) replace the statistic with a descriptive statement of the raw pattern.
- For §6.12 specifically, option (a) is available and preferable: the closed-loop evaluation has 7 cells because only two backbones were run. **Add the third backbone** (see P1-1) and, if feasible, additional injection modes, to take the cell count to 10–12. Even then, do not report a correlation coefficient — report the two facts that are actually supported: the best offline cell on one backbone is the worst deployed channel, and moderate abstention is not associated with failure while the single cell above 11% is.
- Add a lint rule or analysis-time assertion that refuses to emit a correlation coefficient below a configurable minimum n without an explicit override flag.

**Paper changes.** Delete `ρ = −0.46` from the abstract. Restate §6.12's third bullet without coefficients. Same for the in-band-abstention correlation.

**Acceptance criteria.** No correlation coefficient appears anywhere in the paper computed on n < 15.

---

## 2. P1 — Experiments that decide whether the claims hold

### P1-1. Run Qwen2.5-VL-3B and separate the scale confound

**Problem.** §9 concedes it: parameter count, pretraining corpus, and adaptation method all move together across the roster, so "scale" in §6.6 is shorthand for a bundle. The monotone rise in context-blind residue tracking (−0.006 → +0.151 → +0.185) is the paper's most interesting incidental finding and it is currently uninterpretable.

**Implement.**
- Run the Qwen2.5-VL-3B cells. They are described as implemented and omitted; confirm they still run after any collator changes, then execute the full injection grid at the P0-1 seed count.
- Qwen2-VL-2B vs Qwen2.5-VL-3B under **identical LoRA rank, target modules, learning rate, and schedule** is the only clean within-family scale contrast available. Verify identity of all adaptation settings and assert it in the manifest.
- Report the context-blind `gap~κ` value for both, with seed dispersion. If the 2B→3B step is inside the seed floor, the monotone-with-scale claim does not survive and must be restated as a between-family observation only.

**Paper changes.** Add the Qwen2.5-VL-3B rows to Table 4. Rewrite the "Scale changes what context-blind means" paragraph in §6.6 around the within-family contrast, and demote the cross-family comparison to a caveat.

---

### P1-2. Test whether `gap~κ` survives distribution shift

**Problem.** §9 flags the risk directly: 19 scenes, one camera, one table, one pair of cubes. The context-blind residue-tracking values at 450M and 2B may be reading dataset regularities rather than anything about carryover. This decides whether Table 4 measures an ability or an artifact. It is the single highest-value experiment in this brief.

**Implement.**
- Generate a **held-out scene set** that the models never train on: different object colours and sizes, an altered table texture, an added or removed distractor, and — most importantly — a **second camera pose** (a modest translation and rotation off the collection contract, still overhead).
- Evaluate every trained checkpoint on the held-out set. Report Brier gain and `gap~κ` for each cell on both the matched and shifted distributions.
- Report the **degradation** per cell, not just the shifted value. Compare context-blind cells against context-injected cells: if context-blind tracking collapses under shift while context-injected tracking holds, the architectural claim strengthens considerably. If both collapse, Table 4 is about this dataset and must say so.

**Paper changes.** New subsection under §6.6, or a new column pair in Table 4. Rewrite §9's third limitation bullet to report the measured shift result rather than speculating about it.

**Acceptance criteria.** Every cell in Table 4 has a matched-distribution and a shifted-distribution value.

---

### P1-3. Turn the λ-identifiability failure into a design result

**Problem.** §6.2 reports that λ is recovered in 44% of supervisors against 86% for β, and stops there. The machinery to do better already exists in §4.8 — the Fisher-weighted objective over the estimator's own coordinates. Reporting "λ is unidentifiable" is a weaker result than "λ is unidentifiable under the natural schedule; here is the schedule that identifies it, and here is what it costs."

**Implement.**
- Derive and implement the **Fisher information for λ** under Eqs. (3)–(4). λ enters through the elapsed-time exponent, so information about it comes from observing the residue at *many distinct* elapsed times with adequate leverage on the response.
- Implement an **identification-optimal probe schedule**: log-spaced elapsed-time gaps between demonstration and probe, with scene coordinates chosen for maximum residue leverage on the Bernoulli response (near the crossover, where the logit is most sensitive to an offset).
- Add this as a new condition, e.g. **B6 identification-first**, running under the same matched budget.
- Report: (a) the λ-identification rate under B6 versus the current 44%; (b) the **cost** — how much worse B6's estimand MAE× is, since slots spent identifying λ are slots not spent estimating π*; (c) whether a two-phase policy (identify λ early, then exploit it) beats the fixed washout in the high-compliance tercile where §6.5 shows the current adaptive schedule failing.
- Also run the diagnostic **prospectively**: at each slot, log whether the posterior over λ has moved off its prior. A system that can tell it is about to personalise on its prior is the practically useful contribution here — expose it as a first-class output.

**Paper changes.** This becomes a new results subsection and probably a co-headline finding. §6.5's uncomfortable ordering and §D.5's recommendation both change if B6 works.

**Acceptance criteria.** B6 implemented, run at the same N as the primary study, and reported with the identification rate and its cost in estimand precision.

---

### P1-4. Generalise the value-model sensitivity into a procedure

**Problem.** §6.3's "the finding that did not survive re-measuring the physics" is currently one anecdote: we assumed a curve, we measured it, a significance test flipped. That is a warning, not a tool. Nobody can act on it without measuring their own curve, which is exactly the expensive thing most simulated studies won't do.

**Implement.**
- Add a sweep over the **assumed transition width `w`** (and, secondarily, the crossover location `m*`), holding policies, supervisors, and seeds fixed.
- For each value of `w`, recompute the primary contrasts. Produce a **flip plot**: contrast estimate and CI as a function of assumed `w`, with the measured value and its bootstrap CI (from P0-2) marked on the axis.
- Identify and report the **flip point** — the value of `w` at which the always-counter-vs-washout interval stops excluding zero.
- Package this as a reusable diagnostic in the analysis tooling: given a simulated study and a parameterised difficulty curve, report the range of curve parameters over which each headline conclusion holds.

**Paper changes.** Replace the §6.3 narrative paragraph with the flip plot plus a short procedural recommendation. Elevate this to a named methodological contribution — it is the most transferable result in the project and does not require human data to be valid.

---

## 3. P2 — Design and framing changes

### P2-1. Retire the adaptive scheduler as a proposed component

Your own §6.5 and §D.5 argue against it: the full policy is worse than a plain fixed washout in the high-compliance tercile, and its per-supervisor disadvantage trends the wrong way with compliance strength.

**Implement.**
- Restructure the policy family so the **default recommended configuration is corrected-estimator + counter-proposal, with adaptive scheduling off**. Keep the component behind a flag.
- Add an explicit config `policy_recommended` and assert in a test that it does not include adaptive scheduling.
- Keep the scheduler in the results as a **proposed-and-rejected** mechanism, reported alongside B6 (P1-3), which is the version of the scheduling idea that has an identification story behind it.

**Paper framing.** "We proposed three mechanisms. One is load-bearing. One is roughly neutral. One is actively harmful on exactly the population it was designed for, and the identifiability analysis predicted this before we measured it." That is a stronger contribution than three mechanisms of undetermined value.

---

### P2-2. Restate the ask-gate result as an a priori identifiability argument

**Problem.** §6.6 reports the learned ask gate failing across ten cells, then observes on reflection that the gate is being asked to solve the estimation problem in order to decide whether to gather evidence about it. That reasoning does not require ten runs — it is a statement about what the target is a function of, and it should be made *before* the experiment.

**Implement.**
- Write the argument formally: the gate's target (the value of counter-proposing) is a functional of `π*_p`, the very quantity under estimation, so it is not identifiable from data in which `π*_p` is unknown.
- Restructure §6.6's presentation: state the prediction, then report the ten cells as confirmation, then note that this is what motivates the hybrid architecture on principle rather than on results.
- Add a test asserting that the deployed system takes the ask decision from the belief module, not from the learned gate.

---

### P2-3. Promote the dose-tracking result

§6.11's finding that B5's counter-proposal rate tracks the inferred dose without being told it (0.0 → 0.4 → 1.0 → 2.6 → 8.9) is the single clearest demonstration that the mechanism works end to end, and it is currently in a sensitivity subsection. Move it into the main results and give it its own figure. Add a control: verify that the rate does *not* track a placebo parameter the belief module has no access to, which rules out an incidental correlation with session length or scene sequence.

---

## 4. P3 — Preparing the instrument for human participants

### P3-1. Collect a phrase corpus before running the study

**This is the highest-return, lowest-cost de-risking action available.** The lexical grounder and the generative supervisor's phrase set are both tuned on scripted language. §7.5 already makes grounding rate on real speech a pilot go/no-go. Do not discover a grounding failure at participant 12.

**Implement.**
- Recruit ~10 people. Show 5–8 rendered scene images spanning the coordinate range. No robot, no session, no interaction budget — just "what should the robot do here, in your own words." Free text or audio transcript.
- Measure the current grounder's resolution rate on that corpus. Enumerate every failure mode: hedges, conditionals, multi-clause answers, answers that name neither strategy, answers that specify a strategy the design does not implement.
- **Rebuild both** the grounder's phrase inventory and the generative supervisor's utterance sampler from the empirical distribution, including the empirical hedge rate rather than a chosen one.
- Re-run the full simulated study under the rebuilt supervisor. Report whether any conclusion changes.

**Acceptance criteria.** Grounding rate on the real corpus meets the pre-registered 0.85 threshold, or the threshold and the grounder are revised before recruitment with the revision documented.

---

### P3-2. Resolve the identification-versus-deployment tradeoff explicitly

The simulation runs both regimes because it can instantiate the same supervisor twice. A human cannot be run twice. Decide before recruitment whether the human protocol is one-sided (studies the deployment phenomenon, poor θ identification) or alternating (identifies θ, cancels the aggregate bias §3.6 says makes the scheduling question vacuous), or whether session length permits a within-participant split. Document the decision and its consequence for which hypotheses the study can address.

### P3-3. Consider restructuring the human study around H1/H2

Your own power analysis concludes the remedy comparison is unpowerable at any recruitable N. Consider making the primary human contribution **"does compliance carryover exist in supervisory control, how large is it, does it decay, and does it vary between people"** — a clean study the field needs before any remedy is justified — with H3 secondary and model-based. This is a framing decision for the authors, not a task; flag it and let them decide.

---

## 5. P4 — Errata and consistency

Fix all of these. The manuscript's credibility rests on being the kind of work that catches silent defects, which makes each of these disproportionately damaging.

| # | Location | Problem | Fix |
|---|---|---|---|
| 1 | §5.3 heading and body | Titled "eleven defects"; body says "the first six / the next three / the last two" (= 11); enumerates (i)–(ix), then **(xi)**, then **(x)** out of order; summary paragraph says "seven of the **ten**" | Settle on a count, renumber sequentially, fix the ordering, make every internal reference agree |
| 2 | Reproducibility Statement vs Appendix E | "235 tests" vs "234 tests" | Generate the count from the suite at build time; never hand-write it |
| 3 | Reference [62] | Cited as the source for Qwen2-VL-2B and Qwen2.5-VL-3B, but is a different 2026 "Qwen-VLA" paper | Cite the actual Qwen2-VL / Qwen2.5-VL model papers; keep [62] only if the VLA paper is genuinely used |
| 4 | Abstract | Claims "across three backbones" but Table 4 reports three rows while the roster lists four | Reconcile after P1-1 adds the fourth |
| 5 | §6.11 / Table 6 | Reports "MAE at the crossover" while Table 3 reports "crossover-weighted MAE" — verify these are the same quantity | Unify the metric name and definition, or state the difference |
| 6 | Throughout | Verify every cross-reference (§6.3 → §5.4, §6.5 → §6.2, etc.) resolves to the section that contains the claim | Automated cross-reference check |

Add a **numbers audit** to the build: every numeric literal in the manuscript source should be traceable to a generated artifact, and the build should fail on any that is not. This is the natural extension of the discipline already applied to Table 4.

---

## 6. Claims that must be softened

Independent of any new experiment, these overclaim what the current evidence supports.

1. **§6.3: "This is the result that establishes the problem is real."** It does not. It establishes that the estimator can invert a contamination process the authors wrote down. Restate as: *this establishes that the estimator recovers a known injected bias, and that the penalty for ignoring it scales with its magnitude — which is a necessary validation, not evidence about human supervisors.*
2. **§6.5's dose–response framing.** "The cleanest evidence in the paper that the effect is the one the model posits" — it is evidence that the estimator's error scales with the injected parameter, which is a property of the simulator. Keep the result, restate the interpretation.
3. **Abstract, finding (1).** Same correction: the memoryless contrast is a validation result, and the abstract should say so in the same sentence that reports it.
4. **§6.12's ordering claims.** After P0-4, restate without correlation coefficients.

The §3.8 and §5.7 discussions already make these concessions correctly. The problem is that the headline sentences do not carry them. Make the concession travel with the claim rather than living three sections away.

---

## 7. Deliverables checklist

**Code**
- [ ] `--seeds` on the architecture and objective sweeps; seed-level floor computed and reported
- [ ] Physics bootstrap; `m*` and `w` CIs in `physics_report.json`; `--physics-quantile` on the study runner
- [ ] Refitted value model with lapse/isotonic fit; additional tight-end rollouts collected
- [ ] Qwen2.5-VL-3B cells running with adaptation settings asserted identical to the 2B cells
- [ ] Held-out scene set and second camera pose; shifted-distribution evaluation for every checkpoint
- [ ] Fisher information for λ; identification-optimal schedule; condition B6
- [ ] Assumed-`w` sweep and flip-plot tooling, packaged as a reusable diagnostic
- [ ] `policy_recommended` config with adaptive scheduling off, plus a test asserting it
- [ ] Prospective λ-identification diagnostic exposed as a first-class runtime output
- [ ] Minimum-n guard on correlation reporting
- [ ] Build-time numbers audit and cross-reference check

**Data**
- [ ] Phrase corpus from ~10 people; rebuilt grounder inventory and supervisor utterance sampler
- [ ] Full simulated study re-run under the rebuilt supervisor

**Results**
- [ ] Tables 3–7 regenerated under the refitted physics with dispersion estimates throughout
- [ ] New: physics-CI sensitivity panel; flip plot; shifted-distribution columns; B6 results; dose-tracking figure with placebo control

**Paper**
- [ ] Claims in §6.3, §6.5, §6.12 and the abstract softened per §6 above
- [ ] §6.6 mechanism ordering restricted to differences clearing the seed floor
- [ ] Ask-gate result restated as an a priori argument confirmed by experiment
- [ ] Adaptive scheduler reframed as proposed-and-rejected, with B6 as its successor
- [ ] All errata in §5 fixed
- [ ] §9 limitations updated to cite measured intervals rather than speculation

---

## 8. Suggested order of work

1. P4 errata and the numbers audit — cheap, and they protect everything downstream.
2. P0-3 then P0-2 — the value model is upstream of every other number, so fix and bound it before regenerating anything.
3. P0-1 and P0-4 — establishes what the model-level tables can actually claim.
4. P1-2 — the distribution-shift test decides whether the architecture comparison means anything; run it before investing further in that comparison.
5. P1-1 — completes the roster and the closed-loop cell count.
6. P1-3 and P1-4 — the two results most likely to become co-headlines.
7. P2 reframing, once the above determines what the evidence supports.
8. P3-1 phrase corpus, in parallel with anything above; it gates the human study and does not depend on the rest.

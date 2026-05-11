# CoRL 2026 — Reviewer Comments

**Paper:** *When Test-Time Compute Helps Most: Observability-Gated Scaling for Fine-Tuned SmolVLA Under Single-Camera Partial Observability*

**Reviewer:** R2 (simulated peer review — written to be used as an internal pre-submission critique)

**Recommendation:** **Borderline reject in current form (5/10)** — the question is good, the experimental protocol is clean, and the proposed gate is sensible and practical, but (i) the central empirical claim ("observability governs the return on test-time compute") rests on a *single* clean-vs-occluded contrast at *one* occlusion setting with *N=50* episodes and overlapping confidence intervals; (ii) the experiment that would actually establish the thesis — the graded-occlusion sweep — is described as already implemented in the released code but is *not run*; (iii) the work is simulation-only on a single task family, which is a hard sell at CoRL specifically; and (iv) the most relevant adaptive-compute baseline (SCALE) and a faithful MG-Select are absent. None of these are fatal: every one has a well-scoped fix, and if executed I would expect this to become a clear accept (≥6).

---

## Scores

| Axis | Score | Rationale |
|---|---|---|
| **Soundness** | 2 / 4 | The paired-episode protocol is genuinely sound and well-described. But the headline finding is under-powered (two points, N=50, overlapping CIs), one central result (gating *beating* fixed best-of-N on accuracy by +12–13 points) is large and under-explained, and several quantitative entries are flagged in-text as provisional. |
| **Presentation** | 3 / 4 | Well-organized, clearly written, honest limitations section, good signposting. Loses a point for a dense Figure 1, an over-long title, and a results section that mixes a "wiring-check" backbone into the main table. |
| **Contribution** | 2 / 4 | The deployment-vs-method reframing of "does TTC help?" is a nice angle and the released testbed is useful. But the empirical contribution is narrow (one task, one sim, one backbone, two observability points) and the method (twin-forward disagreement gate) is a natural baseline-grade idea rather than a deep one. |
| **Overall rating** | **5 / 10** | Marginally below the acceptance threshold. Promising; not yet convincing. |
| **Confidence** | 4 / 5 | Familiar with VLAs, test-time scaling for embodied agents, and Isaac Lab; have not run this specific codebase. |

---

## Summary of the paper

The paper studies *when* test-time scaling (best-of-N sampling + selection) pays off for VLA policies, and argues the governing variable is *visual observability*. It fixes a Kinova reach-and-lift task in Isaac Lab, fine-tunes SmolVLA on demonstrations exported to LeRobot format, and compares clean overhead RGB against a synthetic 35% bottom-strip occlusion on *paired* episodes (same seeds, only the mask changes). It reports that occlusion roughly doubles the *relative* success gain of fixed best-of-8 over a non-scaling policy (≈+8% → ≈+17%). It then proposes "observability-gated test-time scaling": a twin-forward disagreement probe that triggers up to K=8 sampling only on steps it flags as ambiguous, with a swappable selector. The gated controller is reported to reach 92% (clean) / 74% (occluded) success — above fixed best-of-8 (80–82% / 61–63%) — at an effective K̄≈3.4 and roughly half the per-query latency. The paper releases the pipeline (`vla_lab`).

---

## Strengths

1. **The research question is well-chosen and under-served.** "Does extra inference compute help?" is almost always reported as a single aggregate delta. Reframing it as a property of the *deployment* and asking *which* deployments need it is a genuinely useful contribution of perspective, independent of the specific numbers.
2. **The paired-episode protocol is methodologically sound.** Holding physics, object spawns, domain-randomization seeds, and the language command fixed and varying *only* the camera mask is exactly the right way to isolate the observability axis, and it's clearly described (§3, §4.4). Reporting Wilson intervals and paired McNemar on aligned success bits is appropriate. This is the paper's strongest methodological asset and should be foregrounded even more.
3. **The gate is simple and practical.** A twin-forward disagreement probe that needs no verifier network, no token-native distributions, and no retraining is the kind of thing a practitioner could actually deploy. The argument that it (a) tracks occlusion-induced ambiguity and (b) shields easy steps from selector variance is intuitive.
4. **Honesty about limitations.** The Discussion §6 is candid about sim-only, two observability points, the proxy selector, scripted demos, and the external-camera-only regime. CoRL reviewers genuinely value this, and it is well done here.
5. **Clear writing and structure.** Sections flow, experimental questions are stated up front (Q1–Q4) and answered in order, tables and figures are interpreted rather than dumped, and the contributions are concrete and falsifiable.
6. **Released testbed.** If the artifact is as instrumented as claimed, it lowers the bar for follow-up work on the observability–compute relationship.

---

## Major weaknesses

### W1. The central claim is under-powered, and the decisive experiment is missing.
The thesis is "observability governs the return on test-time compute." The evidence is *one* contrast: clean vs. a single 35% bottom-strip occlusion, N=50 each. The paper itself notes (§5.1) that the per-condition Wilson intervals overlap and "roughly doubles" is "a point-estimate pattern, not a per-condition significance claim." That candor is appreciated, but it also concedes that the headline result is not, as stated, established. Worse: Appendix F advertises that the released testbed already supports graded `bottom_strip(ρ)` at ρ∈{0.15, 0.25, 0.35, 0.50}, `center_box`, and `random_patch` — i.e., the experiment that would turn the point estimate into a curve and substantiate "governing variable" *exists in the code but was not run for the paper*. This is the single biggest gap. A reviewer will reasonably ask: why submit before running the experiment your own paper says is the one that matters?
- **Fix:** Run the graded sweep (≥4 occlusion fractions × clean), each at the largest N you can afford (ideally ≥150–200), and plot relative-TTC-gain vs. occlusion fraction. A monotone trend with non-overlapping or near-non-overlapping intervals at the endpoints would change my assessment substantially. Add `center_box`/`random_patch` as a robustness check that the effect isn't specific to "bottom strip."

### W2. Several quantitative entries are flagged as provisional.
§5 contains the parenthetical "*Numeric entries are produced by the released evaluator from `results_*.json`; refresh against the final run before camera-ready.*" In a submission this reads as "these numbers may move." For the actual submission, either the experiments are complete (remove the disclaimer, report final numbers, fix the seed count) or they are not (the paper isn't ready). Reviewers will also notice that several cross-table numbers are *almost* but not exactly aligned (e.g., the "Boxes ID" column of Table 7 sits ~1–2 points above the "Clean overhead RGB" column of Table 6 for the same configurations) — a careful reviewer flags this as a sign the numbers were assembled rather than dumped from one canonical run.
- **Fix:** Produce all tables/figures from one frozen evaluation run, embed the commit hash, and delete the "refresh later" note. Make Table 6 ("clean") and the largest-budget row of Table 8 identical (they should be the same config) and reconcile Table 7's scene split with Table 6's aggregate.

### W3. Simulation only, single task — a hard sell at CoRL.
Everything is a simulated Kinova on one task family (`reach_to_grasp_VLA`) with two scenes. The `real_robot/` harness is "scaffolded, not yet run." CoRL specifically rewards hardware validation; a sim-only paper needs to be exceptional elsewhere, and the empirical breadth here is not. The motivation (a real fulfillment cell / lab bench with one overhead camera) is exactly the setting the paper does *not* test.
- **Fix (in priority order):** (a) Even a *small* real-robot demonstration — one Kinova, one overhead camera, clean vs. a physical partial occlusion, N≈20–30 — would convert a major objection into a strength; the prediction "the effect shrinks with a wrist camera" is also cheap to test on hardware and would be a striking confirmation. (b) Failing that, add a *second* sim task (different object set / a place task / a different arm) to show the observability effect is not task-specific. (c) If neither is feasible, the paper should be reframed as a *simulation study + released benchmark* and submit to a venue/track where that framing fits, and the title/intro toned down accordingly.

### W4. "Gating beats fixed best-of-N on accuracy" is a large effect that is asserted, not demonstrated.
Table 6: clean K=1 = 74, clean best-of-8 = 80, clean gated = **92**. So gating beats *fixed best-of-8 by +12 points on the clean split* and beats K=1 by +18. If the gate's job is "K=1 on easy steps, resample only on hard ones," then on a mostly-clean split the gated policy should behave close to K=1 (≈74) plus whatever the hard-step resampling buys — getting to 92 implies either (i) a large fraction of *clean* steps are flagged ambiguous *and* the selector helps a lot there, or (ii) the deterministic ξ=0 forward used by the gate is somehow much better than what best-of-8's consensus returns. The paper's explanation ("shielding easy steps from selector variance") would mean consensus-over-8 is *actively worse than a single deterministic forward* on easy steps — which, if true, is itself an important finding about the consensus selector and is more naturally fixed by *including the ξ=0 sample as a candidate in best-of-N*. As written, the mechanism is plausible hand-waving without the supporting measurement.
- **Fix:** (a) Report K̄ *separately* for the clean and occluded splits, not just the pooled 3.4. (b) Break success down by gate decision: success rate on "gate said easy" steps under {K=1, best-of-8} and on "gate said hard" steps under {K=1, best-of-8, gated}. (c) Add the ablation "best-of-8 *including* the deterministic ξ=0 forward as one of the candidates" — if that closes most of the gap, the contribution is the candidate set, not the gate; if it doesn't, you have a real and interesting result, but you need to show it. (d) Add a "fixed best-of-K only on the steps the gate flags, K=1 elsewhere, *without* the twin-probe" condition to isolate budget-allocation from any twin-probe-specific effect.

### W5. No sensitivity analysis for the gate (τ, ε) — unusual for an adaptive-compute method.
The gate has a threshold τ and a probe scale ε. The paper reports results at one (τ, ε) and one resulting K̄≈3.4. Adaptive-compute papers (SCALE, etc.) live or die on the accuracy–compute frontier as the budget knob varies. Without a τ-sweep, the reader cannot tell whether K̄≈3.4 is a sweet spot or a cherry-pick, nor compare the gated frontier against the fixed-K frontier (K∈{1,2,4,8}) on equal footing.
- **Fix:** Add a figure: success vs. K̄ (or vs. median latency) with the *gated* curve (sweeping τ) overlaid on the *fixed-K* curve (sweeping K), per condition. This is the single most informative plot you could add and it directly answers Q2 properly. Also report ε sensitivity (even a small table).

### W6. The MG-Select baseline is a weakened reimplementation, and the headline leans on it.
The abstract's claim is "above fixed best-of-eight (80–82% / 61–63%; paired McNemar p<0.05)" — the "82/63" upper end is the "MG-Select–style histogram KL" selector, which the paper itself says operates "on discretized chunks, not token-native distributions" and "rarely separates from consensus here." So the paper compares against an acknowledged proxy for MG-Select and then reports beating it. A reviewer will discount this.
- **Fix:** Either implement MG-Select faithfully against SmolVLA's flow expert (the paper says the interface accepts a token-native verifier hook — then add one), or stop presenting "we beat MG-Select" as a headline and move that comparison to a clearly-labeled "proxy selector" subsection. If you implement it properly and *still* beat it, that's a much stronger paper.

### W7. The most relevant baseline (adaptive looking, SCALE) is not compared.
The paper positions itself against SCALE [choi2026scale] as the closest adaptive-compute prior work, then never compares against it (or a reimplementation). The baselines actually run are *fixed-budget* (best-of-8, MG-Select-KL). The natural comparison for "adaptive gating" is *another adaptive method*. The bib note says SCALE is contemporaneous, which is a partial excuse — but then the experiments section should say so explicitly and ideally still include a simple "gate on policy self-uncertainty / sample entropy instead of twin-probe disagreement" variant to show the *twin-probe* choice matters.
- **Fix:** Add at least one adaptive baseline: gate on (a) entropy/variance of a small initial sample, or (b) a learned per-step uncertainty head, or (c) a from-the-paper reimplementation of SCALE-style adaptive looking. Show the twin-probe gate is competitive with or better than these, or concede it's one reasonable choice among several.

### W8. "Observability is the governing variable" may be over-abstracted.
What is shown is narrower: a *bottom-strip occlusion* of a *single overhead camera* increases the benefit of resampling on a *grasp* task. The paper itself notes in Q4 that *clutter* behaves the same way ("a second source of the same structural ambiguity"). That suggests the real governing variable is *task/policy ambiguity (multimodality of the action posterior)*, of which limited observability is one cause. The current framing claims more than the experiments isolate; a cleaner and more defensible thesis is "the multimodality of the policy's action distribution governs the return on TTC; we manipulate it via controlled occlusion (and corroborate via clutter)."
- **Fix:** Either (a) directly measure the action-posterior spread (e.g., cross-sample variance of the chunk) and show *that* — not "observability" per se — predicts the TTC gain step-by-step (this would also tie the twin-probe gate to the mechanism beautifully), or (b) soften the framing to "a controlled handle on ambiguity" and stop calling observability *the* governing variable.

---

## Minor weaknesses & presentation

- **M1. TinyVLA in the main results table.** Table 7 includes "TinyVLA + TTC (sanity)" 30+ points below everything. It's repeatedly disclaimed as "a wiring check, not a scientific baseline" — so cut it from the main table (move to appendix) or stop mentioning it in the body. As is, it dilutes the results and invites the question "why is it here?"
- **M2. `macro-F1` for a binary success outcome.** Success/fail → why macro-F1 rather than success rate (already reported) plus, if you want a class-balanced number, balanced accuracy? If macro-F1 is staying, define exactly what the two classes are and why F1 is the right summary; otherwise drop it.
- **M3. Multiple comparisons.** Several pairwise McNemar tests are run (Table 2, plus paired tests referenced elsewhere). With N=50 and p-values near 0.03–0.05, a Bonferroni/Holm correction (or at least an explicit acknowledgment of the family-wise error rate) is warranted before claiming significance.
- **M4. N=50 is small for the key claims.** The "gated > best-of-8, p=0.031" rests on ~7 discordant pairs flipping the right way. Bump N for the headline comparisons specifically; the appendix already says "raise for tighter intervals."
- **M5. Title is too long and bakes in a model name.** "...for Fine-Tuned SmolVLA..." dates the paper and undercuts the "runs on any stochastic VLA" claim made in §4.3. Consider e.g. "When Test-Time Compute Helps a VLA: Observability-Gated Inference Scaling for Single-Camera Manipulation."
- **M6. "Never asks which deployments actually need it" (abstract).** "Never" is a strong universal. Soften to "rarely" or qualify ("rarely isolates which deployments need it") — reviewers reflexively push back on absolutes in related-work claims.
- **M7. Figure 1 is dense.** Two lanes + a camera inset + a controlled-variable callout + the decision diamond is a lot. Consider splitting into (1a) data/training pipeline and (1b) the eval-time gated controller, or trimming the callout text. Also the small dark rectangle in the camera inset overlaps the red object box — looks like an artifact.
- **M8. Missing related work.** No engagement with: classic POMDP / active-perception literature (you say "partial observability" repeatedly); uncertainty estimation in BC/IL (ensembles, MC-dropout) as a reference point for the twin-probe; other adaptive-inference-for-control work. A sentence each would situate the contribution better.
- **M9. Receding-horizon detail.** Chunks are T=8 but consumed one step per query at 5 Hz — so each chunk's later steps are mostly discarded. Worth a sentence on why T=8 then (re-planning stability? training signal?), and whether the gate/selector operate per-query on the full chunk or could exploit the discarded horizon.
- **M10. Reproducibility of the "we release vla_lab" contribution.** As an anonymized submission this can't be assessed; just be aware reviewers may not open the supplementary archive, so don't lean on the artifact as load-bearing for the *scientific* claims — it's a nice-to-have, not a substitute for the missing experiments.
- **M11. Latency is reported on "an RTX 4090-class GPU"** with no batch-size / precision / sequence-length details, and the median-vs-p95 gap for the gated config (112 vs 238 ms) is large — worth a sentence on the bimodality (it's the gate firing) and on whether p95 is the number a real-time controller actually cares about.

---

## Questions for the authors (please address in rebuttal)

1. **Are the reported numbers from a single frozen evaluation run?** If not, when will they be, and will they change? (See W2.)
2. **What is K̄ on the clean split alone vs. the occluded split alone?** And what fraction of steps does the gate flag in each? (See W4/W5.)
3. **Why does the gated controller beat fixed best-of-8 by ~12 points on the *clean* split?** Specifically: does "best-of-8 *including* the deterministic ξ=0 forward as a candidate" close that gap? (See W4.)
4. **How sensitive are the results to τ and ε?** Please provide the success-vs-K̄ frontier (sweeping τ) overlaid on the fixed-K frontier. (See W5.)
5. **Why not run the graded-occlusion sweep that Appendix F says is already supported?** What would it take to include it? (See W1.)
6. **Why is MG-Select reimplemented on discretized chunks rather than on the flow expert's distribution?** Would a faithful implementation change the comparison? (See W6.)
7. **Is "observability" or "action-posterior multimodality" the right name for the governing variable?** Do you have step-level data relating the two? (See W8.)
8. Does the twin-probe gate's behavior depend on SmolVLA's flow-matching head specifically, or have you any evidence it transfers to a different stochastic VLA (beyond the TinyVLA wiring check)?

---

## Detailed comments by section

- **Abstract.** Strong and dense. The "≈+17% vs. ≈+8%" is the fixed-best-of-N relative gain; the "92%/74%" is gated. Consider stating both relative gains side by side so the reader isn't comparing a relative number to an absolute one. "Never asks" → soften (M6).
- **§1 Introduction.** Good motivation; the deployment-vs-method reframing lands. The contributions are concrete. But contribution #1 ("observability governs the return on test-time compute, a controlled finding") is currently *not* a controlled finding at the level of statistical evidence the wording implies (overlapping CIs, two points) — either run the sweep (W1) or weaken the wording to match §5.1's own caveat. The forward pointer to "$\bar K{\approx}3.4$" uses bare $K$ here and $\bar K$ later — make notation consistent.
- **§2 Related Work.** Well-grouped, each paragraph ends with a differentiator — good. Add POMDP/active-perception and IL-uncertainty pointers (M8). The claim "we instead make observability the independent variable" is the paper's distinguishing move and is fine; just make sure §5 actually delivers the variation that justifies "variable" (not two levels).
- **§3 Problem Formulation.** Clean. The (i)/(ii)/(iii) questions overlap with §5's Q1/Q2/Q4 — that's acceptable signposting, but consider whether stating them twice is necessary or whether §5 can just reference §3. Define $\bar K$ here once and use it everywhere.
- **§4 Method.** The backbone choice (SmolVLA, natively stochastic flow head) is well-justified. The three selectors are clearly described, including their failure modes — good. §4.3 (the gate) makes the two key arguments (tracks ambiguity; shields easy steps) but provides no measurement for either *in this section*; the second one in particular (shielding) is the load-bearing claim behind "gating beats fixed best-of-N" and needs evidence (W4). Algorithm 1 is clear. §4.4: state the occlusion fraction(s) you actually evaluate (currently only 0.35 in the body) and forward-reference the graded set.
- **§5 Experiments.** Q1: see W1 — the honesty is good but the experiment is incomplete. Q2: see W4/W5 — the latency win is credible; the accuracy win needs the per-step breakdown and the ξ=0-candidate ablation; add the τ-sweep frontier. Q3: fine, and the "complements data" message is reasonable; with the table now ending at 92% it lines up with Table 6 (good). Q4: the scene breakdown is useful and the clutter↔occlusion observation is actually one of the more interesting things in the paper — lean into it (it supports the W8 reframing). Cut or relocate the TinyVLA row (M1). Failure-mode discussion is good and appropriately humble.
- **§6 Discussion / Limitations.** This is the right content and tone. It does, however, essentially pre-concede W1, W3, W6 — which a reviewer reads as "the authors know the experiments are thin." The fix is not to delete the limitations but to *do the experiments* so the limitations list shrinks.
- **§7 Conclusion.** Fine. "Profile observability, then budget compute where it is lowest" is a clean takeaway — but it's only actionable if you've shown the relationship is smooth and predictable, which is again W1.
- **Statements.** Appropriate.
- **Appendix.** Solid reproducibility detail. Appendix F is, ironically, a list of the experiments that would most strengthen the paper — move some of them into the main results.
- **Figures.** Fig 1: dense (M7). Fig 2: now internally consistent with the caption; consider also annotating the fixed-best-of-8 bracket so the headline (+8%/+17%) is *visible*, not just in the caption. Fig 3: good. **Missing and worth adding:** (a) success vs. K̄ frontier, gated-vs-fixed (W5); (b) gate firing rate vs. occlusion fraction (supports the mechanism); (c) the graded-occlusion curve (W1); (d) one qualitative panel — occluded frame, the ambiguous grasp, what resampling recovers, and a failure where it doesn't.

---

## Prioritized revision plan (what would move my score)

**Must-do for a credible CoRL submission (each addresses a "major"):**
1. **Run the graded-occlusion sweep** (≥4 fractions × clean, larger N) and plot TTC-gain vs. occlusion → turns the headline from a two-point pattern into a curve. *(W1)*
2. **Finalize all numbers from one frozen run; remove the "refresh later" note; reconcile cross-table values.** *(W2)*
3. **Add the τ-sweep frontier** (gated success-vs-K̄ overlaid on fixed-K) and **the per-step / per-gate-decision breakdown**, including the "best-of-8 with ξ=0 as a candidate" ablation, to actually explain the +12-point clean gap. *(W4, W5)*
4. **Either** add a real-robot demonstration (even N≈20–30, one occlusion) **or** a second sim task — something that shows the effect isn't an artifact of this one setup. *(W3)*

**Strongly recommended:**
5. Implement MG-Select faithfully (token-native) or demote the "we beat MG-Select" framing. *(W6)*
6. Add at least one *adaptive* baseline (sample-entropy gate / SCALE-style). *(W7)*
7. Soften "observability is *the* governing variable" to match what's measured, or measure action-posterior multimodality directly and pivot the framing onto it. *(W8)*

**Polish:**
8. Cut TinyVLA from Table 7; reconsider macro-F1; add multiple-comparison correction; bump N for headline tests; trim the title; soften "never"; split/declutter Fig 1; add a qualitative panel; add the missing related-work pointers.

If items 1–4 are done convincingly, this is a 6–7 paper. As submitted today it is a 5.

---

*Note to the authors (out of character): this review is deliberately tough because it's meant as a pre-submission stress test. The idea and the protocol are good — the gap is almost entirely "the experiments that prove the thesis aren't in the paper yet," and that's fixable with the existing testbed.*

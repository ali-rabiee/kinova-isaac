# Execution report — `vla_lab/carryover1.md` (2026-08-23)

What was implemented, what was measured, which of the brief's expectations survived contact
with the data, and what remains. Every number below is reproducible from the result files;
the manuscript quotes them only through generated macros, and the build fails on any prose
literal no artifact vouches for.

---

## The finding that reordered the work: defect (xii)

While preparing P0-3, the existing value-model fit turned out to be wrong for a reason the
brief did not anticipate: `_fit_scaled_logistic` regularised the logistic weights with an
isotropic Gaussian of precision 0.01 **while margins were in metres**. The ~150/m slope the
tight-end data want costs ~110 nats under that prior, so the fit flattened to ~30/m. The
published physics — crossover 8.5 cm, width 3.12 cm — was this artefact; the "two-parameter
logistic cannot follow a near-step" story in §5.4 was the prior, not the form.

**Corrected measurement** (lapse/floor psychometric with a per-cm prior, isotonic check,
2000-replicate within-cell bootstrap; 216 original + 192 new tight-end rollouts = 408):

| | published | corrected |
|---|---|---|
| crossover m\* | 8.5 cm | **4.54 cm**, 95% CI [3.9, 5.2] |
| transition width w | 3.12 cm | **0.50 cm**, 95% CI [0.17, 0.81] (isotonic: 0.50 [0.29, 0.85]) |
| worst in-band residual | 0.205 | **0.068** (brief's target: <0.10) |

The value gap also has a *trivial second crossing* near 0.6 cm (both strategies fail; the
cheaper failure "wins"); the coordinate is defined on the upper crossing and the code now says
so. Everything downstream — scene grid, rendered atlas, every study, every trained cell — was
regenerated; superseded results are intact in `results/legacy_2026-08-23/`. Recorded as defect
(xii) in the paper's §5.3, with the general lesson: *a regulariser's strength has units.*

## P0 — statistical holes

- **P0-1 seeds.** `sup_models.sh` / `sup_ablate.sh` take `--seeds` (5 from-scratch, 3
  pretrained); every cell reports mean ± seed SD; a **seed floor** per backbone (mean |diff|
  between seeds of the same cell, the test–retest floor's construction) is computed, tabulated,
  and orderings inside it are marked and not interpreted (`training/seeds.py`). Audited the
  training path for RNG that escapes the seed: none (CUDA nondeterminism is part of the floor).
- **P0-2 physics uncertainty.** Bootstrap CIs on m\* and w in `physics_report.json`;
  `run_study --physics-quantile lower|upper` rebuilds the grid under the CI-end draws (stamped,
  non-poolable); `sup_physics_ci.sh` reruns the primary study under all three.
  **Result:** the memoryless contrast excludes zero under every draw (+0.049 to +0.051);
  always-counter's contrast excludes zero at the point and upper draws (−0.016, −0.015) and
  not at the lower — see "always-counter" below.
- **P0-3 fit.** Both the lapse psychometric and the isotonic regression, agreeing (m\* 4.54 vs
  4.53 cm, w 0.50 vs 0.50 cm); tight-end sweep collected (0–4 cm at 24 reps, plus 5/7/8 cm
  around the corrected crossover); legacy fit reported beside for the record; one step-budget
  timeout excluded from the duration estimate and counted.
- **P0-4 correlations.** `stats_utils.guarded_correlation` refuses coefficients below n=15
  without an explicit override; ρ=−0.46 and ρ=−0.14 removed from paper and README; the deployed
  evaluation now emits `facts.json` with the descriptive pattern (per-backbone
  best-offline-vs-deployed rank; the >11% in-band abstention flag). The third backbone's
  closed-loop cells are queued (below) to raise the cell count.

## P1 — deciding experiments

- **P1-1 Qwen2.5-VL-3B**: downloaded, queued at 3 seeds under LoRA settings asserted identical
  to the 2B cells (recorded per-run in the manifests; the table generator has them). Citations
  fixed: the roster now cites the Qwen2-VL and Qwen2.5-VL model papers, not the unrelated
  Qwen-VLA paper.
- **P1-2 shift.** Held-out atlas rendered (`sup_frames.sh --shift`): new colours, 4.5 cm cubes,
  a different table surface, a third distractor, plus a second overhead camera pose captured
  alongside. `training/eval_shift.py` re-scores every checkpoint on its own validation
  supervisors under matched / shift / shift+camera; queued after each sweep.
- **P1-3 B6 / Fisher.** The posterior now carries ∂κ/∂λ per cell, so the Fisher information
  for λ is a per-slot output; B6 and two variants implemented and run. **The brief's
  hypothesis is not supported — and the negative result got stronger than expected:**
  1. The brief's 44% λ-identification figure was largely criterion artefact: the
     total-variation flag fires for supervisors with β=0 exactly (the LOO prior correlates λ
     with β), i.e. for people whose λ is unidentifiable *by construction* (19–44% of them
     across runs). The headline criterion is now posterior **contraction** (≥25%), which a
     null case cannot pass, reported with a non-complier control column.
  2. Under that criterion **no schedule identifies λ for a single supervisor at this budget**
     — not B5, not the brief's log-spaced ladder, not the no-wait variant the Fisher analysis
     itself implies. The information is bounded by the residue's magnitude, not the schedule:
     under time decay every slot advances the residue clock, so consecutive probes already
     sample distinct elapsed times and a wait only forfeits an observation.
  3. β·g contracts for ~40% under schedules that probe near demonstrations and for ~0% under
     washout placement — a real schedule effect, on the identifiable parameter.
  4. What ships is the **prospective diagnostic**: every adaptive policy logs per slot whether
     its λ posterior has contracted, the accumulated information, and the e-folding delay —
     the "about to personalise on the prior" readout the brief asked to expose.
- **P1-4 flip diagnostic.** `sup_flip.sh` / `supervisory/flip.py`: every primary contrast as a
  function of the assumed w (and m\*), CI bands, flip points, the measured value and its CI on
  the same axis. **Result:** the memoryless contrast holds at every assumed width (0.17–3.12 cm);
  always-counter's significance is *non-contiguous* along the sweep and flickers inside the
  measured interval — the tool's verdict line says "a conclusion about the measurement error".

## The always-counter nuance (new, must-know)

Under the final physics, B4's paired contrast against the washout is −0.0156 [−0.0290,
−0.0022] — an interval excluding zero, at a sixth of the test–retest floor (0.1008). It loses
the exclusion at the lower physics draw and appears/disappears between adjacent flip-sweep
cells whose grids differ by re-derivation error smaller than the pose noise. The paper reports
it plainly, declines to promote it under the pre-committed §3.8 standard, and tells the human
study's power analysis to treat "asking beats waiting" as a plausible small effect rather than
a null or a finding.

## P2 — design and framing

- **P2-1.** `policy_recommended` (B7): corrected estimator + counter-proposals, adaptive
  schedule off; `CarryoverAwareConfig.recommended()`; a test asserts the schedule is never on.
  **Honesty note:** the first draft's "actively harmful in the high tercile" ordering was
  measured under the defective physics and does not replicate at that size; what replicates is
  the *direction* (schedule-only degrades fastest with compliance: −0.027→+0.028) and the
  identifiability ground got stronger (λ: nobody). The retirement now rests on those, plus
  calibration and parsimony — the paper says exactly this.
- **P2-2.** The ask-gate identifiability argument is stated *before* the experiment (§4.4:
  the gate's target is a functional of π\*, hence unidentifiable from data in which π\* is
  unknown), the ten cells are its confirmation, and a test asserts no grounder's opinion about
  asking can change a single session action.
- **P2-3.** Dose tracking promoted with its control: B5's counter-proposal rate runs
  1.6→3.4→9.8 per session across the dose ladder and 2.7→3.4→3.3 across the lapse-rate placebo
  the belief module cannot see (`sensitivity/placebo.json`; `fig_dose_tracking.pdf`). The
  latency cell is reported as *not* a placebo (it enters the decay clock).

## P3 — human study preparation

- **P3-1 phrase corpus** (`human_study/phrase_corpus.py`): collection packet (rendered scenes +
  verbatim instruction script + response sheet), grounder evaluation against the 0.85 gate with
  every failure mode enumerated (hedge / conditional / multi-clause / names-neither /
  unimplemented-strategy), and a rebuild that regenerates the supervisor's phrase inventory and
  *empirical* hedge rate — `run_study --phrase-corpus` re-runs the study under it, with the
  narration hash preventing pooling. **Needs ~10 real people; no corpus exists and nothing was
  fabricated.** `python -m vla_lab.human_study.phrase_corpus prepare` writes the packet.
- **P3-2 regime**: decided and documented in §7.5 — the human study runs the **alternating**
  regime (it identifies H1/H2, which are now primary), with every one-sided-deployment claim
  labelled model-based.
- **P3-3 framing**: the study is restructured around H1/H2 ("does compliance carryover exist,
  how big, does it decay, how much does it vary") with H3 secondary and model-based, and the
  paper states this as an author decision made on the power analysis. **This is flagged for
  the authors' sign-off** — it is written as recommended by the brief, and it is reversible.

## P4 — errata and the audits

Defect list renumbered (i)–(xii) in order with a consistent count; test counts generated at
build time (`run_tests --count` → currently 266 live + 162 archived); Qwen citations fixed;
"MAE at the crossover" unified to crossover-weighted MAE everywhere; the abstract's roster
claim is generated from readiness flags. New build machinery: `check_numbers.py --strict`
fails the build on any prose literal no artifact or justified allowlist entry vouches for
(`numbers_allowlist.txt` carries the defect-log and historical numbers with reasons);
`lint.py` also writes a cross-reference map (`crossrefs.txt`) and checks claim-refs.

## Replaced claims (ground rule 2 — kept in the paper with what replaced them)

1. "The result that establishes the problem is real" → a validation of the estimator against
   an injected bias, stated in the same sentence everywhere it is quoted (abstract included).
2. λ "identified in 44%" → identified for nobody under an honest criterion; the 44% was
   partly the criterion firing on non-compliers.
3. The §6.5/§D.5 "adaptive schedule actively harmful in the high tercile" → direction
   replicates, size does not; grounds for retirement restated.
4. w = 3.12 cm / m\* = 8.5 cm → defect (xii); both carried with bootstrap CIs now.
5. ρ = −0.46 / ρ = −0.14 → withheld (n=7); descriptive facts instead.

## State of the long-running work — FINISHED (2026-08-25 09:07)

Every stage of the GPU queue completed with zero training failures (one reboot on 2026-08-23
cost a single in-flight cell; `logs/sup/resume_after_reboot.sh` recovered the rest). Final
build: **58 pages, 0 LaTeX errors, 0 dangling references, strict numbers audit clean** — all
readiness flags at 1, no pending blocks. Late-stage findings folded into the manuscript as the
results landed:

- **Seed control dissolved two more first-draft claims**: the from-scratch context-injection
  effect on residue tracking is inside its seed floor (only film>none on Brier gain survives,
  +0.010±0.007), and the "context-blind tracking rises monotonically with scale" chain
  (−0.006→+0.151→+0.185) is formally withdrawn — the four backbones' context-blind cells are
  statistically indistinguishable (+0.05 to +0.07, overlapping dispersions), incl. the clean
  2B→3B within-family contrast (identical LoRA settings, asserted; Δ=0.009 vs floors 0.07–0.11).
- **The shift evaluation worked as designed on every image-sighted backbone** (tiny, both
  Qwens): context-blind cells lose a third or more of their gain on the held-out scenes while
  context-injected cells are unmoved — and it *discovered* that SmolVLA's intent heads are
  image-blind by construction (embed_prefix is tapped before attention fuses the modalities;
  verified zero logit change), so its rows test only the language channel.
- **The closed loop inverted the first draft's R11 in every particular**: best-offline is now
  best-deployed on all three tested backbones, the failing cell moved (SmolVLA/text), and the
  >11%-abstention cell is a top channel. The paper keeps both configurations in view and claims
  only the durable statement: no offline metric predicted deployment stably; measure the loop.
- Ablations at 5 seeds: the no-reference collapse (−0.120) with its gap∼κ spike (+0.44) and the
  no-forward gain improvement (+0.033) both clear the seed floor; the compliance penalty is
  neutral.

## Needs the user

1. **Phrase corpus**: recruit ~10 people, run the packet, then
   `phrase_corpus evaluate` / `rebuild` / `run_study --phrase-corpus ...`.
2. **P3-3 / P3-2 sign-off**: the H1/H2-primary framing and the alternating-regime decision are
   written in; both are flagged in the paper as author decisions.
3. Let the GPU queue finish (or reorder it), then one `./vla_lab/scripts/build_paper.sh`.

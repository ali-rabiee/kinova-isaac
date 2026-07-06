# vla_lab — Project Recommendations & HRI 2027 Assessment

*Written 2026-07-05, after a full code review + cleanup pass of `vla_lab/` (see §8).
Companion docs: [`fable_report.md`](./fable_report.md) (2026-07-04 reframing),
[`README.md`](./README.md), [`data_collection_guide.md`](./data_collection_guide.md).*

---

## 1. Executive verdict

**The research idea is strong and the engineering is essentially done — but the paper
currently has zero measured results and, more importantly for HRI, zero human
participants.** Every claim in the draft is still a `--` scaffold.

- **Is the project a good idea for HRI 2027?** Yes, *conditionally*. The framing —
  uncertainty decomposed by **source** (perception vs. human intent), each routed to a
  different remedy, with mid-execution corrections fused by an online human-reliability
  model — sits squarely on HRI territory and on the 2027 theme (“Innovative HRI”). The
  2026 literature map in `fable_report.md` §2.3 shows the combination is genuinely
  unclaimed. **But HRI full papers live or die on the human study.** A submission whose
  only “human” is `feedback/sim_human.py` will read as a robotics-systems paper at the
  wrong venue and would very likely be rejected regardless of technical quality.
- **The single most important decision this month:** commit to running a real-participant
  study (the infrastructure for it already exists — see §4), or retarget the paper at a
  robotics venue where a simulated-human evaluation is acceptable.
- **Timeline is tight but feasible:** HRI 2027 is March 8–12, 2027 (Santa Clara);
  deadlines are still TBD on the site, but HRI full-paper deadlines historically land in
  **early October** — call it **~13 weeks from today**. The sim results are ~3 weeks of
  push-button work; the human study is the long pole and its longest lead item (ethics
  approval) should be started **this week**.

---

## 2. Where the project actually stands

### Assets (verified during this review)

| Layer | State |
| --- | --- |
| Data collection | `collect_v3` (overhead) and `collect_v4` (+ calibrated wrist cam) stable; scripted expert ≈ 90 % success; `verify_session` gate catches all historical failure modes |
| Policy | Multicam TinyVLA (1.93 M params, shared encoder + camera dropout); SmolVLA fine-tune path through LeRobot; old checkpoints load `strict=True` |
| Inference machinery | K-sample TTC (selection + gating), act/compute/query controllers (`allocator`, `knowno`, `insight`, `scale`, `intent_allocator`, …), conformal calibration, fitted irreducibility detector |
| Human-interaction stack | Counterfactual intent estimator, cross-view disagreement, two-way feedback channel (scripted **and live console**), rule-based parser, Beta-reliability fusion with verify/trust-all ablation |
| Experiment harness | Camera-set / occlusion / ambiguity / feedback-quality sweeps, calibration records → ECE/AUROC/coverage figures, paired-seed comparisons, Wilson CIs, McNemar |
| Study tooling | 5-condition within-subject protocol with Latin-square counterbalancing, NASA-TLX + Jian trust + MDMT instruments, reliance/over-trust quadrants, power analysis CLI |
| Paper | 27-page `acmart[manuscript]` draft, 56 verified references, honest all-scaffold result tables; positioned against the 2026 near-misses (TRIAGE, INSIGHT, BOKBO, PACT, Assistron, SCALE, LIBERO-Occ, YAY Robot) |
| Tests | 83/83 offline tests green (re-verified after today’s cleanup) |

### Gaps (equally real)

1. **No trained model beyond smoke checkpoints.** The 120-episode collect_v4 dataset has
   not been collected; every downstream number depends on it.
2. **τ_intent and the estimator temperature are uncalibrated.** The 2026-07-04 smoke run
   documented the failure mode honestly: on an untrained checkpoint the behavioral
   posterior saturates (entropy ≈ 0.0006 under a fully ambiguous instruction) and the
   query branch never fires. Until this is calibrated on a real checkpoint, the headline
   R-intent experiment cannot run.
3. **No human data of any kind** — not even a pilot with lab members.
4. **`real_robot/` is stubs.** The physical Kinova + wrist camera exist, but no bridge.
5. **Paper format**: `manuscript` (single-column) must become `sigconf, review, anonymous`,
   and ~27 manuscript pages must shrink to HRI’s full-paper budget — a hard cut, not a
   trim. (The `corl_2026.sty` in the paper folder is a leftover; `main.tex` already uses
   `acmart`.)

---

## 3. HRI 2027 fit — honest analysis

### Why it fits

- **The contribution is an interaction contribution.** “When should a robot think harder
  vs. ask, and how should it weigh a human whose input quality varies?” is an HRI
  question. The data-processing corollary (extra compute provably cannot recover intent —
  the missing bits are in the human, not the pixels) is a crisp, defensible theoretical
  nugget that robotics venues would shrug at but HRI reviewers will appreciate.
- **Typed transparency** (“I can’t see” vs. “I don’t know which one you mean”) is a
  measurable interaction variable with direct trust/workload hypotheses — exactly the
  kind of IV HRI likes, and the study protocol for it already exists in `human_study/`.
- **The quality-modeled human is novel at this venue.** YAY-Robot-style systems trust
  every correction; Losey & O’Malley model noise offline. An *online* Beta-reliability
  gate with a verify branch, evaluated across an accuracy × specificity grid, is a fresh
  axis — and the `sweep_feedback_quality.sh` grid is the exact experiment.
- **Theme alignment**: “Innovative HRI” explicitly invites new interaction mechanisms.

### Why it could be rejected

- **No human participants = near-certain rejection at HRI.** This cannot be
  overstated. HRI’s methodological culture expects a study with people (even N≈18–24,
  even with a simulated *robot*). A simulated *human* is a modeling assumption, not a
  study. Everything else on this list is secondary.
- **Sim-only robot.** Acceptable at HRI when the human side is real (screen-based and
  VR studies are routine), but it must be framed as a controlled study environment, not
  as a robotics result. The wrist-camera contract work actually helps here — it shows
  the sim setup mirrors a real, calibrated platform.
- **Toy task / tiny model optics.** Six colored boxes and a 1.9 M-param policy invite a
  “demo-ware” reading. Mitigations: (a) run the SmolVLA leg for the main conditions so
  a pretrained-VLA backbone anchors the claims; (b) lead every results section with the
  human-facing metric (query precision, trust calibration, workload), not success rate.
- **Crowded uncertainty-routing space.** TRIAGE routes by uncertainty type; PACT learns
  ask-or-act; INSIGHT triggers help. The related-work positioning is already written,
  but the *empirical* separation (typed queries land exactly where intent entropy is
  high; compute stops helping) must actually show up in the data. If τ_intent
  calibration fails on the trained model, the fallback signal (`instruction_ambiguous`
  lexical flag + fitted allocator) weakens the “behavioral estimator” story to a
  “feature among features” story — still publishable, less exciting.

### Verdict

**Target HRI 2027 full paper — but make the go/no-go explicit.** Gate the decision on
four things by **end of July 2026** (§6): dataset collected, τ_intent calibrated and
sane on a real checkpoint, ethics application submitted, pilot (N≈6) scheduled. If any
of these slips badly, fall back per §7 rather than submitting a humanless HRI paper.

---

## 4. The pivotal reframe: you already have a human-study apparatus

The `console` feedback channel (`feedback/channel.py`) is the underused crown jewel: a
participant can *already* sit at the machine, watch the Isaac viewport, receive typed
robot questions mid-episode, and type corrections that are parsed, fused, and applied
live. That is a real human-in-the-loop system — no Wizard-of-Oz, no mocks.

**Recommended study (maps onto existing code):**

- **Design**: within-subject, 4 conditions (trim from the 5 in `protocol.py` to keep
  sessions < 60 min): `autonomy` / `compute_gated` (never asks) / `intent_allocator`
  with typed transparency / `intent_allocator` trust-all (`--no-fusion-verify`).
  Latin-square counterbalancing is already implemented.
- **N**: run `./vla_lab/scripts/power_analysis.sh` and put its number in the paper
  (expect ~18–24 for within-subject paired success/trust deltas).
- **Task**: participant states/receives an intent under the ambiguity manipulation
  (`--instruction-ambiguity half`), interacts via keyboard during rollouts. Per-episode
  measures are already logged (`feedback.events`, query counts, reliability trajectory);
  questionnaires (TLX, Jian, MDMT) already scored by `human_study/instruments.py`.
- **Hypotheses**: H1–H5 already drafted in the paper — the study exists to fill them.
- **What to build (small)**: a thin session runner that sequences live eval episodes per
  the `SessionPlan` and prompts the questionnaires between blocks —
  `human_study/session.py` currently drives a *synthetic* episode model; wiring it to
  `eval_isaaclab.py` (subprocess per condition block, parse `results_*.json`) is ~1–2
  days of work and is the **only missing piece of study infrastructure**.
- **Ethics**: minimal-risk, screen-based interaction study → typically expedited review,
  but submit **now**; 2–6 week turnarounds are common and this is the schedule’s long pole.

---

## 5. Critical path to the (expected ~October) deadline

| Weeks (2026) | Work | Owner notes |
| --- | --- | --- |
| **Jul 6–19** | Collect 120–160 eps `collect_v4` in verified 40-ep chunks; train `multicam_v0` (both) + `overhead_only` + `wrist_only`; SmolVLA export `--wrist` + fine-tune; **submit ethics application**; dry-run the console-channel study on yourself | Collection ~2 min/ep headless; all push-button |
| **Jul 20–Aug 2** | **Calibrate τ_intent + estimator temperature on the real checkpoint** (sweep temperature until ambiguous-vs-specific entropy separates; pick τ on a held-out session); run R-camera (`sweep_camera_sets.sh`), calibration sweep + `fit_allocator`; pilot study N≈6 (lab members) | This is the go/no-go gate (§6) |
| **Aug 3–23** | R-intent (`--controller intent_allocator --instruction-ambiguity {none,half,full}` vs. baselines) and R-quality (`sweep_feedback_quality.sh`) on paired seeds; freeze sim results; iterate study protocol from pilot | 50 eps × ~6 conditions × 3 ambiguity levels is days of GPU time — start early |
| **Aug 24–Sep 20** | **Run the human study (N per power analysis)**; analyze with `human_study/analyze.py` (McNemar, Wilson, reliance quadrants, trust calibration) | Recruit continuously; 2 sessions/day is enough |
| **Sep 7–deadline** | Paper: convert to `sigconf,review,anonymous`; cut to the HRI page budget (push method detail to appendix/arXiv version); fill scaffold tables; internal red-team review; watch the CFP — HRI usually wants an **abstract ~1 week before** the paper | Writing overlaps the study |

Contingency: HRI has historically also offered Late-Breaking Reports (~December
deadline) and alt.HRI — both are honorable landing zones for the study-light version
(§7).

---

## 6. Go/no-go gate (propose: **July 31, 2026**)

Submit to HRI 2027 as a full paper **only if all four hold**:

1. ≥120 verified collect_v4 episodes and a `multicam_v0` checkpoint whose plain-eval
   success is meaningfully above chance (the allocation story needs headroom to show
   effects; if the policy fails everywhere, no controller can look good).
2. On that checkpoint, intent entropy **separates** ambiguous from specific instructions
   (the smoke run’s saturation issue resolved by temperature/τ calibration — or
   consciously replaced by the lexical-flag fallback with the framing softened).
3. Ethics application submitted (approval may still be pending — but submission cannot
   wait past July).
4. Pilot participants scheduled.

If 1–2 hold but 3–4 fail → pivot to a robotics venue (§7). If 1–2 fail → the project
needs debugging, not a deadline.

---

## 7. Fallback / portfolio strategy

| Venue | Realistic date | What version fits |
| --- | --- | --- |
| **HRI 2027 full** (primary) | ~Oct 2026 (TBD) | Full system + real-participant study |
| **HRI 2027 LBR / alt.HRI** | ~Dec 2026 (TBD) | System + pilot (N≈6) + sim grids; alt.HRI suits the provocative “compute cannot buy intent” framing |
| **ICRA 2027** | ~mid-Sep 2026 | Sim-only version with simulated-human grids is acceptable; loses the HRI-native audience |
| **RSS / CoRL 2027** | early–mid 2027 | The sim→real version with the real Kinova + wrist camera (the `real_robot/` bridge becomes the new work) |
| **HRI 2028** | Oct 2027 | The strongest version if the study slips — better one great HRI paper than a rushed one |

Note the ICRA deadline likely lands *before* HRI’s — if the human study is clearly not
happening by late August, the ICRA version can be produced from the same result set.

---

## 8. Technical recommendations (prioritized)

### P0 — before/with the experiment sprint

1. **τ_intent + temperature calibration procedure** (the known blocker). Add a small
   offline script: sweep `intent.temperature` over a held-out session’s ticks under
   `none|half|full` ambiguity, plot entropy distributions, pick the operating point
   (max separation), then set τ_intent at e.g. the 90th percentile of the *specific*
   distribution. One day of work; unblocks the headline result.
2. **Wire the live human-study session runner** (§4) — the only missing study infra.
3. **Per-refinement-type reliability** (targets vs. nudges vs. stop) in
   `feedback/fusion.py`. The 2026-07-04 smoke already produced the motivating anecdote
   (verify-override of a *correct* correction by a noisy channel); it is a
   two-Beta-tracker change and directly strengthens R-quality.
4. **Train the SmolVLA leg** for at least `autonomy` vs. `intent_allocator` so the paper
   isn’t “TinyVLA-only” (the exporter now reads fps from session metadata — a silent
   5 Hz/15 Hz timestamp bug fixed in today’s cleanup).

### P1 — quality-of-results improvements (cheap, do opportunistically)

5. **Eval early-stop option** (`eval.early_stop_on_success: false` default): end an
   episode N ticks after the lift threshold holds. Today every episode runs the full
   `max_steps` (4000 physics steps) even when the box was lifted at step 800 — the
   experiment grids in §5 will spend most of their wall-clock on finished episodes.
   Kept out of today’s cleanup because it changes the success metric’s timing
   semantics — make it opt-in per run, and consider a “lifted-and-held ≥ 1 s” criterion.
6. **KnowNo `score_kind` silent no-op** (`allocation/baselines.py:165`): the non-default
   branch reads a feature (`irreducible_score`) that probes never emit, so it always
   falls back to dispersion. Either plumb the fitted score into probe features or delete
   the option; today it can silently invalidate a baseline config.
7. **Single source of truth for the language templates**: `eval_isaaclab._make_language_command`
   mirrors `vla_v1`’s template list by hand-copy. Move the list to a tiny pure-python
   module both import (contract code should not rely on “remember to edit both”).
8. **Unify tokenization** between `feedback/parser.py::_tokens` and
   `intent/estimator.py::lexical_target_candidates` (same algorithm, two copies — they
   must agree on how “the red one” tokenizes).
9. **`results_*.json` writes the episode list twice** (`episodes` and `results` keys —
   only `results` is consumed by `plot_metrics`). Drop one key in the next schema bump
   (`vla_lab_eval/v4`) rather than silently now.
10. **Dispersion scale mismatch**: `twin_dispersion` (RMS of pair difference) and
    `chunk_dispersion` (std about mean) differ by 2× at K=2 yet feed the same thresholds
    and calibration column. Pick one convention, or record which produced each value.

### P2 — after the deadline

11. **`real_robot/` bridge** (Kinova API + the wrist hand-eye slot already reserved in
    `cameras.json`): the natural HRI-2028/RSS follow-up. Do **not** attempt it inside
    the 13-week window; a rushed real-robot demo will consume the study’s weeks.
12. **LLM parser behind `parse_utterance`** (same signature, feature-flagged) — richer
    corrections, but the rule-based parser’s determinism is a *feature* for the study;
    swap only with a regression corpus.
13. **Open-vocabulary candidate proposer** for the intent estimator (currently fixed to
    the 6 colors) — needed the moment the task leaves colored boxes.
14. **Merge `collect_v3.sh`/`collect_v4.sh`** into one script + profile flag (their exec
    blocks are near-identical; their header docs differ and are worth preserving —
    left alone in today’s cleanup deliberately).
15. JEPA/V-JEPA encoder swap: remains correctly **excluded** for HRI (see
    `fable_report.md` §2.1); revisit only for a robotics-venue follow-up.

### What NOT to do before the deadline

- No architecture changes, no new uncertainty signals, no eval-loop refactors beyond
  what landed today. Every remaining week belongs to **data, calibration, the study,
  and writing**. The infrastructure is finished; treat it as frozen.
- Never mix action-rate/camera contracts in one training run (the tooling now enforces
  most of this — keep it that way).
- Keep the no-fabricated-numbers discipline in the paper; it is rarer than it should be
  and reviewers can feel it.

---

## 9. Today’s cleanup (2026-07-05) — what changed and why it matters

Full details in the session summary; headlines:

- **Inference hot-path batching** (no checkpoint/API changes, verified numerically
  identical to 4e-7): the intent estimator’s per-tick counterfactual sweep now runs as
  one batched forward (**13.7 ms → 2.2 ms, 6.3×**), K-sample TTC encodes once and
  decodes K noisy copies in one pass (**K=8: 16.6 ms → 2.3 ms, 7.4×**), cross-view
  disagreement is one batched call. At the 15 Hz (66 ms) tick budget this turns the
  full uncertainty stack from “a third of the budget” into noise — and directly
  supports the paper’s “negligible overhead” claim with measured numbers.
- **Latent-bug fixes**: LeRobot exporter stamped 5 Hz timestamps on 15 Hz data by
  default (now reads session metadata); `include_deterministic_candidate` at K=2
  produced the deterministic chunk twice; per-episode `action_clamp_count` reported the
  cumulative total; `random_patch` occlusion consumed the global RNG stream; SmolVLA
  shell training runs were unseeded.
- **Dead code removed / duplication consolidated**: unused `lerobot_metrics.py` module,
  dead factories/helpers in `allocation/`, triplicated Wilson-CI now in `stats_utils`,
  duplicated calibration-record loaders now `records.read_many`, TTC gating branches
  unified, eval’s inline RGB grab deduplicated, ~25 unused imports.
- **Docs/config coherence**: `fable_report.md` links fixed (file moved into `vla_lab/`),
  ISAACLAB resolution unified across sweep scripts, `vla_lab` added to the root
  `pyproject.toml` packages, stale references corrected.
- **Verification**: 83/83 offline tests green; single-camera checkpoints still load
  `strict=True` with identical parameter count (1,928,263); Isaac closed-loop smoke run
  of the full stack (intent_allocator + ambiguity + scripted noisy human, both cameras).

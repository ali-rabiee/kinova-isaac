# Fable Report — 2026-07-04

**What this is.** `vla_lab/Kye_instruction.md` asked for: a survey of the workspace, a
new research direction now that the real robot has a **wrist camera** (the old
"single-overhead-camera deficiency" framing no longer holds), a calibrated wrist-camera
data-collection profile, a JEPA-2.1 baseline assessment, a July-2026 literature check,
real-time human semantic input with input-quality modeling, camera ablations
(top-down / wrist / both), modular code, README updates, and this report. All of it was
executed on 2026-07-04; the four direction questions were answered by Kye before any
code was written.

---

## 1. The direction (decided with you)

- **Lead framing: uncertainty decomposed by SOURCE.** *Perception* uncertainty (can't
  see: occlusion, missing view) vs *intent* uncertainty (can't tell WHICH object you
  mean). Each source routes to a different remedy — compute / the other view for
  perception, a clarifying query for intent. Compute provably cannot resolve intent
  (the missing bits are in the human, not the pixels), which strengthens the paper's
  existing data-processing argument.
- **Two-way, quality-modeled human channel.** The robot asks (existing query branch,
  now typed by uncertainty source) AND the human interjects unsolicited corrections
  mid-execution; corrections are fused gated by an online reliability estimate.
- **Camera ablations become supporting evidence**, not the headline: "one camera + our
  method ≈ two cameras" is one result block.
- **Paper updated too** (your call): `main.tex` realigned to this framing, keeping its
  honest no-fabricated-numbers scaffold.

## 2. Literature findings (July 2026)

### 2.1 "JEPA 2.1" — decoded, and deliberately NOT integrated

What your coworker most likely referred to is a conflation of two Meta releases:

- **V-JEPA 2.1** (arXiv:2603.14482, March 2026) — real, but **vision-only** (dense
  video features; +20 pts real-robot grasping over V-JEPA 2-AC). It does *not* take
  words as input.
- **VL-JEPA** (arXiv:2512.10942, Dec 2025) — the one that "gives words as input", but
  it is a perception/VQA model with **no action head** and no official weights.
- **V-JEPA 2-AC** (arXiv:2506.09985) — the robot variant; plans toward **image goals**
  (no language) at ~16 s per action, vs this project's 15 Hz closed loop.

**Decision (per your guidance "skip it if it doesn't serve HRI 2027"): excluded.**
Rationale: the paper's contribution is the uncertainty-allocation/HRI layer, not the
policy backbone; a language-free, 16-s-per-action goal-image planner is incomparable to
a 15 Hz language-conditioned loop, and a frozen-encoder variant (JEPA-VLA-style,
arXiv:2602.11832) would read to HRI reviewers as an off-topic architecture comparison —
exactly the confusion you wanted to avoid. If a CoRL-style follow-up ever wants it, the
runnable-on-your-4090 option is: frozen V-JEPA 2/2.1 ViT-L encoder (official HF weights,
github.com/facebookresearch/vjepa2) feeding the existing TinyVLA fusion/action head as a
`policy_backend`. Nothing in the codebase blocks that.

### 2.2 The "Amazon exploration" paper

Could not be verified as Amazon. The two most likely candidates:

1. **"Explore until Confident"** (Ren et al., 2024, arXiv:2403.15941 — Princeton/**Toyota
   Research Institute**, easily misremembered as Amazon): explore until a conformally
   calibrated VLM confidence suffices — the same philosophy as our act/compute/query
   gate, in embodied QA. Cited in the revised paper.
2. **Amazon Vulcan** (Amazon Science, May 2025): tactile probing of cluttered bins
   before/while committing — exploration interleaved with execution.

### 2.3 Where the field is, and the gap we occupy

Every axis of this project now has near-misses, but **no published work combines them**
(as of 2026-07-04): TRIAGE (2603.08128) routes by uncertainty *type* but has no human
and no intent; INSIGHT (2510.01389) triggers help but single-pass, no intent; BOKBO
(2605.30660) calibrates K-sample scaling but *abstains* rather than queries — and warns
that K-sample disagreement can track sampling noise (cite it as a caution for our
dispersion signal); PACT (2605.24350) learns ask-or-act but without visuomotor TTC;
Assistron (2606.23147) solicits interventions but trusts them fully; SCALE (2602.04208)
adapts compute but never asks; LIBERO-Occ (2606.10862) benchmarks occlusion in sim
(this crowds a pure vision-occlusion story — another reason intent leads); YAY Robot
(RSS 2024) takes corrections but trusts every one; Losey & O'Malley (CoRL 2018) model
correction noise but offline. **Unclaimed: source-decomposed (perception vs intent)
uncertainty driving act/compute/query, with mid-execution language feedback fused by an
online human-reliability model, on wrist+overhead cameras.** The human axes are where
this project is first; lead with them at HRI 2027.

---

## 3. What was built

Everything is additive and modular; `collect_v3`, existing configs, checkpoints, and the
51 pre-existing tests are untouched and still pass (now 83/83 total).

### 3.1 Calibrated wrist-camera collection — `collect_v4` (profile `vla_v4`)

- `environments/reach_to_grasp_VLA/config.py`: `WristCameraConfig` /
  `DEFAULT_WRIST_CAMERA` — mount `(0, −0.055, −0.11)` m, rpy `(180°, 0, 0)` in the
  `j2n6s300_end_effector` frame, 640×640, 87° FOV (RealSense-D405-like). On the real
  robot, replace the offsets with your measured hand-eye calibration.
- `environments/utils/camera/wrist.py`: creates the camera prim as a **child of the EE
  link** (tracks the arm for free) + exact pinhole-intrinsics helpers matching the
  repo's 36 mm-aperture focal convention.
- `data_collection/profiles/vla_v1.py`: opt-in `--wrist-camera` path (default OFF —
  collect_v3 output is byte-identical; verified, see §4). Records
  `images/wrist_XXXXXX.png` per tick, `image_wrist{path}` in ticks.jsonl,
  `wrist_images` in episode summaries, and per-episode **`cameras.json`**: pinhole
  intrinsics for both cameras, the **post-domain-randomization** overhead pose, and the
  wrist hand-eye extrinsics. `data_collection/profiles/vla_v4.py` is a ~30-line profile
  (not a fork) that flips the flag; `vla_lab/scripts/collect_v4.sh` is the entrypoint
  with the identical 15 Hz / start-pose / scene contract as collect_v3.
- `vla_lab/verify_session.py`: new wrist checks (every episode has the stream, files
  exist, cameras.json present; mixed-contract sessions hard-fail).

### 3.2 Multi-camera policy + training

- `vla_lab/models.py`: `TinyVLAConfig.cameras` (`("overhead",)` default,
  `("wrist",)`, or both) + `camera_dropout`. One **shared** vision encoder runs per
  view; each stream gets a learned camera-ID embedding (only materialized for
  multi-camera configs, so **old checkpoints load `strict=True` unchanged** — unit
  tested). Train-time camera dropout (0.25 in the shipped config) zeroes random streams
  (never all), which is the mechanism behind single-view robustness at eval; a
  `camera_present` mask does the same thing deterministically at eval.
- `vla_lab/dataset.py`: parses `image_wrist`, `DatasetConfig.cameras`, per-camera sample
  keys + `camera_present`; refuses (with a clear message) to train wrist-consuming
  models on collect_v3 sessions.
- `vla_lab/train.py`: camera plumbing + two new **contract fields in every checkpoint**:
  `camera_set` and `action_rate_hz` (read from session metadata; mixed rates warn).
  New config: `vla_lab/configs/train_multicam.yaml`.

### 3.3 Eval: camera-set ablations + per-view occlusion

`vla_lab/eval_isaaclab.py` builds the wrist sensor when needed, resolves
`--cameras overhead|wrist|both` against the checkpoint's `camera_set` (subsets = the
missing-view ablation), applies occlusion per view (`--occlusion-camera`), threads the
wrist stream through TTC (`vla_lab/ttc.py`) and the controller adapters
(`allocation/policy_adapters.py`), and records `camera_set_model/camera_set_eval` in
results. `vla_lab/scripts/sweep_camera_sets.sh` runs the whole matrix, including the
"one camera + gated compute" claim legs. SmolVLA path: `export_lerobot_dataset.sh`'s
converter gained `--wrist` (wrist → `camera2` slot) and the policy wrapper feeds the
same slots at eval (`smolvla_cameras: both`).

### 3.4 Intent uncertainty — `vla_lab/intent/`

`IntentEstimator`: per tick, run the frozen policy once under the actual instruction and
once per candidate color's canonical instruction ("Pick up the {label} box."); softmax
over negative RMS distances = a **posterior over targets**, anchored by lexical
grounding and (optionally) geometric grounding (does the predicted chunk head toward the
candidate?). Normalized entropy = the intent-uncertainty scalar. ~6 extra forwards of a
1.9 M-param model per tick ≈ negligible. `cross_view_disagreement`: overhead-only vs
wrist-only predictions of the same two-camera model (per-view `camera_present` masks) —
a perception signal invisible to single-view dispersion. Both flow into the allocator's
feature set (`intent_entropy`, `intent_top2_margin`, `instruction_ambiguous`,
`cross_view_disagreement`), so `fit_allocator` can learn on them too.

New eval axis `--instruction-ambiguity none|half|full`: replaces instructions with
colorless templates ("Pick up the box.") so intent uncertainty is a *manipulated
variable* while the human still wants the scheduled target.

### 3.5 Two-way feedback + quality-aware fusion — `vla_lab/feedback/`

- `parser.py`: typed refinements from free text — target override ("no, the red one"),
  avoidance ("not the red one"), directional nudges ("a bit left" → base-frame vector ×
  magnitude), stop/resume, confirmations; chatter degrades to `unknown`, never to a
  wrong strong action.
- `sim_human.py`: **one simulated human serves both directions** — reactive unsolicited
  corrections (fires when the EE visibly heads for a wrong object) and query answers —
  with the quality knobs of the study: `accuracy`, `latency_ticks`, `specificity`
  (color/directional/vague), `noise_rate`, all seeded.
- `channel.py`: `scripted` (sim), `console` (live keyboard: type corrections during a
  rollout; typed lines after a robot question answer it), `none`.
- `fusion.py`: `ReliabilityTracker` (Beta posterior over "this human is right",
  updated online) + `fuse()` → **apply / verify / ignore**. Verify re-asks "which
  object?" through the same channel; corrections that contradict a *confident* visual
  intent posterior get verified even from trusted humans; nudges apply scaled by
  reliability; stop always applies. `--no-fusion-verify` = the trust-all ablation
  (YAY-Robot-style) that the quality study compares against.
- `allocation/baselines.py`: new **`intent_allocator`** controller — intent entropy ≥ τ
  → query (goal clarification); perception (dispersion or cross-view disagreement) high
  → compute best-of-K; else act. Robot queries and human corrections share one human
  model, so `--feedback-accuracy` is the single quality variable.
  `vla_lab/scripts/sweep_feedback_quality.sh` runs the accuracy × specificity grid with
  fusion on vs trust-all.

### 3.6 Paper (`vla_lab/paper/.../main.tex`)

*(Correction: this section was written aspirationally in the afternoon session — the
paper files had NOT actually been touched. The realignment described below was executed
in the evening session of 2026-07-04; see the appendix.)*

Realigned to the framing above (title, abstract, intro/contributions, method
subsections for the intent estimator + cross-view disagreement + fusion, new scaffold
tables for the camera-set / feedback-quality / ambiguity experiments, per-view occlusion
protocol) and the related work now positions against the 2026 near-misses (TRIAGE,
INSIGHT, BOKBO, PACT, Assistron, LIBERO-Occ, SCALE, YAY Robot, Explore-until-Confident,
Losey & O'Malley). The no-fabricated-numbers discipline is preserved — every new table
is a `--` scaffold; the only measured number remains the 7.2%→97.5–100% data-yield fix.

## 4. Verification performed (all on this machine, 2026-07-04)

| Check | Result |
| --- | --- |
| Offline test suite (`run_tests.sh`) | **83/83 pass** (51 pre-existing + 32 new: feedback parser/sim-human/fusion, intent estimator/controller routing, multicam model/dataset/TTC) |
| Old-checkpoint compatibility | single-camera state dict keys unchanged; `strict=True` load passes; param count identical (1,928,263) |
| `collect_v4` Isaac smoke (2 episodes, headless) | 2/2 successful lifts; 156 ticks with paired overhead+wrist PNGs; `cameras.json` written with post-DR overhead pose + wrist extrinsics; **wrist view visually confirmed** (fingers at frame edge, target box centered ahead) |
| `verify_session` on the smoke session | OK — including the new wrist checks (156/156 wrist ticks, 2/2 cameras.json) |
| `collect_v3` regression (1 episode) | tick schema byte-identical to legacy (no `image_wrist`, no wrist files); episode succeeded |
| Two-camera training (`train_multicam.yaml`) | runs end-to-end on the smoke session; camera set + 15 Hz contract recorded in the checkpoint; loss decreases |
| Full-stack eval smoke (both cameras + `intent_allocator` + ambiguous instructions + scripted noisy human) | see note below |

*(Eval-smoke note: run was in flight when this report was written — its outcome is
appended at the bottom.)*

## 5. What was deliberately not changed

- `collect_v3.sh` / profile `vla_v1` defaults, the 15 Hz action contract, start pose,
  top-down camera framing, `eval.sh` behavior for existing configs, all existing
  controllers, the human-study/calibration suites, and every pre-existing test.
- `real_robot/` remains stubs (as before). The wrist config carries the documented
  hand-eye slot for when you calibrate the real camera.
- No git commits were made (per standing instructions).

## 6. Suggested experiment plan (maps 1:1 onto the shipped scripts)

1. **Data**: `NUM_EPISODES=120 ./vla_lab/scripts/collect_v4.sh --headless` (chunks of
   40; verify each chunk).
2. **Policies**: three runs of `train_multicam.yaml` with `cameras` = overhead / wrist /
   both (+ SmolVLA export `--wrist` if desired).
3. **R-camera** (supporting result): `sweep_camera_sets.sh` → value of each view;
   one-camera+gated ≈ both.
4. **R-intent** (headline): `eval.sh --controller intent_allocator
   --instruction-ambiguity {none,half,full}` vs `allocator`/`knowno`/`compute_gated` —
   show queries concentrate exactly where intent entropy is high and compute stops
   helping.
5. **R-quality** (headline): `sweep_feedback_quality.sh` → success vs human accuracy ×
   specificity; fusion-on vs trust-all shows the system covering for bad input.
6. **Calibration**: re-run `sweep_occlusion_eval.sh` (now with `--occlusion-camera`) +
   `fit_allocator.sh` so the fitted irreducibility model consumes the new features.

## 7. Known limitations / next steps

- The intent estimator's counterfactual sweep assumes the candidate set = the 6 colors;
  generalizing to open-vocabulary candidates needs a scene-grounded proposer.
- The feedback parser is rule-based by design (deterministic, testable); an LLM parser
  can swap in behind `parse_utterance` unchanged.
- `console` channel + real wrist camera on the physical Kinova still needs the
  `real_robot/` bridge implemented (unchanged stubs).
- Reliability is a single scalar per session; per-refinement-type reliability (targets
  vs nudges) is a natural extension.
- The smoke-trained checkpoint (2 episodes) is a plumbing test, not a model — collect
  the 120-episode set before drawing any curves.

---

## Appendix — full-stack eval smoke outcome (added 2026-07-04, evening session)

The run referenced in §4 has a small history. The first full-stack smoke **completed at
17:47** (`vla_lab/eval_results/smoke_multicam/`), but its records surfaced two issues;
both were fixed at 17:51 and a rerun was launched — which the session limit killed at
17:52, leaving `smoke_multicam_v2/` empty and this appendix unwritten. The rerun was
re-executed tonight with identical parameters and completed cleanly
(`vla_lab/eval_results/smoke_multicam_v2/results_1783213627.json`); the offline suite
was re-verified at **83/83** afterwards.

**Configuration** (mirrors run 1): 2 episodes × 900 steps, `smoke_multicam` checkpoint
(camera set = overhead+wrist, eval'd on both), `--controller intent_allocator`,
`--instruction-ambiguity full`, scripted human (accuracy 0.6, latency 3 ticks,
specificity color, seed 0), fusion with verify enabled, headless.

**The two fixes between run 1 and the rerun:**

1. **Episode records now carry `instruction_initial`** next to the final `instruction`.
   Run 1's records looked self-contradictory (episode 0: `"Pick up the purple box."`
   with `instruction_ambiguous: true`) because applied feedback rewrites `instruction`
   mid-episode by design — the record was showing the *post-refinement* string. Both are
   now logged (rerun ep 0: initial `"Pick up the box."` → final `"Pick up the red box."`).
2. **The intent estimator's counterfactual distances are normalized by their mean**
   (scale-free posterior): sharpness now depends on *relative* distance differences, not
   the raw action magnitudes of whatever model is loaded.

**What the rerun verified (plumbing — all green):** per-tick intent estimation ran on
every policy tick (57/57 × 2 episodes) with both camera streams; interjections were
parsed, fused, and verified through the same channel as robot queries; the Beta
reliability tracker updated online (0.667 → 0.5 → 0.4 → 0.5 across the four graded
utterances) and persisted across episodes; mid-episode instruction refinements re-encoded
tokens live; `camera_set_model` / `camera_set_eval` and the 15 Hz contract landed in the
results. Success was 0/2 with the target never displaced — expected for the 2-episode
plumbing checkpoint (§7), which mostly plateau-wanders under the step cap.

**Honest negative finding — the query branch did not fire** (`frac_query = 0.0`, both
runs). On this untrained checkpoint the behavioral posterior *saturates*: episode 1's
normalized intent entropy was ≈ 0.0006 under a fully colorless instruction. At softmax
temperature 0.1, any consistent relative gap in counterfactual distances is
winner-take-all, and geometric grounding of the policy's (arbitrary) commitment pins the
posterior further. Consequence: `tau_intent` (0.5) **and the estimator's temperature are
uncalibrated constants until a real checkpoint exists** — calibrate both on the
120-episode model before running R-intent (§6 steps 1–2). The routing logic itself is
covered offline (`test_intent_routed_controller_routes_by_source`), and the lexical
`instruction_ambiguous` flag (correctly 1.0 on every ambiguous tick) already flows to
`fit_allocator` as a fallback feature if behavioral entropy proves insufficient on the
trained model.

**A quality-model microcosm worth remembering for R-quality:** with a 0.6-accuracy
human, the verify path shares the human's noise. Rerun episode 1 (true target blue): the
human wrongly corrected to *"the red box"* → verify re-ask happened to recover *"blue"*
(applied, correct); later the human *correctly* said *"the blue box"* → the verify
re-ask drew *"red"* and overrode it (applied, wrong). This is not a bug — it is the
designed phenomenon `sweep_feedback_quality.sh` quantifies (fusion vs trust-all across
accuracy × specificity), and a concrete argument for the per-refinement-type reliability
extension noted in §7.

*(Also fixed tonight: `--controller`'s CLI help now lists `intent_allocator`.)*

### Paper realignment — actually executed tonight

While §3.6 described the paper realignment as done, **no file under `vla_lab/paper/` had
been modified since 2026-06-17** — the afternoon session wrote that section but never did
the work. It was executed tonight on the live root (`main.tex`, self-contained; the
`sections/` files belong to the legacy roots and were not touched):

- **Title/abstract/intro/contributions** now lead with uncertainty decomposed by SOURCE
  (perception vs intent), the two-way quality-modeled channel, and camera ablations as
  supporting evidence. New title: *"Act, Compute, or Ask: Routing VLA Uncertainty to Its
  Source — Perception or Human Intent — in Human-Robot Teaming."*
- **Theory**: new Corollary (intent defeats sensors too — the data-processing argument
  transfers verbatim when the missing variable is the goal) + a remark on why source ≠
  magnitude.
- **Method**: multicam contract paragraph; cross-view disagreement; new §5.4 intent
  estimator (incl. the scale-free normalization and the honest calibration caveat from
  tonight's smoke); source-routed escalation with intent-first Rule 1 and an updated
  Algorithm 1; new §5.6 feedback/fusion (parser, sim-human quality knobs, Beta
  reliability, verify-shares-the-noisy-channel, trust-all ablation); per-view occlusion.
- **Study**: IV3 instruction-ambiguity, source-routed controller conditions, H1/H2
  extended by source, new H5 (quality-aware fusion); measures extended (typed query
  rates, fusion outcome mix, reliability trajectory).
- **Results scaffold**: three new all-`--` template tables — R-camera (incl. the
  "one camera + allocation vs two" legs), R-intent (typed query rates + ambiguous-episode
  query precision), R-quality (accuracy × specificity × fusion-vs-trust-all, with
  harmful-apply and wrong-verify-override columns). Status inventory updated (83 tests,
  wrist pipeline verified). No numbers anywhere — the yield table remains the only
  measured result.
- **Related work + positioning**: the 2026 near-misses woven in; a new subsection on
  mid-execution corrections and input quality; the "unclaimed combination" stated.
- **References**: 8 new entries (48→56), **each verified against its arXiv abstract page
  tonight** — TRIAGE 2603.08128, BOKBO 2605.30660, PACT 2605.24350, Assistron
  2606.23147, LIBERO-Occ 2606.10862, YAY Robot 2403.12910, Losey & O'Malley 1806.02454,
  Explore-until-Confident 2403.15941.
- **Appendices**: contract table (wrist row, `camera_present`), TinyVLA multicam
  paragraph, evaluator defaults (all new flags + τ defaults), OOD axes (camera set,
  ambiguity, input quality), module inventory (intent/, feedback/, 83 tests), repro
  checklist/statement (schema `vla_lab_eval/v3`, `cameras.json`, `camera_set` in ckpts).
- **Build verified**: `latexmk -pdf` clean — 27 pages, zero undefined references, zero
  overfull boxes, all 56 bib keys resolve. (The draft grew from 18 pp; the header note
  about moving material to the appendix for HRI's page limit stands.)

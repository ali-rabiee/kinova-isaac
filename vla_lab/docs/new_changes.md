# VLA-TTC → CoRL 2026: Strategic Pivot & Codebase Changes

**Author note:** Honest assessment, no sugar-coating. Today is **May 8, 2026**. CoRL 2026 abstract deadline: **May 25, 2026**. Full paper deadline: **May 28, 2026 EOD**. You have ~20 days. The current draft would be desk-rejected. This document specifies what has to change to make a real submission, in priority order.

---

## 0. The hard truth about the competitive landscape

Before any code changes, internalize what you're competing against. The TTS-for-VLA space went from "open problem" to "crowded subfield" in the last 12 months:

| Paper | Date | Method | Verifier? |
|---|---|---|---|
| Steering Your Generalists (Nakamoto et al.) | CoRL 2024 | Value-guided action selection | Trained value fn |
| RoboMonkey (Kwok et al.) | 2025 | Best-of-N + external verifier | Trained verifier |
| SCALE (Choi et al.) | 2025 | Self-uncertainty conditioned looking/execution | Internal |
| **MG-Select** (arXiv 2510.05681) | Oct 2025 | KL-div from masked reference distribution | **Verifier-free** |
| **RoVer** (arXiv 2510.10975) | Oct 2025 | Process reward model + direction-guided sampling | Trained PRM |

**Implication:** Your current pitch ("K Gaussian samples + median consensus, verifier-free") is a strict subset of what MG-Select did seven months ago, with a weaker selection criterion. You cannot submit this framing. Period.

You need a **distinguishing axis** that those papers did not address. The good news: your hardware constraints give you exactly one.

---

## 1. The strategic pivot: what story to tell

### 1.1 Reject the current framing

Drop these from the narrative:
- "We propose TTC for VLAs" — done
- "Verifier-free is novel" — MG-Select did it
- "TinyVLA as the main model" — non-competitive, the authors themselves admit it
- "Engineering specification + roadmap" — not a research contribution
- "Bottleneck Gaussian noise + median consensus" — too simple, no defense

### 1.2 Adopt a defensible angle

Your hardware constraint (Kinova Gen 2 with **only an external/overhead camera, no wrist cam**) is a genuinely under-explored regime in the VLA TTS literature. Almost every recent VLA paper assumes a wrist camera (OpenVLA, π0, SmolVLA, RDT — all multi-view). Single-camera manipulation has fundamentally higher action-space ambiguity from depth and occlusion. **TTS should help more here, and nobody has shown that.**

**Recommended title direction (pick one, refine):**

> **"Test-Time Scaling Compensates for Sensory Deficit: Single-Camera VLA Manipulation Without a Wrist View"**

> **"When Test-Time Compute Helps Most: Disambiguating Partial Observability in Compact VLAs"**

> **"From Sim to Real on a Single Camera: Test-Time Scaling for Compact VLAs Under Partial Observability"**

### 1.3 The thesis (one sentence)

> *Test-time scaling provides disproportionate gains for VLAs operating under partial visual observability — a regime that closely matches real-world deployments on cost-constrained hardware — and we demonstrate this through controlled ablations on a real Kinova Gen 2 manipulator using a fine-tuned SmolVLA.*

### 1.4 Why this beats the existing TTS-VLA papers

- **MG-Select / RoVer / RoboMonkey** all evaluate on benchmarks with full multi-view observation (LIBERO, RoboCasa, real arms with wrist cams). They never isolate the partial-observability axis.
- **You can claim a measurement contribution**: "TTS gain scales with observation ambiguity" — and back it with a controlled ablation (camera ablation: full multi-view → external-only → external + occluded).
- **You have a real Kinova arm.** That alone elevates this above sim-only papers.

### 1.5 Three concrete contributions to claim

1. **An empirical scaling law:** TTS-N (success rate) as a function of observation completeness on identical tasks. Show that ΔSR(N=8) − ΔSR(N=1) is larger for single-camera than for multi-view.
2. **A method tuned for the partial-observability regime:** lightweight uncertainty-gated TTS that triggers more samples only when the model is uncertain (saves compute when unneeded). Build on top of an open VLA (SmolVLA), don't invent your own backbone.
3. **Real-world validation on Kinova Gen 2** with a single overhead RGB camera. Sim (Isaac Lab) + real ablations on the same task family.

This is a plausible CoRL paper. It's not a top-paper-award paper, but it's accept-shaped.

---

## 2. Architecture changes

### 2.1 What to keep

- **Isaac Lab `reach_to_grasp_VLA` environment** — your sim infrastructure is fine
- **Scripted data collection (`collect_v3.sh`)** — useful for sim demos
- **Logging schema (`ticks.jsonl`, `instruction.json`, etc.)** — fine, stable
- **Wilson-CI plotting tooling** — reviewers will appreciate this
- **`smolvla_bridge/`** — this becomes the main path, not an afterthought

### 2.2 What to demote

- **TinyVLA** — keep ONLY as a fast iteration / sanity-check artifact. Drop from the paper as "the method" or "a baseline." Nobody cares about a 1.93M-param model in a CoRL submission. Mention in one line in the appendix as a unit-test backbone.
- **Bottleneck Gaussian + median consensus** — keep as a *baseline you compare against*, not as the contribution.

### 2.3 What to add (in priority order)

#### Priority 1: SmolVLA as the primary policy
- Use `lerobot/smolvla_base` (450M params, fits in 12GB easily)
- Fine-tune on Kinova Gen 2 LeRobot-format data
- This is not optional. A submitted VLA paper without a real VLA backbone will fail review.

#### Priority 2: Uncertainty-gated TTS

The selection criterion has to be more sophisticated than median proximity. Options, ranked by feasibility in 20 days:

1. **Action-token entropy gating** (easiest): if entropy of first-token distribution > threshold, sample N candidates; else use N=1. Aggregate via either MG-Select-style KL or via consistency.
2. **Flow-matching trajectory variance** (medium): SmolVLA uses flow matching for the action expert. Run K parallel flow trajectories from different noise seeds; gate on terminal variance.
3. **VLM-as-verifier** (hard but novel): use a small VLM (e.g., SmolVLM2-256M) as a zero-shot scorer asking "does this end-effector trajectory match the instruction?" — risky, needs careful prompting, but very different from prior work.

Recommend (1) for the sprint, leave (3) as an appendix experiment if you have time.

#### Priority 3: Real Kinova Gen 2 deployment
- Single overhead RGB camera (you already have this constraint)
- ROS bridge to LeRobot policy server (LeRobot supports this; see `lerobot/scripts/control_robot.py`)
- 5–10 task variants (object color, position, distractor count)
- Success criterion: lift ≥ 6 cm + maintained grasp for 2 s

#### Priority 4: A controlled observability ablation

This is the *measurement contribution*. On the same set of episodes, evaluate:
- **Multi-view (sim only):** add a virtual wrist cam to compare
- **External-only:** your real-robot regime
- **External + adversarial occlusion:** drop a virtual occluder over part of the workspace

Show the TTS gain widens as observability shrinks. This plot is the one reviewers will remember.

---

## 3. Experimental setup that CoRL will accept

### 3.1 Mandatory: real-robot results

CoRL's call for papers states explicitly: *"Authors are encouraged to report real-robot experiments or provide convincing evidence that simulation experiments are transferable to real robots."* In practice, sim-only VLA papers are usually rejected unless they introduce a new sim benchmark. You have a Gen 2. Use it.

Minimum real-robot protocol:
- ≥ 30 episodes per condition (gives Wilson 95% CI half-width ≈ ±18% at 50% SR — barely tolerable; 50 episodes is better)
- ≥ 3 conditions (e.g., ID, novel object color, novel object shape)
- ≥ 3 seeds per condition for statistical claims
- Report per-episode latency (median + p95) — this matters for TTS papers

### 3.2 Recommended task suite

Stay narrow. Don't try to cover pick-and-place + insertion + pouring. Pick **one task family** and study it deeply:

- **Primary:** pick-and-lift colored block, instruction-conditioned ("pick up the red block")
- **OOD-1:** novel object colors not in training
- **OOD-2:** distractor objects (1-3 extra blocks of training colors)
- **OOD-3:** position generalization (held-out workspace regions)

This is enough for an 8-page paper.

### 3.3 Baselines you must compare against

| Baseline | Why |
|---|---|
| SmolVLA fine-tuned, N=1 | The "no TTS" anchor |
| SmolVLA + naive Best-of-N (N=4, 8, 16) | Prior-art TTS without verifier |
| SmolVLA + MG-Select (re-implement) | The most direct competitor |
| **Yours: SmolVLA + uncertainty-gated TTS** | Your method |
| TinyVLA + median consensus (sim only) | Honesty about your prior work |

If you skip MG-Select reproduction reviewers will hammer you. Allocate 2 days for this.

### 3.4 Ablations to run

- **K** sweep: {1, 2, 4, 8, 16}
- **Gating threshold:** {off, low, med, high}
- **Camera ablation** (sim): full multi-view vs. external-only vs. external + occluder
- **Demo count:** {50, 100, 200, 500} → SR curve

### 3.5 Statistical reporting

You already have Wilson intervals — use them. For paired comparisons (same scene seeds, different methods), use **McNemar's test on success bits**. Reviewers will explicitly look for this.

---

## 4. File-by-file codebase changes

Files referenced relative to `vla_lab/` unless noted.

### 4.1 Files to delete or heavily demote

| File | Action |
|---|---|
| `vla_ttc_engineering_spec.md` | Move out of the paper appendix. Keep in repo as documentation. |
| Most of `models.py` (TinyVLA classes) | Keep, but no longer the headline. |
| `paper/` draft markdown sections that read as repo docs | Strip — paper should not enumerate CLI flags or YAML keys in the body. |

### 4.2 Files needing major changes

#### `dataset.py`
- Add a **LeRobot-format export path** in addition to your current Kinova-session reader
- Ensure deterministic train/val splits at episode level (you have this)
- Action chunk semantics MUST match SmolVLA's (delta EE pose + gripper, T=8 chunks → check SmolVLA model card)
- Add image normalization that matches SmolVLA's vision encoder (SigLIP-style, not raw [0,1])

#### `train.py`
- Stop training TinyVLA as the headline; instead, write a `train_smolvla.py` wrapper that calls `lerobot-train` with your dataset
- Keep TinyVLA training for fast iteration (debug only)
- Add resume-from-checkpoint properly (you have this, verify with SmolVLA)

#### `ttc.py`
- Replace median-consensus with at least three selection methods, swappable via config:
  1. `naive_consensus` (current)
  2. `mg_select_kl` — re-implementation of KL-div from masked reference, applied to SmolVLA action tokens
  3. `entropy_gated` — your contribution; only sample N>1 when first-token entropy > τ
- Add a `latency_log` argument that records per-call timing (forward, score, total)

#### `eval_isaaclab.py`
- Make policy backend selection truly first-class via `--policy-backend {tiny,smolvla}` (you partially have this)
- Add `--occlusion-mask` flag for the camera-ablation experiment
- Emit results in a schema compatible with `lerobot-eval` so plots are unified

#### `losses.py`
- DINOv2 alignment is a distraction at this stage. Disable by default. If you want to keep the alignment story, make it an appendix experiment, not in the main paper.

### 4.3 New files to add

#### `real_robot/kinova_bridge.py` (NEW — critical)
- ROS or pyKinova bridge from your LeRobot policy server to the Gen 2
- Subscribes to camera topic; publishes joint commands
- Sample structure:
```python
class KinovaPolicyServer:
    def __init__(self, policy, camera_topic, control_rate_hz=5):
        self.policy = policy  # SmolVLA wrapped via policy_wrapper.py
        self.camera = ROSCameraSubscriber(camera_topic)
        self.controller = KinovaController()  # IK + safety bounds

    def step(self, instruction: str):
        img = self.camera.latest()
        state = self.controller.eef_state()
        action_chunk = self.policy(img, state, instruction)  # [T, 7]
        for a in action_chunk:
            self.controller.execute_delta(a)
```

#### `real_robot/safety_envelope.py` (NEW — required)
- Workspace bounds (XYZ AABB)
- Velocity limits
- Force/torque cutoffs
- E-stop integration
- **Reviewers will ask about safety. Have an answer.**

#### `vla_lab/baselines/mg_select.py` (NEW)
- Re-implementation of MG-Select (Nakamoto-style masked reference + KL selection)
- This is your strongest direct competitor; you cannot skip it

#### `vla_lab/ttc_methods/` (NEW directory)
- `entropy_gate.py` — your gating contribution
- `flow_variance.py` — flow-matching variance scoring (if you go that route)
- `vlm_verifier.py` — optional, appendix-only

#### `vla_lab/scripts/run_real_eval.sh` (NEW)
- Shell wrapper for the real-robot evaluation, mirroring `eval.sh`

#### `vla_lab/configs/eval_real.yaml` (NEW)
- Real-robot eval config; bounds, rates, safety, camera intrinsics

### 4.4 Files to keep largely as-is
- `dryrun.py`, `inspect_data.py`, `plot_metrics.py` — keep
- `_path.py` — keep
- Collection profiles `vla_v1`/`vla_v2` — keep (sim training data still useful)

---

## 5. Realistic timeline (May 8 → May 28)

This sprint assumes:
- Kinova Gen 2 is set up, calibrated, and you can already log demos via teleoperation or scripted controllers
- SmolVLA has been fine-tuned at least once on something
- You can dedicate ~10 hr/day to this

### Week 1 (May 8 – May 14): Foundation

| Day | Task |
|---|---|
| **May 8 (today)** | Decide pivot. Lock the thesis sentence. Stop writing the current draft. |
| May 9 | Real-robot data collection rig: teleop + camera sync. Target 50 demos by EOD. |
| May 10 | Continue collection. Convert to LeRobot format via `convert_kinova_to_lerobot.py`. |
| May 11 | First SmolVLA fine-tune run. Baseline N=1 eval on real robot. |
| May 12 | Implement entropy-gated TTS. Re-run eval with K∈{2,4,8}. |
| May 13 | Implement MG-Select baseline. |
| May 14 | First full eval pass: SmolVLA-N1, SmolVLA-BoN, MG-Select, Yours. ID condition only. |

### Week 2 (May 15 – May 21): Experiments

| Day | Task |
|---|---|
| May 15 | OOD-1 (novel colors) eval, all methods. |
| May 16 | OOD-2 (distractors) eval, all methods. |
| May 17 | Camera-ablation experiment (sim). This is the headline plot. |
| May 18 | Latency profiling, ablation on K, gating threshold. |
| May 19 | Buffer day for re-runs, fixing bugs found mid-experiment. |
| May 20 | Plot generation. All figures locked. |
| May 21 | Buffer day. |

### Week 3 (May 22 – May 28): Writing

| Day | Task |
|---|---|
| May 22 | Switch to CoRL PMLR template. Outline 8 pages. Start writing intro + method. |
| May 23 | Write experiments + results sections with real numbers. |
| May 24 | Write related work, limitations, conclusion. |
| **May 25 (Mon)** | **Abstract submission deadline (11:59 UTC).** Submit polished abstract. |
| May 26 | Internal review pass. Co-author comments. |
| May 27 | Polish, fix figures, re-check claims against numbers. Prepare appendix. |
| **May 28 (Thu) EOD** | **Full paper deadline.** Submit. Submit code/video as supplementary. |

### Brutal honesty about this timeline

If any of the following are not already true, the timeline collapses:
- SmolVLA fine-tuning on your Kinova data has been verified end-to-end at least once
- Real-robot teleop / scripted demo collection is operational
- Camera-policy synchronization is solved (no clock drift)
- You can run ≥ 30 real episodes/day with reasonable reset overhead

If two or more of these are not true, **target a workshop, not the main track.** CoRL workshop deadlines typically fall in July-August (announced after main-track decisions); they accept 4-page papers with preliminary results, and many become full papers at the next year's main track.

---

## 6. Backup plans

### 6.1 If main CoRL doesn't work: CoRL workshops

CoRL hosts ~10-15 workshops each year. Likely-relevant ones for 2026:
- Workshop on Foundation Models for Robotics
- Workshop on Embodied Reasoning / VLA scaling
- Workshop on Sim-to-Real

These typically open ~July with deadlines in August. 4-page format. Real-robot results still required for top workshops.

### 6.2 If you can't get real-robot in time

Fall back to: **sim-only with strong sim-to-real transferability evidence**. This means:
- Photorealistic Isaac Lab rendering (already have)
- Domain randomization at training time
- A *small* real-robot validation (10 episodes, 1 condition) showing the sim policy works on hardware. Even 10 episodes of "it didn't immediately fail" is worth a section.

### 6.3 Longer-term targets
- **ICRA 2027** — deadline ~September 2026. Better venue for "system + experiments" papers.
- **CoRL 2027** — deadline ~June 2027. The right target if you want to do this properly: full real-robot study, ≥ 100 episodes per condition, multiple task families.
- **arXiv now** — independent of submission, post the cleaned-up version on arXiv as a tech report. Builds credibility, lets you cite it.

---

## 7. Things to STOP doing immediately

1. **Stop polishing the current draft.** The framing is wrong. New draft, new structure.
2. **Stop adding planned features to the engineering spec.** Roadmaps don't get into CoRL papers.
3. **Stop training TinyVLA as the headline.** It's a tool, not a contribution.
4. **Stop writing CLI flag tables in the paper body.** Move to appendix or repo README.
5. **Stop using placeholder citations.** Every reference must be real before submission.

---

## 8. Things to START doing immediately

1. **Today:** Write the new abstract (250 words). If you can't write it, you don't have a paper. The abstract should answer: What problem? Why does it matter? What did you do? What did you find? What's the takeaway?
2. **Tomorrow:** Get one real-robot demo loop running end-to-end, even crudely. Camera → policy → arm → success/fail logging.
3. **Day after:** Lock the task suite. No scope changes after May 12.
4. **Set up a daily 30-min standup with yourself**: what ran, what broke, what's the unblocker for tomorrow.

---

## 9. Final risk register

| Risk | Probability | Mitigation |
|---|---|---|
| SmolVLA fine-tune doesn't converge on Kinova data | Medium | Pre-test on a tiny subset before May 11. Fallback: OpenVLA-Mini or stay with TinyVLA + downscope claims. |
| Real-robot setup time eats into experiments | High | Start collection today, not after method works. Parallelize. |
| MG-Select reimplementation is buggy | Medium | Use the authors' code if released; if not, contact them. |
| TTS gains under partial observability are not larger than under full observability | **Real risk** | Negative result is still publishable if measured well. Re-frame as "TTS is observation-invariant." |
| Reviewer asks for comparison to RoVer | High | At least cite + discuss in related work. If time permits, run a verifier-based variant. |
| Submission requires a qualifying author-reviewer | Yes | Confirm at least one author has prior CoRL/ICRA/IROS/RSS/ICLR/NeurIPS/ICML accept. |

---

## 10. TL;DR

**The current paper:** No real results, weak novelty, non-competitive backbone, references half-placeholder. Desk-reject probability ~95%.

**The pivot:** Reframe around *single-camera partial-observability* as the hardware-given novelty axis. SmolVLA as the backbone. Uncertainty-gated TTS as the method. Real Kinova Gen 2 results as the credibility anchor. Camera-ablation as the headline measurement.

**The deadline:** May 28, 2026, 11:59 EOD. 20 days from today. Possible if you start the real-robot work now, not after the method is "perfect."

**The backup:** Workshop track or ICRA 2027 if the sprint fails. Don't burn yourself out trying to make a flawed Phase-1 paper work — pivot to the right paper or accept a later venue.
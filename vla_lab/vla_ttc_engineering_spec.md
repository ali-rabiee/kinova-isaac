# VLA-TTC: Engineering specification for CoRL 2026 submission

**Project codename:** `vla-ttc` (Vision-Language-Action with Test-Time Computation)
**Hybrid:** Direction 1 (test-time visual intervention + sample-and-verify) + the feature-alignment loss component from Direction 3 used during fine-tuning so the in-distribution baseline is strong.
**Primary base model:** SmolVLA (0.45B). **Secondary baseline:** π0 with LoRA at 4-bit.
**Hardware contract:** train on V100 cluster, infer on a single 12 GB 4090 laptop, single external camera, Kinova Gen2 7-DOF + 1 gripper.
**Deadline:** May 28, 2026 AoE for CoRL 2026 conference; workshop fallback at ~mid-July.

This document is written so each module is a self-contained "build this" prompt that you can hand to Claude Code or Cursor (or implement yourself). I have flagged every place where a quick early-week sanity check can save you a week later.

---

## 0. Top-level project description (use this as your AI assistant system prompt)

> You are helping build `vla-ttc`, a research codebase for a CoRL 2026 paper. The contribution is a test-time computation pipeline for small Vision-Language-Action (VLA) models that runs on a single 12 GB GPU and uses a single external camera. The base model is SmolVLA (0.45B, flow-matching action expert). The pipeline does three things at inference: (1) parses the scene with a small VLM and masks task-irrelevant regions, (2) samples K candidate action chunks by perturbing the flow-matching initial noise, (3) scores each candidate with a learned verifier and executes the best one. An OOD detector triggers the expensive path only when needed so average latency stays low. During fine-tuning, a feature-alignment loss against frozen DINOv2-base teacher features is used to preserve the VLM's pretrained visual semantics. The validation robot is a Kinova Gen2 doing pick-and-place with novel objects, distractors, and lighting variations. Code must be reproducible (configs in YAML, deterministic seeds, version pins), well-logged (Weights & Biases), and modular so individual components can be ablated. Always preserve the ability to run components independently. Optimize for clarity and speed of iteration over micro-optimization.

---

## 1. VRAM and latency budget — verify this on day 3 before writing anything else

This is the constraint that kills the project if you ignore it. Do a "dry-fit" on day 3.

| Component | Peak VRAM (target) | Latency target | Notes |
|---|---|---|---|
| SmolVLA base, bf16, 1 sample, async | 3.5 GB | 30–60 ms per action chunk | Verified by HF docs |
| K=4 action samples (parallel) | +1.5 GB | +0 ms | Same forward, K noise vectors |
| K=8 action samples | +3.5 GB | +0 ms | May force sequential; budget for it |
| Scene parser VLM (Qwen2.5-VL-3B int4) | 4–4.5 GB | 200–400 ms | Run every N=5–10 steps, not every step |
| SAM2-base for mask refinement | 1.2 GB | 30 ms | Keep loaded |
| Verifier (~10 M params, fp16) | 0.1 GB | 5 ms × K | Negligible |
| OOD detector (RND, ~5 M params) | 0.1 GB | 3 ms | Negligible |
| DINOv2-base (only used at train time) | — | — | Not on inference machine |
| **Sum at inference** | **~9–10 GB** | **~250–500 ms when triggered, 60 ms when not** | Leave 2 GB headroom |

**Day-3 sanity script:** `scripts/vram_dryrun.py` loads SmolVLA + Qwen2.5-VL-3B-int4 + SAM2 + a dummy verifier on the 4090, runs 100 forward passes, and prints peak VRAM + p50/p95 latency. **If this exceeds 11 GB or 800 ms, you have one of three options: drop SDXL inpainting entirely (use SAM2 + median fill), drop the scene parser to SmolVLM2-2.2B, or run scene parsing every N=15 steps instead of 5.**

---

## 2. Repository architecture

```
vla-ttc/
├── README.md
├── pyproject.toml                  # uv or poetry; pin exact versions
├── .python-version                 # 3.11 (LeRobot supports)
├── configs/
│   ├── data/
│   │   ├── kinova_real.yaml        # Real-robot dataset config
│   │   └── kinova_isaacsim.yaml    # Sim dataset config
│   ├── train/
│   │   ├── smolvla_base.yaml       # Vanilla LoRA fine-tune
│   │   ├── smolvla_align.yaml      # + feature alignment (D3 component)
│   │   └── pi0_lora.yaml           # Secondary baseline
│   ├── ttc/
│   │   ├── pipeline_full.yaml      # All TTC components on
│   │   ├── pipeline_no_inpaint.yaml
│   │   ├── pipeline_no_verifier.yaml
│   │   └── pipeline_no_trigger.yaml
│   └── eval/
│       ├── id_eval.yaml
│       ├── ood_objects.yaml
│       ├── ood_distractors.yaml
│       └── ood_lighting.yaml
├── src/vla_ttc/
│   ├── __init__.py
│   ├── data/
│   │   ├── collection/
│   │   │   ├── kinova_teleop.py
│   │   │   ├── isaacsim_collector.py
│   │   │   └── lerobot_writer.py
│   │   ├── augmentation/
│   │   │   ├── background_swap.py    # SDXL-Turbo at train time
│   │   │   └── color_jitter.py
│   │   └── ood_specs.py              # Defines OOD test conditions
│   ├── models/
│   │   ├── smolvla_wrapper.py        # Loads SmolVLA, exposes hooks
│   │   ├── pi0_wrapper.py
│   │   ├── teachers/
│   │   │   └── dinov2.py
│   │   └── feature_hooks.py          # For mid-layer feature taps
│   ├── train/
│   │   ├── finetune.py               # Main entry, dispatches to model
│   │   ├── losses/
│   │   │   ├── flow_matching.py      # SmolVLA's standard loss
│   │   │   └── feature_alignment.py  # D3 hybrid component
│   │   ├── trainer.py
│   │   └── slurm_launcher.py
│   ├── ttc/
│   │   ├── pipeline.py               # Orchestrates inference
│   │   ├── scene_parser.py           # Qwen2.5-VL or SmolVLM2
│   │   ├── inpainter.py              # SAM2 + median fill (fast); SDXL-Turbo (optional)
│   │   ├── action_sampler.py         # Flow-noise perturbation, K samples
│   │   ├── verifier/
│   │   │   ├── model.py
│   │   │   ├── synthetic_pairs.py
│   │   │   └── train_verifier.py
│   │   ├── ood_detector.py           # RND (FIPER-style)
│   │   └── trigger.py                # Conformal threshold logic
│   ├── robot/
│   │   ├── kinova_ros_bridge.py
│   │   ├── action_executor.py
│   │   ├── safety_monitor.py
│   │   └── reset_helpers.py
│   ├── eval/
│   │   ├── real_robot_runner.py
│   │   ├── libero_runner.py
│   │   ├── ood_grids.py
│   │   ├── metrics.py
│   │   └── statistical_tests.py      # Wilson intervals, paired bootstrap
│   ├── analysis/
│   │   ├── plots.py
│   │   ├── tables.py
│   │   ├── failure_taxonomy.py
│   │   └── latency_profiler.py
│   └── utils/
│       ├── seeding.py
│       ├── logging.py                # W&B + structured JSON
│       └── config.py                 # Hydra/OmegaConf loaders
├── scripts/
│   ├── vram_dryrun.py                # Day-3 sanity check
│   ├── collect_demos.sh
│   ├── launch_finetune.sh            # SLURM submission
│   ├── launch_verifier_train.sh
│   ├── deploy_4090.sh                # Sets up inference env
│   ├── run_eval_matrix.sh            # Full eval grid
│   └── make_paper_figures.py
├── tests/
│   ├── unit/
│   │   ├── test_dataset_format.py
│   │   ├── test_flow_perturbation.py
│   │   ├── test_alignment_loss.py
│   │   └── test_ood_calibration.py
│   └── integration/
│       ├── test_pipeline_smoke.py
│       └── test_robot_loop_dryrun.py
├── notebooks/
│   ├── 01_demo_inspection.ipynb
│   ├── 02_baseline_metrics.ipynb
│   └── 03_failure_analysis.ipynb
└── paper/
    ├── main.tex                      # CoRL template
    ├── figures/
    └── tables/
```

---

## 3. Dependencies and environment pinning

Pin everything. The LeRobot/SmolVLA stack churns weekly.

```toml
# pyproject.toml essentials
[project]
requires-python = ">=3.11,<3.12"
dependencies = [
    "lerobot==0.x.y",                  # Pin to whatever is current at project start
    "transformers>=4.45,<4.50",
    "torch==2.4.1",
    "torchvision==0.19.1",
    "accelerate>=1.0",
    "peft>=0.13",                      # LoRA
    "bitsandbytes>=0.44",              # 4-bit for π0
    "diffusers>=0.30",                 # If using SDXL-Turbo for aug
    "omegaconf>=2.3",
    "hydra-core>=1.3",
    "wandb>=0.18",
    "rerun-sdk>=0.18",                 # Optional, great for robot debugging
    "open3d>=0.18",                    # If you need depth visualization
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "matplotlib>=3.9",
    "seaborn>=0.13",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "pytest>=8.0",
    "ruff>=0.6",
    "rospkg",                          # For Kinova ROS
]
```

Robot-side: ROS Noetic or ROS2 Humble depending on your existing Kinova setup. Keep robot code in a separate Python env if necessary — do not entangle ROS with the training env.

VLM/inpainting deps live in a separate optional group: `pip install vla-ttc[ttc]` installs Qwen-VL, SAM2, etc.

---

## 4. Module-by-module specifications

Each subsection below is structured: **Goal → I/O → Implementation notes → Build prompt → Tests**. The build prompt is what you paste into Claude Code/Cursor.

### 4.1 Kinova teleop and demo collection

**Goal.** Collect 120 real-robot demos in LeRobot dataset format across 3 pick-and-place tasks. Use spacemouse or keyboard teleop.

**I/O.** Input: human teleop. Output: a LeRobot-format HuggingFace dataset on disk with per-step `(rgb_image: 224x224x3, state: 7-DoF joint pos + gripper, action: 7-DoF target joint pos + gripper, language: str, task_index: int, episode_index: int, frame_index: int, timestamp: float)`.

**Implementation notes.**
- Standardize action space to **delta end-effector pose + gripper** (8-D total). Joint-space actions on Kinova Gen2 will fight you because of its 6-DoF/7-DoF variants and joint encoder noise. Use Kinova's IK in `kinova-ros`.
- Camera: capture at native resolution then center-crop and resize to 224×224 for SmolVLA. Do not use a wide-angle if avoidable — Kinova workspace is small and warping hurts.
- Sample at 15 Hz (SmolVLA's deployment rate). If your camera is 30 Hz, decimate.
- Three tasks, 40 demos each: `pick_red_block_place_in_bin`, `pick_blue_cup_place_on_plate`, `pick_yellow_banana_place_in_bin`. Vary start positions but keep object identity fixed in-distribution.

**Build prompt.**
> Implement `src/vla_ttc/data/collection/kinova_teleop.py`. It should: connect to the Kinova Gen2 over ROS using `kinova-ros`, subscribe to a single RGB camera topic at 15 Hz, accept teleop commands from a 3DConnexion SpaceMouse via `pyspacemouse` (fall back to keyboard with WASD+QE for translation, IJKL for rotation, space for gripper toggle), record episodes to disk in LeRobot dataset format using `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.create()`. Each episode is bookended by `r` (reset) and `s` (save) keypresses. Action space is 8-D: delta end-effector pose (3 trans + 3 rot in axis-angle) + gripper open/close binary. Include a CLI: `python -m vla_ttc.data.collection.kinova_teleop --task pick_red_block_place_in_bin --num-episodes 40 --out-dir data/kinova_real/`. Add a sanity-check mode that replays the last episode without robot motion to verify recording. Use `rerun` for live visualization of state and image.

**Tests.**
- `test_dataset_format.py`: load a recorded episode, assert all keys present, image shape == (224, 224, 3), action norm reasonable.
- Manual: replay 1 episode in IsaacSim to confirm action format is consistent.

### 4.2 IsaacSim data generator (use your existing codebase)

**Goal.** Generate 200–500 sim demos with scripted scripts to (a) bootstrap the verifier's training data and (b) provide a sim-to-real ablation. **You said you have this already — wire it into LeRobot format.**

**Build prompt.**
> Take the existing IsaacSim Kinova codebase and add a wrapper `src/vla_ttc/data/collection/isaacsim_collector.py` that runs scripted scripted pick-and-place policies (use motion-planning IK in IsaacSim) and writes demos in the same LeRobot format as `kinova_teleop.py`. Match the camera intrinsics and field of view to the real external camera as closely as possible — measure the real camera's FOV with a checkerboard before this step. Generate 200 demos for the same 3 tasks plus 200 demos with **action perturbations** (Gaussian noise with σ=0.05 on each action component) for verifier hard-negative training data. Output to `data/kinova_sim/` and `data/kinova_sim_perturbed/` respectively.

**Why this matters for the project.** The verifier needs hard negatives. Real-robot perturbed rollouts are dangerous and slow. Sim-perturbed rollouts are free. **This is one of the strongest reasons your existing IsaacSim codebase is a project accelerator.**

### 4.3 OOD test set specification

**Goal.** Define the OOD axes precisely so reviewers can replicate.

**Build prompt.**
> Implement `src/vla_ttc/data/ood_specs.py` as a typed config that enumerates evaluation conditions:
>
> - `ID`: training objects, no distractors, training lighting.
> - `OOD-Objects-L1`: 5 unseen objects of similar shape/size to training (e.g., green block, white cup, orange banana, purple block, pink cup). 15 trials per object per task = 225 trials per method.
> - `OOD-Distractors-L1`: training objects but with 2 randomly-placed unseen objects on the table.
> - `OOD-Distractors-L2`: training objects with 4 distractors.
> - `OOD-Lighting`: training objects, training distractors, but lighting changed: warm 2700K, cold 6500K, dim (~30% brightness). One condition tested per trial.
>
> Output: a YAML file plus a Python iterator that yields `(condition_name, scene_setup_instructions, language_prompt)` tuples for the eval harness to consume. Include physical setup checklists per condition (object list, placement diagram, light meter readings).

**Sample size justification:** 15 trials per (method × condition × task) gives a 95% Wilson interval half-width of about ±13 percentage points around 50%, which is the published norm. Pre-register the eval matrix to avoid p-hacking accusations from reviewers.

### 4.4 SmolVLA wrapper with feature hooks

**Goal.** Wrap SmolVLA so we can (a) tap mid-layer features for the alignment loss, (b) inject perturbed flow noise for sampling, (c) run async inference in the TTC pipeline.

**I/O.** Input: image, language, state. Output: action chunk + (optional) intermediate visual features + (optional) the flow-matching latent at t=0.

**Implementation notes.**
- Use LeRobot's `SmolVLAPolicy`. Inspect the model and identify a mid-layer of the SmolVLM-2 base where feature alignment makes sense. The "Don't Blind Your VLA" paper aligns at the visual encoder output (post-projector). Start there; ablate one earlier and one later layer.
- For flow-noise perturbation: SmolVLA's action expert takes Gaussian noise as the t=0 latent. Expose a `sample_actions(obs, lang, noise=None, num_inference_steps=10)` method where `noise` can be a (K, T, action_dim) tensor of K different starting noises; runs K parallel flow-matching trajectories.
- Use forward hooks (`module.register_forward_hook`) for feature extraction so the alignment loss is decoupled from the model code.

**Build prompt.**
> Implement `src/vla_ttc/models/smolvla_wrapper.py`. The class `SmolVLAWrapper` wraps `lerobot.common.policies.smolvla.modeling_smolvla.SmolVLAPolicy` and provides:
>
> 1. `forward_with_features(batch) -> (loss, features_dict)`: standard forward pass that also returns mid-layer features from the visual encoder (post-projector layer by default; configurable via `feature_layer: str` in config).
> 2. `sample_actions(obs, lang, noise: Optional[Tensor]=None, k: int=1) -> Tensor[K, T, A]`: runs K parallel action samples by passing K different noise tensors as the flow-matching t=0 latent. If `noise=None`, samples Gaussian.
> 3. `set_lora(rank: int=16, alpha: int=32, dropout: float=0.05)`: adds LoRA adapters to the language model + action expert (skip the visual encoder so DINOv2 alignment is meaningful).
> 4. Async inference helper: `async_predict(obs, lang)` that returns immediately with the previously-computed chunk and triggers the next chunk computation in a background thread.
>
> Use `register_forward_hook` for feature extraction. Verify on a unit test that the K=4 sampled actions produce 4 distinct trajectories (max pairwise L2 distance > 0).

**Tests.**
- `test_flow_perturbation.py`: pass two different noise tensors, assert resulting actions differ.
- `test_alignment_loss.py`: confirm hook captures expected feature shape.

### 4.5 π0 wrapper (secondary baseline)

**Build prompt.**
> Implement `src/vla_ttc/models/pi0_wrapper.py` analogous to `SmolVLAWrapper`. Use the `openpi` library. Load π0 base, apply LoRA at rank 16 on the action expert and language model, quantize the VLM to 4-bit using `bitsandbytes`. Verify total VRAM at inference is ≤ 11 GB. Implement only `sample_actions` (with K parallel flow noise samples) and `set_lora` — no feature alignment for π0 in this project.

### 4.6 Feature-alignment loss (D3 hybrid component)

**Goal.** During fine-tuning, regularize SmolVLA's mid-layer visual features to stay close to a frozen DINOv2-base teacher's features on the same image. This is the **single component** from D3 you are folding into the project.

**I/O.** Input: student features `[B, N_tokens, D_s]`, teacher features `[B, N_tokens', D_t]`, both from the same RGB image. Output: scalar loss.

**Implementation notes.**
- DINOv2-base ViT-B/14 has 14×14 = 196 patch tokens at 224×224 input. SmolVLA's visual encoder may use a different patchification — handle this by **bilinearly interpolating the teacher features to match the student's spatial grid**, or pool both to a fixed N_tokens=64 grid via adaptive avg-pool over the spatial dims.
- Project student features to teacher dimension via a small learned linear layer (`nn.Linear(D_s, D_t)`) — this projector is initialized fresh and is part of what the alignment loss trains. Do not train DINOv2.
- Loss: token-wise cosine distance, mean over tokens and batch. λ=0.5 default; sweep {0.1, 0.5, 1.0}.

**Build prompt.**
> Implement `src/vla_ttc/train/losses/feature_alignment.py`. The class `FeatureAlignmentLoss(nn.Module)` takes a frozen DINOv2-base in `__init__`, holds a learnable projector `nn.Linear(student_dim, teacher_dim)`, and computes:
>
> ```python
> def forward(self, student_features, image):
>     with torch.no_grad():
>         teacher_features = self.dinov2(image)['x_norm_patchtokens']  # [B, 196, 768]
>     student_proj = self.projector(student_features)  # [B, N, 768]
>     # Resize to match spatial token count
>     teacher_features = self._resize_tokens(teacher_features, target_n=student_proj.shape[1])
>     loss = 1 - F.cosine_similarity(student_proj, teacher_features, dim=-1).mean()
>     return loss
> ```
>
> Add image normalization handling — DINOv2 expects ImageNet stats. Add a unit test that confirms the loss decreases when student features are explicitly set to a (de-projected) version of teacher features.

**Tests.** Unit test as above. Integration test: 1 epoch of fine-tuning on 10 demos with vs. without the loss should both converge.

### 4.7 Fine-tuning trainer

**Goal.** Run SmolVLA fine-tuning with three configurations: (a) base LoRA, (b) LoRA + feature alignment, (c) LoRA + feature alignment + background augmentation. The (b) vs (c) ablation gives you the "alignment alone is enough" or "you need augmentation too" story.

**Build prompt.**
> Implement `src/vla_ttc/train/finetune.py` using HuggingFace `accelerate` for multi-V100 training. Inputs: a Hydra config specifying dataset path, model wrapper class, LoRA rank, alignment loss weight λ (0 disables it), augmentation pipeline, optimizer, batch size, num_steps. Logs to W&B. Saves checkpoints every 1000 steps. Total loss = `flow_matching_loss + lambda_align * alignment_loss + lambda_aug * 0` (augmentation lives in the dataloader, not the loss). Default config: 20k steps, batch 64, AdamW lr=1e-4, cosine schedule, warmup 500. Single-V100 run finishes in ~10 hours; on 4× V100 in ~3 hours via DDP.
>
> Add a `--debug` flag that runs 100 steps on 8 demos to validate the pipeline before launching SLURM jobs.

**Configs to ship:**
- `smolvla_base.yaml`: λ=0, no aug.
- `smolvla_align.yaml`: λ=0.5, no aug.
- `smolvla_align_aug.yaml`: λ=0.5, SDXL-Turbo background swap (5×) + color jitter.
- `pi0_lora.yaml`: λ=0, no aug, 4-bit quant.

**SLURM launcher.**

**Build prompt.**
> Implement `scripts/launch_finetune.sh` and `src/vla_ttc/train/slurm_launcher.py` that submits a SLURM job array to the Unity cluster: 4× V100 per job, 12 hour walltime, partition appropriate, conda env activation, W&B api key from env. Provide a `--sweep` flag that runs a 3-point sweep of λ ∈ {0.1, 0.5, 1.0}.

### 4.8 Background augmentation pipeline

**Goal.** At training time, replace background pixels (everything not in the gripper-or-object region) with synthetic backgrounds.

**Implementation notes.**
- Run **once offline** before training, not in the dataloader. SDXL-Turbo at 512×512 takes ~200 ms per image on a V100; 120 demos × 100 frames × 5 augmentations = 60k images = 3.3 hours offline. Cache to disk.
- Use SAM2 with a point prompt at the gripper TCP (tool center point — known from forward kinematics) to segment the relevant region. Inpaint everything else with SDXL-Turbo using prompts like "a wooden tabletop", "a kitchen counter", "a cluttered desk", etc.
- Save augmented dataset alongside the original. Dataloader randomly samples original or augmented.

**Build prompt.**
> Implement `src/vla_ttc/data/augmentation/background_swap.py`. CLI: `python -m vla_ttc.data.augmentation.background_swap --in-dataset data/kinova_real --out-dataset data/kinova_real_bgswap --num-augmentations 5 --backgrounds wooden_table,kitchen_counter,cluttered_desk,white_studio,outdoor_grass`. Uses SAM2 to mask the foreground (point prompt at gripper TCP from URDF + forward kinematics on each frame's joint state), then SDXL-Turbo to inpaint the background region. Saves new episodes with `episode_index` offset by 1000 to keep them separable. Adds a manifest JSON listing aug provenance per frame.

### 4.9 Verifier model and training

**Goal.** Train a small network that, given (image features, language, candidate action chunk), outputs a scalar score correlated with whether the action chunk leads to task success.

**Architecture.**
- Inputs: SmolVLA's visual encoder output (frozen at this stage, ~768-D × N tokens), SigLIP-encoded language (768-D), action chunk (T=10 × 8-D). Total input ~768×64 + 768 + 80 ≈ 50k features after flattening, but use cross-attention not flattening.
- Body: 4-layer transformer, hidden dim 256, 4 heads. ~10 M params, ~40 MB at fp16.
- Output: scalar score via linear → sigmoid.

**Training data.**
- Positives: (state, action_chunk) pairs from successful demos (real + sim).
- Hard negatives, four kinds:
  1. **Gaussian-perturbed**: same state, action chunk + N(0, σ=0.1) noise. Sim has confirmed these often fail.
  2. **Time-shifted**: action chunk from 10 timesteps ahead in the same demo (wrong action for current state).
  3. **Cross-trajectory**: action chunk from a different demo at visually-similar state (use a CLIP-distance-based hard-negative miner).
  4. **Cross-instruction**: same image, action chunk from a *different task's* demo at similar state.
- Ratio: 1 positive : 4 negatives (one per type).
- Loss: binary cross-entropy.

**Build prompt.**
> Implement `src/vla_ttc/ttc/verifier/`:
>
> - `model.py`: a `Verifier(nn.Module)` with cross-attention from action tokens (linearly projected per-step) to image+language tokens. Final pool to a scalar via mean + linear. ~10 M params. Use LayerNorm and GELU.
> - `synthetic_pairs.py`: takes a LeRobot dataset, generates a parallel dataset of positive/negative (state, action_chunk, label) tuples. For "cross-trajectory" hard negatives, build a kNN index over CLIP image embeddings of demo frames using `faiss-cpu`. For sim-perturbed, load `data/kinova_sim_perturbed/` and use rollout outcomes (success flag) as labels directly.
> - `train_verifier.py`: trains the verifier for 50k steps on a single V100 (~4 hours), batch 256, AdamW lr=3e-4. Logs accuracy on a held-out split. Target: >80% accuracy distinguishing positives from each negative type.
>
> Provide `tests/unit/test_verifier_calibration.py` that confirms verifier scores correlate with held-out rollout success rate (Spearman ρ > 0.5).

**Risk flag.** If verifier accuracy stalls at <70%, the entire TTC pipeline collapses. **Validate by day 18.** If failing, fallback: use SmolVLA's own log-likelihood of the action chunk under its noise prior (no verifier needed) — this is a cheaper but weaker scoring function and a respectable ablation either way.

### 4.10 Scene parser and inpainter

**Goal.** Detect target object, distractors, and background. Mask non-task regions.

**Implementation notes.**
- **Cheap path (default):** SAM2 with grounded prompts. Use Qwen2.5-VL-3B-int4 once per N=10 steps to produce object names → use Grounding-DINO-Tiny for bounding boxes → SAM2 for masks → median-color background fill outside the object/gripper region. Total ~250 ms per call, but only 1× per 10 steps.
- **Skip the SDXL-Turbo inpainter at inference time.** It will not fit in the VRAM budget alongside everything else. Median-color fill is empirically nearly as effective at OOD robustness because the VLA's vision encoder is sensitive to texture/color, not to perfectly photorealistic backgrounds. Save SDXL-Turbo for *training-time* augmentation only (Section 4.8).

**Build prompt.**
> Implement `src/vla_ttc/ttc/scene_parser.py`:
>
> - Class `SceneParser` loads Qwen2.5-VL-3B-Instruct at int4 (via `bitsandbytes`) and Grounding-DINO-Tiny.
> - Method `parse(image, instruction) -> SceneParse(target_bbox, distractor_bboxes, gripper_bbox)`. Pipeline: VLM produces "what is the target object? what are distractors?" parse → Grounding-DINO produces bounding boxes for each named object → return.
> - Caches the last parse for K_skip=10 steps unless the OOD detector triggers.
>
> Implement `src/vla_ttc/ttc/inpainter.py`:
>
> - Class `MaskedInpainter` loads SAM2-base.
> - Method `mask_and_fill(image, scene_parse) -> Tensor[3,H,W]`. Pipeline: SAM2 refines bboxes to masks → keep target+gripper region → fill rest with median color computed from a 4-pixel border around the kept region → return.
> - Total inference: <100 ms on 4090.
> - Optional flag `use_sdxl_turbo: bool = False` for ablation in sim only.

### 4.11 OOD detector (RND, FIPER-style)

**Goal.** Decide when to invoke the expensive TTC pipeline.

**Implementation notes.**
- Two networks: a fixed random target `f_target: ℝ^d → ℝ^k` and a trainable predictor `f_pred: ℝ^d → ℝ^k`. Both 2-layer MLPs.
- Input: SmolVLA's visual encoder pooled output + state.
- Train predictor on in-distribution data only to match target output. Loss: MSE.
- At test time: `ood_score = ||f_target(x) - f_pred(x)||²`. High = OOD.
- **Calibration:** compute `ood_score` on a held-out 20% of training data; set threshold τ at the 90th percentile so 10% of in-distribution inputs trigger the expensive path (false-positive budget).

**Build prompt.**
> Implement `src/vla_ttc/ttc/ood_detector.py`. Class `RNDDetector` with `target_mlp` and `predictor_mlp`, both 2-layer 512-hidden, output dim 64. Method `train(features_train: Tensor[N, D], num_steps=5000)` and `score(features: Tensor[B, D]) -> Tensor[B]`. Method `calibrate(features_val: Tensor[N, D], target_fpr: float = 0.1) -> float` returns the threshold τ. Save target+predictor weights and τ.

### 4.12 Trigger logic and the orchestrator pipeline

**Goal.** Glue everything together with the right control flow.

```python
# Pseudocode for the inference loop
class TTCPipeline:
    def step(self, image, state, instruction):
        features = self.smolvla.visual_encode(image)
        ood_score = self.ood_detector.score(features)

        if ood_score < self.tau:
            # Fast path: single sample, no TTC
            action = self.smolvla.sample_actions(image, instruction, k=1)[0]
            self.last_path = "fast"
        else:
            # Slow path: scene parsing + sample-and-verify
            if self._steps_since_parse >= K_SKIP or ood_score > self.tau_high:
                self.last_parse = self.scene_parser.parse(image, instruction)
                self._steps_since_parse = 0
            self._steps_since_parse += 1

            cleaned_image = self.inpainter.mask_and_fill(image, self.last_parse)
            candidates = self.smolvla.sample_actions(cleaned_image, instruction, k=self.K)
            scores = self.verifier.score(features, instruction, candidates)
            best = candidates[scores.argmax()]
            action = best
            self.last_path = "slow"

        return action

    def reset(self):
        self._steps_since_parse = 0
        self.last_parse = None
        self.smolvla.reset_action_buffer()  # Clear async chunk
```

**Build prompt.**
> Implement `src/vla_ttc/ttc/pipeline.py` containing `TTCPipeline` as above. Add structured logging of every decision (fast vs slow, ood_score, candidate scores, latency per stage) to a JSON-lines file for post-hoc analysis. Add a `--log-rerun` flag that streams to a Rerun visualizer for debugging. Include `from_config(cfg)` factory and a CLI `python -m vla_ttc.ttc.pipeline --config configs/ttc/pipeline_full.yaml --checkpoint <path> --episode-num 0` for offline replay on recorded demos.

**Tests.**
- `test_pipeline_smoke.py`: feed 100 random images + dummy state, assert no crashes and latency p95 < 500 ms on 4090.
- `test_pipeline_ablations.py`: confirm `pipeline_no_inpaint`, `pipeline_no_verifier`, `pipeline_no_trigger` configs each disable the right component.

### 4.13 Robot interface and execution

**Build prompt.**
> Implement `src/vla_ttc/robot/`:
>
> - `kinova_ros_bridge.py`: subscribes to camera, publishes EE pose targets and gripper commands to `kinova-ros`. Wraps `kortex_driver` if you're on Gen3-style API or `kinova-ros` for original Gen2. Provides `get_obs() -> Obs(image, state)` and `execute(action: ndarray[8])`.
> - `action_executor.py`: takes an action chunk from TTCPipeline and dispatches to the bridge at the right cadence. Handles SmolVLA's async-chunk paradigm (next chunk computed while current chunk executes).
> - `safety_monitor.py`: enforces a workspace box (configurable XYZ limits), max joint velocity, max EE velocity, and emergency stop on a keypress. **This is non-optional. Kinova will hit itself if you let it.**
> - `reset_helpers.py`: a one-keypress reset to a home pose between trials.

### 4.14 Evaluation harness

**Build prompt.**
> Implement `src/vla_ttc/eval/real_robot_runner.py`:
>
> - CLI: `python -m vla_ttc.eval.real_robot_runner --method smolvla_align_ttc --condition ood_objects_l1 --task pick_red_block --num-trials 15 --out-dir results/`.
> - Loop: for each trial, prompt the human to set up the scene per `ood_specs.py` checklist, press space when ready, run the policy for up to 200 steps, mark success/failure via human input, log latency and trajectory.
> - Saves per-trial JSON: `{trial_id, condition, task, method, success: bool, num_steps, mean_latency_ms, p95_latency_ms, ood_score_trace, fast_path_fraction, fail_reason: Optional[str]}`.
> - Add `scripts/run_eval_matrix.sh` that runs every (method, condition, task) combination with proper randomization of trial order to avoid time-of-day confounds.
>
> Implement `src/vla_ttc/eval/libero_runner.py` that runs LIBERO-Spatial / LIBERO-Object / LIBERO-Goal benchmark using LeRobot's LIBERO integration. This gives sim numbers that reviewers expect.
>
> Implement `src/vla_ttc/eval/metrics.py` with: `wilson_interval`, `paired_bootstrap_diff`, `mcnemar_test` for paired success/failure comparisons.

**Eval matrix size:**
- Methods: `smolvla_base`, `smolvla_align`, `smolvla_align_ttc` (full), `smolvla_align_ttc_no_inpaint`, `smolvla_align_ttc_no_verifier`, `pi0_lora` (cross-model), `pi0_lora_ttc`. 7 methods.
- Conditions: ID, OOD-Objects-L1, OOD-Distractors-L1, OOD-Distractors-L2, OOD-Lighting (3 lighting conditions averaged). 5 conditions.
- Tasks: 3.
- Trials: 15.
- Total real-robot rollouts: 7 × 5 × 3 × 15 = **1575 trials** in the worst case. **At 30 s each, this is 13 hours of pure execution but realistically 4 days with resets and failures.**

**Reduction strategy if you're behind schedule.** Prune to: 4 methods (base, align, align+ttc, ttc-only-no-align ablation) × 5 conditions × 2 tasks × 12 trials = **480 trials** ≈ 2 days. The cross-model π0 and the no-inpaint/no-verifier ablations move to LIBERO-only.

### 4.15 Analysis and figures

**Build prompt.**
> Implement `src/vla_ttc/analysis/`:
>
> - `plots.py`: bar charts of success rate per method per condition with Wilson CIs; per-method latency distribution box plots; "fast vs slow path" pie chart per condition; sample qualitative rollouts (image grids).
> - `tables.py`: emits LaTeX tables of the main results, ablations, and LIBERO scores in CoRL paper format.
> - `failure_taxonomy.py`: parses per-trial logs, clusters failures by reason (collision, miss-grasp, wrong-object, dropped), produces a stacked bar.
> - `latency_profiler.py`: detailed per-stage latency breakdown.
>
> Implement `scripts/make_paper_figures.py` that takes a results dir and produces every figure and table needed for the paper in one go.

---

## 5. Critical-path schedule (30 days, starting May 1)

This Gantt assumes solo work. With one collaborator the eval phase compresses by 30%.

```
Day 1-2:   Repo setup, environment pinning, dataset format scaffolding,
           Kinova ROS bridge smoke test (just publish a hello-world EE pose).
Day 3:     VRAM dry-run on 4090. *Hard checkpoint:* if pipeline doesn't fit, drop SDXL-Turbo
           inpainter and re-budget.
Day 4-7:   Demo collection (120 real demos). Run IsaacSim collector in parallel for sim demos.
Day 5-8:   SmolVLA fine-tuning runs on Unity (3 configs in parallel, each ~10h on 4× V100).
           pi0 LoRA fine-tune in parallel.
Day 8-10:  Verifier dataset generation (synthetic pairs from real + sim).
Day 9-12:  Verifier training. *Hard checkpoint day 12:* verifier accuracy >80% on held-out?
           If no, fallback to action log-likelihood scoring.
Day 11-14: Scene parser + inpainter implementation. End-to-end TTC pipeline integration.
Day 13-15: LIBERO sim evaluation of all methods (cheap, run while real-robot setup is finalized).
Day 14:    *Hard checkpoint:* full pipeline runs on 4090 at ≥2 Hz. If not, simplify scene parser.
Day 15-22: Real-robot evaluation matrix. Mornings: ID and OOD-Objects (4 days).
           Afternoons: OOD-Distractors and OOD-Lighting (3 days). One full day of buffer.
Day 23-25: Failure analysis, additional ablations to plug review-bait holes (e.g., ID-vs-OOD
           latency tradeoff, demo-budget curve at 50/100/150 demos for one method).
Day 26-29: Paper writing. Use the CoRL 2026 LaTeX template. Sections in order:
           Method (start day 23), Experiments (day 25), Intro/Related Work (day 27),
           Limitations (day 28, mandatory at CoRL 2026), Abstract last.
Day 30:    Submission. Submit at least 12 hours before AoE deadline.
```

**Hard checkpoints summarized** (write these down, they save you):
1. **Day 3:** VRAM dry-run passes on 4090 → if not, drop components.
2. **Day 12:** verifier > 80% accuracy → if not, switch to log-likelihood scoring.
3. **Day 14:** full pipeline runs at ≥2 Hz end-to-end on 4090.
4. **Day 18:** pivot decision: if real-robot ID success of `smolvla_align` baseline is below 70%, the OOD comparison is meaningless. Stop adding TTC complexity, debug fine-tuning first.

---

## 6. Configs you will actually need (sample)

**`configs/train/smolvla_align.yaml`:**

```yaml
defaults:
  - _self_
  - data: kinova_real

model:
  name: smolvla
  base_checkpoint: lerobot/smolvla_base
  lora:
    rank: 16
    alpha: 32
    dropout: 0.05
    target_modules: [q_proj, v_proj, action_expert.q_proj, action_expert.v_proj]
  feature_alignment:
    enabled: true
    teacher: facebook/dinov2-base
    student_layer: visual_projector_out
    lambda: 0.5
    project_dim: 768

train:
  num_steps: 20000
  batch_size: 64
  optimizer: adamw
  lr: 1e-4
  weight_decay: 0.01
  warmup_steps: 500
  schedule: cosine
  grad_clip: 1.0
  precision: bf16

logging:
  wandb_project: vla-ttc-corl2026
  wandb_run_name: smolvla_align_lambda0p5
  log_interval: 50
  ckpt_interval: 1000
  ckpt_dir: ./ckpts/smolvla_align/

seed: 42
```

**`configs/ttc/pipeline_full.yaml`:**

```yaml
smolvla:
  checkpoint: ./ckpts/smolvla_align/step_20000
  precision: bf16
  k_action_samples: 4
  num_inference_steps: 10

ood_detector:
  checkpoint: ./ckpts/ood_rnd/final.pt
  threshold_calibration: 0.9   # 10% FPR
  threshold_high: 0.99          # Force re-parse

scene_parser:
  vlm: Qwen/Qwen2.5-VL-3B-Instruct
  vlm_quantization: int4
  detector: IDEA-Research/grounding-dino-tiny
  parse_every_n_steps: 10

inpainter:
  use_sdxl_turbo: false  # Inference-time: use SAM2+median fill
  sam2_checkpoint: facebook/sam2-hiera-base-plus

verifier:
  checkpoint: ./ckpts/verifier/final.pt
  precision: fp16

pipeline:
  log_jsonl: ./logs/ttc_pipeline.jsonl
  log_rerun: false
  async_inference: true
```

---

## 7. Pre-flight checklist (before day 1)

- [ ] Unity cluster account active, SLURM access verified, GPU quota confirmed.
- [ ] Kinova Gen2 calibrated, gripper functional, ROS bridge tested.
- [ ] External camera intrinsics measured (you'll need this to match IsaacSim).
- [ ] 4090 laptop has CUDA 12.x, Python 3.11, ≥40 GB free disk.
- [ ] W&B account, project created, API keys in `.env` files (cluster + laptop).
- [ ] LeRobot version pinned and installed in both envs.
- [ ] OOD test objects ordered/printed (give Amazon 5 days).
- [ ] Two backup lighting setups (different bulb temperatures + a dimmer).
- [ ] CoRL 2026 LaTeX template downloaded.
- [ ] HuggingFace tokens for gated models (Qwen-VL, etc.) configured.

---

## 8. Brutally honest things to know

1. **Verifier training is the riskiest single component.** If sim hard negatives don't transfer to real, your scores correlate with nothing. Day-12 checkpoint is real, not a formality. The fallback (action log-likelihood) is honest and publishable, just less novel — be prepared to take it.

2. **Single-camera + Kinova Gen2 noise will hurt absolute success rates.** Don't be alarmed by ID success of 65–80% on `smolvla_base` — that's normal for this hardware. What matters is **the relative gap closed by your method**, not absolute numbers.

3. **The async inference loop is where bugs hide.** SmolVLA's chunked execution means the action being executed is N steps stale. If your safety monitor or scene parser uses fresh observations against stale actions, you'll get jittery behavior. Test the loop in IsaacSim with fake observations first (day 11).

4. **The "what about more demos?" question is unavoidable.** Pre-bake a demo-efficiency curve (50/100/150 demos for one method × one task) in the eval matrix from the start. It is one extra fine-tuning run and one extra eval block — cheap insurance against rejection.

5. **Reviewers will fixate on one of three things:** (a) "Your method is just RoboMonkey scaled down" — answer: you operate at 12 GB on a single camera, with a fundamentally different (flow-matching noise) sampling primitive. (b) "Your verifier is trained on sim" — answer: cite real-robot ablations and sim-to-real generalization analysis. (c) "Why not also try OpenVLA-OFT?" — answer: it doesn't fit in 12 GB at deployment, which is the regime you target. Have these three rebuttals literally written into the paper before submission.

6. **One month is tight but feasible if you protect the critical path.** The only way this fails is if you spend week 1 polishing infrastructure. Get an end-to-end "stupid version" of every component running in week 1, even if individually weak — then improve. Do not write the verifier before you have a working SmolVLA loop on the robot.

7. **Budget 1 full day for the paper figures.** Researchers consistently underestimate this. `make_paper_figures.py` saves you that day.

---

## 9. What to skip, intentionally

- **3D / point-cloud inputs.** You have one RGB camera. Don't fake it.
- **Bimanual or dexterous tasks.** One gripper, period.
- **RL fine-tuning.** Out of scope for 30 days, crowded space anyway.
- **Cross-embodiment claims.** One robot. Frame as "single-platform deep evaluation," not breadth.
- **Architectural changes to SmolVLA.** You said you don't want this; don't drift into it.
- **Custom CUDA kernels.** Use whatever LeRobot ships; profile only if a bottleneck blocks you.

---

## 10. After submission

- Push the codebase to GitHub at submission time (anonymized fork). CoRL increasingly weights reproducibility.
- Tag a `corl2026-submission` git tag.
- Prepare a 4-page workshop version (the same paper, compressed) for the parallel CoRL workshop submission window in mid-July. Workshop CFPs typically open in May–June.
- Plan the v2 (ICRA 2027 or RSS 2027): adds the missing camera, expands to 2 robots, replaces the median-fill inpainter with proper diffusion inpainting now that latency budget is relaxed.

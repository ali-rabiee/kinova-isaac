# Bring-up prompt: the carryover-aware supervisory study on the physical Kinova JACO2, with a hardcoded surrogate standing in for the VLA

> **For the human operator (read before pasting).** This file is a complete, self-contained prompt for Claude
> Code running on the computer that drives the robot. It assumes no network, no git and no GitHub on that
> machine: you copy the code over yourself, and the prompt verifies the copy against the manifest in
> Appendix F. Paste everything below the horizontal rule into a fresh Claude Code session started **inside
> the copied repository directory**. Before you paste it:
>
> 1. Copy the repository tree from the laptop, uncommitted changes included, skipping the heavy trees
>    (270 GB of checkpoints under `vla_lab/results/`, 58 GB under `logs/`, 31 GB under `vla_lab/old_demos/`,
>    15 GB of `.git`). From the laptop, with `DST` either an ssh target or a mounted drive:
>    ```bash
>    SRC=/home/kye/Desktop/Depo/Code/CORL/kinova-isaac
>    DST=user@robot-pc:~/kinova-isaac                     # or /media/<drive>/kinova-isaac
>    rsync -av --exclude='/.git/' --exclude='/vla_lab/results/' --exclude='/logs/' \
>          --exclude='/vla_lab/old_demos/' --exclude='__pycache__/' --exclude='*.pyc' "$SRC/" "$DST/"
>    rsync -av "$SRC/vla_lab/results/physics/" "$DST/vla_lab/results/physics/"   # 59 MB: the physics fit + rendered scene frames
>    rsync -av "$SRC/vla_lab/results/reach/"   "$DST/vla_lab/results/reach/"     # 136 KB: the simulated reach envelope
>    ```
>    The second and third lines are optional: they carry the only result files the study code reads. If you
>    skip them the prompt installs the physics from its Appendix A (byte-identical) and the phrase-corpus
>    packet uses real photographs instead of renders. The paper directory `vla_lab/paper/` comes along with
>    the first line; add `--exclude='/vla_lab/paper/'` if you would rather it did not. Tell Claude the path.
> 2. Nothing else moves. Section 7, Appendix A (the study contract) and Appendix C (the preregistered
>    analysis plan) of the paper are the specification of the participant experience, and everything the
>    prompt needs from them is restated inside the prompt.
> 3. Have ready on the robot machine: the arm powered and its driver launchable, the cameras plugged in,
>    the e-stop reachable, the two 5 cm cubes (red target, blue blocker) plus any distractor cubes, a
>    tape measure or steel rule with millimetre marks, and a printer for the placement template.
> 4. Sit with the session. The prompt tells Claude to stop and ask before the arm moves for the first time
>    and before every new kind of motion. Answer its questions; do not let it guess about hardware.

---

## 0. Who you are and what you are building

You are Claude Code on the computer that drives a **Kinova JACO2 (Gen2, `j2n6s300`, three-finger hand)**.
Your job is to bring up, on this machine and this robot, the human-study apparatus of the
**carryover-aware supervisory control** project, so that a real person can sit behind a barrier, watch the
robot fetch a red box from beside a blue box, type instructions when the robot asks, and be measured exactly
the way the study design specifies.

The project's simulation results are finished. They were produced with a Carryover-Aware VLA, a neural
policy wrapper that did two things inside the study loop: it **listened** (its intent heads turned the
supervisor's utterance into a strategy label and an "unprompted" estimate) and, in principle, it **moved the
arm**. On this robot **both of those roles are replaced by hardcoded, heuristic code**, and **nothing else
changes**:

- the interaction protocol (blocks, matched budget, counterbalancing, demonstrations, probes,
  counter-proposals, waits),
- the robot's behaviour and its exact words,
- the data-collection contract and the event-locked session logs,
- the participant's experience (what they see, hear, are asked, and are asked to fill in),
- the belief module, the schedulers, the estimators, the session gate, and the analysis.

You will implement **several surrogate scenarios** behind one runtime switch, so the operator can choose
which one a session runs. They are specified in §5. The default is the one that keeps the robot's decision
logic identical to the paper's system and only stubs the two roles the VLA played.

The codebase is `vla_lab/` inside the `kinova-isaac` repository. It already contains the whole study: the
single session runner, the schedulers, the estimand and estimators, the contract, the protocol, the logger,
the gate, the analysis, the questionnaire scoring, the phrase-corpus tool, and a scripted expert whose
waypoint geometry you will port to the physical arm. The code was written so that the robot, the human, and
the listener sit behind three narrow seams (`Apparatus`, `SupervisorChannel`, `Grounder`). Your work is
almost entirely on the far side of those seams.

### Ground rules (they override your defaults)

1. **The arm does not move until the operator says so, and each new kind of motion is confirmed
   separately.** Homing, a Cartesian move, a gripper close, a first push, a first grasp: each is a
   separate confirmation. Before any motion, print what will move, to where, at what speed, and how to
   stop it. Assume a person may be within reach of the arm at any time unless the operator has confirmed
   otherwise for that specific step.
2. **Do not rewrite the study.** The modules listed in §4.1 are frozen. If you believe a change inside
   them is unavoidable, make it minimal, backwards-compatible, behind a default-off argument, covered by a
   test that shows default behaviour is unchanged, and report it explicitly. Never change the narration
   wording, the contract semantics, the scheduler decisions, the estimators, or the log schema.
3. **Same words, same numbers, same files.** The participant must hear and read the strings that
   `vla_lab/supervisory/narration.py` produces, and the session directory must contain the same files
   with the same keys that the simulated study writes, plus whatever extra hardware telemetry you add in
   *additional* files. The equivalence checks in §9 are pass/fail.
4. **Measure, do not assume.** Frame conventions, table height, reachable envelope, gripper values, camera
   calibration, timing overheads: each is measured on this hardware and written down with the command
   that measured it. Anything you could not measure is labelled `prior` or `unverified` in code, logs and
   docs, the way `ScenePhysics.source` already does for the physics.
5. **Move, never delete.** Superseded files go into a dated `legacy_*` folder. Existing tests must stay
   green; new invariants get new tests that run with no ROS and no hardware.
6. **Nothing fabricated.** No placeholder participant data, no invented calibration numbers, no
   "typical" success rates. Empty is fine; made up is not.
7. **Ask when hardware facts are unknown.** The operator does not know for certain which driver stack or
   which camera setup this machine has ("I think ROS 1", "I think colour segmentation"). Discover first
   (§3). Where discovery cannot settle something, ask the questions in Appendix E rather than guessing.
8. **Write long, complete documents and code comments** that explain why, including the dead ends. The
   project's house style is that a reader should never discover a defect three weeks later that a
   comment could have warned them about.
9. **Report honestly at the end.** State what was built, what was measured, what was verified on
   hardware, and what was not. Never describe a hardware step as done unless it ran on the hardware.

---

## 1. The system in one page

**The phenomenon.** A robot that demonstrates a manipulation strategy and narrates it ("Clearing the path
first keeps the grasp safe") teaches the person supervising it what to say. When it then asks, on a
genuinely ambiguous scene, "How should I approach this one?", the answer is a mixture of what the person
actually prefers and what they were just shown. The study estimates the **unprompted** preference map and
models the residue of the robot's own coaching as a signed, decaying latent state.

**The task.** Fetch the **red box** (target) when a **blue box** (blocker) sits beside it. Two strategies,
one axis (`plan`):

- **A = CLEAR_FIRST** (cautious): push the blocker aside to a drop-off, then grasp the target top-down.
- **B = DIRECT** (efficient): detour around the blocker and grasp the target where it stands.

The single independent variable is the **free clearance gap between the two cube faces** (cubes are 5 cm;
centre distance = gap + 5 cm). The **ambiguity coordinate** `c` is the value margin between the strategies
in units of the transition width: `c > 0` means A is objectively better, `c < 0` means B is, `c = 0` is the
crossover where the answer is a genuine preference. Under the measured physics (Appendix A) the crossover is
at a **4.54 cm** gap and the transition width is **0.50 cm**; the 19-scene grid (Appendix B) has 15 probe
scenes between 3.34 cm and 5.73 cm, 9 of them inside the crossover band 4.09 to 4.99 cm, plus 4
demonstration scenes at 2.94 cm and 6.13 cm where one strategy is unambiguously right. **Adjacent band
scenes differ by 1.5 mm of gap.** Read §6.3 for what that means for physical placement.

**The four actions**, one per interaction slot:

| action | what the robot does | who decides |
|---|---|---|
| `COACH` | narrates and executes a strategy on an unambiguous scene | the protocol (count, positions, directions fixed per participant) |
| `PROBE` | presents an ambiguous scene, asks the neutral question, executes what it is told | the condition's scheduler |
| `WAIT` | says a neutral filler, idles for the contract's wait time | the condition's scheduler |
| `COUNTER` | asks the question and names the option it did **not** just demonstrate | the condition's scheduler |

When an answer cannot be grounded to A or B, the robot executes the value-optimal strategy for that scene and
the slot contributes no observation. The participant is never told this happened.

**The conditions the human study runs** (`vla_lab/supervisory/scheduler/`):

- `no_coach` (B0): reference and retest blocks, no demonstrations, every slot probes.
- `memoryless` (B1): probes immediately after every demonstration, no carryover model, pooled estimator.
- `recommended` (B7): corrected estimator plus counter-proposals, adaptive schedule off. This is
  `CarryoverAwareConfig.recommended()`; a test asserts the schedule is never on.

Others exist (`fixed_washout` B2, `random_static` B3, `always_counter` B4, `carryover_aware` B5,
`identification_first` B6) and must keep working; two of them double as "hardcoded schedule" surrogates
(§5).

**The session** (`vla_lab/supervisory/protocol.py`, contract in Appendix B): a 40-slot no-coach
**reference** block, then the compared condition blocks in balanced-Latin-square order (60 slots each, 10
demonstrations each, identical scene sequence and demonstration positions across conditions by
construction), then a 30-slot no-coach **retest** block. The human study runs the **alternating**
demonstration regime (direction flips every demonstration) because its primary hypotheses are H1 (the
carryover exists and decays) and H2 (it varies between people); the remedy contrast (B1 vs B7) is
secondary and model-based. Questionnaires (NASA-TLX, a 12-item trust scale, a short burden item) are
administered at every block boundary. A pilot of about six people gates recruitment.

**The three seams** (`vla_lab/supervisory/apparatus/base.py`):

```python
class Apparatus(Protocol):
    def reset_scene(self, scene: SceneSpec) -> None: ...
    def execute(self, scene: SceneSpec, strategy: str) -> ExecutionOutcome: ...   # strategy, success, duration_s, notes
    def say(self, text: str) -> None: ...
    def describe(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...

class SupervisorChannel(Protocol):
    def ask(self, query: str, scene: SceneSpec, *, action: str, session_progress: float) -> SupervisorTurn: ...  # utterance, latency_s, truth=None
    def observe_demonstration(self, scene: SceneSpec, strategy: str, narration: str, *, strength: float) -> None: ...
    def elapse(self, delta: float) -> None: ...
    def describe(self) -> Dict[str, Any]: ...

class Grounder(Protocol):
    name: str
    def ground(self, utterance: str, scene: SceneSpec) -> str: ...   # "A" | "B" | "unresolved"
```

The session runner (`vla_lab/supervisory/session.py`, `run_block` / `run_session`) is the only slot loop.
It calls `apparatus.reset_scene`, then for a demonstration `apparatus.say(narration)` and
`apparatus.execute(scene, strategy)` and `channel.observe_demonstration(...)`; for a probe or counter
`apparatus.say(query)`, `channel.ask(...)`, `grounder.ground(...)`, `apparatus.execute(scene, executed)`;
for a wait `apparatus.say(filler)`. It writes `trials.jsonl`, `beliefs.jsonl`, `events.jsonl`,
`contract.json`, `protocol.json`, `meta.json` through `vla_lab/supervisory/logging.py`. You will not
change this loop.

**What the VLA did, and what replaces it here.**

| VLA role in the simulation | surrogate on this robot |
|---|---|
| intent head "said": utterance to A/B | `LexicalGrounder` (`narration.ground`), the study's reference channel, unchanged |
| intent head "unprompted": de-biased estimate | the explicit belief module (already how the deployed policy decides); optionally a rule-based `HeuristicIntentGrounder` logged as a secondary channel (§5) |
| ask gate | already taken from the belief module, never from a learned gate |
| motor policy | the scripted strategies from `environments/supervisory_fetch/experts.py`, executed on the real arm through a driver bridge |

---

## 2. Phase 0: get the code, verify the version, make the study run here (no robot yet)

1. **Locate the copied tree.** The operator copied the `kinova-isaac` repository here by hand (rsync or a
   drive), uncommitted changes included, normally without `.git` and without the heavy `results/`,
   `logs/` and `old_demos/` trees. Ask for the path if it is not the current directory. Nothing in this
   prompt needs network access, git, or GitHub. The code copes without a `.git` (the session logger's
   `_git_commit()` simply records `None`); if one is present, do not fetch or push anything.
2. **Verify the copy is complete and is the laptop's version.** Run the manifest check of Appendix F
   (sha256 prefix and byte size of every file that defines the study). Every listed file must exist and
   match. Then check the must-have list below. If anything is missing or differs, **stop and tell the
   operator which files**; do not build on a partial or older tree:
   ```bash
   for f in vla_lab/supervisory/physics_fit.py vla_lab/supervisory/scheduler/identification_first.py \
            vla_lab/supervisory/flip.py vla_lab/supervisory/physics_ci.py vla_lab/human_study/phrase_corpus.py \
            vla_lab/training/seeds.py vla_lab/training/eval_shift.py vla_lab/docs/carryover1_execution.md \
            vla_lab/docs/real_robot_bringup_prompt.md vla_lab/stats_utils.py; do
     [ -f "$f" ] && echo "ok  $f" || echo "MISSING $f"; done
   grep -q 'def recommended' vla_lab/supervisory/scheduler/carryover_aware.py && echo "ok recommended()"
   grep -q 'CONDITION_RECOMMENDED' vla_lab/supervisory/scheduler/__init__.py && echo "ok B7 registered"
   grep -q 'lapse' vla_lab/supervisory/physics_fit.py && echo "ok lapse fit"
   grep -q 'guarded_correlation' vla_lab/stats_utils.py && echo "ok min-n guard"
   ```
3. **Python.** `pyproject.toml` requires Python >= 3.11 and the study modules use 3.10+ syntax. ROS 1
   Noetic ships Python 3.8. Therefore: create a dedicated **Python 3.11 (or newer) environment** for the
   study process (`uv venv`, `conda create`, or `python3.11 -m venv`, whichever this machine supports),
   install `numpy matplotlib opencv-python fastapi uvicorn` (and later a TTS package), and **never
   import ROS into it**. The ROS side runs as a separate process in the system Python (§6.1). If the
   machine is offline, `numpy` alone is enough for the tests, the smoke run and the hash check; list the
   remaining wheels you need and ask the operator to bring them rather than stalling. Do **not** install
   torch, Isaac Sim, Isaac Lab, or LeRobot on this machine; nothing in the study path needs them.
4. **Make sure the physics the code reads is the laptop's.** `build_scene_grid()` reads
   `vla_lab/results/physics/physics.json` automatically; without it the code silently falls back to a
   prior physics and every hash below differs. If the operator copied `vla_lab/results/physics/`, `diff`
   its `physics.json` against Appendix A (they must be identical). If not, write the four JSON files from
   Appendix A to exactly these paths, byte content as given: `vla_lab/results/physics/physics.json`,
   `vla_lab/results/physics/physics_lower.json`, `vla_lab/results/physics/physics_upper.json`,
   `vla_lab/results/reach/reach.json`.
5. **Run the offline study tests that do not need torch** from the repo root with `PYTHONPATH=.`:
   ```bash
   python -m vla_lab.tests.run_tests --only test_supervisory_scenes test_supervisory_carryover \
     test_supervisory_estimand test_supervisory_scheduler test_supervisory_session test_supervisory_protocol \
     test_supervisory_logging test_supervisory_supervisor test_supervisory_audit test_supervisory_physics_fit \
     test_human_study
   ```
   Modules that import torch (`test_policy_grounder`, `test_supervisory_models`, `test_intent`,
   `test_multicam`, `test_feedback`, `test_allocation`, `test_fit_allocator`, `test_calibration`) are
   expected to be unavailable here; record which ran and which were skipped, do not install torch to make
   them run.
6. **Smoke-run the synthetic study**, which exercises the real session code with the surrogate apparatus
   and a generative supervisor: `SUPERVISORS=6 OUT=vla_lab/results/smoke ./vla_lab/scripts/sup_study.sh`.
7. **Equivalence hashes.** With the embedded physics installed, this must print exactly:
   ```bash
   python - <<'EOF'
   from vla_lab.supervisory.contract import Contract, BudgetConfig
   print(Contract().hash(), Contract().narration_hash())                       # 30df16367d79b38d 30e0921cb87a7589
   print(Contract(budget=BudgetConfig(coach_regime="alternating")).hash())      # 8254f4654c1d8256
   EOF
   ```
   If the first line differs, the physics file or the narration differs from the laptop's and nothing you
   run here is comparable with the simulation results; find out why before continuing.
8. If a `.git` exists, make sure `logs/human/` is ignored. Either way: **participant data never leaves
   this machine**, not in a copy, not in a handoff folder, not in a commit.

---

## 3. Phase 1: discovery (read-only; the arm does not move)

The operator believes the arm is driven by **ROS 1** and that object localisation might be **colour
segmentation from a camera**, but is not certain of either. Other projects on this machine have used the
same robot; their code, launch files and calibration files are evidence. Find out, with commands, and write
the findings to `vla_lab/docs/real_robot_bringup.md` §1 ("What this machine has") before designing anything.

Inventory, each item with the command and its output pasted or summarised:

1. **OS and Python.** `lsb_release -a`, `python3 --version`, every other Python (`ls /usr/bin/python3*`,
   `conda env list`, `pyenv versions`, `uv python list`).
2. **ROS.** `echo $ROS_DISTRO`, `ls /opt/ros`, `rosversion -d`, whether `roscore` is running, the catkin
   workspace(s) (`ls ~/catkin_ws ~/*_ws`), `rospack find kinova_driver kinova_msgs kinova_bringup`.
   With the driver launched (ask the operator to launch it the way they normally do, or find the launch
   command in shell history / project READMEs): `rostopic list`, `rosservice list`, `rosnode list`,
   `rostopic echo -n1 /j2n6s300_driver/out/tool_pose`, `rostopic echo -n1 /j2n6s300_driver/out/finger_position`,
   `rosmsg show kinova_msgs/ArmPoseGoal`, `rosmsg show kinova_msgs/SetFingersPositionGoal`,
   `rostopic hz /j2n6s300_driver/out/joint_state`. If ROS 2 or the raw JACO SDK is what exists instead,
   record that and adapt §6.1 accordingly (the bridge design is driver-agnostic on purpose).
3. **Other projects that drove this arm.** Search home directories and workspaces for `j2n6s300`,
   `kinova_msgs`, `ArmPoseAction`, `SetFingersPosition`, `cartesian_velocity`, `home_arm`, `PoseVelocity`,
   `kinova_api`, `Kinova.API`. For each hit: what it does, whether it ran recently (mtime, logs), and
   whether it contains a **tool-pointing-down orientation**, a **home or park pose**, **finger open/closed
   values**, a **table calibration**, or a **camera-to-robot calibration**. These are the numbers you would
   otherwise have to measure; reuse them with attribution and re-verify each on hardware.
4. **Cameras.** `ls /dev/video*`, `v4l2-ctl --list-devices`, `rs-enumerate-devices` (RealSense),
   `rostopic list | grep -i image`, any `apriltag`, `aruco`, `cv2` or `pyrealsense2` code found in step 3,
   any saved intrinsics or homographies. Establish which camera can see the whole workspace from above
   (the placement verifier and the participant's feed can be different cameras), its resolution and
   frame rate, and how frames are obtained (ROS topic vs direct V4L2/RealSense in the 3.11 env).
5. **Audio and displays.** `aplay -l`, `pactl list short sinks`, available TTS engines (`which piper
   espeak-ng espeak festival`, `python -c "import pyttsx3"`), number of displays (`xrandr`), which one the
   participant will face.
6. **Safety hardware.** Where the e-stop is, what it does electrically (power cut vs driver stop), whether
   the driver exposes an e-stop or "stop" service (`/j2n6s300_driver/in/stop`, `/in/start`), whether a
   second emergency stop exists for the participant side (the paper's design says the participant is
   behind a barrier and does not touch the robot; confirm with the operator).
7. **The physical workspace.** Ask the operator to photograph the table from the participant's position
   and from above, and to report: table height relative to the arm's base plate, whether the arm is bolted
   to the table top (the simulation assumed the base sits on the table so the table top is `z = 0` in the
   base frame), the cubes' actual edge length and colours, the barrier, where the participant sits, where
   the experimenter sits, ambient lighting (colour segmentation cares).
8. **Disk and network.** Free disk (`df -h ~`), whether the machine is offline (affects package installs
   and TTS model downloads).

Then write §2 of the same document, "Decisions", each with its evidence: driver path (ROS 1 bridge, ROS 2,
SDK), image path (ROS relay vs direct capture), object localisation method (colour segmentation with a
table homography, fiducials, or template-only; see §6.3), Python split, and TTS engine. Present the
discovery report to the operator, ask the open questions from Appendix E that discovery did not settle, and
then continue with §5 and §6 (software; no motion). Do not wait for hardware answers to write software
that does not depend on them.

Also run a **quick physics reality check plan** past the operator (it needs the arm, so it belongs to
Phase 4): the simulation's crossover and transition width come from a scripted expert with 4 mm pose
noise in Isaac. On the real hand the transition will almost certainly be wider and the crossover will move.
§7 step 11 measures it; §6.3 explains why that decides the scene grid.

---

## 4. Phase 2: architecture and invariants

### 4.1 Frozen (do not edit except under ground rule 2)

`vla_lab/supervisory/{__init__,session,contract,protocol,narration,strategies,scenes,carryover,estimand,
supervisor,logging,verify_session,analyze,physics_fit}.py`, everything in `vla_lab/supervisory/scheduler/`,
`vla_lab/supervisory/apparatus/{base,surrogate}.py`, `vla_lab/human_study/{instruments,phrase_corpus,protocol}.py`,
`environments/supervisory_fetch/{config,experts}.py` (you **import** these; the real-arm overrides live in
your apparatus config, never in these defaults, because the simulation results were produced with them).

One permitted core change, if you want it: an optional `between_blocks` callback argument on
`run_session` (default `None`, called after each block with the `Block` and its `BlockResult`) so the
participant runner can administer questionnaires and enforce the rest without duplicating the block loop.
Add a test that `run_session` output is byte-identical with and without the argument when the callback is
`None`. If you prefer not to touch `session.py`, drive `run_block` yourself from the runner, reproducing
`run_session`'s bookkeeping exactly (scheduler per block via `build_scheduler`, `block_start`/`block_end`
events, `channel.elapse(inter_block_rest_deltas)`, `contract.save`, `protocol.save`, `logger.close(meta)`).

### 4.2 New modules (create these paths)

```
vla_lab/real_robot/gen2_bridge/            ROS-side process (system Python): NDJSON-over-socket server that owns the driver
    bridge_node.py                         ops: connect, state, home, park, move_pose, fingers, stop, start, estop_state,
                                           capture, ping, close   (wire format in §6.1)
    fake_bridge.py                         pure-Python fake with configurable faults, for tests and dry runs (no ROS)
    README.md                              how to launch, which topics it uses, measured constants
vla_lab/real_robot/transport.py            UnixSocketTransport / LoopbackTransport (copy the shape from
                                           vla_lab/old_direction/rehab/apparatus/kinova_gen2.py)
vla_lab/real_robot/perception.py           table homography, HSV segmentation, gap measurement, lift/displacement checks,
                                           FakeCamera for tests
vla_lab/real_robot/calibration/            JSON files written by the calibration procedures (dated, hashed)
vla_lab/real_robot/safety.py               limits, dwell watchdog, halt taxonomy, e-stop state (pure Python, tested)
vla_lab/real_robot/run_sweep_real.py       the real-arm margin sweep, same rollouts.jsonl rows as run_sweep.py (+ scene_id)
vla_lab/real_robot/run_reach_real.py       the real-arm reach probe, same reach.json schema
vla_lab/real_robot/make_template.py        1:1 printable placement template (SVG/PDF) for every scene id
vla_lab/supervisory/apparatus/kinova_gen2.py   KinovaGen2Apparatus(Apparatus): the real robot behind the seam
vla_lab/supervisory/apparatus/grounders.py     HeuristicIntentGrounder (secondary channel surrogate)
vla_lab/supervisory/scheduler/scripted.py      ScriptedTimetableScheduler, WizardScheduler (+ registration, see §5)
vla_lab/human_study/live/                  the study UI and the human channel
    app.py                                 FastAPI app: /participant, /experimenter, /stream.mjpg, /api/*
    channel.py                             HumanSupervisorChannel(SupervisorChannel)
    tts.py                                 speak(text) -> onset/end timestamps; text-only fallback flagged in meta
    static/                                HTML/JS/CSS (vanilla; no CDN, machine may be offline)
vla_lab/human_study/run_participant.py     ★ the participant session runner (pre-flight, consent, blocks, questionnaires, debrief, gate)
vla_lab/human_study/contracts/human_v1.json    the frozen human contract (hash recorded in docs)
vla_lab/human_study/materials/             instructions, consent placeholder, debrief script, experimenter checklist (drafts for IRB)
vla_lab/human_study/analyze_sessions.py    the preregistered analysis over logs/human/*
vla_lab/supervisory/power.py               Monte-Carlo power for H1/H2 under the human protocol (the module map promises it; it does not exist yet)
vla_lab/tests/test_real_robot_*.py, test_live_ui.py, test_scripted_schedulers.py, test_run_participant.py
vla_lab/docs/real_robot_bringup.md         discovery, decisions, calibrations, procedures, measurements, open items
```

### 4.3 Runtime switches (all on `run_participant.py`; also on the sweep and rehearsal tools where relevant)

```
--backend  null | real          null = SurrogateApparatus (measured curves, instant); real = KinovaGen2Apparatus via the bridge
--channel  human | simulated    human = the UI; simulated = SimulatedSupervisorChannel(draw_supervisor(...)) for rehearsals
--brain    belief | heuristic-ear | fixed | wizard        (§5)
--conditions memoryless recommended                       (default; any registered condition name is accepted)
--contract vla_lab/human_study/contracts/human_v1.json    (default; hash checked; --allow-contract-change for rehearsals only)
--participant P001 --participant-idx 1                    (idx drives the Latin square and the first demonstration's direction)
--slots-per-block / --reference-slots / --retest-slots    (rehearsal knobs; changing them changes the contract hash and is refused
                                                          without --allow-contract-change)
--tts piper|espeak|none   --camera <source>   --bridge-socket /tmp/gen2_bridge.sock
```

Every combination must run. The four that matter:

| purpose | flags |
|---|---|
| logic rehearsal (seconds) | `--backend null --channel simulated` |
| UI rehearsal with a lab member, no arm | `--backend null --channel human` |
| arm rehearsal, no person | `--backend real --channel simulated` |
| the study | `--backend real --channel human` |

---

## 5. The surrogate scenarios (the operator chooses at run time)

All four share the protocol, the narration, the UI, the logs, and the analysis. Only what fills the free
slots and what grounds the utterance differs. The `condition` field in `trials.jsonl` must always name what
actually decided; never label a wizard's choice `recommended`.

**5.1 `belief` (default; the paper's system with the VLA stubbed).** Conditions run their own schedulers
(`build_scheduler`), the primary grounder is `LexicalGrounder(contract.axis)`, execution is the scripted
strategy on the real arm. This is what the simulation ran with the reference channel, so it is the
scenario whose numbers are directly comparable with the paper.

**5.2 `heuristic-ear` (a rule-based stand-in for the learned two-head listener).** As 5.1, plus a
`HeuristicIntentGrounder` passed as `second_grounder` to `run_block`, so its label lands in
`grounded_secondary` on every probe and the analysis can report agreement (`narration.grounding_agreement`)
exactly as it did for the trained heads. Rules, all logged in its `describe()`:
`said` = lexical grounding; an **echo flag** when the utterance contains words from the most recent coach
narration's rationale (`COACH_WHY`) or names the strategy demonstrated within the last `k` slots (default
3) while the scene is inside the crossover band; a `possible_echo` count per block. It is **never
authoritative** by default. `--primary-grounder heuristic` makes it primary for an explicit "what if the
listener were this heuristic" scenario; then `grounder` in the logs says `heuristic`, and the lexical
label is logged as secondary. Its abstention behaviour (returning `unresolved` on an echo) is a switch,
default off, because abstaining in the band starves the estimand, which is exactly what R11 in the paper
found for one learned channel.

**5.3 `fixed` (no belief module at all).** Deterministic timetables that a person could run with a
stopwatch: `fixed_washout` (B2, wait `w` free slots after each demonstration then probe),
`always_counter` (B4), and a new `scripted_timetable` scheduler in `scheduler/scripted.py` whose per-slot
action table is generated from the block budget at `reset()` (parameters: washout `w`, counter every
`k`-th free slot, both from the CLI) and logged in its `describe()`. Register `scripted_timetable` in
`scheduler/__init__.py` (`DISPLAY_NAMES`, `estimator_for` = psychometric, `build_scheduler`), add it to
nothing else (it is not a compared condition of the paper). Under this brain `--conditions` accepts only
these three names.

**5.4 `wizard` (experimenter-in-the-loop, robot still autonomous in execution and speech).** A
`WizardScheduler` whose `decide()` blocks on the experimenter page: for each free slot the experimenter
picks PROBE, COUNTER or WAIT; coach slots stay protocol-fixed and are not offered. A passive B7 instance
observes the same history and its rationale is shown to the experimenter as **advice** and logged in
`beliefs.jsonl` under `advisor`; the experimenter's choice is what runs and is logged with
`"decided_by": "wizard"`. The participant cannot tell this scenario from the others.

Tests: for one participant seed, `protocol.json` is byte-identical across all four brains; the set of keys
in `trials.jsonl` rows is identical across brains and across `--backend null|real`; coach slots, coach
directions and scene sequence are identical across brains; `verify_session` passes on a rehearsal of each.

---

## 6. Phase 3: build (no arm motion needed for any of this)

### 6.1 The driver bridge (ROS-side process) and its client

Follow the design already in the repo for the previous direction
(`vla_lab/old_direction/rehab/apparatus/kinova_gen2.py`): the study process never imports ROS; a bridge
process in the system Python owns the driver and speaks **newline-delimited JSON over a Unix socket**,
request/response matched by `id`, every response `{"id", "ok"}` plus payload or `{"error": ...}`.
Reuse that file's `Transport` protocol, `UnixSocketTransport`, `LoopbackTransport` and the fake-driver
pattern; put the copies in `vla_lab/real_robot/transport.py` and `gen2_bridge/fake_bridge.py`. Ops:

```
connect      -> {"driver": "...", "firmware": "...", "frame": "j2n6s300_link_base"}
state        -> {"ee_xyz": [..], "ee_quat_xyzw": [..], "fingers": [f1,f2,f3], "moving": bool, "estop": bool,
                 "fault": null|str, "joint_currents": [..], "t_ms": int}
home         -> blocking driver home (kinova-ros: service /j2n6s300_driver/in/home_arm); returns when settled
park         -> blocking move to the PARK pose (§6.2), the pose the arm rests in while cubes are placed
move_pose    -> {"xyz": [x,y,z], "quat_xyzw": [..] | "tool_down": true, "tol_m": 0.015, "timeout_s": 20,
                 "settle_dwell_ms": 500, "max_lin_speed": 0.10}  blocking; returns pose_error_m, elapsed_ms,
                 "reached": bool, "halted": null|reason
fingers      -> {"command": "open"|"closed"|"value", "value": [..], "timeout_s": 5} blocking; returns final finger values
stop / start -> driver stop and restart (kinova-ros: /j2n6s300_driver/in/stop, /in/start); stop is also what halt calls
estop_state  -> {"estop": bool, "source": ...}
capture      -> {"camera": "overhead"|"participant", "path": "<file>.jpg"} writes the latest frame; returns t_ms
ping / close
```

kinova-ros (ROS 1) expectations to **verify** against `rostopic list` / `rosmsg show` before writing a
line: Cartesian moves through the action `/j2n6s300_driver/pose_action/tool_pose` (`kinova_msgs/ArmPoseAction`,
goal is a `geometry_msgs/PoseStamped` in `j2n6s300_link_base`), fingers through
`/j2n6s300_driver/fingers_action/finger_positions` (`kinova_msgs/SetFingersPositionAction`, three raw
finger values, fully closed somewhere near 6800 for this hand; **measure** open, closed-empty and
closed-on-cube values), pose feedback on `/j2n6s300_driver/out/tool_pose`, fingers on
`/j2n6s300_driver/out/finger_position`, joints on `/j2n6s300_driver/out/joint_state`, homing and stop via
the services above. If the machine has `rosbridge_server` and `roslibpy` already working from another
project, using them for transport is acceptable, but keep the op vocabulary above so the fake bridge and
the tests stay valid.

Bridge behaviour that is not optional: a **dwell watchdog** (no motion op may run longer than its
`timeout_s`; on expiry, `stop`, return `halted: dwell_timeout`), **speed caps** below the driver
defaults (start at 0.10 m/s linear, raise only with the operator), a **workspace box** in the base frame
that rejects any target outside it before sending, a **heartbeat** (`ping` answered within 500 ms or the
client halts), and **typed halt reasons** from one taxonomy (reuse the names in
`vla_lab/old_direction/rehab/apparatus/base.py`: `estop_experimenter`, `dwell_timeout`,
`workspace_violation`, `speed_limit`, `torque_limit`, `driver_fault`, `heartbeat_lost`,
`participant_request`, `experimenter_request`). Every halt is an `events.jsonl` row
`{"kind": "halt", "reason": ...}`; `verify_session` fails a session whose halt has no reason.

### 6.2 `KinovaGen2Apparatus` (the robot behind the seam)

Geometry comes from the simulation's own functions, so it is identical by construction:
`layout_for_margin(scene.margin_m)` from `environments/supervisory_fetch/config.py` gives target, blocker,
drop-off and distractor positions in the base frame (table top `z = 0`, target at `(0.48, -0.10)` m,
blocker toward the robot along `x`, drop-off perpendicular on the `+y` side); `waypoints_for(strategy,
layout, cfg=ExpertConfig(...))` from `experts.py` gives the ordered waypoints with per-waypoint gripper
command, phase name and tolerance. DIRECT: `transit, detour, pregrasp, descend, close, lift`. CLEAR_FIRST:
`transit_to_push, push_start, push, push_retract, transit_to_target, pregrasp, descend, close, lift`.

The real arm will need different heights than the simulated one (the push height 0.075 m, grasp depth
-0.02 m below the cube's top face, transit 0.22 m, pregrasp 0.12 m, lift 0.18 m, detour 0.07 m were all
**measured in Isaac** and each has a comment explaining the measurement). Put real-arm overrides in a
`RealExpertOverrides` dataclass inside your apparatus config, stamped into `describe()`, and change them
only by the measurements of §7. Never edit the defaults in `experts.py`.

Per-slot choreography (the participant sees the robot side of this; the experimenter sees the console):

```
reset_scene(scene):
  1. arm at PARK (if not, park; refuse if a person is flagged in the workspace)
  2. experimenter console shows: scene id, gap in cm, target/blocker/distractor positions in cm from the base origin,
     the template row to use, and (if perception is calibrated) a live overlay of measured cube centres and the measured gap
  3. experimenter places cubes; console shows measured gap vs nominal, green when |error| <= tolerance (§6.3)
  4. experimenter presses CONFIRM ("cubes placed, hands clear")   -> placements.jsonl row: nominal, measured, error, image path, t_ms
  5. return

say(text):
  TTS speaks the exact string and the participant page shows it at the same moment; events.jsonl row
  {"kind": "robot_say", "text": text, "sha1": ..., "onset_ms": ..., "end_ms": ...}

execute(scene, strategy):
  1. safety: workspace check for every waypoint; bridge ping; e-stop clear
  2. run waypoints in order through move_pose / fingers with the waypoint's tolerance; log each phase's start/end and pose error
  3. success determination (§6.4) and blocker displacement (perception, before/after)
  4. return ExecutionOutcome(strategy, success, duration_s = first motion command -> lift complete,
        notes = {phases: [...], halted: null|reason, success_source: "auto"|"experimenter", measured_margin_m, blocker_displacement_m,
                 target_lifted_evidence: {...}, real_overrides_hash, calibration_hash})
  5. park (the target is still in the hand: open the fingers over the drop zone the operator designates, then park)
```

The WAIT action needs no motion: `say(filler)` is called by the session runner; the apparatus does nothing
else, and the runner accounts the contract's `wait_s`. Make the participant page show a neutral "the robot
is busy" state during waits and executions so the participant is never left wondering whether to type.

`describe()` must include: backend `kinova_gen2`, driver/firmware strings from `connect`, the bridge
version, the calibration file hashes, the `RealExpertOverrides`, the perception config, the TTS engine and
voice, the speed caps, and the git SHA. `meta.json` picks this up through `run_session`.

### 6.3 Placement precision, the scene grid, and the physics decision

This is the part of the physical study that is genuinely hard, and it must be surfaced to the operator, not
solved silently. The 9 crossover-band scenes span 4.09 to 4.99 cm of gap in **1.5 mm** steps, because the
measured transition width is 0.50 cm. A person placing two cubes by eye cannot realise 1.5 mm steps.
Three things follow:

1. **Placement is jig-based and measured.** `make_template.py` prints a 1:1 template (target square fixed,
   one blocker outline per scene id with the gap in mm printed beside it, drop-off marked, distractor rings
   marked) for the experimenter to tape to the table; recommend a simple mechanical stop (a 3D-printed or
   laser-cut comb with one slot per scene, or a slide with a vernier) if the operator can make one. In all
   cases the **measured gap** from perception goes into `placements.jsonl` and into the execution notes,
   and the analysis is prepared to use `measured_margin_m` (recomputing `c` through the contract's
   physics) instead of the nominal scene id. Placement tolerance: start at 3 mm; report the achieved
   distribution after rehearsal.
2. **The real physics decides the grid.** The transition width on the real hand is unknown and probably
   larger, which would make the band wider and the placement problem easier. §7 step 11 is a short
   reality check (2 strategies × 5 gaps × 6 repetitions, about an hour) followed, if the operator agrees,
   by the full sweep (`run_sweep_real.py`, same gaps and repetitions as the Isaac sweep: 0 to 16 cm plus
   a dense tight end, 12 or more repetitions per (strategy, gap)). The sweep writes `rollouts.jsonl` in
   the schema `run_sweep.py` writes (one JSON object per rollout with `margin_m, strategy, rep, success,
   duration_s` plus the apparatus notes flattened in; add `scene_id` as well, `fidelity_report` groups on
   it), so the existing fitter runs unchanged:
   `python -m vla_lab.supervisory.apparatus.measure fit <rollouts.jsonl> --bootstrap 2000 --out vla_lab/results/physics_real/physics.json`
   (the `--out` is the physics **file**; the report and figure land beside it)
   and `Contract(grid=build_scene_grid(physics=load_physics(...)))` rebuilds the grid. Any session run
   under a different physics has a different contract hash and is not poolable with the others; that is
   the intended protection, not a bug.
3. **Until the real physics exists, sessions may use the simulation grid** (Appendix A) with `source`
   recorded as `measured` (Isaac) and a note in `contract.notes` saying so. Rehearsals should. Whether the
   **pilot** runs on the Isaac grid or waits for the real fit is the operator's decision; put it to them
   with the reality-check numbers in hand (Appendix E, Q8).

### 6.4 Perception (`vla_lab/real_robot/perception.py`)

If colour segmentation is the method (confirm in discovery; fiducials are an acceptable alternative with the
same outputs):

- **Table homography** from >= 4 correspondences between overhead-image pixels and base-frame table
  points. Two ways to get correspondences, use both if you can: a printed calibration sheet placed
  against the arm's base with known offsets, and **touch-off** (jog the tool tip to a point, read the base
  frame pose from `state`, click the pixel). Solve with the DLT + SVD routine already in
  `vla_lab/old_direction/rehab/observation/calibration.py` (pure NumPy; import it or copy it with
  attribution). Save `calibration/table_homography_<date>.json` with residuals; the apparatus refuses to
  run if the calibration is older than a configurable age or its residual exceeds 3 mm.
- **Segmentation**: HSV thresholds for red and blue (and the distractor colours, to reject them),
  morphological cleanup, largest blob per colour, centroid to table plane through the homography, cube
  centre offset by half the cube height accounted for by the calibration procedure (calibrate on the cube
  top face, not the table). Outputs: `target_xy_m`, `blocker_xy_m`, `measured_margin_m = centre distance −
  cube_size`, confidence, image path.
- **Checks used by the apparatus**: placement error before a slot; `blocker_displacement_m` and
  `target_displaced` after execution; `target_lifted` evidence (target absent from its table position
  after the lift waypoint, combined with finger values; §6.5).
- **Validation procedure** (§7 step 8): five gaps set with a steel rule, ten frames each; report mean and
  worst error; target < 2 mm mean, < 3 mm worst. Record the lighting condition; re-validate if lighting
  changes.
- `FakeCamera` renders synthetic frames with coloured squares at given base-frame positions through the
  inverse homography, so segmentation and gap measurement are unit-tested without hardware.

### 6.5 Success determination

`success` in the simulation meant the target was lifted at the end of the routine. On hardware, decide it
from evidence and log the evidence:

- fingers after `close`: final values inside the calibrated **closed-on-cube band** (between fully open and
  closed-empty, both measured in §7 step 6), and
- after `lift`: perception no longer finds the red cube at its pre-grasp table position (moved by more
  than 2 cm or absent), and no `halted` reason.

If the two disagree, or perception confidence is low, the experimenter console asks "Did the robot lift the
red box? [y/n]" and the outcome records `success_source: "experimenter"`. Never silently default. Also log
`blocker_displacement_m` (the simulation's value model charges a disturbance penalty; on hardware it is a
measured quantity the physics refit can use).

### 6.6 The participant UI and the human channel (`vla_lab/human_study/live/`)

A local web app (FastAPI + uvicorn, vanilla HTML/JS, no external assets; the machine may be offline). Two
pages on two displays:

**`/participant`** (the participant's screen, behind the barrier):
- the live camera feed (MJPEG from `/stream.mjpg`; source per discovery: direct capture in the 3.11 env or
  frames relayed by the bridge),
- the robot's speech panel: the current utterance in large type, the last few below it,
- a status line: "The robot is working", "The robot is asking you", "Please answer", "Break",
- a text box and a Send button, enabled only while a question is pending; Enter sends; empty input is
  refused with a neutral hint ("Please type what the robot should do"); no suggestions, no option
  buttons, no autocomplete, nothing that names either strategy (that would answer the question the study
  asks),
- questionnaire pages at block boundaries: NASA-TLX six subscales as 0 to 100 sliders with the standard
  anchors, the 12-item trust scale on 1 to 7, the short burden items; scored by
  `vla_lab/human_study/instruments.py` and written to `questionnaires.jsonl`,
- a **Pause** button (logs `participant_pause`, the runner finishes the current motion, parks, waits) and
  a **Stop** button (logs `participant_stop`; the session ends cleanly and the gate still runs),
- nothing that reveals the block kind, the condition, the scene id, the coordinate, the belief, or
  whether an answer was grounded.

**`/experimenter`** (the experimenter's laptop or second display):
- session header: participant id, block index and kind, condition name, slot counter, elapsed time,
- the placement panel of §6.2 with the live overlay and CONFIRM,
- HALT (typed reason picker) and a big e-stop mirror showing the driver's e-stop state,
- the last utterance and its grounding result (visible to the experimenter only),
- the wizard action panel when `--brain wizard`, with the advisor's suggestion,
- the success confirmation prompt of §6.5 when needed,
- a free-text notes field written to `events.jsonl` as `experimenter_note`.

`HumanSupervisorChannel.ask(query, scene, *, action, session_progress)` posts the question to the
participant page (the apparatus has already spoken it), blocks until Send, returns
`SupervisorTurn(utterance=text, latency_s=submit_time − speech_onset, truth=None)`. `observe_demonstration`
and `elapse` do nothing except an `events.jsonl` row; the human is watching and living in real time.
`describe()` returns `{"channel": "human", "modality": "typed", "ui_version": ..., "tts": ...}`.

TTS (`tts.py`): use what discovery found (prefer an offline neural voice such as piper if present, else
`espeak-ng`); one fixed voice and rate for the whole study, recorded in `meta.json`; `speak()` returns
onset and end timestamps; if no audio device works, fall back to text-only and set
`"tts": "none (text only)"` so the analysis knows. The spoken string and the displayed string are the same
object, the one the session runner passed to `say()`.

Latency and timing: the session runner adds the contract's nominal overheads (`probe_overhead_s`,
`counter_overhead_s`, `inter_slot_s`) to `duration_s`; the realised times are all in `t_ms` of
`trials.jsonl` and in your `robot_say` and `placement` events, so the analysis can recompute deltas from
wall-clock if the nominal constants turn out to be far off. Measure the real overheads in rehearsal (§7 step
9) and, before the pilot, set `TimingConfig` in the human contract to the measured means, then freeze it.

### 6.7 The participant runner (`vla_lab/human_study/run_participant.py`)

Order of operations, enforced in code:

1. **Pre-flight** (all must pass or the runner refuses): contract file loads, `contract.check()` is empty,
   its hash equals the frozen hash in `contracts/human_v1.json` (or `--allow-contract-change` for
   rehearsals, which stamps `rehearsal: true` into `meta.json`); bridge `connect` + `ping`; e-stop clear;
   arm homed and parked; calibration present, fresh, within residual; camera streaming; TTS test phrase
   spoken; disk space > 5 GB; `logs/human/<PID>/` does not already exist (never overwrite a participant).
2. **Consent and instructions** on the participant page (materials from `human_study/materials/`; drafts
   written by you from the paper's §7, marked "DRAFT, IRB approval required"; the operator edits them).
   The instructions explain: the robot will pick up the red box many times; sometimes it will announce what
   it is going to do; sometimes it will ask you how to approach it; answer in your own words by typing;
   there is no right answer; you may pause or stop at any time. They do **not** mention demonstrations
   as a manipulation (the debrief does).
3. **Protocol** drawn and written before slot 1:
   `build_protocol(supervisor_id=PID, contract=contract, seed=<derived from participant idx and a study
   seed recorded in the contract notes>, conditions=[...], order_index=participant_idx)`; save
   `protocol.json`; print the block plan to the experimenter page only.
4. **Blocks**, through `run_session` (or your faithful reproduction of it), with the real apparatus and
   channel; between blocks: questionnaire, then the enforced rest (`contract.timing.block_rest_s` of
   wall-clock and the runner's `inter_block_rest_deltas` decay units, whichever is longer, shown as a
   countdown labelled "Break"), then `block_start` of the next block. The condition name never appears on
   the participant page.
5. **Debrief** page (the deception disclosure script, questions invited, logged `debrief_shown`).
6. **Gate**: run `vla_lab.supervisory.verify_session.verify(log_root)` automatically, print the result to
   the experimenter, write `gate.json`. A failing gate does not delete anything; it marks the session
   non-poolable.
7. Files in `logs/human/<PID>/`: the standard six (`trials.jsonl`, `beliefs.jsonl`, `events.jsonl`,
   `contract.json`, `protocol.json`, `meta.json`) with exactly the standard keys, plus
   `questionnaires.jsonl`, `placements.jsonl`, `robot.jsonl` (bridge telemetry per phase), `gate.json`,
   and `media/` (one overhead frame per slot before and after execution; add `logs/human/**/media/` to
   `.gitignore`, the pattern for the previous direction's media is already there).

Pause/stop semantics: a pause finishes the current motion, parks, and holds the slot open; a stop ends the
session after parking; both are typed events; a halted slot is recorded with `halted` in the execution notes
and the runner asks the experimenter whether to repeat the slot (repeat = same slot index, event
`slot_repeated`) or end the session. `verify_session` will flag inter-slot gaps over ten minutes; that is
expected after a pause and should be annotated in the experimenter notes.

### 6.8 The human contract (`vla_lab/human_study/contracts/human_v1.json`)

Start from `Contract(budget=BudgetConfig(coach_regime="alternating"))` (hash `8254f4654c1d8256` with the
Isaac physics) and record in `notes`: physics provenance (Isaac measured, or real sweep file and date),
placement tolerance, TTS engine, study seed. After §7 step 9 set `TimingConfig` to the measured overheads.
Then freeze: write the file, record its hash in `docs/real_robot_bringup.md`, and make the runner refuse a
different hash unless `--allow-contract-change`. Every participant in a cohort must share one hash;
`verify_session --pool` checks it.

### 6.9 Phrase corpus packet from real photographs

`python -m vla_lab.human_study.phrase_corpus prepare --frames <dir>` expects
`<dir>/figure/scene_<id:03d>/*.png` (or `topdown/`) for the six packet scenes it picks from the grid. If
the operator copied `vla_lab/results/physics/frames/`, `prepare` works as it is with the Isaac renders;
real photographs of this table are better and are what the corpus should use: add a small tool that places
the cubes for each of those six scene ids (experimenter, template), captures a three-quarter-view
photograph with the participant camera and an overhead frame, and writes them into that layout; then run
the unchanged `prepare --frames <that dir>`, `evaluate`, `rebuild`. The instruction text, the response sheet and the 0.85
grounding-rate gate stay as they are. The corpus needs about ten real people; it involves no robot motion
in front of them and no deception. It is the gating step before any pilot, per the paper.

### 6.10 Power for H1/H2 (`vla_lab/supervisory/power.py`)

The package docstring promises a Monte-Carlo power module and it does not exist. Write it: draw cohorts
with `vla_lab.supervisory.supervisor.draw_cohort` at several compliance strengths, run the **human
protocol** (alternating regime, the human contract's budget) with `SurrogateApparatus` and
`SimulatedSupervisorChannel` through `run_session`, fit each participant with
`vla_lab.supervisory.estimand.joint_carryover_posterior`, and report, as a function of N: the probability
that the post-demonstration elevation at band scenes is detected (H1), the precision on the between-person
spread of `beta·g` (H2), and the fraction of participants whose posterior meets the contraction criterion
(`CarryoverPosterior.identifiability`). Deterministic seeds, a table and a figure, no correlation
coefficient below n = 15 (`vla_lab.stats_utils.guarded_correlation`). This tells the operator what N the
pilot's measured effect implies.

### 6.11 The analysis over human sessions (`vla_lab/human_study/analyze_sessions.py`)

Implements the preregistered order exactly, over `logs/human/*` that pass the gate, and runs unchanged on
rehearsal sessions from `--channel simulated` so it is tested before any person sits down:

1. manipulation checks (realised budget, demonstration count, scene-sequence equality across condition
   blocks, grounding rate, inter-channel agreement when a secondary grounder ran, placement error
   distribution);
2. reference stability (test-retest between the initial and terminal no-coach blocks, with interval);
3. H1, H2: post-demonstration elevation and decay at band scenes; per-participant `beta·g` posteriors;
   contraction and total-variation identifiability side by side with the non-complier control column;
4. H3 model-based: off-policy contrast recommended vs memoryless on each participant's fitted parameters,
   labelled model-based;
5. H4 burden: counter-proposal counts and wall-clock;
6. H5 coverage and calibration;
7. secondary: estimation error, task success, NASA-TLX, trust, burden;
8. heterogeneity: per-participant effects plotted individually; the code refuses a correlation coefficient
   below fifteen participants.

Use the estimators as they are (`sequence_from_records`, `reference_map_from_observations`, `fit_all`,
`joint_carryover_posterior`, `evaluate`, `decision_regret`); provide a `--use-measured-margin` switch that
recomputes each observation's `c` from `measured_margin_m` through the contract physics.

### 6.12 Tests and docs

Every new module has tests that run with no ROS, no camera, no arm, no browser: the fake bridge with
configurable faults (settle failure, driver fault, timeout, e-stop mid-move, heartbeat loss), `FakeCamera`
frames for segmentation and gap error, a `ScriptedHuman` channel, FastAPI's `TestClient` for the UI
endpoints, the four-brain equivalence tests of §5, a `run_participant` end-to-end test with
`--backend null --channel simulated` on a shortened contract that asserts the session directory contains
the standard files with the standard keys and passes the gate. Register the modules in
`vla_lab/tests/run_tests.py` so `run_tests --count` includes them. Keep the existing suite green.

`vla_lab/docs/real_robot_bringup.md` holds: discovery (§3), decisions, the bridge protocol, the frame and
calibration procedures with their measured numbers, the gripper values, the real expert overrides and how
each was measured, the timing measurements, the placement-tolerance results, the reality-check physics and
the sweep (if run), the frozen contract hash, the rehearsal log, the pilot checklist status, and an honest
"not yet verified" list. Add a short section to `vla_lab/README.md` pointing at it and at
`run_participant.py`.

---

## 7. Phase 4: hardware bring-up (the arm moves; the operator is present; each step confirmed)

Do these in order. Before each step print the plan, the speed cap, and how to stop. After each, write the
measurement into the docs and, where it is a constant, into a calibration file or the apparatus config.

1. **Bridge smoke, no motion**: launch the driver as the operator does; start the bridge; `connect`,
   `state`, `ping` from the 3.11 env; confirm the reported frame and units (metres, quaternion order).
2. **Stop path**: with the arm still, exercise `stop`/`start` and the physical e-stop; confirm the bridge
   reports `estop` and that a `move_pose` issued during e-stop is refused with `halted: estop_experimenter`.
3. **Home and PARK**: `home` (driver homing; watch the whole motion). Choose the PARK pose with the
   operator: out of the camera's view of the table, clear of the placement area, reachable in one move
   from every waypoint's transit height. Record it.
4. **Frame calibration by touch-off**: jog (through `move_pose` at low speed, or by hand if the driver
   supports it) to three or four marked table points and to the intended target position; read
   `state.ee_xyz`; solve the base-frame to study-frame offset and yaw and the table height `z_table`
   (the simulation assumes `z = 0` at the table top; on this arm it will be some measured value; the
   apparatus adds it to every waypoint). Verify the target `(0.48, −0.10)` and the widest blocker
   position and the drop-off are all inside the reachable envelope; if not, choose a new corridor with the
   operator and record that the Isaac physics then no longer applies (real sweep required).
5. **Tool-down orientation**: capture the quaternion of a tool-pointing-down pose (from another project's
   code, or posed by hand); confirm `move_pose(tool_down=True)` holds it during a slow horizontal move
   (the Isaac work found that the wrist drifted when orientation was not actively held; check for the
   same on hardware).
6. **Gripper**: measure finger values open, closed on nothing, closed on a cube; derive the closed-on-cube
   band; measure close and open durations.
7. **First executions at a safe gap**: a 12 cm gap, DIRECT, at 0.05 m/s; then CLEAR_FIRST at 12 cm. Tune
   the real overrides (transit, pregrasp, grasp depth relative to the cube's top face, push height with a
   closed hand, push backoff and overshoot, detour) exactly as the Isaac work did, and write each
   measurement down (the comments in `experts.py` show the format and the pitfalls: the push height was
   found by a sweep, the grasp depth references the top face, a closed hand touches the table earlier
   than an open one). Then a 6 cm gap and a 3 cm gap.
8. **Camera calibration and validation** (§6.4), then placement rehearsal: ten placements of scene 7
   (4.54 cm) by the experimenter with the template; report the measured-gap error distribution.
9. **Timing**: measure the realised probe overhead (speech onset to answer submit, with a lab member),
   counter overhead, inter-slot placement time, a full COACH, a full PROBE with each strategy; set the
   human contract's `TimingConfig`; note what it implies for session length (the simulation's own timing
   model puts a 60-slot block at 28 to 34 minutes; the full 190-slot session with placement time is
   likely above two hours; tell the operator, because a shorter budget is a contract-level decision).
10. **Rehearsals**: `--backend real --channel simulated` on a shortened contract (arm only); then
    `--backend real --channel human` with a lab member on the frozen contract; gate must pass; save the
    session as `logs/human/REHEARSAL_*/`.
11. **Physics reality check**, then the sweep if the operator agrees (§6.3): 2 strategies × gaps
    {2, 3.5, 4.5, 5.5, 8} cm × 6 repetitions first; compare against the Isaac curves (Appendix A); then the
    full sweep and `measure fit --bootstrap 2000`; report crossover, width and their intervals next to the
    Isaac values; rebuild the grid and the human contract if the operator chooses the real physics.
12. **Freeze the contract** (§6.8).

---

## 8. Phase 5: pilot readiness (the go/no-go the paper commits to)

Produce `docs/real_robot_bringup.md` §"Pilot readiness" with each item marked done / not done / not
applicable, with evidence:

- phrase corpus collected from about ten people, grounding rate >= 0.85, or the documented revision of
  grounder or threshold before recruitment;
- the frozen human contract, its hash, its physics provenance;
- a full rehearsal session that passes `verify_session`, with realised timing and session length;
- placement error distribution within tolerance; calibration residuals;
- safety demonstrations recorded (e-stop during motion, dwell timeout, workspace rejection, heartbeat
  loss), each as an `events.jsonl` halt with a typed reason;
- the analysis script run end-to-end on rehearsal sessions;
- power table for H1/H2 at the pilot's compliance strength (§6.10);
- materials (instructions, consent, debrief) drafted and handed to the operator for IRB; IRB status is the
  operator's item, record what they tell you;
- the open items you could not close, stated plainly.

---

## 9. The participant's experience, as the equivalence checklist

Walk through this yourself in a `--backend null --channel human` rehearsal and tick each line:

1. The participant sits behind the barrier, sees the robot and the table directly and a screen with the
   live feed, the robot's words, and a text box. They are told there is no right answer.
2. **Reference block (40 slots).** The robot never announces a plan. Each slot it says "How should I
   approach this one?", the participant types, the robot does what it understood (or the value-optimal
   strategy if it could not ground the answer, without saying so), the experimenter re-places the cubes.
   Gaps vary between roughly 3.3 and 5.7 cm.
3. Questionnaire, then a timed break.
4. **Condition block 1 (60 slots, 10 of them demonstrations).** On a demonstration the robot says one of
   the three fixed coach sentences (for example "I'll move the blocking box aside first, then pick up the
   target. Clearing the path first keeps the grasp safe.") on a clearly wide or clearly tight scene, and
   does it. Directions alternate demonstration by demonstration. On the other slots the condition decides:
   under `memoryless` the robot always asks the plain question right away; under `recommended` it
   sometimes asks the counter-proposal form ("How should I approach this one? I could also go straight
   for the target without moving anything, which would you prefer?") and never waits; under a fixed
   washout it sometimes says a filler ("Give me a moment while I re-check the workspace.") and idles.
5. Questionnaire, break, **condition block 2**, questionnaire, break.
6. **Retest block (30 slots)**, like the reference block.
7. Final questionnaire, debrief (the demonstrations were a manipulation; what was measured; questions).
8. At no point did the screen show a condition name, a scene number, a probability, or whether an answer
   was understood; the robot's words were exactly the strings in `narration.py`; every utterance, answer,
   placement, motion phase and halt is in the session directory with a monotonic timestamp.

Pass/fail checks to run in code: E1 the hashes of §2.7; E2 `protocol.json` identical across brains and
backends for one participant seed; E3 `trials.jsonl` keys identical between a `null/simulated` session and
a `real/human` session; E4 every `robot_say` text equals a string producible by `narration.py` (regenerate
from the axis and compare); E5 `verify_session` passes on every rehearsal; E6 the participant page's HTML
contains none of the strings `clear`, `direct`, `push`, `straight`, `condition`, `memoryless`,
`recommended`, `scene`, `belief` outside the robot's own utterances.

---

## 10. Deliverables and how to report

- Code under the paths in §4.2, tests green (`python -m vla_lab.tests.run_tests` for the modules that run
  here; list the skipped torch modules), the four brains runnable in all four backend/channel combinations.
- `vla_lab/docs/real_robot_bringup.md` complete, with every measurement and its command, and the "not yet
  verified" list.
- `vla_lab/human_study/materials/` drafts.
- The frozen contract and its hash.
- A final message to the operator that states, in this order: what runs on hardware today and was seen to
  run; what was built but only tested against fakes; what the operator must decide (Appendix E items
  still open); and the readiness checklist status. No number in that message that did not come from a
  file on this machine.
- If the tree has a `.git`, commit your work on a branch named `robot-bringup`; otherwise write
  `vla_lab/docs/CHANGELOG_robot_bringup.md` listing every file you added or changed, so the operator can
  carry the work back to the laptop by hand. Never push anything from this machine.

---

## Appendix A. Files to install under `vla_lab/results/` (only if the operator did not copy `results/physics` and `results/reach`)

`vla_lab/results/physics/physics.json`  (point estimate; 408 Isaac rollouts; lapse-logistic fit; crossover
0.04537 m, width 0.00499 m; bootstrap 95% CI on the crossover 3.9 to 5.2 cm, on the width 0.17 to 0.81 cm):
```json
{
  "p_a_asym": 0.95,
  "a_slope": 256.5047677449502,
  "a_mid": 0.01035951205106347,
  "p_b_asym": 1.0,
  "b_slope": 200.45676599087412,
  "b_mid": 0.03816938549036824,
  "p_a_floor": 0.1,
  "p_b_floor": 0.2,
  "t_a": 13.39745484400657,
  "t_b": 8.998611111111112,
  "t_b_tight": 0.0,
  "reward": 1.0,
  "time_cost": 0.012,
  "disturbance_penalty": 0.05,
  "source": "measured",
  "n_measured": 408,
  "fit_method": "lapse_logistic",
  "quantile": "point"
}
```

`vla_lab/results/physics/physics_lower.json`:
```json
{
  "p_a_asym": 0.95,
  "a_slope": 211.1572719557476,
  "a_mid": 0.012837428438085751,
  "p_b_asym": 1.0,
  "b_slope": 577.4523568345459,
  "b_mid": 0.038223586325808925,
  "p_a_floor": 0.2,
  "p_b_floor": 0.2,
  "t_a": 13.411666666666667,
  "t_b": 9.027006172839506,
  "t_b_tight": 0.0,
  "reward": 1.0,
  "time_cost": 0.012,
  "disturbance_penalty": 0.05,
  "source": "measured",
  "n_measured": 408,
  "fit_method": "lapse_logistic",
  "quantile": "lower"
}
```

`vla_lab/results/physics/physics_upper.json`:
```json
{
  "p_a_asym": 0.95,
  "a_slope": 339.7877980462833,
  "a_mid": 0.013393467019966626,
  "p_b_asym": 1.0,
  "b_slope": 123.02671888298697,
  "b_mid": 0.03487846876837479,
  "p_a_floor": 0.2,
  "p_b_floor": 0.1,
  "t_a": 13.383825944170773,
  "t_b": 9.010493827160493,
  "t_b_tight": 0.0,
  "reward": 1.0,
  "time_cost": 0.012,
  "disturbance_penalty": 0.05,
  "source": "measured",
  "n_measured": 408,
  "fit_method": "lapse_logistic",
  "quantile": "upper"
}
```

`vla_lab/results/reach/reach.json` (the simulated arm's reachable envelope at grasp height; replace with
the real probe's output from `run_reach_real.py` when it exists, keeping this file as
`reach_isaac.json`):
```json
{
  "per_height": {
    "0.020": {"n": 60, "reached": 55, "fraction": 0.9166666666666666, "x_min_reached": 0.22, "x_max_reached": 0.66, "median_err_m": 0.018851090262279685},
    "0.120": {"n": 60, "reached": 57, "fraction": 0.95, "x_min_reached": 0.22, "x_max_reached": 0.66, "median_err_m": 0.018255802458454908}
  },
  "recommendation": {
    "reachable_x_at_lowest_z": [0.22, 0.66],
    "lowest_z_probed": 0.02,
    "suggested_target_x": 0.5720000000000001,
    "suggested_blocker_x_at_14cm": 0.38200000000000006,
    "blocker_inside_band": true,
    "note": "blocker fits on the near side"
  },
  "n": 120, "n_probed": 120, "n_dropped_bad_home": 0, "home_tol_m": 0.05
}
```

---

## Appendix B. The contract as it stands (regenerate from code; do not retype into the code)

Hashes with the Appendix A physics: simulation contract `30df16367d79b38d`, narration `30e0921cb87a7589`,
alternating-regime candidate for the human study `8254f4654c1d8256`.

Scene grid (`build_scene_grid()` under the Appendix A physics; gap = free gap between cube faces, cubes 5 cm,
centre distance = gap + 5 cm; target fixed at `(0.48, −0.10)` m, blocker toward the robot along `x`):

| scene id | gap (cm) | c | clutter | role | in band | value-optimal |
|---|---|---|---|---|---|---|
| 0 | 5.73 | −2.40 | 2 | probe | | B |
| 1 | 5.53 | −2.00 | 4 | probe | | B |
| 2 | 5.33 | −1.60 | 2 | probe | | B |
| 3 | 5.14 | −1.20 | 4 | probe | | B |
| 4 | 4.99 | −0.90 | 2 | probe | yes | B |
| 5 | 4.84 | −0.60 | 4 | probe | yes | B |
| 6 | 4.69 | −0.30 | 2 | probe | yes | B |
| 7 | 4.54 | 0.00 | 4 | probe | yes | B |
| 8 | 4.39 | +0.30 | 2 | probe | yes | A |
| 9 | 4.24 | +0.60 | 4 | probe | yes | A |
| 10 | 4.09 | +0.90 | 2 | probe | yes | A |
| 11 | 3.94 | +1.20 | 4 | probe | | A |
| 12 | 3.74 | +1.60 | 2 | probe | | A |
| 13 | 3.54 | +2.00 | 4 | probe | | A |
| 14 | 3.34 | +2.40 | 2 | probe | | A |
| 15 | 6.13 | −3.20 | 4 | demonstration (B) | | B |
| 16 | 6.13 | −3.20 | 2 | demonstration (B) | | B |
| 17 | 2.94 | +3.20 | 4 | demonstration (A) | | A |
| 18 | 2.94 | +3.20 | 2 | demonstration (A) | | A |

Contract constants (from `vla_lab/supervisory/contract.py`; the human contract changes only
`coach_regime` and, after measurement, `TimingConfig`):
budget `slots_per_block 60, coach_per_block 10, reference_slots 40, retest_slots 30, coach_regime
one_sided (simulation) / alternating (human)`; timing `probe_overhead_s 12, counter_overhead_s 26, wait_s 45,
inter_slot_s 4, block_rest_s 60, probe_s 42, counter_s 58, coach_s 45`; carryover `decay_mode time,
time_unit_s 30, rho_counter 0.6 (prior)`; dose `moderate` (one demonstration with rationale); gates
`min_grounding_rate 0.85, min_band_scenes 6`.

Lexical grounder anchors (`strategies.py`, `plan` axis): A if the utterance contains any of
`clear, aside, out of the way, move the, push, first` and none of the B anchors; B if it contains any of
`direct, directly, straight, around, thread, leave the, just grab, just take` and none of the A anchors;
otherwise `unresolved`. The phrase corpus exists to find out whether real people's words hit these anchors.

Robot speech (`narration.py`; hashed into the contract): coach templates `"I'll {what}. {why}"`,
`"{why} So I'll {what}."`, `"Approaching this one as follows: I'll {what}. {why}"`; for A `what` = "move
the blocking box aside first, then pick up the target", `why` = "Clearing the path first keeps the grasp
safe."; for B `what` = "go straight for the target and leave the other box where it is", `why` = "Going
direct saves time and leaves the workspace as it is."; probe "How should I approach this one?"; counter
templates "How should I approach this one? I could also {alt} -- which would you prefer?" and "How should I
approach this one? Either works here; I can {alt} instead if you'd rather." with `alt` = "clear the blocking
box out of the way first" (A) / "go straight for the target without moving anything" (B); wait fillers "Give
me a moment while I re-check the workspace.", "Recalibrating my cameras, one moment.", "Logging the last
result. Stand by.", "Just resetting the scene."

---

## Appendix C. Simulation lessons that transfer to the physical arm (read before tuning anything)

From the Isaac bring-up, each recorded at the line of code that fixes it:

1. The scripted routines' heights were all measured, not derived: the push is done with a **closed** hand
   at 0.075 m because a closed three-finger hand reaches the table well before its tool frame does; the
   grasp closes 2 cm **below the cube's top face** (measured against the cube centre it lands below the
   table and the descent stalls at a fixed height no matter the step budget); the direct approach detours
   7 cm away from the blocker so the fingers come down on the far side. Expect every one of these to move
   on the real hand; measure them the same way (a small sweep, write down each value's outcome).
2. Sweeping the blocker **sideways** (perpendicular to the target-blocker axis) is what keeps clearing
   feasible at tight gaps; pushing along the corridor moved the blocker 5 mm at a 2 cm gap and made both
   strategies fail at the same boundary. `layout_for_margin` already computes the perpendicular drop-off;
   do not "simplify" it.
3. Orientation must be actively held during translation or the wrist drifts.
4. A waypoint that only changes the gripper is done when the gripper has moved, not when a position
   tolerance is met.
5. Starting pose matters: the simulated study begins every routine from a neutral pose; on hardware PARK
   plays that role, choose it so both strategies start equally.
6. Placement is verified from live measurement, never assumed: a 23 cm object displacement went undetected
   in simulation until the poses were read back. Read the cubes back from the camera every slot.
7. Regularisers have units: the original physics fit flattened a per-metre slope with a prior meant for
   per-centimetre coefficients and reported a 3.12 cm width that was really 0.50 cm. If you fit anything on
   real data, check the prior's strength in the coefficient's natural units.
8. Only the upper crossing of the value gap is the crossover; near 0.6 cm both strategies fail and the
   cheaper failure "wins", which is a second, trivial crossing. `crossover_margin()` returns the upper one.

---

## Appendix D. What to send back to the operator's laptop

Nothing needs to go back for the simulation work. For the paper, the operator will want: the bring-up doc,
the frozen human contract, the real physics fit if it was run (`physics_real/physics_report.json`), the
timing measurements, and the placement-error and calibration numbers. Put them in one dated folder
`handoff_<date>/` and say so in the final message.

---

## Appendix E. Questions to ask the operator when discovery cannot settle them

Ask these when you reach them, not all at once, and record the answers in the bring-up doc:

- Q1. Which command launches the arm driver, and does anyone else's code assume it is already running?
- Q2. Is the arm bolted to the table top? What is the table height relative to the base plate? Which way
  does the base frame's `x` axis point relative to the participant?
- Q3. Which camera should be the participant's live feed, and which one is the overhead verifier? Is the
  lighting fixed during sessions?
- Q4. Cube dimensions and colours as they really are; are there distractor cubes, and how many?
- Q5. Where exactly does the participant sit, where is the barrier, where does the experimenter sit, where
  is the e-stop, and is there a second stop the participant can reach?
- Q6. Maximum speed you are comfortable with for the arm in front of a participant?
- Q7. TTS voice and language; is audio acceptable in the room, or should the robot's speech be text only?
- Q8. Pilot on the Isaac-measured grid, or wait for the real physics sweep (one to two days of arm time
  with an experimenter placing cubes)? Present the reality-check numbers first.
- Q9. Can a placement jig be made (3D print or laser cut)? If not, what placement tolerance is acceptable?
- Q10. Session length: is a participant session above two hours acceptable, or should the budget be
  shortened (which changes the contract and the power analysis)?
- Q11. IRB status, consent form source, and whether the debrief script needs institutional wording.
- Q12. Which surrogate brain should be the default for the pilot (the recommendation is `belief`).

---

## Appendix F. Copy-integrity manifest (run this before anything else in Phase 0)

These are the files that define the study, as they were on the laptop on 2026-09-03: the first 16 hex
characters of each file's sha256 and its size in bytes. Run the loop from the repository root. Every line
must print `ok`. A `MISSING` or `DIFFERS` line means the copy is incomplete or is not the laptop's
version; report the exact lines to the operator and stop. (This check is for the copy as delivered; you
will later edit `vla_lab/tests/run_tests.py` and `vla_lab/supervisory/scheduler/__init__.py` yourself, so
do not re-run it as a regression test afterwards.)

```bash
while read -r h n f; do
  [ -f "$f" ] || { echo "MISSING $f"; continue; }
  hh=$(sha256sum "$f" | cut -c1-16); nn=$(wc -c < "$f")
  if [ "$hh" = "$h" ] && [ "$nn" = "$n" ]; then echo "ok  $f"; else echo "DIFFERS $f (got $hh $nn, want $h $n)"; fi
done <<'MANIFEST'
4b72ac00b6ccabae 4880 vla_lab/supervisory/__init__.py
62851ad900ffd069 13992 vla_lab/supervisory/session.py
d2a8c76456d4211c 8928 vla_lab/supervisory/contract.py
c8176151b69b6b61 11337 vla_lab/supervisory/protocol.py
0b82c4391daa087b 10584 vla_lab/supervisory/narration.py
d7bec8d2fe816882 6733 vla_lab/supervisory/strategies.py
e25c83ae06531b5f 28661 vla_lab/supervisory/scenes.py
18f33f46a4767a9e 33616 vla_lab/supervisory/carryover.py
bfac85cffa57906a 30033 vla_lab/supervisory/estimand.py
ca25ec604ed323a8 13474 vla_lab/supervisory/supervisor.py
8cb4d9eb8594894a 4934 vla_lab/supervisory/logging.py
557bcc11fc453678 7961 vla_lab/supervisory/verify_session.py
9161e8947174fde6 22697 vla_lab/supervisory/analyze.py
f4708cf607f734bf 19608 vla_lab/supervisory/physics_fit.py
3353b446e7d8438a 4721 vla_lab/supervisory/_numerics.py
8d2bac211fd29893 9468 vla_lab/supervisory/scheduler/__init__.py
cc050e72d36af9ba 8901 vla_lab/supervisory/scheduler/base.py
7bb394c80c0541fb 5168 vla_lab/supervisory/scheduler/baselines.py
15525af18a5412b0 28529 vla_lab/supervisory/scheduler/carryover_aware.py
2e0d508841f302a6 9725 vla_lab/supervisory/scheduler/identification_first.py
d88c2bba72286268 1028 vla_lab/supervisory/apparatus/__init__.py
63dbd90b0c2dab4c 2990 vla_lab/supervisory/apparatus/base.py
27d711bb0dc59c44 4850 vla_lab/supervisory/apparatus/surrogate.py
db9136041a612cbe 11667 vla_lab/supervisory/apparatus/measure.py
e3b8d249b604d6ef 12333 vla_lab/supervisory/apparatus/isaac.py
3b0a173ab04c76d8 38119 vla_lab/supervisory/run_study.py
897e558918584e67 7324 vla_lab/supervisory/run_sweep.py
40d101cef6ccfbb8 13359 vla_lab/human_study/instruments.py
a241cfeae55d8b54 13403 vla_lab/human_study/phrase_corpus.py
ee279354cd9c43b1 8657 vla_lab/human_study/protocol.py
47c0ef477b68678d 4120 vla_lab/stats_utils.py
c956913db8cfdefc 5602 vla_lab/tests/run_tests.py
947997debee40b04 652 environments/supervisory_fetch/__init__.py
7e72adf75e3f9dc6 13408 environments/supervisory_fetch/config.py
e7c43d0d7e29730a 11735 environments/supervisory_fetch/experts.py
e76f3d56bd25a061 18616 vla_lab/old_direction/rehab/apparatus/kinova_gen2.py
116383568660a71c 6280 vla_lab/old_direction/rehab/apparatus/base.py
7a9838f251a02c5a 11407 vla_lab/old_direction/rehab/observation/calibration.py
d2203d6ab3375583 13792 vla_lab/old_direction/rehab/safety.py
MANIFEST
```

The four `old_direction` files are the previous study's real-robot scaffolding (bridge transport and fake
driver, halt taxonomy, table homography, human-proximate safety state machine). They are reference
material to copy patterns from, not modules to import into the live study.

---

## Appendix G. The geometry and the numeric procedures, restated so nothing depends on reading them elsewhere

You should import these from the copied code (`environments/supervisory_fetch/config.py` and
`experts.py`, both pure Python). They are restated here so you can verify what you import, and so the
prompt survives a missing file.

### G.1 Frames and the scene layout (`layout_for_margin`)

- Frame: the robot **base frame**; table top at `z = 0` in the simulation because the arm is mounted on
  the table. On the real arm, measure `z_table` by touch-off (§7 step 4) and add it to every `z` below.
  `x` points from the base toward the target, `y` to the left when facing along `x`; confirm the real base
  frame's yaw against this by touch-off as well.
- Cubes: edge `0.05` m; cube centre height `z = 0.025`; **top face** `z_top = 0.05`. Three-finger hand
  width `0.09` m.
- Target: fixed at `(tx, ty) = (0.48, −0.10)` m.
- Blocker: `d = gap + 0.05` (centre distance); bearing `π` (toward the robot along `−x`):
  `(bx, by) = (tx − d, ty)`.
- Drop-off for the clear-first sweep: `0.24` m from the blocker, **perpendicular** to the target-blocker
  axis, on the `+y` side: `(bx, by + 0.24)`. The perpendicular direction is the whole reason clearing
  stays feasible at tight gaps; keep it.
- Distractors: `n = scene.clutter` candidates drawn with `random.Random(1000 + scene_id)` on a ring of
  radius `0.20 × U(0.8, 1.4)` m around the target, minimum separation `0.09` m from every other cube,
  never inside the corridor between target and blocker; the simulator physically placed at most the
  first three (its object count was fixed). Call
  `layout_for_margin(scene.margin_m, n_distractors=int(scene.clutter), rng=random.Random(1000 + scene.scene_id))`
  so the physical layout for a scene id is the same every time and the same as the renders.
- Placement jitter in the simulation was `0.008` m on position and `0.25` rad on yaw for the objects,
  **never on the margin**. On hardware the measured placement is the record.
- Finger clearance of the direct approach (the geometry behind `p_success_B`):
  `gap − (0.09 − 0.05) / 2 = gap − 0.02` m; negative below a 2 cm gap, which is why DIRECT fails
  outright there.

### G.2 The two routines (`waypoints_for`), Isaac-measured heights in the `ExpertConfig` defaults

Each waypoint is `(x, y, z), gripper, phase, tolerance_m`. Gripper `open`/`closed` is the state to hold
while reaching the point; a waypoint that only changes the gripper is complete when the fingers have
moved. Defaults: `transit 0.22`, `pregrasp 0.12`, `grasp_depth −0.02` (below the top face), `lift 0.18`,
`push_height 0.075` (closed hand), `push_backoff 0.09`, `push_overshoot 0.04`, `direct_detour 0.07`.

**DIRECT (B).** Let `a` = unit vector from blocker to target (for bearing `π`, `a = (+1, 0)`), detour point
`D = target + 0.07·a`:

```
1. (D,      0.22)        open    transit    0.03
2. (D,      0.12)        open    detour     0.02
3. (target, z_top+0.12)  open    pregrasp   0.015
4. (target, z_top−0.02)  open    descend    0.015
5. (target, z_top−0.02)  closed  close      0.015
6. (target, z_top+0.18)  closed  lift       0.03
```

**CLEAR_FIRST (A).** Let `p` = unit vector from blocker to drop-off (`(0, +1)`), push start
`S = blocker − 0.09·p`, push end `E = dropoff + 0.04·p`, `zc = 0.075`:

```
1. (S,      0.22)        closed  transit_to_push    0.03
2. (S,      zc)          closed  push_start         0.02
3. (E,      zc)          closed  push               0.025
4. (E,      0.22)        closed  push_retract       0.03
5. (target, 0.22)        open    transit_to_target  0.03
6. (target, z_top+0.12)  open    pregrasp           0.015
7. (target, z_top−0.02)  open    descend            0.015
8. (target, z_top−0.02)  closed  close              0.015
9. (target, z_top+0.18)  closed  lift               0.03
```

Worked example, scene 7 (gap 4.54 cm): `d = 0.0954`, blocker `(0.3846, −0.10)`, drop-off
`(0.3846, +0.14)`, `D = (0.55, −0.10)`, `S = (0.3846, −0.19)`, `E = (0.3846, +0.18)`.

The simulated expert deliberately perceived object poses with 4 mm Gaussian noise (with oracle poses the
success curves were step functions and the coordinate degenerated). On hardware your perception has its
own noise; do not add artificial noise.

### G.3 Table homography (from `old_direction/rehab/observation/calibration.py`, pure NumPy)

```python
def solve_table_homography(image_points, table_points):
    """3x3 H mapping image pixels -> table-plane metres, >= 4 correspondences, DLT + SVD, H[2,2] = 1."""
    ip = np.asarray(image_points, float); tp = np.asarray(table_points, float)
    A = []
    for (u, v), (x, y) in zip(ip, tp):
        A.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        A.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]

def apply_homography(h, uv):
    p = h @ np.array([uv[0], uv[1], 1.0]); return (p[0] / p[2], p[1] / p[2])
```

Calibrate on the cubes' **top faces** (that is the plane the segmented centroids lie on), report the
reprojection residual over held-out points, and store `H`, the residual, the camera id, the resolution and
the date in `vla_lab/real_robot/calibration/`.

### G.4 The bridge wire format (repeat of §6.1 in one place)

Newline-delimited JSON over a Unix socket, one object per line, `{"id": int, "op": str, ...}` in,
`{"id": int, "ok": bool, ...}` or `{"id": int, "ok": false, "error": str}` out. Ops: `connect, state, home,
park, move_pose, fingers, stop, start, estop_state, capture, ping, close`. Every motion op takes
`timeout_s` and returns `elapsed_ms`, `reached`, `pose_error_m`, `halted` (null or a typed reason). The
client halts on a missed `ping` (500 ms). The fake bridge implements the same ops with configurable faults
so every client code path is tested without hardware.

### G.5 What the session files must contain (so you can check a session without reading the runner)

`trials.jsonl` rows always carry `slot, block, block_kind, condition, action, scene_id, c, clutter, margin_m,
coach_direction, coach_strength, delta, t_ms`; demonstration rows add `demonstrated_strategy, narration,
repeats, executed_strategy, success, duration_s`; wait rows add `narration, duration_s`; probe and counter
rows add `query, utterance, instructed_strategy, grounded, grounder, fallback_used, executed_strategy,
success, duration_s` and, with a secondary grounder, `grounded_secondary, grounder_secondary`.
`beliefs.jsonl` rows carry `slot, block, action` plus the scheduler's rationale. `events.jsonl` rows carry
`t_ms, kind` plus payload; the runner emits `session_open, block_start, block_end, session_close`, and you add
`robot_say, placement, halt (with reason), participant_pause, participant_stop, questionnaire,
experimenter_note, debrief_shown, slot_repeated`. `meta.json` carries `supervisor_id, condition, n_trials,
clock, git_commit, python, platform, contract_hash, narration_hash, apparatus, channel, grounder, blocks`.
`contract.json` and `protocol.json` are written before slot 1. The clock is `time.monotonic()` with the
wall-clock offset recorded once; the gate fails a session whose trial clock runs backwards.

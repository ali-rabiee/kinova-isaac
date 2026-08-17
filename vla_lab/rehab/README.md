# `rehab/` — Phase 0 quick start

**The question.** With **healthy adults** and the **Kinova Gen2 (JACO 2, `j2n6s300`)**, under a
**matched interaction budget**, can a **personalized, carryover-aware COACH / WAIT / ASSESS
policy** estimate an individual's **unprompted arm-choice map** more accurately, and with
better-calibrated uncertainty, than naive assessment schedules?

**The specification is [`../rehab.md`](../rehab.md).** This file is the run order. Every section
below points back at it by number.

> **Claim boundary (§1.6) — this belongs verbatim in the paper.** Phase 0 is a **healthy-adult
> physical-HRI measurement and scheduling** result. It is **not** evidence of stroke nonuse,
> rehabilitation efficacy, clinical construct validity, or recovery. "Nonpreferred arm" is a
> handedness-defined label, **not a paretic limb**.

---

## Start here (no robot, no participants, seconds)

```bash
./vla_lab/scripts/rehab_pilot.sh                    # a full synthetic study + analysis + figures
./vla_lab/scripts/run_tests.sh                      # both tracks' suites, one gate
```

`rehab_pilot.sh` runs simulated participants through the **real** session code path — the same
protocol, schedulers, observers, safety envelope, phase machine, and event-locked logging a real
session uses — with a generative participant behind the observer seam and a null apparatus
behind the apparatus seam. That is what makes every offline work item verifiable before hardware
or IRB approval exists.

It is a **rehearsal, not evidence about people.** Every number it produces follows from the
population prior in `sim_participant.py`, which the lab pilot (M4) is meant to replace.

## The five day-to-day commands

| Command | What it does |
| --- | --- |
| `rehab_pilot.sh` | synthetic study end-to-end; no robot, no humans |
| `rehab_twin_dryrun.sh` | Isaac twin: reachability, participant clearance, wrist framing (gate M2) |
| `rehab_session.sh` | one real participant session |
| `rehab_verify.sh` | the session gate — **run after every session, before any analysis** |
| `rehab_analyze.sh` | outcomes, tables, and every paper figure |

Plus `rehab_calibrate.sh` (per-participant frame + cameras) and `rehab_power.sh` (the power memo).

## Real-study run order

```bash
# 0. once per rig / per participant — the participant frame defines the crossover band (§9)
./vla_lab/scripts/rehab_calibrate.sh --points rig_points.json --participant P001

# 1. before anyone is near the arm (M2)
./vla_lab/scripts/rehab_twin_dryrun.sh

# 2. the session. The handedness inventory is administered FIRST and is required:
#    it defines "nonpreferred arm", the label the estimand is expressed in.
./vla_lab/scripts/rehab_session.sh --participant P001 --participant-idx 1 \
    --calibration logs/rehab/calibration_P001.json

# 3. the gate. Exit 1 = do not analyze this session as-is.
./vla_lab/scripts/rehab_verify.sh logs/rehab/participant_P001/session_*

# 4. after coding the video, re-verify (kappa vs the coded gold standard), then analyze
./vla_lab/scripts/rehab_verify.sh --root logs/rehab --pool
SESSION_ROOT=logs/rehab ./vla_lab/scripts/rehab_analyze.sh
```

---

## The five things that are easy to get wrong

1. **The handedness inventory comes before trial 1.** `pi*` is `P(nonpreferred arm)`; without
   the inventory that label is undefined. Mixed handedness (|LQ| < 40) is an **exclusion**, not
   a rounding decision — the session runner refuses to start.
2. **`protocol.json` is written before trial 1**, so the preregistered analysis can be checked
   against the realized assignment. The logger enforces it.
3. **The contract is hashed.** Changing geometry, timing, budget, or COACH wording changes the
   hash, and sessions with different hashes are **not poolable**. `rehab_verify.sh --pool` is
   what catches it.
4. **Observer labels are never overwritten.** The online label is what the scheduler acted on;
   the coded label is what the analysis uses. Both live in `observers.jsonl`, and their
   disagreement is a reported quantity.
5. **Run the gate.** It is the Phase 0 counterpart of the VLA track's `verify_session.py`, and
   the same discipline: a hard stop before anything is analyzed.

## What the conditions are (§1.4)

| | Rule |
| --- | --- |
| **B0** `no_prompt` | never COACH — defines the reference map, and the cleanest baseline |
| **B1** `immediate` | never WAIT — maximum contamination |
| **B2** `fixed_washout` | wait a **population** constant `w` after each COACH |
| **B3** `random_static` | B2's budget split, placed independently of history |
| **B4** `carryover_aware` | *(proposed)* per-person posterior over `(lambda, beta, g)`; probe when contamination is low **or** correctable |
| — `ablation_schedule_only` | B4's schedule, no de-biasing |
| — `ablation_estimator_only` | B2's schedule, with de-biasing |

The protocol fixes `T`, the target sequence, and **which slots are COACH slots**, identically for
every compared condition. The scheduler only chooses ASSESS vs WAIT on the rest — so the budget
is matched by construction, and the two ablations separate B4's two mechanisms
(*probe placement* vs *de-biasing*).

## Session layout (§12.2)

```
reference block (B0)  ->  compared blocks, counterbalanced, with washout  ->  retest block (B0)
      defines tilde-pi*                                                    test-retest + residual check
```

The analysis reports **test-retest reliability first**: it bounds how much of the measured
estimation error is irreducible drift. A study that cannot show test-retest stability of
`tilde-pi*` cannot interpret its primary outcome.

## On-disk layout (§10)

```
logs/rehab/participant_<PID>/session_<TS>/
├── contract.json    hashed Phase 0 contract + provenance
├── participant.json handedness, participant-frame calibration, condition assignment, clock offset
├── protocol.json    block layout, seeds, target sequence — written BEFORE trial 1
├── trials.jsonl     one record per trial (the unit of analysis)
├── events.jsonl     audit trail: phases, halts, prompts, pauses, faults
├── observers.jsonl  every observer's label, kept separately — never merged
└── media/           wrist + front video (GITIGNORED: identifiable data, §11)
```

## Two model notes worth knowing before you read the numbers

- **The online and offline carryover posteriors are different objects.** The scheduler's
  posterior plugs in its current `pi*` estimate — cheap, sequential, and *biased*, because a
  contaminated `pi*` absorbs the carryover (on synthetic data with true `beta*g = 1.2` it
  recovers ~0.45). The analysis refits with `pi*` **marginalized out**
  (`estimand.joint_carryover_posterior`), which recovers ~1.14. The logged posterior is an
  operational belief kept for auditability; the reported one is refit offline.
- **`beta` and `g` are only weakly separable.** Report their **product** `beta*g` — the
  immediate logit shift one COACH produces. It is what is identified, and it is what §12.7's
  go/no-go asks about.

## Environment

Everything here is **pure Python + NumPy** (matplotlib for figures, PyYAML for configs) — no
torch, no Isaac, no LeRobot. Two exceptions, both isolated behind the `Apparatus` protocol:

- the **Isaac twin** needs Isaac Lab (`conda activate riften`), imported lazily so this package
  still imports without it;
- the **real Gen2 backend** talks to an out-of-process driver bridge over a narrow JSON socket,
  so `vla_lab.rehab` never imports ROS and the Isaac environment's `numpy<2` pin stays
  untouched (§12.4). The bridge itself is an environment dependency, not repo code —
  `apparatus/kinova_gen2.py`'s `BRIDGE_CONTRACT` says exactly what it must implement, and
  `FakeGen2Driver` exercises every code path without hardware.

## Where the long poles are (§13)

| | | |
| --- | --- | --- |
| **M0** | IRB protocol submitted; Gen2 driver bring-up started | **start both in week 1** — neither compresses by working harder on Python |
| **M1** | Tier A offline: `rehab_pilot.sh` green, all `test_rehab_*` pass | ✅ implemented |
| **M2** | twin dry-run: 100% reachable, zero proxy collisions, wrist view per target | needs Isaac |
| **M3** | hardware + safety: 200 fault-free presentations, every interlock demonstrated | needs the arm |
| **M4** | lab pilot (N≈4–6): **is there a measurable, decaying carryover effect at all?** | the go/no-go (§12.7) |
| **M5** | power memo + preregistration | before the first non-pilot participant |
| **M6–M7** | data collection; analysis + figures | every session passes `rehab_verify.sh` |

**M4 is the one that can end the study.** Healthy adults have less room to move than stroke
survivors, and a purely verbal prompt may leave carryover too small to schedule around. That is
why COACH here is a prompt **plus an effort manipulation**, why targets are densified in the
crossover band, and why "is there a detectable, decaying carryover effect?" is an explicit
go/no-go rather than something discovered during analysis.

## What Phase 0 does *not* need (§15)

No learned visuomotor policy, no `ticks.jsonl`, no object manipulation by the robot, no clinical
measures, no patients, no trust/reliance outcomes, and none of the long-term action library
beyond COACH / WAIT / ASSESS. Those belong to the VLA track (which keeps them) or to later
phases.

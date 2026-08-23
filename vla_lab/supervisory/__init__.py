"""Carryover-aware supervisory control: de-biasing human intent in shared autonomy.

**The question.** A robot that repeatedly demonstrates and narrates one manipulation strategy
teaches the person watching it what to say. When the robot then asks *"how should I approach
this one?"* on a genuinely ambiguous scene, the answer it gets is a mixture of what the person
actually prefers and what the robot just spent three episodes showing them. This package
treats that mixture as the object of study: it estimates the supervisor's **unprompted
strategy-preference map**, models the residue of the robot's own coaching as a latent decaying
state, and lets the robot schedule (and re-open) its own queries accordingly.

**The four actions.** Each interaction slot the system emits one of

``COACH``
    Execute a strategy autonomously on an *unambiguous* scene and narrate it
    ("Clearing the path first so the grasp is safe."). This is the manipulation; its count and
    direction are fixed by the protocol, not chosen by the policy.
``PROBE``
    Present an ambiguous scene, ask the neutral query, execute what the supervisor says. One
    Bernoulli draw from the estimand -- clean only if the carryover state is near zero.
``WAIT``
    A strategy-neutral filler interaction. Costs budget and wall-clock; decays the residue.
``COUNTER``
    A PROBE plus a *counter-proposal*: the robot names the alternative it did not just
    demonstrate ("I can also reach around it directly -- which do you want?"). This re-opens
    the option the coaching closed, so the observation is less contaminated (attenuation
    ``rho``), at the price of extra interaction burden. This is the active de-biasing action.

**Sign convention, everywhere.** Strategy **A** is the *cautious* member of a strategy axis
(clear-the-path-first; top-down grasp) and **B** the *efficient* one (direct reach; lateral
grasp). Outcomes are coded ``1`` for A. The scene coordinate ``c`` is signed so that ``c > 0``
means A is objectively the better strategy for that scene and ``c < 0`` means B is; ``c = 0``
is the crossover, where the two are equally good and the supervisor's answer is genuinely
theirs. The carryover state ``kappa`` is likewise **signed**: coaching A pushes it positive,
coaching B negative. That is what makes the design counterbalanced -- if compliance is real,
priming in either direction must move commands in that direction with the same decay.

Module map::

    strategies.py   the strategy axes and their narration keys
    scenes.py       SceneSpec / SceneGrid, the ambiguity coordinate c, the task-value model
    carryover.py    the signed latent residue and its sequential posterior over (lambda, beta, g)
    estimand.py     pi*(c) and its three estimators, error / calibration / regret metrics
    supervisor.py   the generative supervisor: preference map + compliance bias + utterances
    narration.py    COACH narration, neutral filler, counter-proposals -- content-hashed
    scheduler/      B0-B5 and B5's ablations
    apparatus/      surrogate (measured outcome tables) and Isaac closed-loop backends
    protocol.py     block layout, counterbalancing, matched budget
    session.py      the one session runner
    logging.py      event-locked session records
    contract.py     the hashed study contract
    verify_session.py  the session gate
    analyze.py      outcomes and figures
    power.py        Monte-Carlo power
    run_study.py    the Tier-1 study runner (surrogate apparatus, many sessions)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
COACH = "coach"
PROBE = "probe"
WAIT = "wait"
COUNTER = "counter"

ACTIONS = (COACH, PROBE, WAIT, COUNTER)
#: Actions that yield an observation of the supervisor's instruction.
OBSERVING_ACTIONS = (PROBE, COUNTER)

# ---------------------------------------------------------------------------
# Strategy labels (see strategies.py for the axes that use them)
# ---------------------------------------------------------------------------
STRATEGY_A = "A"  # the cautious member of the axis
STRATEGY_B = "B"  # the efficient member of the axis
STRATEGY_UNRESOLVED = "unresolved"  # the utterance could not be grounded to either

#: Numeric coding used by every likelihood in the package.
STRATEGY_CODE = {STRATEGY_A: 1, STRATEGY_B: 0}

# ---------------------------------------------------------------------------
# Coaching direction
# ---------------------------------------------------------------------------
COACH_A = +1
COACH_B = -1

__all__ = [
    "COACH",
    "PROBE",
    "WAIT",
    "COUNTER",
    "ACTIONS",
    "OBSERVING_ACTIONS",
    "STRATEGY_A",
    "STRATEGY_B",
    "STRATEGY_UNRESOLVED",
    "STRATEGY_CODE",
    "COACH_A",
    "COACH_B",
]

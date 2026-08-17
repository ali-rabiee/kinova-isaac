"""W14 — run ONE real participant session (or a twin rehearsal of one).

    ./vla_lab/scripts/rehab_session.sh --participant P001 --participant-idx 1

A minimal experimenter console around :class:`~vla_lab.rehab.session.Phase0Session`. It exists
so the real study runs through the *same* code the synthetic pilot rehearses — the only things
that change are the apparatus backend and the observer.

Order of operations, enforced (``rehab.md`` §9, §10, §11):

1. the **handedness inventory** is administered first; it defines "nonpreferred arm", which is
   the label the estimand is expressed in, and mixed-handedness is an exclusion;
2. the **calibration** is loaded and checked (participant frame, cameras) before the arm moves;
3. ``protocol.json`` is written before trial 1;
4. the session runs, with the keyed observer *always* live;
5. questionnaires at every block boundary.

**Keyed input.** A background thread reads stdin lines, so the experimenter can type the
side-key for each trial without blocking the control loop. Defaults are ``z`` = participant's
LEFT, ``m`` = participant's RIGHT, ``x`` = saw a reach but could not call it, ``n`` = no reach.
The experimenter sits facing the participant and keys the side *from the participant's point of
view* — that convention is recorded in the contract and is worth rehearsing before a real
participant is in the chair.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..human_study.instruments import EHI_ITEMS, NASA_TLX_SUBSCALES, SESSION_BURDEN_ITEMS
from .apparatus import BACKEND_NULL, BACKEND_REAL, BACKEND_TWIN, make_apparatus
from .carryover import CarryoverConfig
from .contract import Phase0Contract
from .observation import CompositeObserver, KeyedObserver, VisionObserver
from .observation.calibration import CalibrationBundle
from .observation.vision import ScriptedHandDetector, VisionConfig
from .protocol import Phase0Protocol
from .safety import SafetyLimits
from .session import (
    OBSERVER_BOTH,
    OBSERVER_KEYED,
    OBSERVER_VISION,
    Phase0Session,
    SessionConfig,
)


class StdinKeySource:
    """Non-blocking keypress source: a background thread reads stdin lines into a queue.

    The queue is **bounded**. A human types a handful of keys per trial; a stuck key, or stdin
    redirected from a file, would otherwise grow it without limit and starve the control loop.
    Overflow is dropped rather than buffered — a key the experimenter pressed thousands of
    presses ago is not a selection anyone wants applied to the current trial.
    """

    MAX_PENDING = 64

    def __init__(self) -> None:
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=self.MAX_PENDING)
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in sys.stdin:
            if self._stop.is_set():
                return
            for ch in line.strip().lower():
                try:
                    self.q.put_nowait(ch)
                except queue.Full:
                    self.dropped += 1

    def __call__(self) -> Optional[str]:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()


class ConsoleQuestionnaireProvider:
    """Collects questionnaire responses from the experimenter console at block boundaries."""

    def __init__(self, *, skip: bool = False) -> None:
        self.skip = bool(skip)

    def __call__(self, instrument: str, block_kind: str) -> Optional[Any]:
        if self.skip:
            return None
        name = str(instrument)
        print(f"\n--- {name} ({block_kind} block) ---")
        if name in ("nasa_tlx", "tlx"):
            return {k: _ask_float(f"  {k} (0-100): ", 0.0, 100.0) for k in NASA_TLX_SUBSCALES}
        if name in ("session_burden", "burden"):
            prompts = {
                "arm_fatigue": "How tired do your arms feel right now? (0 none .. 10 maximal)",
                "arm_heaviness": "How heavy do your arms feel? (0 none .. 10 maximal)",
                "effort": "How much effort did that block take? (0 none .. 10 maximal)",
                "discomfort": "Any discomfort in arms/shoulders? (0 none .. 10 maximal)",
                "willing_to_continue": "How willing are you to continue? (0 not at all .. 10 very)",
            }
            return {k: _ask_float(f"  {prompts[k]}: ", 0.0, 10.0) for k in SESSION_BURDEN_ITEMS}
        return None


def _ask_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            v = float(raw)
        except ValueError:
            print(f"    (need a number between {lo} and {hi})")
            continue
        if lo <= v <= hi:
            return v
        print(f"    (need a number between {lo} and {hi})")


def administer_handedness(path: Optional[str]) -> List[float]:
    """Load or collect the Edinburgh Handedness Inventory. Required before any trial."""

    if path:
        d = json.loads(Path(path).read_text())
        if isinstance(d, dict) and "responses" in d:
            d = d["responses"]
        if isinstance(d, dict):
            return [float(d[k]) for k in EHI_ITEMS]
        return [float(x) for x in d]
    print("\n=== Edinburgh Handedness Inventory ===")
    print("For each activity: -2 always left, -1 usually left, 0 no preference, "
          "+1 usually right, +2 always right\n")
    return [_ask_float(f"  {item}: ", -2.0, 2.0) for item in EHI_ITEMS]


def build_observer_factory(
    kind: str,
    nonpreferred_side: str,
    *,
    key_source: Any,
    hand_detector: Any = None,
):
    """The keyed observer is ALWAYS in the composite: it is the fallback and the gold standard."""

    def factory(_: str):
        keyed = KeyedObserver(nonpreferred_side, key_source)
        if kind == OBSERVER_KEYED:
            return keyed
        if hand_detector is None:
            print(
                "[rehab.session] no hand detector supplied; running keyed-only. That is the "
                "pre-registered fallback (W8), but record it as such."
            )
            return keyed
        vision = VisionObserver(nonpreferred_side, hand_detector, cfg=VisionConfig())
        if kind == OBSERVER_VISION:
            return CompositeObserver([vision, keyed], primary=0)
        return CompositeObserver([vision, keyed], primary=0)  # OBSERVER_BOTH

    return factory


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default="vla_lab/configs/rehab_phase0.yaml")
    ap.add_argument("--participant", type=str, required=True, help="pseudonymous study ID, e.g. P001")
    ap.add_argument("--participant-idx", type=int, required=True, help="enrollment index (drives counterbalancing)")
    ap.add_argument("--apparatus", type=str, default=BACKEND_REAL, choices=[BACKEND_NULL, BACKEND_TWIN, BACKEND_REAL])
    ap.add_argument("--gen2-socket", type=str, default="/tmp/kinova_gen2_bridge.sock")
    ap.add_argument("--observer", type=str, default=OBSERVER_BOTH, choices=[OBSERVER_VISION, OBSERVER_KEYED, OBSERVER_BOTH])
    ap.add_argument("--handedness", type=str, default=None, help="JSON with the EHI responses (else asked at the console)")
    ap.add_argument("--calibration", type=str, default=None, help="JSON from rehab_calibrate.sh")
    ap.add_argument("--log-root", type=str, default="logs/rehab")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-questionnaires", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    contract = Phase0Contract.from_yaml(cfg_path) if cfg_path.exists() else Phase0Contract()
    protocol = Phase0Protocol.from_yaml(cfg_path) if cfg_path.exists() else Phase0Protocol()
    problems = contract.validate() + protocol.validate()
    if problems:
        print("[rehab.session] configuration is invalid:")
        for p in problems:
            print(f"  - {p}")
        return 2

    hand = administer_handedness(args.handedness)
    from ..human_study.instruments import edinburgh_handedness

    scored = edinburgh_handedness(hand)
    print(f"[rehab.session] handedness: {scored['handedness']} (LQ={scored['lq']:.1f}); "
          f"nonpreferred arm = {scored['nonpreferred_arm']}")
    if scored["nonpreferred_arm"] is None:
        print("[rehab.session] EXCLUSION: mixed handedness — pi* is undefined for this participant (§9).")
        return 3
    side = str(scored["nonpreferred_arm"])

    calibration = None
    if args.calibration and Path(args.calibration).exists():
        calibration = CalibrationBundle.from_dict(json.loads(Path(args.calibration).read_text()))
        cal_problems = calibration.check()
        if cal_problems:
            print("[rehab.session] calibration check FAILED:")
            for p in cal_problems:
                print(f"  - {p}")
            return 4
    elif args.apparatus == BACKEND_REAL:
        print("[rehab.session] WARNING: no calibration file. The participant frame defines the "
              "crossover band; running without it makes the estimand's informative region a guess (W13).")

    apparatus = make_apparatus(
        args.apparatus, contract, socket_path=args.gen2_socket, headless=True
    )
    keys = StdinKeySource()
    factory = build_observer_factory(args.observer, side, key_source=keys)

    print("\n[rehab.session] keyed input is LIVE: z = participant's LEFT, m = participant's RIGHT, "
          "x = ambiguous, n = no reach. Press and hit Enter.")
    print("[rehab.session] the participant may stop at any time; that is a logged event, not an abort.\n")

    session = Phase0Session(
        contract,
        protocol,
        SessionConfig(
            participant_id=str(args.participant),
            participant_idx=int(args.participant_idx),
            log_root=str(args.log_root),
            seed=int(args.seed),
            carryover=CarryoverConfig(),
            safety_limits=SafetyLimits(),
        ),
        apparatus=apparatus,
        observer_factory=factory,
        observer_kind=str(args.observer),
        questionnaire_provider=ConsoleQuestionnaireProvider(skip=bool(args.no_questionnaires)),
        handedness_responses=hand,
        calibration=calibration,
    )
    try:
        result = session.run()
    finally:
        keys.stop()

    s = result.summary()
    print(f"\n[rehab.session] {s}")
    if keys.dropped:
        print(f"[rehab.session] NOTE: {keys.dropped} keypress(es) dropped as overflow — check "
              "for a stuck key; the keyed labels for this session may be unreliable.")
    print(f"[rehab.session] now run:  ./vla_lab/scripts/rehab_verify.sh {result.session_dir}")
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())

"""CLI: run the human-study pipeline **offline** with the bundled simulators.

This is the memo's July "pilot the human study (2–3 participants) to debug the protocol"
step, made runnable with **no robot and no human subjects**. It generates counterbalanced
session plans for ``--participants`` synthetic participants, runs each through the
:class:`~vla_lab.human_study.session.SimEpisodeRunner` +
:class:`~vla_lab.human_study.session.SimQuestionnaireProvider`, and writes a study JSONL that
``python -m vla_lab.human_study.analyze`` consumes — so the whole
``generate → run → analyze → figures`` path is exercisable end-to-end (and in CI).

To run the **real** study, swap :class:`SimEpisodeRunner` for the Kinova JACO loop and
:class:`SimQuestionnaireProvider` for the study front-end in
:class:`~vla_lab.human_study.session.SessionRunner`; nothing else here changes.

Usage:
    python -m vla_lab.human_study.run_pilot \
        --participants 12 --design within \
        --out vla_lab/study_runs/pilot/study.jsonl --analyze
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .protocol import StudyProtocol, generate_session_plan
from .session import (
    SessionRunner,
    SimEpisodeRunner,
    SimQuestionnaireProvider,
    SimWorldParams,
    StudyLogger,
)


def run_pilot(
    participants: int,
    out_path: str,
    *,
    design: str = "within",
    reps: int = 1,
    seed: int = 1337,
    instruments: Optional[List[str]] = None,
    world: Optional[SimWorldParams] = None,
    world_seed: Optional[int] = None,
    append: bool = False,
) -> Tuple[Path, int]:
    """Generate + run ``participants`` synthetic sessions; return ``(log_path, n_trials)``."""

    proto = StudyProtocol(
        design=str(design),
        reps_per_cell=int(reps),
        seed=int(seed),
        n_participants=int(participants),
        instruments=list(instruments or ["nasa_tlx", "jian_trust"]),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not append:
        out.unlink()  # fresh log per run unless --append

    n_trials = 0
    for pi in range(int(participants)):
        plan = generate_session_plan(pi, proto)
        ws = (int(world_seed) + pi) if world_seed is not None else pi
        runner = SessionRunner(
            plan,
            SimEpisodeRunner(params=world, seed=ws),
            questionnaire_provider=SimQuestionnaireProvider(seed=ws),
            instruments=proto.instruments,
            logger=StudyLogger(out),  # append-mode; closed by run()
        )
        result = runner.run()
        n_trials += len(result.reliance_trials)
    return out, n_trials


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline human-study pilot (synthetic robot + human)")
    ap.add_argument("--participants", type=int, default=8)
    ap.add_argument("--out", type=str, default="vla_lab/study_runs/pilot/study.jsonl")
    ap.add_argument("--design", type=str, default="within", choices=["within", "between"])
    ap.add_argument("--reps", type=int, default=1, help="repetitions per (condition × observability) cell")
    ap.add_argument("--seed", type=int, default=1337, help="protocol/counterbalancing seed")
    ap.add_argument("--world-seed", type=int, default=None, help="base seed for the synthetic world (default: participant idx)")
    ap.add_argument("--instruments", type=str, default="nasa_tlx,jian_trust")
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    # Optional: chain straight into analysis + figures.
    ap.add_argument("--analyze", action="store_true", help="run vla_lab.human_study.analyze on the log afterwards")
    ap.add_argument("--out-dir", type=str, default="vla_lab/results/human_study", help="analysis output dir (with --analyze)")
    ap.add_argument("--reference", type=str, default="allocator_type", help="reference condition for McNemar (with --analyze)")
    ap.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    instruments = [s for s in str(args.instruments).split(",") if s.strip()]
    out, n_trials = run_pilot(
        args.participants,
        args.out,
        design=args.design,
        reps=args.reps,
        seed=args.seed,
        instruments=instruments,
        world_seed=args.world_seed,
        append=args.append,
    )
    print(f"[run_pilot] {args.participants} participants × design={args.design} → {n_trials} trials")
    print(f"[run_pilot] wrote {out}")

    if args.analyze:
        from . import analyze as _analyze

        an_argv = ["--log", str(out), "--out-dir", args.out_dir, "--reference", args.reference, "--format", args.format]
        if args.no_figures:
            an_argv.append("--no-figures")
        return _analyze.main(an_argv)
    else:
        print(f"[run_pilot] analyze with:\n"
              f"    python -m vla_lab.human_study.analyze --log {out} --out-dir {args.out_dir} --format {args.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

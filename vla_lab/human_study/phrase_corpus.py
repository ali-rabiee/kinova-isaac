"""The phrase corpus: what real people say, before any of them meets the robot.

The lexical grounder and the generative supervisor's phrase set are both tuned on scripted
language, and the pilot's go/no-go makes grounding rate on real speech a gate. This is the
cheapest de-risking action available and it needs no robot, no session and no interaction
budget: show about ten people a handful of rendered scenes spanning the coordinate range and
ask, in their own words, what the robot should do. Three commands:

``prepare``
    Write the collection packet: the scene images (the three-quarter view, since that is how a
    supervisor sees the task), a response sheet to fill in, and the exact instruction text.
``evaluate``
    Run the current grounder over the collected responses, report the grounding rate against
    the pre-registered threshold, and enumerate every failure mode: hedges, conditionals,
    multi-clause answers, answers that name neither strategy, answers that name a strategy the
    design does not implement.
``rebuild``
    Rebuild the grounder's phrase inventory and the generative supervisor's utterance sampler
    from the empirical distribution -- including the empirical hedge rate rather than a chosen
    one -- into a JSON the study runner can load with ``--phrase-corpus``. Re-running the study
    under it is then one flag, and the narration hash changes, so sessions under the rebuilt
    supervisor can never be pooled with sessions under the scripted one.

    python -m vla_lab.human_study.phrase_corpus prepare
    python -m vla_lab.human_study.phrase_corpus evaluate vla_lab/results/phrase_corpus/responses.csv
    python -m vla_lab.human_study.phrase_corpus rebuild  vla_lab/results/phrase_corpus/responses.csv

**No corpus exists yet (2026-08-23).** Nothing here fabricates one: ``prepare`` writes an empty
response sheet, and the tests use a handful of clearly labelled fixture utterances.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..supervisory import STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED
from ..supervisory.narration import ground
from ..supervisory.strategies import AXIS_PLAN, StrategyAxis, get_axis

ROOT = Path("vla_lab/results/phrase_corpus")
THRESHOLD = 0.85                                  # the pre-registered grounding-rate gate
INSTRUCTION = (
    "The robot has to pick up the RED box. The BLUE box is in the way to some degree -- the gap "
    "between them is different in each picture. In your own words, tell the robot how it should "
    "go about it. There is no right answer; say what you would actually say."
)

HEDGE_WORDS = ("either", "not sure", "don't know", "dont know", "whatever", "up to you", "don't mind",
               "dont mind", "no preference", "you decide", "you choose", "i guess", "maybe", "hmm")
CONDITIONAL_WORDS = ("if ", "unless", "depends", "as long as", "provided", "in case", "otherwise")
UNIMPLEMENTED_WORDS = ("lift", "over the top", "over it", "from above", "ask", "wait", "both", "slide",
                       "rotate", "turn", "other side", "behind", "nudge", "tip", "carry", "stack", "two")


def _band_scenes(n: int = 6) -> List[Any]:
    """Scenes spanning the coordinate range: flanks, band edges, crossover."""
    from ..supervisory.scenes import build_scene_grid

    grid = build_scene_grid()
    probes = sorted(grid.probe_scenes(), key=lambda s: s.c)
    if len(probes) <= n:
        return probes
    idx = [round(i * (len(probes) - 1) / (n - 1)) for i in range(n)]
    return [probes[i] for i in idx]


def prepare(out: Path = ROOT, *, frames: Path = Path("vla_lab/results/physics/frames"), n_scenes: int = 6) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    packet = out / "packet"
    packet.mkdir(exist_ok=True)
    scenes = _band_scenes(n_scenes)
    rows = []
    for k, s in enumerate(scenes):
        src = None
        for view in ("figure", "topdown"):
            cands = sorted((Path(frames) / view / f"scene_{s.scene_id:03d}").glob("*.png"))
            if cands:
                src = cands[0]
                break
        dst = packet / f"scene_{k + 1}.png"
        if src is not None:
            shutil.copy(src, dst)
        rows.append({"packet_index": k + 1, "scene_id": int(s.scene_id), "c": float(s.c),
                     "gap_cm": float(s.margin_m * 100), "image": str(dst), "rendered": src is not None})
    (packet / "INSTRUCTIONS.txt").write_text(
        "Phrase corpus collection -- read to each participant verbatim.\n\n" + INSTRUCTION +
        "\n\nShow the pictures in the order numbered. Record the answer exactly as spoken or typed, "
        "including hedges and false starts. Do not prompt with either option.\n")
    sheet = out / "responses.csv"
    if not sheet.exists():
        with sheet.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["participant_id", "packet_index", "scene_id", "response_text", "modality"])
    (out / "packet_manifest.json").write_text(json.dumps({"instruction": INSTRUCTION, "scenes": rows}, indent=2) + "\n")
    return {"packet": str(packet), "sheet": str(sheet), "scenes": rows}


def read_responses(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def classify(text: str, axis: StrategyAxis) -> Dict[str, Any]:
    """Ground one utterance and, if it fails, say why -- every mode the brief lists."""
    t = f" {str(text).lower().strip()} "
    hit_a = any(k in t for k in axis.keywords_a)
    hit_b = any(k in t for k in axis.keywords_b)
    g = ground(text, axis)
    modes: List[str] = []
    if g == STRATEGY_UNRESOLVED:
        if any(h in t for h in HEDGE_WORDS):
            modes.append("hedge")
        if any(c in t for c in CONDITIONAL_WORDS):
            modes.append("conditional")
        if hit_a and hit_b:
            modes.append("multi_clause_both_sides")
        if not hit_a and not hit_b and not modes:           # a hedge is a hedge, not also "names neither"
            if any(u in t for u in UNIMPLEMENTED_WORDS):
                modes.append("unimplemented_strategy")
            else:
                modes.append("names_neither")
        if not modes:
            modes.append("other")
    return {"text": text, "grounded": g, "hit_a": hit_a, "hit_b": hit_b, "modes": modes,
            "n_words": len(t.split())}


def evaluate(path: Path, *, axis_name: str = AXIS_PLAN, threshold: float = THRESHOLD) -> Dict[str, Any]:
    axis = get_axis(axis_name)
    rows = read_responses(path)
    items = [{**classify(r.get("response_text", ""), axis), "participant_id": r.get("participant_id"),
              "scene_id": r.get("scene_id")} for r in rows if str(r.get("response_text", "")).strip()]
    n = len(items)
    grounded = [i for i in items if i["grounded"] in (STRATEGY_A, STRATEGY_B)]
    modes = Counter(m for i in items for m in i["modes"])
    rate = len(grounded) / n if n else float("nan")
    per_p: Dict[str, List[int]] = {}
    for i in items:
        per_p.setdefault(str(i["participant_id"]), []).append(int(i["grounded"] != STRATEGY_UNRESOLVED))
    return {
        "n_responses": n,
        "n_participants": len(per_p),
        "grounding_rate": rate,
        "threshold": float(threshold),
        "meets_threshold": bool(n and rate >= threshold),
        "share_a_among_grounded": (sum(1 for i in grounded if i["grounded"] == STRATEGY_A) / len(grounded)) if grounded else None,
        "failure_modes": dict(modes),
        "hedge_rate": (modes.get("hedge", 0) / n) if n else None,
        "per_participant_rate": {k: sum(v) / len(v) for k, v in per_p.items()},
        "unresolved": [{"participant_id": i["participant_id"], "scene_id": i["scene_id"], "text": i["text"],
                        "modes": i["modes"]} for i in items if i["grounded"] == STRATEGY_UNRESOLVED],
        "verdict": ("go: grounding rate meets the pre-registered threshold" if (n and rate >= threshold) else
                    "no-go: revise the grounder (rebuild) and/or the threshold BEFORE recruitment, and document it"),
    }


def rebuild(path: Path, *, axis_name: str = AXIS_PLAN, min_count: int = 1) -> Dict[str, Any]:
    """An empirical axis: phrases and hedges from people, keywords extended from what grounded.

    Keywords are not mined from the failures -- that would tune the grounder to this corpus and
    report an inflated rate -- only the *phrase inventory* the supervisor samples from and the
    hedge rate are empirical. The grounded utterances become the sampler; the unresolved ones
    become the hedge pool; the ratio becomes the ungrounded rate.
    """
    axis = get_axis(axis_name)
    rows = read_responses(path)
    a_ph: Counter = Counter()
    b_ph: Counter = Counter()
    hedges: Counter = Counter()
    for r in rows:
        text = str(r.get("response_text", "")).strip()
        if not text:
            continue
        g = ground(text, axis)
        (a_ph if g == STRATEGY_A else b_ph if g == STRATEGY_B else hedges)[text.lower()] += 1
    n = sum(a_ph.values()) + sum(b_ph.values()) + sum(hedges.values())
    out = {
        "axis": axis_name,
        "source": str(path),
        "n_responses": n,
        "phrases_a": [p for p, c in a_ph.most_common() if c >= min_count],
        "phrases_b": [p for p, c in b_ph.most_common() if c >= min_count],
        "hedges": [p for p, c in hedges.most_common() if c >= min_count],
        "ungrounded_rate": (sum(hedges.values()) / n) if n else 0.0,
        "keywords_a": list(axis.keywords_a),
        "keywords_b": list(axis.keywords_b),
    }
    return out


def load_empirical_axis(path: Path) -> Tuple[StrategyAxis, Tuple[str, ...], float]:
    """``(axis with empirical phrases, hedge pool, empirical ungrounded rate)`` from a rebuild file."""
    d = json.loads(Path(path).read_text())
    base = get_axis(d["axis"])
    if not d["phrases_a"] or not d["phrases_b"]:
        raise ValueError("an empirical axis needs at least one grounded phrase per strategy")
    axis = StrategyAxis(
        name=base.name, label_a=base.label_a, label_b=base.label_b, coordinate=base.coordinate,
        margin_units=base.margin_units, phrases_a=tuple(d["phrases_a"]), phrases_b=tuple(d["phrases_b"]),
        keywords_a=tuple(d.get("keywords_a", base.keywords_a)), keywords_b=tuple(d.get("keywords_b", base.keywords_b)),
        command_a=base.command_a, command_b=base.command_b,
    )
    return axis, tuple(d.get("hedges") or ()), float(d.get("ungrounded_rate", 0.0))


def install_empirical_axis(path: Path) -> Dict[str, Any]:
    """Make the empirical axis the one every component uses for the rest of the process.

    Replaces the registry entry and the supervisor's hedge pool, so the generative supervisor
    speaks from people's phrases and hedges at people's rate; the contract's narration hash
    picks the change up automatically.
    """
    from ..supervisory import strategies, supervisor

    axis, hedges, rate = load_empirical_axis(path)
    strategies.AXES[axis.name] = axis
    if hedges:
        supervisor._HEDGES = tuple(hedges)
    return {"axis": axis.name, "n_phrases_a": len(axis.phrases_a), "n_phrases_b": len(axis.phrases_b),
            "n_hedges": len(hedges), "ungrounded_rate": rate}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, default=ROOT)
    p.add_argument("--frames", type=Path, default=Path("vla_lab/results/physics/frames"))
    p.add_argument("--scenes", type=int, default=6)
    e = sub.add_parser("evaluate")
    e.add_argument("responses", type=Path)
    e.add_argument("--threshold", type=float, default=THRESHOLD)
    r = sub.add_parser("rebuild")
    r.add_argument("responses", type=Path)
    r.add_argument("--out", type=Path, default=ROOT / "axis_plan_empirical.json")
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        info = prepare(args.out, frames=args.frames, n_scenes=args.scenes)
        print(json.dumps(info, indent=2))
        print(f"\nfill in {info['sheet']}; then: python -m vla_lab.human_study.phrase_corpus evaluate {info['sheet']}")
        return 0
    if args.cmd == "evaluate":
        rep = evaluate(args.responses, threshold=args.threshold)
        out = Path(args.responses).with_name("grounding_report.json")
        out.write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "unresolved"}, indent=2))
        print(f"\n{len(rep['unresolved'])} unresolved utterances listed in {out}")
        return 0 if rep["meets_threshold"] else 1
    rep = rebuild(args.responses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("phrases")}, indent=2))
    print(f"\nwrote {args.out}; re-run the study with: python -m vla_lab.supervisory.run_study --phrase-corpus {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

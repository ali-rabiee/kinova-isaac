"""Rule-based parser: free-text interjection -> Refinement.

Deliberately small and deterministic (stdlib only) so its behavior is fully
unit-testable and its failure mode is `unknown`, never a wrong strong action.
Swap in an LLM parser later behind the same function signature if needed.
"""

from __future__ import annotations

from typing import List, Sequence

from .types import DIRECTION_VECTORS, Refinement

_STOP_WORDS = {"stop", "wait", "pause", "freeze", "halt"}
_RESUME_WORDS = {"resume", "continue", "proceed", "go", "carry"}
_YES_WORDS = {"yes", "yeah", "yep", "correct", "exactly", "confirmed"}
_NO_WORDS = {"no", "nope", "wrong", "incorrect"}
_NEGATORS = {"not", "dont", "don", "isnt", "avoid"}
_MAG_SMALL = {"slightly", "bit", "little", "tiny", "tad"}
_MAG_LARGE = {"more", "further", "much", "way", "lot"}


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in str(text).lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def parse_utterance(text: str, known_labels: Sequence[str]) -> Refinement:
    """Parse one utterance against the scene's candidate labels (e.g. colors)."""

    toks = _tokens(text)
    if not toks:
        return Refinement(kind="unknown", raw_text=text)
    tokset = set(toks)
    labels_lc = {str(l).lower(): str(l) for l in known_labels}
    hit_labels = [labels_lc[t] for t in toks if t in labels_lc]

    # 1. Safety words win outright.
    if tokset & _STOP_WORDS:
        return Refinement(kind="stop", raw_text=text, matched_tokens=tuple(tokset & _STOP_WORDS))

    # 2. Target talk. "no, the red one" -> target red; "not the red one" -> avoid red.
    if hit_labels:
        label = hit_labels[0]
        lab_idx = next(i for i, t in enumerate(toks) if t in labels_lc)
        window = set(toks[max(0, lab_idx - 2) : lab_idx])
        if window & _NEGATORS:
            return Refinement(kind="avoid", target_label=label, raw_text=text,
                              matched_tokens=(label,))
        return Refinement(kind="target", target_label=label, raw_text=text,
                          matched_tokens=(label,))

    # 3. Directional nudges ("a bit to the left", "up and left", "much closer").
    dirs = [DIRECTION_VECTORS[t] for t in toks if t in DIRECTION_VECTORS]
    if dirs:
        v = [sum(d[k] for d in dirs) for k in range(3)]
        norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
        if norm > 1e-9:
            v = [x / norm for x in v]
            mag = 1.0
            if tokset & _MAG_SMALL:
                mag = 0.5
            elif tokset & _MAG_LARGE:
                mag = 1.5
            return Refinement(
                kind="nudge",
                direction=(v[0], v[1], v[2]),
                magnitude=mag,
                raw_text=text,
                matched_tokens=tuple(t for t in toks if t in DIRECTION_VECTORS),
            )

    # 4. Bare confirmations (checked after directions: "right" is a direction).
    if tokset & _YES_WORDS:
        return Refinement(kind="confirm_yes", raw_text=text, matched_tokens=tuple(tokset & _YES_WORDS))
    if tokset & _NO_WORDS and len(tokset) <= 3:
        # "not that one" — a contentless rejection of the current target.
        return Refinement(kind="confirm_no", raw_text=text, matched_tokens=tuple(tokset & _NO_WORDS))
    if tokset & _NEGATORS and ("that" in tokset or "this" in tokset or "one" in tokset):
        return Refinement(kind="confirm_no", raw_text=text, matched_tokens=tuple(tokset & _NEGATORS))

    # 5. Resume (after stop).
    if tokset & _RESUME_WORDS:
        return Refinement(kind="resume", raw_text=text, matched_tokens=tuple(tokset & _RESUME_WORDS))

    return Refinement(kind="unknown", raw_text=text)

"""Render the scene driver's measured status as a figure.

The instrument is not finished, and a reader is entitled to see how far from finished it is
rather than take a sentence for it. Everything plotted here was measured by an instrumented
probe and is stored in ``diagnostics.json`` next to the figure, so the panel can be regenerated
and, more importantly, *falsified* by re-running the probes.

    python -m vla_lab.supervisory.apparatus.diagnostics vla_lab/results/isaac_status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def figure(diag: Dict[str, Any], out: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 160, "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5))

    # (a) the orientation-hold sweep: five configurations, one of them qualitatively different
    sw = diag["orientation_hold_sweep"]
    labels = [c["config"] for c in sw]
    vals = [c["descent_m"] for c in sw]
    cols = ["#c0392b" if v <= 0 else "#2c6fbb" for v in vals]
    axes[0].barh(range(len(vals)), vals, color=cols, height=0.62)
    axes[0].axvline(0.0, color="#444", lw=0.9)
    axes[0].set_yticks(range(len(vals)), labels, fontsize=6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("descent achieved (m)")
    axes[0].set_title("(a) holding the wrist orientation\nstops the descent entirely",
                      loc="left", fontsize=8)

    # (b) the root cause, before and after the alignment phase
    wa = diag["wrist_alignment"]
    axes[1].bar([0, 1], [wa["downwardness_at_home"], wa["downwardness_after_alignment"]],
                color=["#c0392b", "#2c6fbb"], width=0.55)
    axes[1].axhline(0.0, color="#444", lw=0.9)
    axes[1].axhline(wa["target"], color="#16a085", lw=0.9, ls="--")
    axes[1].text(1.35, wa["target"], "target", fontsize=6, va="center", color="#16a085")
    axes[1].set_xticks([0, 1], ["at home", "after\nalignment"], fontsize=7)
    axes[1].set_ylabel("tool-axis downwardness")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("(b) the gripper points UP at the\nconfigured home pose", loc="left", fontsize=8)

    # (c) per-phase completion, and the floor the last one hits
    tr = diag["waypoint_trace"]
    names = [t["phase"] for t in tr]
    steps = [t["steps"] if t["steps"] else 0 for t in tr]
    cols = ["#2c6fbb" if t["ok"] else ("#c0392b" if t["ok"] is False else "#bbbbbb") for t in tr]
    axes[2].barh(range(len(names)), np.log10(np.array(steps) + 1.0), color=cols, height=0.62)
    axes[2].set_yticks(range(len(names)), names, fontsize=6.5)
    axes[2].invert_yaxis()
    axes[2].set_xlabel(r"$\log_{10}$ steps to complete")
    axes[2].set_title("(c) five of eight phases complete;\nthe sixth hits a hard floor",
                      loc="left", fontsize=8)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_isaac_status.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, nargs="?", default=Path("vla_lab/results/isaac_status"))
    args = ap.parse_args(argv)
    diag = json.loads((args.root / "diagnostics.json").read_text())
    p = figure(diag, args.root)
    print(p if p else "matplotlib unavailable; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

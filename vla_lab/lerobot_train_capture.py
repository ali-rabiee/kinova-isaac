"""Wrap `lerobot-train`: tee logs, parse training metrics, write CSV/JSONL, plot curves under `vla_lab/results/<UTC-date>/<run>/`.

LeRobot logs `MetricsTracker` lines like:
  step:50 smpl:1.6K ep:26.3 epch:0.10 loss:0.234 grdn:1.234 lr:1.0e-04 updt_s:0.05 data_s:0.01

Disabled with:  VLA_LAB_CAPTURE_RESULTS=0  lerobot-train ...
Or run wrapper directly:
  python -m vla_lab.lerobot_train_capture -- --policy.path=... ...

Outputs (per run folder):
  - train_console.log     — full stdout/stderr
  - train_metrics.csv     — one row per log step (loss, lr, grad norm, epoch coverage, timings)
  - train_metrics.jsonl   — same rows as JSON
  - eval_events.jsonl     — raw eval-related log lines (e.g. sim success if eval_freq>0)
  - run_meta.json         — command, paths, git rev, metric glossary
  - figures/              — loss, lr, grad norm, epoch coverage, timing plots (+ eval success if parsed)
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parent

METRIC_GLOSSARY: Dict[str, str] = {
    "step": "Optimizer step (LeRobot training step index).",
    "smpl": "Approximate number of samples seen (batch_size × step × world_size).",
    "ep": "Approximate equivalent episodes seen (coverage heuristic).",
    "epch": "Dataset epoch coverage (~1.0 = one full pass over num_frames).",
    "loss": "Policy training loss (BC / flow objective; policy-specific).",
    "grdn": "Gradient norm after clipping (L2).",
    "lr": "Learning rate at this step.",
    "updt_s": "Wall time for forward+backward+optimizer (seconds).",
    "data_s": "Time spent loading the batch (seconds).",
    "eval_success_pct": "Sim eval success rate (%) if mid-train eval is enabled (eval_freq > 0).",
}


def _parse_human_number(raw: str) -> float:
    raw = raw.strip().rstrip("%")
    if not raw or raw.lower() in ("nan", "inf"):
        return float("nan")
    suffix = raw[-1]
    if suffix in "KMBTQ":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}[suffix]
        try:
            return float(raw[:-1]) * mult
        except ValueError:
            return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _parse_metrics_tracker_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse LeRobot MetricsTracker.__str__ tail from a log line."""
    if "step:" not in line or "loss:" not in line:
        return None
    idx = line.find("step:")
    blob = line[idx:].strip()
    out: Dict[str, Any] = {}
    for part in blob.split():
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if k in ("step", "smpl", "ep", "epch", "loss", "grdn", "lr", "updt_s", "data_s"):
            if k == "lr":
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = _parse_human_number(v)
            else:
                out[k] = _parse_human_number(v)
    if "loss" not in out or "step" not in out:
        return None
    return out


def _parse_eval_success(line: str) -> Optional[Tuple[int, float]]:
    """Best-effort: find success rate in eval log lines."""
    # e.g. pc_success in dict repr, or "success': 85.0"
    m = re.search(r"['\"]pc_success['\"]\s*:\s*([^\s,}]+)", line)
    if m:
        return (-1, float(m.group(1)))  # step unknown
    m2 = re.search(r"success[\":\s]+([0-9.]+)\s*%?", line, re.I)
    if m2:
        return (-1, float(m2.group(1)))
    return None


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    i = 2
    while True:
        c = Path(f"{base}_{i}")
        if not c.exists():
            return c
        i += 1


def _git_rev() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _write_run_meta(
    path: Path,
    *,
    cmd: List[str],
    exit_code: int,
    results_dir: Path,
    train_rows: int,
    eval_rows: int,
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "repo_root": str(REPO_ROOT),
        "git_rev": _git_rev(),
        "results_dir": str(results_dir),
        "lerobot_invocation": cmd,
        "train_metric_rows": train_rows,
        "eval_event_rows": eval_rows,
        "metric_glossary": METRIC_GLOSSARY,
        "notes": (
            "BC / flow policies do not report classification accuracy during training. "
            "Use 'loss' and 'epch' (data coverage); add mid-train sim eval (--eval_freq > 0 + env) for success curves."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _write_figures(rows: List[Dict[str, Any]], fig_dir: Path, fmt: str = "pdf") -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    steps = [float(r["step"]) for r in rows]

    def _plot_series(
        name: str,
        key: str,
        ylabel: str,
        title: str,
    ) -> None:
        ys = [float(r.get(key, float("nan"))) for r in rows]
        if all(v != v for v in ys):  # all nan
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, ys, color="#1f77b4", linewidth=1.2)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.{fmt}", dpi=150)
        plt.close(fig)

    _plot_series("loss_vs_step", "loss", "Loss", "Training loss (LeRobot)")
    _plot_series("lr_vs_step", "lr", "Learning rate", "Learning rate schedule")
    _plot_series("grad_norm_vs_step", "grdn", "Gradient norm", "Gradient norm (after clip)")
    _plot_series("epoch_coverage_vs_step", "epch", "Dataset epochs seen", "Approximate dataset epoch coverage")
    _plot_series("update_time_vs_step", "updt_s", "Seconds", "Optimizer step wall time")
    _plot_series("data_load_time_vs_step", "data_s", "Seconds", "Data loading time")


def _plot_eval_success(eval_points: List[Tuple[int, float]], fig_dir: Path, fmt: str) -> None:
    if not eval_points:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps_s: List[int] = []
    succ: List[float] = []
    for st, sc in eval_points:
        if st >= 0:
            steps_s.append(st)
            succ.append(sc)
    if not steps_s and eval_points:
        # no step parsed — index by order
        steps_s = list(range(len(eval_points)))
        succ = [p[1] for p in eval_points]
    if not steps_s:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_s, succ, "o-", color="#2ca02c", linewidth=1.2)
    ax.set_xlabel("Training step (or eval index)")
    ax.set_ylabel("Success (%)")
    ax.set_title("Mid-train eval success (if sim eval enabled)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"eval_success_vs_step.{fmt}", dpi=150)
    plt.close(fig)


def _forward_argv(argv: List[str]) -> List[str]:
    if argv[:1] == ["--"]:
        return argv[1:]
    return argv


def main() -> int:
    if os.environ.get("VLA_LAB_CAPTURE_RESULTS", "1") in ("0", "false", "False", "no"):
        argv = _forward_argv(sys.argv[1:])
        exe = shutil.which("lerobot-train")
        if not exe:
            print("[lerobot_train_capture] lerobot-train not on PATH", file=sys.stderr)
            return 2
        return subprocess.call([exe, *argv])

    argv = _forward_argv(sys.argv[1:])
    exe = shutil.which("lerobot-train")
    if not exe:
        print("[lerobot_train_capture] lerobot-train not on PATH", file=sys.stderr)
        return 2

    run_name = os.environ.get("VLA_LAB_RUN_NAME", "").strip() or None
    out_dir_arg: str | None = None
    for a in argv:
        if a.startswith("--output_dir="):
            out_dir_arg = a.split("=", 1)[1]
            break
    job_name_arg: str | None = None
    for a in argv:
        if a.startswith("--job_name="):
            job_name_arg = a.split("=", 1)[1]
            break
    slug = run_name or job_name_arg or out_dir_arg and Path(out_dir_arg).name or "lerobot_run"
    for ch in ("/", "\\", " "):
        slug = slug.replace(ch, "_")

    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = RUN_ROOT / "results" / date_part / slug
    results_dir = _unique_dir(base)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    console_log = results_dir / "train_console.log"
    csv_path = results_dir / "train_metrics.csv"
    jsonl_path = results_dir / "train_metrics.jsonl"
    eval_jsonl = results_dir / "eval_events.jsonl"

    cmd = [exe, *argv]

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[str] = []
    eval_points: List[Tuple[int, float]] = []
    last_step_for_eval = 0

    fieldnames = ["step", "smpl", "ep", "epch", "loss", "grdn", "lr", "updt_s", "data_s"]

    with open(console_log, "w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fp.write(line)
            sys.stdout.write(line)
            log_fp.flush()
            sys.stdout.flush()
            stripped = line.strip()
            row = _parse_metrics_tracker_line(stripped)
            if row:
                train_rows.append(dict(row))
                last_step_for_eval = int(row.get("step", last_step_for_eval))
                _append_jsonl(jsonl_path, {"type": "train_log", **row})
            if "Eval policy" in stripped or "Suite " in stripped or "aggregated" in stripped:
                eval_rows.append(stripped)
                _append_jsonl(eval_jsonl, {"type": "eval_log_line", "line": stripped})
                ev = _parse_eval_success(stripped)
                if ev:
                    st, pct = ev
                    st_use = st if st >= 0 else last_step_for_eval
                    eval_points.append((st_use, pct))

        proc.wait()
        code = int(proc.returncode or 0)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in train_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    if eval_rows:
        with open(results_dir / "eval_console_snippets.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(eval_rows))

    if eval_points:
        with open(results_dir / "eval_success.csv", "w", newline="", encoding="utf-8") as f:
            ww = csv.writer(f)
            ww.writerow(["train_step", "success_pct"])
            for st, pct in eval_points:
                ww.writerow([st, pct])

    _write_run_meta(
        results_dir / "run_meta.json",
        cmd=cmd,
        exit_code=code,
        results_dir=results_dir,
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
    )

    _write_figures(train_rows, fig_dir, fmt="pdf")
    _write_figures(train_rows, fig_dir, fmt="png")
    _plot_eval_success(eval_points, fig_dir, "pdf")
    _plot_eval_success(eval_points, fig_dir, "png")

    print(
        f"\n[vla_lab] Captured metrics → {results_dir}\n"
        f"  metrics: {csv_path.name}, {jsonl_path.name}\n"
        f"  figures: {fig_dir}/",
        file=sys.stderr,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Export `vla_v1` Kinova logs to a LeRobot v2 dataset (for SmolVLA fine-tuning).

Requires the `lerobot` package (see `vla_lab/requirements-smolvla.txt`).

Example:
  python -m vla_lab.smolvla_bridge.convert_kinova_to_lerobot \\
    --session-roots logs/data_collection/session_20260506_232450 \\
    --out-dir vla_lab/datasets/lerobot_kinova_v0 \\
    --repo-id kinova_isaac_vla
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from vla_lab.dataset import EpisodeRecord, discover_episodes, split_episodes

from .action_obs_contract import (
    ACTION_KEY,
    CAMERA_KEYS,
    IMAGE_HWC_SHAPE,
    STATE_KEY,
    pack_action_six,
    pack_state_xyz_quat,
    smolvla_base_dataset_features,
)


def _load_image_hwc_uint8(ep: EpisodeRecord, rel: str | None, size_hw: tuple[int, int]) -> np.ndarray | None:
    if not rel:
        return None
    path = ep.folder / rel
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def _git_rev(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _lerobot_version() -> str | None:
    try:
        import lerobot

        return getattr(lerobot, "__version__", None)
    except Exception:
        return None


def convert(
    *,
    session_roots: Sequence[Path],
    out_dir: Path,
    repo_id: str,
    fps: int,
    val_fraction: float,
    split_seed: int,
    train_only: bool,
    drop_no_image: bool,
    overwrite: bool,
    success_only: bool = True,
    use_wrist: bool = False,
) -> Dict[str, Any]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The `lerobot` package is required for export. Install with:\n"
            "  pip install -r vla_lab/requirements-smolvla.txt"
        ) from exc

    roots = [Path(p) for p in session_roots]
    episodes = discover_episodes(roots, success_only=success_only)
    if not episodes:
        raise RuntimeError(f"No episodes found under roots: {[str(r) for r in roots]}")

    train_eps, val_eps = split_episodes(episodes, val_fraction=val_fraction, seed=split_seed)
    to_export = train_eps if train_only else train_eps + val_eps

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists (use --overwrite): {out_dir}")
        shutil.rmtree(out_dir)
    # LeRobotDataset.create() requires `root` to not exist yet (it mkdirs with exist_ok=False).
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    features = smolvla_base_dataset_features(use_video=False)
    h, w, _ = IMAGE_HWC_SHAPE

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=out_dir,
        robot_type="kinova_gen3",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    frames_written = 0
    episodes_written = 0
    for ep in to_export:
        n = len(ep.ticks)
        if n < 2:
            continue
        added = False
        for t in range(n - 1):
            tick_t = ep.ticks[t]
            tick_next = ep.ticks[t + 1]
            if drop_no_image and not tick_t.image_rel:
                continue
            img = _load_image_hwc_uint8(ep, tick_t.image_rel, (h, w))
            if img is None:
                if drop_no_image:
                    continue
                img = np.zeros((h, w, 3), dtype=np.uint8)
            wrist_img = None
            if use_wrist:
                wrist_img = _load_image_hwc_uint8(ep, tick_t.wrist_image_rel, (h, w))
                if wrist_img is None:
                    if drop_no_image:
                        continue  # wrist requested but frame missing -> skip (mixed session?)
                    wrist_img = np.zeros((h, w, 3), dtype=np.uint8)
            act = pack_action_six(tick_next.action_from_prev)
            if act is None:
                continue
            st = pack_state_xyz_quat(tick_t.ee_pos_b, tick_t.ee_quat_b)
            task = ep.instruction or "reach and grasp"
            frame = {
                STATE_KEY: st,
                ACTION_KEY: act,
                "task": task,
            }
            # Slot routing (matches policy_wrapper): camera2 = wrist when exported,
            # otherwise all three slots duplicate the overhead view (legacy).
            for i, key in enumerate(CAMERA_KEYS):
                frame[key] = wrist_img.copy() if (wrist_img is not None and i == 1) else img.copy()
            ds.add_frame(frame)
            frames_written += 1
            added = True
        if added:
            ds.save_episode(parallel_encoding=False)
            episodes_written += 1

    ds.finalize()

    repo_root = Path(__file__).resolve().parents[2]
    manifest: Dict[str, Any] = {
        "kind": "kinova_to_lerobot_export",
        "repo_id": repo_id,
        "out_dir": str(out_dir.resolve()),
        "session_roots": [str(Path(p).resolve()) for p in roots],
        "fps": fps,
        "val_fraction": val_fraction,
        "train_only": train_only,
        "episodes_source": int(len(episodes)),
        "episodes_exported": episodes_written,
        "frames_exported": frames_written,
        "git_rev": _git_rev(repo_root),
        "lerobot_version": _lerobot_version(),
        "use_wrist": bool(use_wrist),
        "notes": (
            ("Action is 6D delta pose (no gripper). camera2 = WRIST view, camera1/camera3 = overhead. "
             if use_wrist else
             "Action is 6D delta pose (no gripper). Three camera keys duplicate the same RGB. ")
            + "Pairing: obs at tick t, action = action_from_prev at tick t+1."
        ),
    }
    (out_dir / "kinova_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--session-roots",
        nargs="+",
        required=True,
        help="Session or data-collection root paths (see vla_lab.dataset.discover_episodes).",
    )
    p.add_argument("--out-dir", type=str, required=True, help="e.g. vla_lab/datasets/lerobot_kinova_v0")
    p.add_argument("--repo-id", type=str, default="kinova_isaac_vla", help="LeRobot dataset repo_id / name in meta")
    p.add_argument("--fps", type=int, default=5, help="Policy logging rate (nominal; used for timestamps).")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Held-out fraction at episode level (info only)")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Export only the train split (episodes), excluding val episodes.",
    )
    p.add_argument("--keep-empty-image-frames", action="store_true", help="Allow black frames when PNG missing")
    p.add_argument("--overwrite", action="store_true", help="Delete out-dir if it exists.")
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Also export failed/truncated/unknown-outcome episodes (default: successful lifts only).",
    )
    p.add_argument(
        "--wrist",
        action="store_true",
        help="Route the wrist view (vla_v4/collect_v4 sessions) into camera2; overhead fills camera1/camera3. "
        "Eval must then run with `smolvla_cameras: both` so the wrapper feeds the same slots.",
    )
    args = p.parse_args()

    manifest = convert(
        session_roots=args.session_roots,
        out_dir=Path(args.out_dir),
        repo_id=str(args.repo_id),
        fps=int(args.fps),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        train_only=bool(args.train_only),
        drop_no_image=not bool(args.keep_empty_image_frames),
        overwrite=bool(args.overwrite),
        success_only=not bool(args.include_failed),
        use_wrist=bool(args.wrist),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

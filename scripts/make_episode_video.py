"""Stitch a logged episode's camera streams into a side-by-side demo video.

Reads the per-camera PNG folders an episode writes (``images/front/``,
``images/wrist/``, ...), pairs them up by tick index, labels each pane, and
writes an mp4 — a quick way to see what the policy will actually be trained
on, and to eyeball a collection run without opening hundreds of PNGs.

Pure post-processing: no Isaac, no GPU. Run in any env with imageio + PIL.

    python scripts/make_episode_video.py \\
        --episode-dir logs/demo_two_cam/session_X/episode_0000 \\
        --cameras front,wrist --fps 5 --out outputs/demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode-dir", type=Path, required=True)
    p.add_argument("--cameras", type=str, default="front,wrist",
                   help="comma-separated camera folder names under images/")
    p.add_argument("--fps", type=float, default=5.0,
                   help="playback rate; the logger ticks at 5 Hz, so 5 = real time")
    p.add_argument("--pane", type=int, default=512, help="height of each pane in px")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--slowdown", type=float, default=1.0,
                   help="divide fps by this (2 = half speed)")
    return p.parse_args()


def main() -> int:
    import imageio.v2 as iio
    import numpy as np
    from PIL import Image, ImageDraw

    args = _parse_args()
    ep = args.episode_dir
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    # collect frames per camera, keyed by the tick index in the filename so
    # streams stay aligned even if one camera dropped a frame
    per_cam: dict[str, dict[int, Path]] = {}
    for cam in cams:
        cam_dir = ep / "images" / cam
        if not cam_dir.is_dir():
            print(f"ERROR: no such camera folder: {cam_dir}")
            return 2
        frames = {}
        for f in sorted(cam_dir.glob("image_*.png")):
            try:
                frames[int(f.stem.split("_")[-1])] = f
            except ValueError:
                continue
        if not frames:
            print(f"ERROR: no frames in {cam_dir}")
            return 2
        per_cam[cam] = frames
        print(f"  {cam}: {len(frames)} frames")

    ticks = sorted(set.intersection(*(set(v.keys()) for v in per_cam.values())))
    if not ticks:
        print("ERROR: cameras share no common tick indices")
        return 2
    print(f"  {len(ticks)} ticks common to all cameras")

    out_path = args.out or (ep / "demo.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(0.1, args.fps / max(1e-6, args.slowdown))

    pane = int(args.pane)
    label_h = 34
    writer = iio.get_writer(str(out_path), fps=fps, macro_block_size=1)
    try:
        for t in ticks:
            panes = []
            for cam in cams:
                im = Image.open(per_cam[cam][t]).convert("RGB").resize((pane, pane), Image.LANCZOS)
                canvas = Image.new("RGB", (pane, pane + label_h), (18, 21, 26))
                canvas.paste(im, (0, label_h))
                d = ImageDraw.Draw(canvas)
                d.text((10, 9), f"{cam}", fill=(224, 152, 58))
                d.text((pane - 90, 9), f"t={t:04d}", fill=(139, 147, 163))
                panes.append(np.asarray(canvas))
            writer.append_data(np.concatenate(panes, axis=1))
    finally:
        writer.close()

    print(f"\nwrote {out_path}  ({len(ticks)} frames @ {fps:g} fps, {len(ticks)/fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

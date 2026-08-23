"""Scene images for the dialogue samples: Isaac frames when they exist, a schematic otherwise.

The intent head has to look at *something*: whether a supervisor's "clear it first" is a
preference or an echo depends on how tight the gap actually is, and a model given only the
utterance and the residue cannot know that. So every dialogue sample carries an image of its
scene.

Two sources, and which one was used is recorded per sample:

``isaac``
    Frames captured during the margin sweep, one or more per scene id, with the real camera
    contract. This is what the reported models train on.
``schematic``
    A procedurally drawn top-down diagram of the same geometry. Deliberately crude -- flat
    colours, no shading, no texture -- so that nobody can mistake a model trained on it for a
    model trained on rendered images. It exists so the entire training and evaluation pipeline
    can be exercised end-to-end on a laptop with no simulator, which is how the pipeline gets
    debugged before it costs GPU-hours. Results from schematic training are labelled as such
    everywhere they appear and never enter the headline table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SOURCE_ISAAC = "isaac"
SOURCE_SCHEMATIC = "schematic"

_COLORS = {
    "red": (0.85, 0.18, 0.16),
    "blue": (0.16, 0.35, 0.82),
    "green": (0.20, 0.62, 0.28),
    "yellow": (0.92, 0.78, 0.16),
    "table": (0.78, 0.74, 0.68),
    "robot": (0.35, 0.35, 0.38),
}


def render_schematic(
    layout: Dict[str, Any],
    *,
    size: int = 224,
    view_m: float = 0.80,
    center_xy: Tuple[float, float] = (0.40, 0.0),
    jitter: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Top-down schematic of one scene as a (3, size, size) float array in [0, 1].

    The view is centred on the workspace and spans ``view_m`` metres, matching the framing of
    the top-down camera closely enough that a schematic-trained model's failures are about the
    rendering and not about the crop.
    """
    img = np.zeros((3, size, size), dtype=np.float32)
    for ch in range(3):
        img[ch] = _COLORS["table"][ch]
    px_per_m = size / float(view_m)
    cx, cy = center_xy

    def to_px(x: float, y: float) -> Tuple[float, float]:
        # +x forward -> up the image; +y left -> left in the image.
        col = size / 2.0 - (y - cy) * px_per_m
        row = size / 2.0 - (x - cx) * px_per_m
        return row, col

    dx = dy = 0.0
    if jitter is not None:
        dx, dy = float(jitter.normal(0, 0.006)), float(jitter.normal(0, 0.006))

    # Robot base, as an anchor so the model can tell which side the arm comes from.
    _disc(img, *to_px(0.0 + dx, 0.0 + dy), 0.055 * px_per_m, _COLORS["robot"])

    half = 0.5 * float(layout.get("cube_size_m", 0.05)) * px_per_m
    for d in layout.get("distractors", []):
        x, y = d["xy"]
        _square(img, *to_px(x + dx, y + dy), half, _COLORS["green"])
    bx, by = layout["blocker"]["xy"]
    _square(img, *to_px(bx + dx, by + dy), half, _COLORS.get(layout["blocker"].get("color", "blue"), _COLORS["blue"]))
    tx, ty = layout["target"]["xy"]
    _square(img, *to_px(tx + dx, ty + dy), half, _COLORS.get(layout["target"].get("color", "red"), _COLORS["red"]))

    if jitter is not None:
        img += jitter.normal(0.0, 0.015, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _square(img: np.ndarray, row: float, col: float, half: float, color: Sequence[float]) -> None:
    s = img.shape[-1]
    r0, r1 = int(max(0, row - half)), int(min(s, row + half + 1))
    c0, c1 = int(max(0, col - half)), int(min(s, col + half + 1))
    for ch in range(3):
        img[ch, r0:r1, c0:c1] = color[ch]


def _disc(img: np.ndarray, row: float, col: float, radius: float, color: Sequence[float]) -> None:
    s = img.shape[-1]
    rr, cc = np.ogrid[:s, :s]
    m = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    for ch in range(3):
        img[ch][m] = color[ch]


class SceneAtlas:
    """Scene id -> image. Prefers real frames, falls back to schematics, and says which."""

    def __init__(
        self,
        grid,
        *,
        frames_dir: Optional[Path] = None,
        size: int = 224,
        scene_cfg: Optional[Any] = None,
    ) -> None:
        from environments.supervisory_fetch import layout_for_margin

        self.grid = grid
        self.size = int(size)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self._index: Dict[int, List[Path]] = {}
        self._layouts: Dict[int, Dict[str, Any]] = {}
        self.scene_cfg = scene_cfg
        import random as _random

        for s in grid.scenes:
            lay = layout_for_margin(s.margin_m, cfg=scene_cfg, n_distractors=int(s.clutter),
                                    rng=_random.Random(1000 + s.scene_id))
            lay["cube_size_m"] = getattr(scene_cfg, "cube_size_m", 0.05) if scene_cfg else 0.05
            self._layouts[s.scene_id] = lay
        if self.frames_dir and self.frames_dir.exists():
            for p in sorted(self.frames_dir.glob("scene_*/**/*.png")):
                try:
                    sid = int(p.parent.name.split("_")[-1]) if p.parent.name.startswith("scene_") else int(
                        p.parts[-2].split("_")[-1]
                    )
                except ValueError:
                    continue
                self._index.setdefault(sid, []).append(p)

    @property
    def source(self) -> str:
        return SOURCE_ISAAC if self._index else SOURCE_SCHEMATIC

    def coverage(self) -> Dict[str, Any]:
        ids = [s.scene_id for s in self.grid.scenes]
        have = [i for i in ids if self._index.get(i)]
        return {
            "source": self.source,
            "scenes": len(ids),
            "scenes_with_frames": len(have),
            "frames": int(sum(len(v) for v in self._index.values())),
            "missing": [i for i in ids if i not in have],
        }

    def image(self, scene_id: int, *, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, str]:
        paths = self._index.get(int(scene_id))
        if paths:
            idx = int(rng.integers(len(paths))) if rng is not None else 0
            return _load_png(paths[idx], self.size), SOURCE_ISAAC
        return render_schematic(self._layouts[int(scene_id)], size=self.size, jitter=rng), SOURCE_SCHEMATIC


def _load_png(path: Path, size: int) -> np.ndarray:
    from PIL import Image  # type: ignore

    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0


__all__ = ["SOURCE_ISAAC", "SOURCE_SCHEMATIC", "SceneAtlas", "render_schematic"]

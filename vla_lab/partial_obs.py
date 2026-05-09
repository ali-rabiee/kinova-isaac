"""Partial-observability helpers for sim/real camera streams (occlusion / masking).

Used by `eval_isaaclab.py` for the **single-camera / degraded sensing** ablations
described in `new_changes.md` (external-only vs occluded workspace regions).
"""

from __future__ import annotations

import torch


def apply_occlusion_rgb_chw(
    rgb_chw: torch.Tensor,
    mode: str,
    *,
    fraction: float = 0.35,
) -> torch.Tensor:
    """Return a copy of `rgb_chw` (3,H,W float) with a simple spatial mask.

    Modes:
      - ``none`` / empty: no-op
      - ``bottom_strip``: zero bottom ``fraction`` of rows
      - ``top_strip``: zero top ``fraction`` of rows
      - ``center_box``: zero a central ``fraction`` wide/tall box
      - ``random_patch``: zero a random rectangle (~`fraction` area, stochastic)
    """

    m = str(mode or "none").lower().strip()
    if m in ("", "none", "off", "false", "0"):
        return rgb_chw

    x = rgb_chw.clone()
    _, h, w = x.shape
    frac = float(fraction)
    frac = max(0.05, min(0.95, frac))

    if m == "bottom_strip":
        cut = int(round(h * frac))
        if cut > 0:
            x[:, h - cut :, :] = 0.0
        return x
    if m == "top_strip":
        cut = int(round(h * frac))
        if cut > 0:
            x[:, :cut, :] = 0.0
        return x
    if m == "center_box":
        ch = max(1, int(round(h * frac)))
        cw = max(1, int(round(w * frac)))
        y0 = max(0, (h - ch) // 2)
        x0 = max(0, (w - cw) // 2)
        x[:, y0 : y0 + ch, x0 : x0 + cw] = 0.0
        return x
    if m == "random_patch":
        side_h = max(1, int(round(h * (frac**0.5))))
        side_w = max(1, int(round(w * (frac**0.5))))
        y0 = int(torch.randint(0, max(1, h - side_h + 1), (1,), device=x.device).item())
        x0 = int(torch.randint(0, max(1, w - side_w + 1), (1,), device=x.device).item())
        x[:, y0 : y0 + side_h, x0 : x0 + side_w] = 0.0
        return x

    return x

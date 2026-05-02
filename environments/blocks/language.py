from __future__ import annotations

from typing import Optional


BOX_COLORS: list[tuple[str, tuple[float, float, float]]] = [
    ("red", (0.85, 0.20, 0.20)),
    ("blue", (0.20, 0.35, 0.90)),
    ("yellow", (0.95, 0.85, 0.20)),
    ("purple", (0.65, 0.25, 0.80)),
    ("orange", (0.95, 0.55, 0.15)),
    ("cyan", (0.15, 0.80, 0.85)),
]


def box_idx_from_leaf(leaf: str) -> Optional[int]:
    try:
        if "_" in leaf:
            return int(str(leaf).split("_")[-1])
        return None
    except Exception:
        return None


def box_desc_from_prim(prim_path: str) -> tuple[str, Optional[str], Optional[int]]:
    """Return (human_label, color_name, box_idx) for box mode."""
    leaf = str(prim_path).split("/")[-1]
    idx = box_idx_from_leaf(leaf)
    if idx is not None and len(BOX_COLORS) > 0:
        color_name = BOX_COLORS[(idx - 1) % len(BOX_COLORS)][0]
        return f"{color_name} box {idx}", color_name, idx
    return f"box {leaf}", None, idx


def make_language_command(
    *,
    ep_idx: int,
    target_prim: str,
    id_to_label: dict[str, str],
    spawn_mode: str,
) -> tuple[str, dict]:
    """Generate a natural-language instruction for VLA training."""
    leaf = str(target_prim).split("/")[-1]
    human_label = str(id_to_label.get(leaf, "")) or leaf
    color_name = None
    box_idx = None
    if str(spawn_mode) == "box":
        human_label, color_name, box_idx = box_desc_from_prim(str(target_prim))

    templates = [
        "Pick up the {label}.",
        "Reach and pick up the {label}.",
        "Go to the {label} and grasp it.",
        "Grab the {label} and lift it.",
        "Please pick up the {label}.",
        "Move to the {label} and pick it up.",
    ]
    if box_idx is not None:
        templates += [
            "Pick up box number {box_idx}.",
            "Go for box {box_idx} and pick it up.",
            "Grasp box {box_idx} and lift it.",
        ]
    if color_name is not None:
        templates += [
            "Pick up the {color} box.",
            "Reach to the {color} box and pick it up.",
            "Grab the {color} box and lift it.",
        ]
    if (box_idx is not None) and (color_name is not None):
        templates += [
            "Pick up the {color} box (box {box_idx}).",
            "Go to box {box_idx} - the {color} one - and pick it up.",
        ]

    try:
        import random as _random

        rng = _random.Random(int(ep_idx) + 1337)
        tmpl = rng.choice(templates)
    except Exception:
        tmpl = templates[0]

    cmd = tmpl.format(label=human_label, color=str(color_name), box_idx=str(box_idx))
    meta = {
        "target_leaf": leaf,
        "target_label": human_label,
        "box_idx": box_idx,
        "color": color_name,
    }
    return cmd, meta


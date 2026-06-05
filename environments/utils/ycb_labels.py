"""Curated YCB label sets for Isaac Nucleus ``Props/YCB`` spawning.

Labels match ``ObjectLoader._derive_label_from_usd_path`` (basename without
``NNN_`` prefix or extension), e.g. ``003_cracker_box.usd`` → ``cracker_box``.
"""

from __future__ import annotations

# Hand-sized props suitable for Kinova tabletop pick-and-place (~≤10–12 cm).
YCB_SMALL_LABELS: tuple[str, ...] = (
    # Small cans / bottles
    "master_chef_can",
    "tomato_soup_can",
    "tuna_fish_can",
    "potted_meat_can",
    "mustard_bottle",
    # Small boxes
    "sugar_box",
    "pudding_box",
    "gelatin_box",
    # Mug / cup-scale
    "mug",
    # Fruits (compact)
    "banana",
    "apple",
    "lemon",
    "orange",
    "peach",
    "plum",
    "pear",
    "strawberry",
    # Small misc
    "marble",
    "golf_ball",
    "racquetball",
    "tennis_ball",
    "baseball",
    "softball",
    "wood_block",
    "foam_brick",
    "rubiks_cube",
    "dice",
    "fork",
    "knife",
    "spoon",
    "flat_screwdriver",
    "phillips_screwdriver",
)

# Reference only (not used directly unless you add exclude_labels support).
YCB_LARGE_LABELS: tuple[str, ...] = (
    "cracker_box",
    "pitcher",
    "bleach_cleanser",
    "bowl",
    "plate",
    "skillet",
    "wine_glass",
    "power_drill",
    "extra_large_clamp",
    "large_marker",
    "cola",
    "chips_can",
)

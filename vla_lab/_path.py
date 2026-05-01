"""Internal sys.path helper.

The repository's existing scripts (e.g. `data_collection/collect_data.py`)
do path manipulation to make `repo_root/` importable as the project root
even when running under Isaac Lab's bundled Python (which can shadow
`utils`, `environments`, etc. via Kit-bundled modules).

We replicate the same pattern so that `vla_lab/eval_isaaclab.py` can be
invoked via `./IsaacLab/isaaclab.sh -p vla_lab/eval_isaaclab.py ...` and
still import `environments`, `controllers`, `data_collection`, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_repo_imports() -> Path:
    """Ensure the kinova-isaac repo root is at the front of sys.path.

    Returns the resolved repo root path.
    """
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    env_mod = sys.modules.get("environments")
    if env_mod is not None and not hasattr(env_mod, "__path__"):
        del sys.modules["environments"]

    utils_mod = sys.modules.get("utils")
    if utils_mod is not None:
        utils_file = str(getattr(utils_mod, "__file__", "") or "")
        if utils_file and root_str not in utils_file:
            for k in list(sys.modules.keys()):
                if k == "utils" or k.startswith("utils."):
                    del sys.modules[k]

    return root

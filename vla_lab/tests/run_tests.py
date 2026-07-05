"""No-pytest test runner for the HRI-pivot suite.

Runs every ``test_*`` function across the test modules and reports a summary. Exits non-zero
if any test fails, so it works as a CI gate without pytest::

    python -m vla_lab.tests.run_tests
"""

from __future__ import annotations

import importlib
import sys
from typing import List

from vla_lab.tests import run_namespace

MODULES: List[str] = [
    "test_allocation",
    "test_calibration",
    "test_human_study",
    "test_fit_allocator",
    # 2026-07 intent/feedback/wrist additions. test_feedback is torch-free;
    # test_intent/test_multicam skip their torch-dependent cases when torch
    # is unavailable (the suite stays runnable anywhere).
    "test_feedback",
    "test_intent",
    "test_multicam",
]


def main() -> int:
    total_failed = 0
    total_run = 0
    for mod_name in MODULES:
        mod = importlib.import_module(f"vla_lab.tests.{mod_name}")
        ns = {k: getattr(mod, k) for k in dir(mod)}
        n_tests = sum(1 for k in ns if k.startswith("test_") and callable(ns[k]))
        total_run += n_tests
        total_failed += run_namespace(ns, label=mod_name)

    print("\n" + "=" * 60)
    status = "OK" if total_failed == 0 else "FAILED"
    print(f"[{status}] {total_run - total_failed}/{total_run} tests passed across {len(MODULES)} modules")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())

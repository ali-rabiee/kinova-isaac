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
    # --- VLA / act-compute-query track ---------------------------------------
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
    # --- rehab Phase 0 track (2026-08 pivot; see vla_lab/rehab.md §3, §6) -----
    # One test gate covers the whole repository, so ./vla_lab/scripts/run_tests.sh
    # still answers "is anything broken?" for both tracks at once.
    "test_rehab_workspace",
    "test_rehab_carryover",
    "test_rehab_estimand",
    "test_rehab_scheduler",
    "test_rehab_sim_participant",
    "test_rehab_protocol",
    "test_rehab_safety",
    "test_rehab_logging",
    # rehab.md §4 lists the eight modules above; these two cover W8's offline
    # "done when" (agreement machinery on fixtures) and W15's ("each failure
    # mode has a fixture that triggers it, and a passing session that does not").
    "test_rehab_observation",
    "test_rehab_session",
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

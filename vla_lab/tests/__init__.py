"""Offline test suite for the HRI-2027 pivot (allocation / calibration / human_study).

Every test here is **pure Python + NumPy** with no torch / Isaac / LeRobot / pytest
dependency, so the whole suite runs anywhere with::

    python -m vla_lab.tests.run_tests          # run everything
    python -m vla_lab.tests.test_allocation    # run one module

The files are also ``test_*``-function shaped, so ``pytest vla_lab/tests`` collects them if
pytest happens to be installed. ``run_tests`` is the canonical entry point since pytest is
not part of the Isaac env.
"""

from __future__ import annotations

import traceback
from typing import Dict


def run_namespace(ns: Dict[str, object], *, label: str = "") -> int:
    """Run every ``test_*`` callable in ``ns``; return the number that failed."""

    prefix = f"{label}." if label else ""
    failed = 0
    for name in sorted(ns):
        if name.startswith("test_") and callable(ns[name]):
            try:
                ns[name]()  # type: ignore[operator]
                print(f"PASS {prefix}{name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                print(f"FAIL {prefix}{name}: {exc}")
                traceback.print_exc()
    return failed

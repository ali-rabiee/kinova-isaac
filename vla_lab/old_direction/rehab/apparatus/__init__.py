"""W10/W11/W14 — apparatus backends: null, Isaac twin, real Kinova Gen2.

One protocol, three implementations, one session code path (``rehab.md`` §4). Only
:mod:`~vla_lab.rehab.apparatus.null` and the *client half* of
:mod:`~vla_lab.rehab.apparatus.kinova_gen2` import cleanly with no simulator and no ROS; the
Isaac backend resolves its imports lazily inside its methods, so importing this package is
always free.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import (
    HALT_REASONS,
    Apparatus,
    ApparatusFault,
    ApparatusState,
    BaseApparatus,
    PresentResult,
)
from .kinova_gen2 import (
    BRIDGE_CONTRACT,
    FakeGen2Driver,
    Gen2Config,
    KinovaGen2Apparatus,
    LoopbackTransport,
    UnixSocketTransport,
    connect_gen2,
)
from .null import NullApparatus

BACKEND_NULL = "null"
BACKEND_TWIN = "twin"
BACKEND_REAL = "real"
BACKENDS = (BACKEND_NULL, BACKEND_TWIN, BACKEND_REAL)


def make_apparatus(
    backend: str,
    contract: Any,
    *,
    clock: Optional[Any] = None,
    manual_clock: Optional[Any] = None,
    socket_path: Optional[str] = None,
    headless: bool = True,
    **kw: Any,
):
    """Build the named backend. ``real`` needs a running bridge (see ``kinova_gen2``)."""

    name = str(backend)
    if name == BACKEND_NULL:
        return NullApparatus(contract.timing, clock=clock, manual_clock=manual_clock, **kw)
    if name == BACKEND_TWIN:
        from .isaac_apparatus import IsaacApparatus  # lazy: pulls Isaac only when asked for

        return IsaacApparatus(contract, headless=headless, clock=clock, **kw)
    if name == BACKEND_REAL:
        if not socket_path:
            raise ValueError("the real Gen2 backend needs --gen2-socket pointing at the driver bridge")
        return KinovaGen2Apparatus(
            UnixSocketTransport(socket_path), timing=contract.timing, clock=clock, **kw
        )
    raise ValueError(f"unknown apparatus backend {backend!r}; known: {BACKENDS}")


__all__ = [
    "Apparatus",
    "BaseApparatus",
    "ApparatusState",
    "ApparatusFault",
    "PresentResult",
    "HALT_REASONS",
    "NullApparatus",
    "KinovaGen2Apparatus",
    "Gen2Config",
    "UnixSocketTransport",
    "LoopbackTransport",
    "FakeGen2Driver",
    "BRIDGE_CONTRACT",
    "connect_gen2",
    "make_apparatus",
    "BACKENDS",
    "BACKEND_NULL",
    "BACKEND_TWIN",
    "BACKEND_REAL",
]

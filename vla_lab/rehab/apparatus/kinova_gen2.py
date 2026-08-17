"""W11 — the real Kinova Gen2 (JACO 2, ``j2n6s300``) apparatus backend.

**Why nothing in ``vla_lab/real_robot/`` was reused.** That stub is written for a **Gen3 /
Kortex** arm (``arm_namespace: /my_gen3``) and its interface shape is *policy-chunk stepping* —
both wrong for Phase 0, which needs blocking, settle-verified moves on a Gen2 (``kinova-ros``
``j2n6s300_driver``, or the JACO SDK). It is left untouched for the VLA track; this is a fresh
file (``rehab.md`` §5).

**§12.4 is settled here: the driver runs out-of-process.** ``vla_lab.rehab`` never imports ROS.
A narrow request/response surface (below) talks to a bridge process that owns the vendor
driver, which keeps the ROS 1 dependency out of the conda environment Isaac Sim and the VLA
track's ``numpy<2`` pin live in. If ROS proves hostile, the JACO SDK's Python binding is the
fallback — and the swap is local to the *bridge*, not to this file.

The wire protocol (newline-delimited JSON, one object per line, request/response by ``id``):

.. code-block:: text

    -> {"id": 7, "op": "present", "xy_participant_m": [0.28, -0.05],
        "settle_tolerance_m": 0.01, "settle_dwell_ms": 700, "timeout_ms": 8000}
    <- {"id": 7, "ok": true, "settled": true, "pose_error_m": 0.004, "settle_time_ms": 1830}

    ops: connect | home | present | prompt | go | effort | halt | state | ping | close
    every response carries {"id", "ok"} and, on failure, {"error": "<driver fault code>"}

The transport is **injected** (:class:`Transport`), so:

- :class:`UnixSocketTransport` talks to the real bridge, and
- :class:`LoopbackTransport` + :class:`FakeGen2Driver` exercise every code path here — settle
  verification, fault surfacing, timeouts, e-stop, heartbeat — with no hardware and no ROS.

What this file does **not** do: it does not implement the bridge process. That needs the ROS
workspace or the JACO SDK on the machine, which is an environment dependency, not repo code
(``rehab.md`` §8). :data:`BRIDGE_CONTRACT` documents exactly what the bridge must provide.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..contract import TimingConfig
from ..workspace import PlanarTransform, TargetSpec
from .base import (
    HALT_DRIVER_FAULT,
    HALT_HEARTBEAT_LOST,
    ApparatusFault,
    ApparatusState,
    BaseApparatus,
    PresentResult,
)

#: What the out-of-tree bridge process must implement. Kept as data so the README, the tests,
#: and the eventual bridge author are all reading the same list.
BRIDGE_CONTRACT: Dict[str, str] = {
    "connect": "open the driver, report firmware/driver versions",
    "home": "blocking move to the home posture; returns when settled",
    "present": "blocking Cartesian move to a table-plane point, hold position tolerance for "
               "settle_dwell_ms, return pose_error_m and settle_time_ms",
    "prompt": "play the named audio asset; return its onset timestamp",
    "go": "emit the GO cue (tone); return its onset timestamp",
    "effort": "stage the named effort level; return whether an experimenter action is required",
    "halt": "stop all motion immediately, latch the reason",
    "state": "moving/settled/homed, EE pose, joint currents, fault code, e-stop state",
    "ping": "heartbeat; must answer within heartbeat_timeout_ms",
    "close": "release the driver",
}


@runtime_checkable
class Transport(Protocol):
    """One request/response exchange with the bridge."""

    def request(self, payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class UnixSocketTransport:
    """Newline-delimited JSON over a Unix domain socket (or TCP with ``host``/``port``)."""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        host: Optional[str] = None,
        port: int = 0,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if path:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(float(connect_timeout_s))
            self._sock.connect(str(path))
        elif host:
            self._sock = socket.create_connection((str(host), int(port)), timeout=float(connect_timeout_s))
        else:
            raise ValueError("UnixSocketTransport needs either a socket path or host/port")
        self._fp = self._sock.makefile("rwb")

    def request(self, payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
        self._sock.settimeout(float(timeout_s))
        self._fp.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._fp.flush()
        line = self._fp.readline()
        if not line:
            raise ApparatusFault("bridge closed the connection")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApparatusFault(f"malformed bridge response: {line!r}") from exc

    def close(self) -> None:
        for closer in (self._fp, self._sock):
            try:
                closer.close()
            except Exception:  # noqa: BLE001 - closing must not mask a study outcome
                pass


class LoopbackTransport:
    """In-process transport onto a driver object. Same wire shape, no socket."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self.log: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def request(self, payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
        resp = self.driver.handle(dict(payload), timeout_s=float(timeout_s))
        self.log.append((dict(payload), dict(resp)))
        return resp

    def close(self) -> None:
        handler = getattr(self.driver, "close", None)
        if callable(handler):
            handler()


# ---------------------------------------------------------------------------
# The fake driver (tests, dry-runs, and bridge-author reference)
# ---------------------------------------------------------------------------


@dataclass
class FakeGen2Driver:
    """A JACO 2 that exists only in Python, speaking the bridge protocol.

    Configurable failure modes, because the interesting code paths in
    :class:`KinovaGen2Apparatus` are the ones that only run when hardware misbehaves:
    ``fail_settle_on`` (the arm never converges), ``fault_on`` (a driver fault code),
    ``timeout_on`` (the bridge stops answering), and ``estop_after`` (someone hits the button).
    """

    settle_time_ms: int = 1800
    pose_error_m: float = 0.004
    fail_settle_on: Tuple[int, ...] = ()
    fault_on: Dict[int, str] = field(default_factory=dict)
    timeout_on: Tuple[int, ...] = ()
    estop_after: Optional[int] = None
    driver_version: str = "fake-kinova-ros/1.2.1"

    n_present: int = 0
    connected: bool = False
    estop: bool = False
    closed: bool = False
    _xy: Tuple[float, float] = (0.60, 0.0)
    _halt_reason: Optional[str] = None

    def handle(self, req: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
        op = str(req.get("op", ""))
        rid = req.get("id")
        if op == "connect":
            self.connected = True
            return {"id": rid, "ok": True, "driver_version": self.driver_version, "robot": "j2n6s300"}
        if op == "ping":
            return {"id": rid, "ok": True, "t_ms": int(time.monotonic() * 1000)}
        if op == "home":
            self._xy = (0.60, 0.0)
            return {"id": rid, "ok": True, "settled": True}
        if op == "present":
            self.n_present += 1
            n = self.n_present
            if n in self.timeout_on:
                raise TimeoutError("fake bridge stopped answering")
            if n in self.fault_on:
                return {"id": rid, "ok": False, "error": self.fault_on[n]}
            if self.estop_after is not None and n > int(self.estop_after):
                self.estop = True
                return {"id": rid, "ok": False, "error": "estop_engaged"}
            xy = req.get("xy_participant_m", [0.0, 0.0])
            settled = n not in self.fail_settle_on
            if settled:
                self._xy = (float(xy[0]), float(xy[1]))
            return {
                "id": rid,
                "ok": True,
                "settled": bool(settled),
                "pose_error_m": (self.pose_error_m if settled else 0.09),
                "settle_time_ms": int(self.settle_time_ms),
            }
        if op in ("prompt", "go"):
            return {"id": rid, "ok": True, "t_ms": int(time.monotonic() * 1000)}
        if op == "effort":
            return {"id": rid, "ok": True, "requires_experimenter": True}
        if op == "halt":
            self._halt_reason = str(req.get("reason", ""))
            return {"id": rid, "ok": True}
        if op == "state":
            return {
                "id": rid,
                "ok": True,
                "connected": self.connected,
                "moving": False,
                "settled": True,
                "homed": True,
                "ee_xy_participant_m": list(self._xy),
                "joint_currents_a": [0.4] * 6,
                "fault": self._halt_reason,
                "estop_engaged": self.estop,
            }
        if op == "close":
            self.closed = True
            self.connected = False
            return {"id": rid, "ok": True}
        return {"id": rid, "ok": False, "error": f"unknown op {op!r}"}

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


@dataclass
class Gen2Config:
    """Client-side limits. The bridge enforces its own; these are the second line."""

    request_timeout_s: float = 12.0
    present_timeout_s: float = 15.0
    heartbeat_interval_ms: int = 1000
    heartbeat_timeout_ms: int = 3000
    max_settle_retries: int = 1
    #: Speed/acceleration caps, passed to the bridge on every move. Well below JACO 2
    #: defaults because a human's hands are in the workspace (§6/W12).
    max_cartesian_speed_ms: float = 0.12
    max_cartesian_accel_ms2: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_timeout_s": float(self.request_timeout_s),
            "present_timeout_s": float(self.present_timeout_s),
            "heartbeat_interval_ms": int(self.heartbeat_interval_ms),
            "heartbeat_timeout_ms": int(self.heartbeat_timeout_ms),
            "max_settle_retries": int(self.max_settle_retries),
            "max_cartesian_speed_ms": float(self.max_cartesian_speed_ms),
            "max_cartesian_accel_ms2": float(self.max_cartesian_accel_ms2),
        }


class KinovaGen2Apparatus(BaseApparatus):
    """Blocking, settle-verified presentation on a real JACO 2, via the out-of-process bridge."""

    name = "kinova_gen2"

    def __init__(
        self,
        transport: Transport,
        *,
        timing: Optional[TimingConfig] = None,
        cfg: Optional[Gen2Config] = None,
        clock: Optional[Any] = None,
    ) -> None:
        super().__init__(clock=clock)
        self.transport = transport
        self.timing = timing or TimingConfig()
        self.cfg = cfg or Gen2Config()
        self._req_id = 0
        self.driver_version = ""
        self.settle_times_ms: List[int] = []

    # -- wire --------------------------------------------------------------
    def _call(self, op: str, *, timeout_s: Optional[float] = None, **kw: Any) -> Dict[str, Any]:
        self._req_id += 1
        payload = {"id": self._req_id, "op": str(op), **kw}
        try:
            resp = self.transport.request(payload, timeout_s=float(timeout_s or self.cfg.request_timeout_s))
        except (TimeoutError, socket.timeout) as exc:
            self.halt(HALT_HEARTBEAT_LOST)
            raise ApparatusFault(f"bridge did not answer {op!r} in time") from exc
        if int(resp.get("id", self._req_id)) != self._req_id:
            raise ApparatusFault(f"bridge response id mismatch: sent {self._req_id}, got {resp.get('id')}")
        if not resp.get("ok", False):
            err = str(resp.get("error", "unknown"))
            self.halt(HALT_DRIVER_FAULT)
            self._state.fault = err
            raise ApparatusFault(f"{op} failed: {err}")
        return resp

    # -- protocol ----------------------------------------------------------
    def connect(self) -> None:
        resp = self._call("connect")
        self.driver_version = str(resp.get("driver_version", ""))
        self._state.connected = True
        self._state.last_heartbeat_ms = self.now_ms()

    def home(self) -> None:
        self._call("home", timeout_s=self.cfg.present_timeout_s)
        self._state.homed = True
        self._state.moving = False
        self._state.settled = False

    def present(self, target: TargetSpec) -> PresentResult:
        """Move, stop, verify settle. Retries once on a failed settle, then reports failure.

        A failed settle is **not** an exception: the session runner turns it into a halted
        trial with a recorded reason, and :mod:`vla_lab.rehab.verify_session` counts them.
        Silently presenting an unsettled target would put the participant's hand next to a
        moving arm.
        """

        t0 = self.now_ms()
        self._state.moving = True
        self._state.settled = False
        last: Optional[Dict[str, Any]] = None
        for attempt in range(int(self.cfg.max_settle_retries) + 1):
            last = self._call(
                "present",
                timeout_s=self.cfg.present_timeout_s,
                xy_participant_m=[float(target.x_m), float(target.y_m)],
                target_id=int(target.target_id),
                settle_tolerance_m=float(self.timing.settle_tolerance_m),
                settle_dwell_ms=int(self.timing.settle_dwell_ms),
                timeout_ms=int(self.timing.present_timeout_ms),
                max_speed_ms=float(self.cfg.max_cartesian_speed_ms),
                max_accel_ms2=float(self.cfg.max_cartesian_accel_ms2),
            )
            if bool(last.get("settled", False)):
                break
        t1 = self.now_ms()
        settled = bool((last or {}).get("settled", False))
        self._state.moving = False
        self._state.settled = settled
        self._state.last_heartbeat_ms = t1
        if settled:
            self._state.ee_xy_participant_m = (float(target.x_m), float(target.y_m))
            self.settle_times_ms.append(int((last or {}).get("settle_time_ms", t1 - t0)))
        return PresentResult(
            settled=settled,
            t_present_ms=t0,
            t_settled_ms=t1,
            pose_error_m=float((last or {}).get("pose_error_m", 0.0)),
            settle_time_ms=int((last or {}).get("settle_time_ms", t1 - t0)),
            fault=None if settled else "settle_failed",
        )

    def prompt(self, kind: str, text: str = "") -> int:
        self._call("prompt", kind=str(kind), text=str(text))
        return self.now_ms()

    def go_signal(self) -> int:
        self._call("go")
        return self.now_ms()

    def configure_effort(self, level: str) -> bool:
        resp = self._call("effort", level=str(level))
        self._state.effort_level = str(level)
        return bool(resp.get("requires_experimenter", False))

    def halt(self, reason: str) -> None:
        super().halt(reason)
        try:
            self._call("halt", reason=str(reason))
        except ApparatusFault:
            # A halt that cannot be delivered is exactly when the physical e-stop matters;
            # do not mask the original reason with a transport error.
            pass

    def heartbeat(self) -> bool:
        """Ping the bridge. Halts on a missed heartbeat; returns liveness."""

        try:
            self._call("ping", timeout_s=float(self.cfg.heartbeat_timeout_ms) / 1000.0)
        except ApparatusFault:
            self.halt(HALT_HEARTBEAT_LOST)
            return False
        self._state.last_heartbeat_ms = self.now_ms()
        return True

    def state(self) -> ApparatusState:
        try:
            resp = self._call("state", timeout_s=2.0)
        except ApparatusFault:
            return self._state
        xy = resp.get("ee_xy_participant_m", list(self._state.ee_xy_participant_m))
        self._state = ApparatusState(
            connected=bool(resp.get("connected", False)),
            moving=bool(resp.get("moving", False)),
            settled=bool(resp.get("settled", False)),
            homed=bool(resp.get("homed", False)),
            ee_xy_participant_m=(float(xy[0]), float(xy[1])),
            joint_currents_a=tuple(float(v) for v in resp.get("joint_currents_a", ())),
            fault=resp.get("fault"),
            estop_engaged=bool(resp.get("estop_engaged", False)),
            last_heartbeat_ms=self.now_ms(),
            effort_level=self._state.effort_level,
        )
        return self._state

    def close(self) -> None:
        try:
            self._call("close", timeout_s=2.0)
        except ApparatusFault:
            pass
        self.transport.close()
        self._state.connected = False

    def describe(self) -> Dict[str, Any]:
        n = len(self.settle_times_ms)
        mean = (sum(self.settle_times_ms) / n) if n else float("nan")
        var = (sum((x - mean) ** 2 for x in self.settle_times_ms) / n) if n else float("nan")
        return {
            "apparatus": self.name,
            "driver_version": self.driver_version,
            "cfg": self.cfg.to_dict(),
            "n_presentations": n,
            "settle_time_ms_mean": (None if n == 0 else round(float(mean), 1)),
            "settle_time_ms_sd": (None if n == 0 else round(float(var ** 0.5), 1)),
        }


def connect_gen2(
    socket_path: str,
    *,
    timing: Optional[TimingConfig] = None,
    cfg: Optional[Gen2Config] = None,
) -> KinovaGen2Apparatus:
    """Convenience: connect to a running bridge over a Unix socket."""

    app = KinovaGen2Apparatus(UnixSocketTransport(socket_path), timing=timing, cfg=cfg)
    app.connect()
    return app


__all__ = [
    "BRIDGE_CONTRACT",
    "Transport",
    "UnixSocketTransport",
    "LoopbackTransport",
    "FakeGen2Driver",
    "Gen2Config",
    "KinovaGen2Apparatus",
    "connect_gen2",
]

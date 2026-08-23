"""W1 — bilateral target geometry: frames, the target grid, and reachability.

Every Phase 0 quantity is indexed by a target location, so target placement is a
*scientific* decision, not a layout detail (``rehab.md`` §6/W1):

- The estimand ``pi*(l)`` is a probability map over targets, and nearly all its information
  lives in the **crossover band** near the participant's midline where ``pi* ~ 0.5``. The
  grid therefore densifies there (:attr:`WorkspaceConfig.densification`).
- No target may require **trunk displacement**: leaning shifts the effective midline and
  would confound arm choice. Every target must be reachable by *both* arms from a seated
  posture (:meth:`TargetGrid.human_reachable`).
- Every target must also be presentable by the JACO 2 from its mounting pose
  (:meth:`TargetGrid.robot_reachable`); the twin dry-run (W10) is what validates that
  envelope against the real arm.

Frames (``rehab.md`` §9). Three planar frames chained by :class:`PlanarTransform`:

``participant``
    Origin at the participant's sternum projection on the table plane, ``+x`` forward (away
    from the participant), ``+y`` to the **participant's left**. Solved per participant
    (:mod:`vla_lab.rehab.observation.calibration`) because the midline defines the crossover
    band and therefore the informative region of the estimand.
``table``
    Fixed apparatus frame; the table plane is ``z = table_height_m``.
``robot``
    JACO 2 base frame. The mounting pose is a fixed transform recorded in the contract.

Targets are defined in the **participant** frame, which is what makes them comparable across
participants; the apparatus converts to the robot frame at presentation time.

Handedness enters through :func:`nonpreferred_lateral`: the signed lateral coordinate ``s``
points toward the participant's **nonpreferred** side, so ``pi*`` is monotone increasing in
``s`` for every participant regardless of handedness.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SIDE_LEFT = "left"
SIDE_RIGHT = "right"


# ---------------------------------------------------------------------------
# Planar rigid transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanarTransform:
    """A rigid transform with yaw-only rotation plus a z offset.

    Tabletop apparatus geometry is planar: every frame here shares the table's vertical
    axis, so a full SO(3) rotation would add unidentifiable parameters without adding
    expressiveness. ``apply`` maps a point expressed in the *child* frame into the *parent*
    frame: ``p_parent = R_z(yaw) @ p_child + t``.
    """

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    yaw_rad: float = 0.0

    def apply(self, p: Sequence[float]) -> Tuple[float, float, float]:
        x, y = float(p[0]), float(p[1])
        z = float(p[2]) if len(p) > 2 else 0.0
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (c * x - s * y + self.tx, s * x + c * y + self.ty, z + self.tz)

    def inverse(self) -> "PlanarTransform":
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        # R^T @ (-t)
        return PlanarTransform(
            tx=-(c * self.tx + s * self.ty),
            ty=-(-s * self.tx + c * self.ty),
            tz=-self.tz,
            yaw_rad=-self.yaw_rad,
        )

    def compose(self, other: "PlanarTransform") -> "PlanarTransform":
        """``self ∘ other``: apply ``other`` first, then ``self``."""

        ox, oy, oz = self.apply((other.tx, other.ty, other.tz))
        return PlanarTransform(tx=ox, ty=oy, tz=oz, yaw_rad=_wrap_pi(self.yaw_rad + other.yaw_rad))

    def to_dict(self) -> Dict[str, float]:
        return {"tx": float(self.tx), "ty": float(self.ty), "tz": float(self.tz), "yaw_rad": float(self.yaw_rad)}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PlanarTransform":
        d = d or {}
        return cls(
            tx=float(d.get("tx", 0.0)),
            ty=float(d.get("ty", 0.0)),
            tz=float(d.get("tz", 0.0)),
            yaw_rad=float(d.get("yaw_rad", 0.0)),
        )


def _wrap_pi(a: float) -> float:
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    """One standardized target location, in the participant frame.

    ``effort_index`` is a **geometric** difficulty proxy in ``[0, 1]``: the mean of the two
    shoulder-to-target distances, normalized by the seated reach limit. It is a property of
    the target, not of the arm; the *choice-relevant* quantity is the effort **asymmetry**
    between arms (:func:`effort_asymmetry`), which is ~0 near the midline — which is exactly
    why the crossover band sits there.
    """

    target_id: int
    x_m: float
    y_m: float
    lateral_bin: int
    depth_bin: int
    effort_index: float

    @property
    def xy(self) -> Tuple[float, float]:
        return (float(self.x_m), float(self.y_m))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["x_m"] = round(float(self.x_m), 4)
        d["y_m"] = round(float(self.y_m), 4)
        d["effort_index"] = round(float(self.effort_index), 4)
        return d


@dataclass
class WorkspaceConfig:
    """Geometry of the bilateral reaching workspace. Part of the hashed contract (§9)."""

    # --- lateral axis (the informative one) --------------------------------
    lateral_extent_m: float = 0.34       # targets span y in [-extent, +extent]
    n_lateral: int = 9                   # lateral bins per depth row
    crossover_center_m: float = 0.0      # expected crossover location (participant frame y)
    crossover_halfwidth_m: float = 0.12  # the band the estimand's information lives in
    densification: float = 1.8           # >=1: relative target density inside the band
    min_spacing_m: float = 0.05          # below a puck's width, two targets are one target

    # --- depth axis --------------------------------------------------------
    depths_m: Tuple[float, ...] = (0.28, 0.38)

    # --- human reach model (no trunk displacement) -------------------------
    shoulder_halfwidth_m: float = 0.19   # half the biacromial width; shoulders at (0, +-w)
    shoulder_forward_m: float = 0.0      # shoulder line offset from the sternum origin
    human_max_reach_m: float = 0.65      # seated arm reach from the shoulder
    human_min_reach_m: float = 0.18      # too close = awkward/compressed posture

    # --- robot presentation envelope ---------------------------------------
    table_height_m: float = 0.75
    robot_reach_min_m: float = 0.20
    robot_reach_max_m: float = 0.70

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["depths_m"] = [float(x) for x in self.depths_m]
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "WorkspaceConfig":
        d = dict(d or {})
        if "depths_m" in d:
            d["depths_m"] = tuple(float(x) for x in d["depths_m"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Handedness-relative geometry
# ---------------------------------------------------------------------------


def nonpreferred_lateral(y_m: float, nonpreferred_side: str) -> float:
    """Signed lateral coordinate pointing toward the participant's nonpreferred side.

    ``pi*(s)`` is monotone increasing in ``s`` for every participant: far toward the
    nonpreferred side ``pi* -> 1``, far toward the preferred side ``pi* -> 0``.
    """

    side = str(nonpreferred_side).lower()
    if side == SIDE_LEFT:
        return float(y_m)     # participant's left is +y
    if side == SIDE_RIGHT:
        return -float(y_m)
    raise ValueError(f"nonpreferred_side must be 'left' or 'right'; got {nonpreferred_side!r}")


def shoulder_positions(cfg: WorkspaceConfig) -> Dict[str, Tuple[float, float]]:
    """Left/right shoulder positions in the participant frame."""

    return {
        SIDE_LEFT: (float(cfg.shoulder_forward_m), float(cfg.shoulder_halfwidth_m)),
        SIDE_RIGHT: (float(cfg.shoulder_forward_m), -float(cfg.shoulder_halfwidth_m)),
    }


def reach_distances(target: TargetSpec, cfg: WorkspaceConfig) -> Dict[str, float]:
    """Distance from each shoulder to the target, in the table plane."""

    out: Dict[str, float] = {}
    for side, (sx, sy) in shoulder_positions(cfg).items():
        out[side] = math.hypot(float(target.x_m) - sx, float(target.y_m) - sy)
    return out


def effort_asymmetry(target: TargetSpec, cfg: WorkspaceConfig, nonpreferred_side: str) -> float:
    """``(d_preferred - d_nonpreferred) / max_reach``: how much cheaper the nonpreferred arm is.

    Positive means the nonpreferred arm has the shorter (cheaper) reach, which is when the
    literature says a healthy adult starts choosing it (arm choice is effort-sensitive;
    Nguyen et al. 2023). ~0 in the crossover band, by construction.
    """

    d = reach_distances(target, cfg)
    pref_side = SIDE_RIGHT if str(nonpreferred_side).lower() == SIDE_LEFT else SIDE_LEFT
    return float((d[pref_side] - d[str(nonpreferred_side).lower()]) / max(1e-6, cfg.human_max_reach_m))


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def _densified_lateral_positions(cfg: WorkspaceConfig) -> List[float]:
    """Lateral positions from the inverse CDF of a piecewise-constant density.

    Density is ``densification`` inside the crossover band and ``1`` outside it, over
    ``[-extent, +extent]``. Points sit at the ``(i + 0.5)/n`` quantiles, so:

    - positions are deterministic given the config (stable target IDs across runs), and
    - the number of targets inside the band is **weakly monotone** in ``densification``.
    """

    y0, y1 = -float(cfg.lateral_extent_m), float(cfg.lateral_extent_m)
    b_lo = max(y0, float(cfg.crossover_center_m) - float(cfg.crossover_halfwidth_m))
    b_hi = min(y1, float(cfg.crossover_center_m) + float(cfg.crossover_halfwidth_m))
    if b_hi < b_lo:
        b_lo = b_hi = float(cfg.crossover_center_m)
    dens = max(1e-6, float(cfg.densification))

    # Segments: (start, end, density)
    segs = [(y0, b_lo, 1.0), (b_lo, b_hi, dens), (b_hi, y1, 1.0)]
    segs = [(a, b, d) for (a, b, d) in segs if b > a]
    masses = [(b - a) * d for (a, b, d) in segs]
    total = sum(masses)
    if total <= 0:
        return [0.0 for _ in range(int(cfg.n_lateral))]

    n = max(1, int(cfg.n_lateral))
    out: List[float] = []
    for i in range(n):
        q = (i + 0.5) / n * total
        acc = 0.0
        pos = y1
        for (a, b, d), m in zip(segs, masses):
            if q <= acc + m:
                pos = a + (q - acc) / d
                break
            acc += m
        out.append(float(pos))
    return out


class TargetGrid:
    """The fixed target set ``L`` with stable integer IDs.

    Target IDs are ``depth_bin * n_lateral + lateral_bin`` — a pure function of the
    workspace config, hence stable across runs for a given contract hash.
    """

    def __init__(self, cfg: Optional[WorkspaceConfig] = None) -> None:
        self.cfg = cfg or WorkspaceConfig()
        self.targets: List[TargetSpec] = self._build()
        self._by_id: Dict[int, TargetSpec] = {t.target_id: t for t in self.targets}

    # -- construction ------------------------------------------------------
    def _build(self) -> List[TargetSpec]:
        cfg = self.cfg
        lat = _densified_lateral_positions(cfg)
        n_lat = max(1, int(cfg.n_lateral))
        out: List[TargetSpec] = []
        for di, depth in enumerate(cfg.depths_m):
            for li, y in enumerate(lat):
                t = TargetSpec(
                    target_id=di * n_lat + li,
                    x_m=float(depth),
                    y_m=float(y),
                    lateral_bin=li,
                    depth_bin=di,
                    effort_index=0.0,
                )
                out.append(t)
        # effort_index needs the whole grid to normalize against.
        scale = max(1e-6, float(cfg.human_max_reach_m))
        rebuilt: List[TargetSpec] = []
        for t in out:
            d = reach_distances(t, cfg)
            mean_d = 0.5 * (d[SIDE_LEFT] + d[SIDE_RIGHT])
            rebuilt.append(
                TargetSpec(
                    target_id=t.target_id,
                    x_m=t.x_m,
                    y_m=t.y_m,
                    lateral_bin=t.lateral_bin,
                    depth_bin=t.depth_bin,
                    effort_index=float(min(1.0, max(0.0, mean_d / scale))),
                )
            )
        return rebuilt

    # -- lookup ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self):
        return iter(self.targets)

    def get(self, target_id: int) -> TargetSpec:
        try:
            return self._by_id[int(target_id)]
        except KeyError as exc:
            raise KeyError(f"unknown target_id {target_id}; grid has {sorted(self._by_id)}") from exc

    def ids(self) -> List[int]:
        return [t.target_id for t in self.targets]

    # -- geometry ----------------------------------------------------------
    def in_crossover_band(self, target: TargetSpec) -> bool:
        c, w = float(self.cfg.crossover_center_m), float(self.cfg.crossover_halfwidth_m)
        return bool(abs(float(target.y_m) - c) <= w + 1e-9)

    def crossover_targets(self) -> List[TargetSpec]:
        return [t for t in self.targets if self.in_crossover_band(t)]

    def crossover_weights(self) -> Dict[int, float]:
        """Weights for the crossover-weighted primary outcome: 1.0 in-band, 0.25 out.

        The primary outcome is "averaged over L and weighted toward the crossover band"
        (``rehab.md`` §1.5). Out-of-band targets still count — a policy that got the easy
        extremes wrong would be broken — but they carry little information, so they carry
        little weight.
        """

        return {t.target_id: (1.0 if self.in_crossover_band(t) else 0.25) for t in self.targets}

    def human_reachable(self, target: TargetSpec) -> bool:
        """Reachable by **both** arms from a seated posture (no trunk displacement)."""

        d = reach_distances(target, self.cfg)
        return all(self.cfg.human_min_reach_m <= v <= self.cfg.human_max_reach_m for v in d.values())

    def robot_reachable(self, target: TargetSpec, participant_to_robot: PlanarTransform) -> bool:
        """Presentable by the arm, given the participant->robot transform.

        ``participant_to_robot`` maps a participant-frame point into the robot base frame —
        i.e. ``contract.robot_base_in_participant.inverse()``. The envelope is a radial annulus
        in the table plane: a coarse but honest stand-in for the true JACO 2 workspace, which
        W10's twin sweep is what actually validates.
        """

        x, y, _ = participant_to_robot.apply((target.x_m, target.y_m, 0.0))
        r = math.hypot(x, y)
        return bool(self.cfg.robot_reach_min_m <= r <= self.cfg.robot_reach_max_m)

    def min_pairwise_spacing(self) -> float:
        """Smallest distance between any two targets (m); ``inf`` for a single target."""

        best = float("inf")
        ts = self.targets
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                best = min(best, math.hypot(ts[i].x_m - ts[j].x_m, ts[i].y_m - ts[j].y_m))
        return best

    # -- validation --------------------------------------------------------
    def validate(
        self,
        *,
        participant_to_robot: Optional[PlanarTransform] = None,
        min_crossover_targets: int = 3,
    ) -> List[str]:
        """Return a list of problems; empty means the grid satisfies the contract."""

        problems: List[str] = []
        spacing = self.min_pairwise_spacing()
        if spacing < float(self.cfg.min_spacing_m) - 1e-9:
            problems.append(
                f"minimum inter-target spacing {spacing*1000:.0f} mm < required "
                f"{self.cfg.min_spacing_m*1000:.0f} mm (reduce n_lateral or densification)"
            )
        unreachable = [t.target_id for t in self.targets if not self.human_reachable(t)]
        if unreachable:
            problems.append(
                f"{len(unreachable)} targets are not reachable by both arms without trunk "
                f"displacement: {unreachable[:8]}{'...' if len(unreachable) > 8 else ''}"
            )
        if participant_to_robot is not None:
            not_robot = [t.target_id for t in self.targets if not self.robot_reachable(t, participant_to_robot)]
            if not_robot:
                problems.append(
                    f"{len(not_robot)} targets are outside the robot presentation envelope: "
                    f"{not_robot[:8]}{'...' if len(not_robot) > 8 else ''}"
                )
        n_band = len(self.crossover_targets())
        if n_band < int(min_crossover_targets):
            problems.append(
                f"only {n_band} targets in the crossover band (need >= {min_crossover_targets}); "
                "the estimand's information lives there"
            )
        return problems

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.cfg.to_dict(),
            "n_targets": len(self.targets),
            "targets": [t.to_dict() for t in self.targets],
        }


__all__ = [
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "PlanarTransform",
    "TargetSpec",
    "WorkspaceConfig",
    "TargetGrid",
    "nonpreferred_lateral",
    "shoulder_positions",
    "reach_distances",
    "effort_asymmetry",
]

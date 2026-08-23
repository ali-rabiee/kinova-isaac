"""W13 — physical calibration: cameras, the table plane, and the participant frame.

Three solves, in increasing order of how much the study depends on them:

1. **Camera intrinsics** (:func:`pinhole_intrinsics`) — the same 36 mm-aperture convention the
   VLA track's wrist camera uses, reimplemented here rather than imported so that Phase 0
   never pulls an Isaac-tainted module into a pure-Python session.
2. **Table homography** (:func:`solve_table_homography`) — image points to table-plane metres,
   from >= 4 known correspondences. This is what lets
   :mod:`vla_lab.rehab.observation.vision` work in metres and never see pixels.
3. **The participant frame** (:func:`solve_participant_frame`) — origin at the sternum
   projection, ``+x`` forward, ``+y`` to the participant's left. **Re-solved for every
   participant**, because the midline defines the crossover band and therefore the informative
   region of the estimand (§9). A 2 cm error in the midline moves the whole crossover band by
   2 cm, which at a typical psychometric slope is a ~0.06 shift in ``pi*`` right where the
   primary outcome is measured — so its uncertainty is reported, not assumed away.

Everything here is pure NumPy: a homography from a DLT + SVD, and a frame solve from two
shoulder points. The camera capture and marker detection that *produce* the correspondences
are environment dependencies (``rehab.md`` §8), not repo code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..workspace import PlanarTransform, SIDE_LEFT, SIDE_RIGHT


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


def pinhole_intrinsics(fov_deg: float, resolution: Tuple[int, int]) -> Dict[str, Any]:
    """Pinhole intrinsics implied by the repo's 36 mm-aperture convention.

    ``fx = W / (2 tan(hfov/2))``; square pixels; principal point at the image centre. Matches
    ``environments/utils/camera/wrist.py`` so twin renders and real captures are comparable.
    """

    w, h = int(resolution[0]), int(resolution[1])
    fx = float(w / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0)))
    return {
        "model": "pinhole",
        "width_px": w,
        "height_px": h,
        "fx_px": fx,
        "fy_px": fx,
        "cx_px": w / 2.0,
        "cy_px": h / 2.0,
        "fov_deg_horizontal": float(fov_deg),
        "sensor_aperture_mm": 36.0,
    }


# ---------------------------------------------------------------------------
# Table homography
# ---------------------------------------------------------------------------


def solve_table_homography(
    image_points: Sequence[Sequence[float]],
    table_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """3x3 homography mapping image pixels to table-plane metres (>= 4 correspondences).

    Direct linear transform with an SVD null-space solve. Normalized so ``H[2, 2] == 1``.
    """

    ip = np.asarray(image_points, dtype=np.float64)
    tp = np.asarray(table_points, dtype=np.float64)
    if ip.shape[0] < 4 or ip.shape != tp.shape or ip.shape[1] != 2:
        raise ValueError("need >= 4 matching 2-D correspondences to solve a homography")
    A: List[List[float]] = []
    for (u, v), (x, y) in zip(ip, tp):
        A.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
        A.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (collinear correspondences?)")
    return h / h[2, 2]


def apply_homography(h: np.ndarray, image_point: Sequence[float]) -> Tuple[float, float]:
    p = np.asarray([float(image_point[0]), float(image_point[1]), 1.0])
    q = np.asarray(h, dtype=np.float64) @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("point maps to the horizon under this homography")
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def homography_residuals(
    h: np.ndarray,
    image_points: Sequence[Sequence[float]],
    table_points: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """Reprojection error of a solved homography, in metres. The repeatability check."""

    errs = [
        math.dist(apply_homography(h, ip), (float(tp[0]), float(tp[1])))
        for ip, tp in zip(image_points, table_points)
    ]
    a = np.asarray(errs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_m": float(a.mean()) if a.size else float("nan"),
        "max_m": float(a.max()) if a.size else float("nan"),
        "rms_m": float(math.sqrt(float((a ** 2).mean()))) if a.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# The participant frame
# ---------------------------------------------------------------------------


@dataclass
class ParticipantFrame:
    """The per-participant frame solve, written into ``participant.json`` (§9, §10)."""

    #: Participant frame pose expressed in the table frame.
    participant_in_table: PlanarTransform = field(default_factory=PlanarTransform)
    shoulder_halfwidth_m: float = 0.19
    #: Uncertainty of the midline, propagated from the marker/landmark noise. Reported, not
    #: assumed away: it bounds how precisely the crossover band can be placed.
    midline_sd_m: float = 0.0
    method: str = "shoulders"
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_in_table": self.participant_in_table.to_dict(),
            "shoulder_halfwidth_m": round(float(self.shoulder_halfwidth_m), 5),
            "midline_sd_m": round(float(self.midline_sd_m), 5),
            "method": str(self.method),
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ParticipantFrame":
        d = dict(d or {})
        return cls(
            participant_in_table=PlanarTransform.from_dict(d.get("participant_in_table")),
            shoulder_halfwidth_m=float(d.get("shoulder_halfwidth_m", 0.19)),
            midline_sd_m=float(d.get("midline_sd_m", 0.0)),
            method=str(d.get("method", "shoulders")),
            n_samples=int(d.get("n_samples", 0)),
        )

    def table_to_participant(self) -> PlanarTransform:
        return self.participant_in_table.inverse()


def solve_participant_frame(
    left_shoulder_table: Sequence[Sequence[float]],
    right_shoulder_table: Sequence[Sequence[float]],
    *,
    sternum_forward_offset_m: float = 0.0,
) -> ParticipantFrame:
    """Solve the participant frame from repeated shoulder observations in table coordinates.

    Origin = midpoint of the two shoulders, pushed forward by ``sternum_forward_offset_m``
    (0 puts the origin on the shoulder line, which is the convention §9 fixes). ``+y`` points
    at the participant's **left** shoulder; ``+x`` is 90 degrees from it, forward.

    ``midline_sd_m`` is the standard deviation of the per-sample midpoint's lateral position —
    the honest uncertainty of the quantity the whole estimand is indexed against.
    """

    L = np.asarray(left_shoulder_table, dtype=np.float64).reshape(-1, 2)
    R = np.asarray(right_shoulder_table, dtype=np.float64).reshape(-1, 2)
    if L.shape[0] == 0 or L.shape != R.shape:
        raise ValueError("need matching, non-empty left/right shoulder samples")

    mids = 0.5 * (L + R)
    mid = mids.mean(axis=0)
    lr = (L - R).mean(axis=0)              # right -> left, i.e. the +y direction
    halfwidth = float(np.linalg.norm(lr) / 2.0)
    if halfwidth < 1e-6:
        raise ValueError("shoulder points coincide; cannot define a midline")
    y_axis = lr / np.linalg.norm(lr)
    # +x is +y rotated by -90 degrees (right-handed with z up), i.e. forward from the chair.
    x_axis = np.asarray([y_axis[1], -y_axis[0]])
    yaw = math.atan2(float(x_axis[1]), float(x_axis[0]))
    origin = mid + float(sternum_forward_offset_m) * x_axis

    # Lateral scatter of the midpoint, measured along the frame's own y axis.
    lateral = (mids - mid) @ y_axis
    midline_sd = float(np.std(lateral)) if mids.shape[0] > 1 else 0.0

    return ParticipantFrame(
        participant_in_table=PlanarTransform(tx=float(origin[0]), ty=float(origin[1]), tz=0.0, yaw_rad=float(yaw)),
        shoulder_halfwidth_m=halfwidth,
        midline_sd_m=midline_sd,
        method="shoulders",
        n_samples=int(mids.shape[0]),
    )


@dataclass
class CalibrationBundle:
    """Everything one session's calibration produces. Stamped into ``participant.json``."""

    participant_frame: ParticipantFrame = field(default_factory=ParticipantFrame)
    intrinsics: Dict[str, Dict[str, Any]] = field(default_factory=dict)     # camera -> intrinsics
    homographies: Dict[str, List[List[float]]] = field(default_factory=dict)  # camera -> 3x3
    residuals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Hand-eye extrinsics of the wrist camera, re-aimed for Phase 0 (W11).
    wrist_hand_eye: Dict[str, Any] = field(default_factory=dict)
    robot_base_in_table: PlanarTransform = field(default_factory=PlanarTransform)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_frame": self.participant_frame.to_dict(),
            "intrinsics": self.intrinsics,
            "homographies": self.homographies,
            "residuals": self.residuals,
            "wrist_hand_eye": self.wrist_hand_eye,
            "robot_base_in_table": self.robot_base_in_table.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CalibrationBundle":
        d = dict(d or {})
        return cls(
            participant_frame=ParticipantFrame.from_dict(d.get("participant_frame")),
            intrinsics=dict(d.get("intrinsics", {})),
            homographies={k: [list(map(float, r)) for r in v] for k, v in (d.get("homographies") or {}).items()},
            residuals=dict(d.get("residuals", {})),
            wrist_hand_eye=dict(d.get("wrist_hand_eye", {})),
            robot_base_in_table=PlanarTransform.from_dict(d.get("robot_base_in_table")),
        )

    def check(self, *, max_reprojection_m: float = 0.01, max_midline_sd_m: float = 0.02) -> List[str]:
        """Problems that should block a session from starting."""

        out: List[str] = []
        for cam, res in self.residuals.items():
            if float(res.get("max_m", 0.0)) > float(max_reprojection_m):
                out.append(
                    f"{cam} homography reprojection error {1000*float(res['max_m']):.0f} mm exceeds "
                    f"{1000*max_reprojection_m:.0f} mm"
                )
        sd = float(self.participant_frame.midline_sd_m)
        if sd > float(max_midline_sd_m):
            out.append(
                f"participant midline uncertainty {1000*sd:.0f} mm exceeds {1000*max_midline_sd_m:.0f} mm; "
                "the crossover band cannot be placed reliably (§9)"
            )
        return out


__all__ = [
    "pinhole_intrinsics",
    "solve_table_homography",
    "apply_homography",
    "homography_residuals",
    "ParticipantFrame",
    "solve_participant_frame",
    "CalibrationBundle",
]

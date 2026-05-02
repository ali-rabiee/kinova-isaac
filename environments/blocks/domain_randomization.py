from __future__ import annotations

import time
from typing import Optional


class BlocksDomainRandomizer:
    """Applies per-episode camera and lighting randomization for the blocks environment."""

    def __init__(self, *, args, top_down_camera, enable_cameras: bool, light_prim_path: str = "/World/Light") -> None:
        self.args = args
        self.top_down_camera = top_down_camera
        self.enable_cameras = bool(enable_cameras)
        self.light_prim_path = str(light_prim_path)
        self.base_cam_pos = None
        self.base_cam_fov = None
        self.cam_prim_path = None
        try:
            if top_down_camera is not None:
                self.base_cam_pos = tuple(float(v) for v in getattr(top_down_camera, "position", (0.4, 0.0, 4.0)))
                self.base_cam_fov = float(getattr(top_down_camera, "fov", 65.0))
                self.cam_prim_path = str(getattr(top_down_camera, "prim_path", ""))
        except Exception:
            self.base_cam_pos = None
            self.base_cam_fov = None
            self.cam_prim_path = None

        try:
            if getattr(args, "domain_rand_seed", None) is not None:
                self.seed_base = int(getattr(args, "domain_rand_seed"))
            else:
                self.seed_base = int(time.time() * 1000) & 0x7FFFFFFF
        except Exception:
            self.seed_base = 0

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.args, "domain_rand", False))

    def apply(self, *, ep_idx: int, logger: Optional[object] = None) -> Optional[dict]:
        """Apply domain randomization once and return the sampled parameters."""
        if not self.enabled:
            return None
        try:
            import importlib
            import random

            omni_usd = importlib.import_module("omni.usd")
            UsdGeom = importlib.import_module("pxr.UsdGeom")
            UsdLux = importlib.import_module("pxr.UsdLux")
            Gf = importlib.import_module("pxr.Gf")
            stage = omni_usd.get_context().get_stage()
        except Exception:
            return None

        seed = int((self.seed_base or 0) + int(ep_idx))
        rng = random.Random(seed)
        out: dict = {"enabled": True, "seed": int(seed)}

        try:
            prim = stage.GetPrimAtPath(str(self.light_prim_path))
            if prim.IsValid():
                dome = UsdLux.DomeLight(prim)
                try:
                    base_int = float(dome.GetIntensityAttr().Get())
                except Exception:
                    base_int = 2000.0
                try:
                    c = dome.GetColorAttr().Get()
                    base_col = (float(c[0]), float(c[1]), float(c[2]))
                except Exception:
                    base_col = (0.75, 0.75, 0.75)

                mult_min = float(getattr(self.args, "domain_rand_light_intensity_mult_min", 0.5))
                mult_max = float(getattr(self.args, "domain_rand_light_intensity_mult_max", 1.5))
                mult = float(rng.uniform(min(mult_min, mult_max), max(mult_min, mult_max)))
                intensity = float(max(0.0, base_int * mult))

                jitter = float(getattr(self.args, "domain_rand_light_color_jitter", 0.15))

                def _clamp01(x: float) -> float:
                    return float(max(0.0, min(1.0, x)))

                color = (
                    _clamp01(base_col[0] + rng.uniform(-jitter, jitter)),
                    _clamp01(base_col[1] + rng.uniform(-jitter, jitter)),
                    _clamp01(base_col[2] + rng.uniform(-jitter, jitter)),
                )

                dome.GetIntensityAttr().Set(float(intensity))
                dome.GetColorAttr().Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
                out["light"] = {
                    "prim_path": str(self.light_prim_path),
                    "intensity": float(intensity),
                    "intensity_mult": float(mult),
                    "color_rgb": [float(color[0]), float(color[1]), float(color[2])],
                }
        except Exception:
            pass

        try:
            if self.enable_cameras and self.cam_prim_path and self.base_cam_pos is not None:
                cam_prim = stage.GetPrimAtPath(str(self.cam_prim_path))
                if cam_prim.IsValid():
                    xy = float(getattr(self.args, "domain_rand_camera_xy_m", 0.02))
                    z_j = float(getattr(self.args, "domain_rand_camera_z_m", 0.10))
                    dx = float(rng.uniform(-xy, xy))
                    dy = float(rng.uniform(-xy, xy))
                    dz = float(rng.uniform(-z_j, z_j))
                    x0, y0, z0 = self.base_cam_pos
                    pos = (float(x0 + dx), float(y0 + dy), float(max(0.5, z0 + dz)))

                    yaw_rng = float(getattr(self.args, "domain_rand_camera_yaw_deg", 20.0))
                    pitch_rng = float(getattr(self.args, "domain_rand_camera_pitch_deg", 0.0))
                    roll_rng = float(getattr(self.args, "domain_rand_camera_roll_deg", 0.0))
                    yaw = float(rng.uniform(-yaw_rng, yaw_rng))
                    pitch = float(rng.uniform(-pitch_rng, pitch_rng)) if pitch_rng > 0 else 0.0
                    roll = float(rng.uniform(-roll_rng, roll_rng)) if roll_rng > 0 else 0.0

                    xform = UsdGeom.Xformable(cam_prim)
                    xform.ClearXformOpOrder()
                    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                    rotate_op.Set(Gf.Vec3f(float(roll), float(pitch), float(yaw)))

                    base_fov = float(self.base_cam_fov) if self.base_cam_fov is not None else 65.0
                    fov_j = float(getattr(self.args, "domain_rand_camera_fov_deg", 5.0))
                    fov = float(base_fov + rng.uniform(-fov_j, fov_j))
                    fov = float(max(15.0, min(120.0, fov)))
                    try:
                        cam = UsdGeom.Camera(cam_prim)
                        import math as _math

                        sensor_size_mm = 36.0
                        focal_length_mm = sensor_size_mm / (2.0 * _math.tan(_math.radians(fov) / 2.0))
                        cam.GetFocalLengthAttr().Set(float(focal_length_mm))
                    except Exception:
                        pass

                    out["camera"] = {
                        "prim_path": str(self.cam_prim_path),
                        "pos_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "rpy_deg": [float(roll), float(pitch), float(yaw)],
                        "fov_deg": float(fov),
                    }
        except Exception:
            pass

        try:
            if logger is not None:
                logger.log_event("domain_randomization", out)
        except Exception:
            pass
        return out


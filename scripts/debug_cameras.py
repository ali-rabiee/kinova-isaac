"""Debug/confirmation script for the front + wrist cameras (M3.5).

Builds the YCB reach-to-grasp scene, spawns a couple of boxes, attaches
whichever cameras are requested, drives the arm through a few poses, and
saves labeled PNGs + the exact config values used -- so the front/wrist
camera geometry can be visually confirmed (or iterated via CLI overrides)
before it's trusted for real data collection.

Does NOT touch the existing top-down camera's behavior; by default this
script only builds front + wrist (pass --cameras top_down,front,wrist to
include top-down too).

Run under Isaac Lab python (use -u: Kit swallows buffered stdout on crashes):
    conda run -n kinova python -u scripts/debug_cameras.py --headless
    conda run -n kinova python -u scripts/debug_cameras.py --headless \\
        --cameras wrist --wrist-offset-pos 0.0 0.0 0.08 --wrist-focal-length-mm 24.0
    conda run -n kinova python -u scripts/debug_cameras.py --headless \\
        --cameras front --front-position 1.0 0.0 1.1 --front-target 0.3 0.0 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cameras", type=str, default="front,wrist",
        help="comma-separated subset of top_down,front,wrist",
    )
    parser.add_argument("--num-objects", type=int, default=2)
    parser.add_argument("--box-size", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/debug_cameras"))
    parser.add_argument(
        "--num-poses", type=int, default=4,
        help="arm poses to sample (home pose + up to 3 perturbed workspace configs)",
    )

    parser.add_argument("--front-position", type=float, nargs=3, default=None)
    parser.add_argument("--front-target", type=float, nargs=3, default=None)
    parser.add_argument("--front-fov", type=float, default=None)
    parser.add_argument(
        "--front-candidate", type=float, nargs=6, action="append", default=None,
        metavar=("X", "Y", "Z", "TX", "TY", "TZ"),
        help="extra front-camera placement to capture in the same run (repeatable); "
             "each gets its own prim + frames, so several placements can be compared "
             "from one Isaac launch",
    )

    parser.add_argument("--wrist-parent-body-name", type=str, default=None)
    parser.add_argument("--wrist-offset-pos", type=float, nargs=3, default=None)
    parser.add_argument("--wrist-offset-rot-wxyz", type=float, nargs=4, default=None)
    parser.add_argument("--wrist-focal-length-mm", type=float, default=None)
    parser.add_argument("--wrist-clipping-range", type=float, nargs=2, default=None)
    parser.add_argument("--wrist-resolution", type=int, nargs=2, default=None)

    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        pass
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Ensure project root on sys.path for modular imports.
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    code = 1
    try:
        code = _run(args)
    finally:
        import os
        import threading

        # Kit sometimes wedges in close() headless; give it 60 s then force-exit
        # with the outcome code (all results were already printed/written).
        t = threading.Thread(target=simulation_app.close, daemon=True)
        t.start()
        t.join(timeout=60)
        os._exit(code)


def _save_png(arr, path: Path) -> None:
    try:
        from PIL import Image

        Image.fromarray(arr).save(str(path))
        return
    except Exception as e_pil:
        try:
            import cv2

            cv2.imwrite(str(path), arr[..., ::-1])  # RGB -> BGR
            return
        except Exception as e_cv2:
            import numpy as np

            np.save(str(path.with_suffix(".npy")), arr)
            print(f"[WARN] PNG save failed (PIL: {e_pil}; cv2: {e_cv2}); wrote raw npy instead")


def _capture(sensor) -> "object | None":
    import numpy as np

    data = sensor.data
    rgb = data.output.get("rgb") if data.output is not None else None
    if rgb is None:
        return None
    arr = rgb[0].cpu().numpy() if hasattr(rgb, "dim") and rgb.dim() == 4 else np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr[..., :3]


def _report_scene(robot, spawned_paths, cam_prim_paths: dict, ee_body_name: str) -> None:
    """Print where everything actually is in world coordinates.

    The camera's resolved world translation + look direction is the ground
    truth for "is this camera in front of or behind the robot" -- read it
    here rather than inferring it from the rendered frame.
    """
    import numpy as np
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()

    def _world_xform(path: str):
        prim = stage.GetPrimAtPath(str(path))
        if not prim.IsValid():
            return None
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    print("\n=== scene report (world frame) ===")
    root = robot.data.root_pose_w[0, :3].cpu().numpy()
    print(f"  robot base      : {np.round(root, 3).tolist()}")
    try:
        ids, _ = robot.find_bodies([ee_body_name])
        ee = robot.data.body_pose_w[0, int(ids[0]), :3].cpu().numpy()
        print(f"  end effector    : {np.round(ee, 3).tolist()}")
    except Exception as e:
        print(f"  end effector    : <unavailable: {e}>")

    for p in spawned_paths:
        xf = _world_xform(p)
        if xf is not None:
            t = xf.ExtractTranslation()
            print(f"  box {str(p).split('/')[-1]:<12}: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

    for label, path in cam_prim_paths.items():
        xf = _world_xform(path)
        if xf is None:
            print(f"  camera {label:<9}: <prim not found at {path}>")
            continue
        t = xf.ExtractTranslation()
        fwd = xf.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
        print(
            f"  camera {label:<9}: pos [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]  "
            f"look dir [{fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f}]"
        )
    print("  (+X is the side the boxes spawn on, i.e. the robot's working side)\n")


def _run(args) -> int:
    import torch

    from environments.base import default_jaco2_home_pose
    from environments.utils.camera import (
        CAMERA_CONFIGS,
        FrontCameraConfig,
        WristCameraConfig,
        build_camera,
        find_prim_path_by_name,
        list_all_descendant_prim_names,
    )
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import object_loader_kwargs_from_physix
    from environments.ycb_reach_to_grasp import YCBReachToGraspEnv

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    for c in cameras:
        if c not in CAMERA_CONFIGS:
            print(f"ERROR: unknown camera {c!r}; choices: {list(CAMERA_CONFIGS)}")
            return 2

    front_cfg = FrontCameraConfig()
    if args.front_position is not None:
        front_cfg.position = tuple(args.front_position)
    if args.front_target is not None:
        front_cfg.target = tuple(args.front_target)
    if args.front_fov is not None:
        front_cfg.fov = args.front_fov

    # One config per front placement to capture. Multiple candidates each get
    # their own prim so several placements can be compared from one Isaac
    # launch (startup dominates runtime).
    front_cfgs: list[tuple[str, FrontCameraConfig]] = []
    if args.front_candidate:
        for i, cand in enumerate(args.front_candidate):
            cfg_i = FrontCameraConfig()
            cfg_i.prim_path = f"/World/Origin1/FrontCamera_{i}"
            cfg_i.position = tuple(cand[0:3])
            cfg_i.target = tuple(cand[3:6])
            cfg_i.fov = front_cfg.fov
            cfg_i.resolution = front_cfg.resolution
            front_cfgs.append((f"front{i}", cfg_i))
    else:
        front_cfgs.append(("front", front_cfg))

    wrist_cfg = WristCameraConfig()
    if args.wrist_parent_body_name is not None:
        wrist_cfg.parent_body_name = args.wrist_parent_body_name
    if args.wrist_offset_pos is not None:
        wrist_cfg.offset_pos = tuple(args.wrist_offset_pos)
    if args.wrist_offset_rot_wxyz is not None:
        wrist_cfg.offset_rot_wxyz = tuple(args.wrist_offset_rot_wxyz)
    if args.wrist_focal_length_mm is not None:
        wrist_cfg.focal_length_mm = args.wrist_focal_length_mm
    if args.wrist_clipping_range is not None:
        wrist_cfg.clipping_range = tuple(args.wrist_clipping_range)
    if args.wrist_resolution is not None:
        wrist_cfg.resolution = tuple(args.wrist_resolution)

    env = YCBReachToGraspEnv(device=str(args.device))
    sim = env.build_simulation()
    env.design_scene()
    robot = env.robot

    if "wrist" in cameras:
        found = find_prim_path_by_name(robot, wrist_cfg.parent_body_name)
        if found is None:
            print(f"ERROR: no prim named {wrist_cfg.parent_body_name!r} found under the robot.")
            print("Available descendant prim names (pick one with --wrist-parent-body-name):")
            for n in list_all_descendant_prim_names(robot):
                print(f"  {n}")
            return 3
        print(f"[debug_cameras] resolved wrist parent prim: {found}")

    loader_cfg = ObjectLoaderConfig(
        dataset_dirs=[],
        bounds=SpawnBounds(min_xyz=(0.30, -0.30, 0.90), max_xyz=(0.55, 0.30, 0.95)),
        min_distance=0.20,
        min_distance_xy_only=True,
        spawn_mode="box",
        box_size_min=(args.box_size, args.box_size, args.box_size),
        box_size_max=(args.box_size, args.box_size, args.box_size),
        box_color_palette=[(0.85, 0.20, 0.20), (0.20, 0.35, 0.90)],
        box_color_names=["red", "blue"],
        **object_loader_kwargs_from_physix(env.physics_cfg),
    )
    loader = ObjectLoader(loader_cfg)
    spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=args.num_objects)

    sensors = {}
    for name in cameras:
        if name == "front":
            for label, cfg_i in front_cfgs:
                sensors[label] = build_camera("front", cfg=cfg_i)
        elif name == "wrist":
            sensors["wrist"] = build_camera("wrist", robot=robot, cfg=wrist_cfg)
        else:
            sensors[name] = build_camera(name)

    env.reset()
    for s in sensors.values():
        s.reset()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # settle physics so box/robot poses are real before reporting them
    for _ in range(30):
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

    cam_prim_paths = {}
    for name in cameras:
        if name == "front":
            for label, cfg_i in front_cfgs:
                cam_prim_paths[label] = cfg_i.prim_path
        elif name == "top_down":
            cam_prim_paths["top_down"] = env.top_down_camera_cfg.prim_path
    _report_scene(robot, spawned_paths, cam_prim_paths, wrist_cfg.parent_body_name)

    # A few arm poses spanning the workspace, driven via direct joint writes
    # (same idiom as lsteer/isaac/runtime.py::set_arm_joint_positions) -- this
    # is a static visual check, not a motion/executor test, so no controller
    # is needed.
    home = default_jaco2_home_pose()
    perturbations = [(0.0, 0.0), (-0.3, 0.0), (0.3, 0.2), (0.0, -0.3)]
    pose_variants = []
    for dj2, dj4 in perturbations[: max(1, args.num_poses)]:
        variant = dict(home)
        variant["j2n6s300_joint_2"] = home["j2n6s300_joint_2"] + dj2
        variant["j2n6s300_joint_4"] = home["j2n6s300_joint_4"] + dj4
        pose_variants.append(variant)

    names = list(pose_variants[0].keys())
    for step, pose in enumerate(pose_variants):
        if step > 0:  # step 0 is already the post-reset home pose
            ids, _ = robot.find_joints(names, preserve_order=True)
            q = robot.data.joint_pos.clone()
            q[0, ids] = torch.tensor([pose[n] for n in names], dtype=q.dtype, device=q.device)
            robot.write_joint_state_to_sim(q, torch.zeros_like(robot.data.joint_vel))
            robot.reset()

        for _ in range(10):
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            for s in sensors.values():
                s.update(sim.get_physics_dt())

        for name, sensor in sensors.items():
            arr = _capture(sensor)
            if arr is None:
                print(f"[step {step}] {name}: no frame")
                continue
            out_path = args.out_dir / f"{name}_step{step:02d}.png"
            _save_png(arr, out_path)
            print(f"[step {step}] saved {out_path}")

    print("\nConfigs used:")
    if "front" in cameras:
        for label, cfg_i in front_cfgs:
            print(f"  {label}: {cfg_i}")
    if "wrist" in cameras:
        print(f"  wrist: {wrist_cfg}")
    print(f"\nInspect the saved frames under {args.out_dir} before trusting these values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check applied gripper drive gains on USD/PhysX joints.")

    # Gripper gains to apply
    parser.add_argument("--stiffness", type=float, default=40.0, help="Gripper drive stiffness (N-m/rad or N/m)")
    parser.add_argument("--damping", type=float, default=2.0, help="Gripper drive damping (N-m-s/rad or N-s/m)")
    parser.add_argument("--effort-limit", type=float, default=50.0, help="Max effort/force limit")
    parser.add_argument("--max-velocity", type=float, default=None, help="Optional max joint velocity")

    # Scene / controller specifics
    parser.add_argument(
        "--joint-regex",
        type=str,
        default=".*_joint_finger_.*|.*_joint_finger_tip_.*",
        help="Regex to select gripper joints",
    )
    # Add AppLauncher args after our custom args to avoid the warning and let it own --device/--headless
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        pass
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure project root on sys.path for modular imports
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    # Launch Omniverse app (must be done before importing sim/pxr)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        # Deferred imports that require a running Kit app
        from environments.ycb_reach_to_grasp import YCBReachToGraspEnv
        from kinova import GripperConfig, GripperController

        env = YCBReachToGraspEnv(device=str(args.device))
        sim = env.build_simulation()
        env.design_scene()
        robot = env.robot
        env.reset()

        # Instantiate a gripper controller with desired gains
        g_cfg = GripperConfig(
            joint_regex=str(args.joint_regex),
            stiffness=float(args.stiffness),
            damping=float(args.damping),
            effort_limit=None if args.effort_limit is None else float(args.effort_limit),
            max_velocity=None if args.max_velocity is None else float(args.max_velocity),
        )
        gripper = GripperController(g_cfg, num_envs=1, device=str(sim.device))
        gripper.resolve_joints(robot)

        prim_path: Optional[str] = getattr(getattr(robot, "cfg", None), "prim_path", None)
        if isinstance(prim_path, str):
            print(f"[Check] Applying drive gains at prim: {prim_path}")
            gripper.set_drive_gains(prim_path)
        else:
            print("[Check][WARN] robot.cfg.prim_path is missing; attempting to proceed without applying gains")

        # Inspect USD for each selected gripper joint
        from pxr import Usd, UsdPhysics, PhysxSchema  # type: ignore[import-not-found]
        from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]

        stage = get_current_stage()
        joint_ids, joint_names = robot.find_joints(g_cfg.joint_regex)
        dof_paths = robot.root_physx_view.dof_paths[0]

        def _drive_api_name(usd_joint_prim: Usd.Prim) -> Optional[str]:
            if usd_joint_prim.IsA(UsdPhysics.RevoluteJoint):
                return "angular"
            if usd_joint_prim.IsA(UsdPhysics.PrismaticJoint):
                return "linear"
            return None

        print("\n[Check] Reading back USD DriveAPI/PhysxJointAPI attributes for gripper joints:\n")
        header = (
            f"{'Joint':40s} | {'Type':7s} | {'stiffness':>10s} | {'damping':>10s} | "
            f"{'maxForce':>10s} | {'maxJointVel':>12s}"
        )
        print(header)
        print("-" * len(header))

        for jid, jname in zip(joint_ids, joint_names):
            usd_joint_path = dof_paths[int(jid)]
            jprim = stage.GetPrimAtPath(usd_joint_path)
            if not jprim.IsValid():
                print(f"{jname:40s} | {'N/A':7s} | invalid prim: {usd_joint_path}")
                continue

            drive_kind = _drive_api_name(jprim)
            if drive_kind is None:
                print(f"{jname:40s} | {'N/A':7s} | not a revolute/prismatic joint: {usd_joint_path}")
                continue

            drive_api = UsdPhysics.DriveAPI(jprim, drive_kind)
            physx_joint_api = PhysxSchema.PhysxJointAPI(jprim)

            # Fetch values from USD
            stiffness = drive_api.GetStiffnessAttr().Get() if drive_api else None
            damping = drive_api.GetDampingAttr().Get() if drive_api else None
            max_force = drive_api.GetMaxForceAttr().Get() if drive_api else None
            max_joint_vel = (
                physx_joint_api.GetMaxJointVelocityAttr().Get() if physx_joint_api else None
            )

            print(
                f"{jname:40s} | {drive_kind:7s} | "
                f"{str(round(stiffness, 4)) if stiffness is not None else 'None':>10s} | "
                f"{str(round(damping, 4)) if damping is not None else 'None':>10s} | "
                f"{str(round(max_force, 4)) if max_force is not None else 'None':>10s} | "
                f"{str(round(max_joint_vel, 4)) if max_joint_vel is not None else 'None':>12s}"
            )

        # Helpful note on unit conversion for angular joints
        print(
            "\n[Note] For revolute joints, USD stores stiffness/damping/max velocity in degree units.\n"
            "       Expected USD values = input_in_rad_units * (pi/180) for stiffness/damping,\n"
            "       and input_max_velocity_rad_per_s * (180/pi) for maxJointVelocity."
        )

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()



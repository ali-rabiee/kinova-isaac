from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _ensure_repo_on_syspath() -> Path:
    """
    This file lives at <repo-root>/grasp-vla/ground_truth_test.py.
    Make sure <repo-root> is on sys.path so we can import controllers/environments/etc.
    """
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    # Isaac/Omni may preload a non-package module named `environments`, which breaks our imports.
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]

    # Isaac may preload `cv2.utils` which can shadow this repo's `utils/` package.
    _utils_mod = sys.modules.get("utils")
    if _utils_mod is not None:
        _utils_file = str(getattr(_utils_mod, "__file__", "") or "")
        if _utils_file and root_str not in _utils_file:
            for _k in list(sys.modules.keys()):
                if _k == "utils" or _k.startswith("utils."):
                    del sys.modules[_k]

    return root


@dataclass
class _TickAction:
    """One action over a single logged tick interval (typically ~0.2s at 5Hz)."""

    dt_s: float
    a7: np.ndarray  # (7,) float32, base-frame ee delta + gripper (usually 0)
    tick_idx: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _load_episode(
    *,
    session_dir: Path,
    episode_idx: int,
) -> tuple[list[dict[str, Any]], str | None]:
    ep_dir = session_dir / f"episode_{int(episode_idx):04d}"
    ticks_path = ep_dir / "ticks.jsonl"
    if not ticks_path.exists():
        raise FileNotFoundError(f"ticks.jsonl not found: {ticks_path}")

    instruction: str | None = None
    instr_path = ep_dir / "instruction.json"
    if instr_path.exists():
        try:
            instruction = json.loads(instr_path.read_text(encoding="utf-8")).get("instruction")
        except Exception:
            instruction = None

    return _read_jsonl(ticks_path), instruction


def _extract_objects_tick0(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ticks:
        return []
    objs = ticks[0].get("objects", [])
    if not isinstance(objs, list):
        return []
    out: list[dict[str, Any]] = []
    for o in objs:
        try:
            oid = str(o.get("id", ""))
            lbl = str(o.get("label", oid))
            pos = [float(x) for x in o["pose_w"]["position_m"]]
            quat = [float(x) for x in o["pose_w"].get("orientation_wxyz", [1, 0, 0, 0])]
            out.append({"id": oid, "label": lbl, "pos_w": pos, "quat_wxyz": quat})
        except Exception:
            continue
    return out


def _extract_tick_actions(
    ticks: list[dict[str, Any]],
    *,
    action_source: str,
    tick_start: int,
    max_ticks: int | None,
) -> list[_TickAction]:
    out: list[_TickAction] = []

    def _dt_fallback(i: int) -> float:
        try:
            t_ms0 = int(ticks[i - 1]["t_ms"])
            t_ms1 = int(ticks[i]["t_ms"])
            return max(1e-6, (t_ms1 - t_ms0) / 1000.0)
        except Exception:
            return 0.2

    # We start from i=1 because action_from_prev describes motion from prev->cur tick.
    for i in range(1, len(ticks)):
        tick_idx = int(ticks[i].get("tick_idx", i))
        if tick_idx < int(tick_start):
            continue
        if max_ticks is not None and len(out) >= int(max_ticks):
            break

        if action_source == "action_from_prev":
            ap = (ticks[i].get("policy") or {}).get("action_from_prev")
            if not ap:
                continue
            try:
                dt_s = float(ap.get("dt_s", _dt_fallback(i)))
                dp = [float(x) for x in ap["ee_delta_pos_b"]]
                dr = [float(x) for x in ap["ee_delta_rotvec_b"]]
                g = float(ap.get("gripper_action", 0.0))
                a7 = np.array(dp + dr + [g], dtype=np.float32)
            except Exception:
                continue
        elif action_source == "user_cmd_7d":
            try:
                dt_s = _dt_fallback(i)
                u = ticks[i]["user"]["joystick"]["cartesian_vel_cmd_7d"]
                a7 = np.array([float(x) for x in u], dtype=np.float32)
            except Exception:
                continue
        else:
            raise ValueError(f"Unknown --action-source: {action_source}")

        if a7.shape != (7,):
            a7 = a7.reshape(-1)[:7]
            if a7.shape[0] < 7:
                a7 = np.pad(a7, (0, 7 - a7.shape[0]))
            a7 = a7.astype(np.float32, copy=False)

        out.append(_TickAction(dt_s=float(dt_s), a7=a7, tick_idx=tick_idx))

    return out


class _ConstantCmdInput:
    """Controller input provider: returns the last command tensor you set."""

    def __init__(self, device: str):
        self.device = torch.device(device)
        self._cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self._cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)

    def set(self, cmd: torch.Tensor) -> None:
        if cmd.ndim == 1:
            cmd = cmd.view(1, -1)
        self._cmd = cmd.to(self.device, dtype=torch.float32)

    def advance(self) -> torch.Tensor:
        return self._cmd


def _spawn_boxes_from_log(
    *,
    objects: list[dict[str, Any]],
    parent_prim_path: str,
    box_size_m: float,
) -> tuple[list[str], dict[str, str]]:
    """Spawn box prims exactly at the positions logged in tick 0."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
    from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
    from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

    import importlib

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")
    prim_utils.create_prim(parent_prim_path, "Xform")

    color_map = {
        "red": (0.85, 0.15, 0.15),
        "blue": (0.20, 0.35, 0.95),
        "green": (0.20, 0.85, 0.20),
        "yellow": (0.90, 0.85, 0.20),
        "purple": (0.65, 0.25, 0.85),
    }

    prim_paths: list[str] = []
    id_to_label: dict[str, str] = {}
    for o in objects:
        oid = str(o.get("id", "")).strip()
        if not oid:
            continue
        label = str(o.get("label", oid))
        rgb = (0.7, 0.7, 0.7)
        for k, v in color_map.items():
            if k in label.lower():
                rgb = v
                break

        prim_path = f"{parent_prim_path.rstrip('/')}/{oid}"
        pos_w = tuple(float(x) for x in o.get("pos_w", [0.5, 0.0, 0.82]))
        quat_wxyz = tuple(float(x) for x in o.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]))

        mat_vis = PreviewSurfaceCfg(diffuse_color=rgb)
        mat_phys = RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0, restitution=0.0)
        cube = CuboidCfg(
            size=(float(box_size_m), float(box_size_m), float(box_size_m)),
            visual_material=mat_vis,
            physics_material=mat_phys,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0015),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=600.0),
        )
        cube.func(prim_path, cube, translation=pos_w, orientation=quat_wxyz)

        prim_paths.append(prim_path)
        id_to_label[oid] = label

    return prim_paths, id_to_label


def main() -> int:
    repo_root = _ensure_repo_on_syspath()

    # Unbuffered-ish prints for headless/background runs.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Replay logged ground-truth actions inside Isaac Sim (sanity test).")
    ap.add_argument(
        "--session-dir",
        type=str,
        default=str(
            (repo_root / "logs" / "data_collection" / "session_20251231_214301").resolve()
        ),
        help="Path to a data_collection session directory (contains episode_XXXX folders).",
    )
    ap.add_argument("--episode", type=int, default=0, help="Episode index (0-based, episode_0000 = 0).")
    ap.add_argument(
        "--action-source",
        type=str,
        default="action_from_prev",
        choices=["action_from_prev", "user_cmd_7d"],
        help=(
            "Which field to treat as the ground-truth action.\n"
            " - action_from_prev: measured EE delta per tick (matches xvla training stats)\n"
            " - user_cmd_7d: last commanded joystick/teleop cmd (often much smaller)\n"
        ),
    )
    ap.add_argument("--tick-start", type=int, default=1, help="First tick_idx to replay from (default: 1).")
    ap.add_argument("--max-ticks", type=int, default=None, help="Replay at most N tick actions.")
    ap.add_argument("--box-size", type=float, default=0.08)
    ap.add_argument("--spawn-from-log", action="store_true", help="Spawn boxes at tick0 logged poses (recommended).")
    ap.add_argument("--speed", type=float, default=0.7, help="Controller linear speed (m/s) for clamping.")
    ap.add_argument("--rot-speed", type=float, default=2.0, help="Rotation speed (rad/s) for clamping.")
    ap.add_argument("--action-scale", type=float, default=1.0, help="Scale ground-truth action before replay.")
    ap.add_argument("--action-ema", type=float, default=0.0, help="EMA smoothing for motion command (0..1).")
    ap.add_argument("--settle-steps", type=int, default=180, help="Physics settle steps before replay.")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--print-every-s", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")

    # IsaacLab / AppLauncher args (headless, enable_cameras, device, etc.)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()

    # Start Kit / Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Kit may mutate sys.path; re-pin repo root (for environments/controllers imports).
    repo_root = _ensure_repo_on_syspath()

    # Heavy imports must happen after Kit is started.
    import carb
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    # Enable cameras if requested.
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", enable_cameras)

    from environments.reach_to_grasp_VLA.config import DEFAULT_SCENE, DEFAULT_TOP_DOWN_CAMERA
    from environments.reach_to_grasp_VLA.utils import design_scene
    from environments.utils.camera import create_topdown_camera
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg
    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController

    # Load episode logs (no LeRobot involved)
    session_dir = Path(str(args.session_dir)).expanduser().resolve()
    ticks, instruction = _load_episode(session_dir=session_dir, episode_idx=int(args.episode))
    actions = _extract_tick_actions(
        ticks,
        action_source=str(args.action_source),
        tick_start=int(args.tick_start),
        max_ticks=None if args.max_ticks is None else int(args.max_ticks),
    )
    if not actions:
        raise RuntimeError(
            f"No actions extracted (source={args.action_source}) from session={session_dir} episode={args.episode}."
        )

    if bool(getattr(args, "debug", False)):
        a_stack = np.stack([a.a7 for a in actions], axis=0)
        print(f"[GT] Loaded {len(ticks)} ticks, extracted {len(actions)} actions.")
        print(f"[GT] action7 min={a_stack.min(axis=0)} max={a_stack.max(axis=0)} std={a_stack.std(axis=0)}")
        sys.stdout.flush()

    # Sim setup (match vla_v1 style)
    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Top-down camera prim + sensor
    camera_sensor = None
    if enable_cameras and DEFAULT_TOP_DOWN_CAMERA is not None:
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)
        camera_cfg = CameraCfg(
            prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=None,
            data_types=["rgb"],
            width=256,
            height=256,
        )
        camera_sensor = Camera(cfg=camera_cfg)

    # Objects: either spawn from log poses or skip (replay motion still valid).
    prim_paths: list[str] = []
    id_to_label: dict[str, str] = {}
    if bool(getattr(args, "spawn_from_log", False)):
        objs0 = _extract_objects_tick0(ticks)
        prim_paths, id_to_label = _spawn_boxes_from_log(
            objects=objs0,
            parent_prim_path="/World/Origin1/Objects",
            box_size_m=float(args.box_size),
        )
        print(f"[GT] Spawned {len(prim_paths)} boxes from log tick0 poses.")
    else:
        print("[GT] Not spawning objects from log (use --spawn-from-log to match training episode).")

    # Reset sim + robot to defaults (important for stability)
    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    if camera_sensor is not None:
        try:
            camera_sensor.reset()
        except Exception:
            camera_sensor = None

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name="j2n6s300_end_effector",
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(args.speed),
        workspace_min=(0.20, -0.45, 0.0),
        workspace_max=(0.6, 0.45, 0.35),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.reset(robot)
    controller.set_mode("translate")
    cmd_input = _ConstantCmdInput(device=str(sim.device))
    controller.set_input_provider(cmd_input)  # type: ignore[arg-type]

    # Optional settle
    dt = float(sim.get_physics_dt())
    settle_steps = max(0, int(args.settle_steps))
    if settle_steps > 0:
        zero = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
        for _ in range(settle_steps):
            cmd_input.set(zero)
            controller.step(robot, dt)
            sim.step(render=bool(enable_cameras))
            robot.update(dt)
        if bool(getattr(args, "debug", False)):
            print(f"[GT] Settled for {settle_steps} physics steps.")

    # Replay loop
    max_dpos = float(ctrl_cfg.linear_speed_mps) * dt
    max_drot = float(args.rot_speed) * dt
    ema = float(getattr(args, "action_ema", 0.0))
    ema = ema if (0.0 <= ema < 1.0) else 0.0

    t0 = time.time()
    last_print = 0.0
    steps = 0
    a_idx = 0
    steps_left_in_tick = 0
    last_cmd = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)

    print(f"[GT] instruction={instruction!r}")
    print(f"[GT] Replaying {len(actions)} tick actions from episode_{int(args.episode):04d} ({args.action_source}).")

    while simulation_app.is_running() and (time.time() - t0) < float(args.max_seconds):
        steps += 1

        # Load next tick action if needed
        if steps_left_in_tick <= 0:
            if a_idx >= len(actions):
                break
            ta = actions[a_idx]
            a_idx += 1
            steps_per_tick = max(1, int(round(float(ta.dt_s) / max(1e-9, dt))))
            steps_left_in_tick = steps_per_tick

            a = ta.a7.astype(np.float32, copy=False)
            # Scale and distribute across physics steps (matching rollout_xvla_isaac behavior).
            a_step = a.copy()
            a_step[0:6] = a_step[0:6] * float(args.action_scale) / float(steps_per_tick)

            # Clamp per physics step.
            a_step[0:3] = np.clip(a_step[0:3], -max_dpos, max_dpos)
            a_step[3:6] = np.clip(a_step[3:6], -max_drot, max_drot)

            # Gripper action (rare/usually 0 in these logs)
            g_val = float(a_step[6])
            if abs(g_val) > 0.2:
                controller.set_mode("gripper")
                last_cmd = torch.zeros(1, 7, dtype=torch.float32, device=sim.device)
                last_cmd[0, 6] = 1.0 if g_val > 0.0 else -1.0
            else:
                controller.set_mode("translate")
                new_cmd = torch.tensor(a_step, dtype=torch.float32, device=sim.device).view(1, 7)
                if ema > 0.0:
                    new_cmd[0, 0:6] = ema * last_cmd[0, 0:6] + (1.0 - ema) * new_cmd[0, 0:6]
                last_cmd = new_cmd

            if bool(getattr(args, "debug", False)):
                print(
                    f"[GT] tick={ta.tick_idx:04d} dt_s={ta.dt_s:.3f} steps_per_tick={steps_per_tick} "
                    f"a7={ta.a7.tolist()} a7_step={a_step.tolist()}"
                )

        # Apply current command every physics step
        cmd_input.set(last_cmd)
        controller.step(robot, dt)
        sim.step(render=bool(enable_cameras))
        robot.update(dt)
        steps_left_in_tick -= 1

        now = time.time() - t0
        if float(getattr(args, "print_every_s", 0.0)) > 0.0 and (now - last_print) >= float(args.print_every_s):
            last_print = now
            cmd_np = last_cmd.detach().view(-1).to("cpu").numpy()
            cmd_norm = float(np.linalg.norm(cmd_np[:6]))
            print(f"[GT] t={now:5.1f}s steps={steps} a_idx={a_idx}/{len(actions)} cmd_norm={cmd_norm:.4f}")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



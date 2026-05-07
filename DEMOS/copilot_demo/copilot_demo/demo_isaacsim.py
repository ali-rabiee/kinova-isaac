"""IsaacSim reach-to-grasp demo with grasp-copilot assistance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from isaaclab.app import AppLauncher

# Ensure repo roots are importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Ensure this package is importable when running the file directly:
#   python kinova-isaac/copilot_demo/copilot_demo/demo_isaacsim.py
COPILOT_ROOT = ROOT / "copilot_demo"
if COPILOT_ROOT.exists() and str(COPILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(COPILOT_ROOT))
GP_ROOT = ROOT.parent / "grasp-copilot"
if GP_ROOT.exists() and str(GP_ROOT) not in sys.path:
    sys.path.insert(0, str(GP_ROOT))

from controllers import (
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
    Se3KeyboardInput,
    ModeManager,
)
from data_collection.core.input_mux import CommandMuxInputProvider
from data_collection.core.objects import ObjectsTracker
from data_generator.oracle import validate_tool_call  # type: ignore

from copilot_demo.backends import HFBackend, OracleBackend, _strip_choice_label
from copilot_demo.extractor import ExtractorConfig, InputExtractor
from copilot_demo.executor import ActionExecutor, ExecutorConfig


def _mode_for_oracle(mode: str | None) -> str:
    """Map controller/UI mode to oracle's expected UserState.mode values."""
    m = str(mode or "translation").lower().strip()
    if m == "translate":
        return "translation"
    if m == "rotate":
        return "rotation"
    if m in {"translation", "rotation", "gripper"}:
        return m
    return "translation"


def _choice_to_user_content(choice_str: str) -> str:
    s = choice_str.strip()
    semantic = _strip_choice_label(s).strip().upper()
    if semantic in {"YES", "NO"}:
        return semantic
    return _strip_choice_label(s).strip()


def _apply_none_exclusion(memory: Dict[str, Any], last_prompt: Dict[str, Any], objects: Sequence[Dict[str, Any]]) -> None:
    labels: List[str] = []
    for c in list(last_prompt.get("choices") or []):
        lab = _strip_choice_label(str(c)).strip()
        if lab.upper() in {"YES", "NO"} or lab.lower() == "none of them":
            continue
        labels.append(lab)
    ex = set(memory.get("excluded_obj_ids") or [])
    if labels:
        for lab in labels:
            for o in objects:
                if str(o.get("label")) == lab:
                    ex.add(str(o.get("id")))
    memory["excluded_obj_ids"] = sorted(ex)


def _apply_oracle_user_reply(
    user_content: str,
    *,
    memory: Dict[str, Any],
    state,
    objects: Sequence[Dict[str, Any]],
    update_candidates_cb,
) -> bool:
    """Port of gui_assist_demo oracle reply handler."""
    ctx = state.last_prompt_context or {}
    t = ctx.get("type")

    def reset_conversation_only() -> None:
        memory["n_interactions"] = 0
        memory["past_dialogs"] = []
        memory["last_tool_calls"] = []
        memory["excluded_obj_ids"] = []
        memory["last_action"] = {}
        update_candidates_cb()
        state.intended_obj_id = ctx.get("obj_id") or (objects[0]["id"] if objects else "obj0")
        state.selected_obj_id = None
        state.pending_action_obj_id = None
        state.pending_mode = None
        state.awaiting_confirmation = False
        state.awaiting_help = False
        state.awaiting_choice = False
        state.awaiting_intent_gate = False
        state.awaiting_anything_else = False
        state.awaiting_mode_select = False
        state.terminate_episode = False
        state.last_prompt_context = None

    def set_selected_by_label(label: str) -> None:
        for o in objects:
            if str(o.get("label")) == label:
                state.selected_obj_id = o["id"]
                state.intended_obj_id = o["id"]
                return

    auto_continue = True

    if t == "intent_gate_candidates":
        if user_content.upper() == "YES":
            state.awaiting_choice = True
            state.awaiting_intent_gate = False
            action = str(ctx.get("action") or "APPROACH").upper()
            state.pending_mode = action if action in {"APPROACH", "ALIGN_YAW"} else "APPROACH"
        else:
            state.awaiting_intent_gate = False
            state.awaiting_choice = False
            state.awaiting_anything_else = True
            state.pending_mode = None
            state.selected_obj_id = None
    elif t == "intent_gate_yaw":
        if user_content.upper() == "YES":
            state.awaiting_help = True
            state.awaiting_intent_gate = False
            state.pending_mode = "ALIGN_YAW"
            obj_id = ctx.get("obj_id")
            if isinstance(obj_id, str):
                state.selected_obj_id = obj_id
        else:
            state.awaiting_intent_gate = False
            state.awaiting_help = False
            state.awaiting_anything_else = True
            state.pending_mode = None
            state.selected_obj_id = None
    elif t == "candidate_choice":
        labels: List[str] = list(ctx.get("labels") or [])
        obj_ids: List[str] = list(ctx.get("obj_ids") or [])
        none_index = int(ctx.get("none_index") or (len(labels) + 1))
        if user_content.strip().lower() == "none of them":
            ex = set(memory.get("excluded_obj_ids") or [])
            for oid in obj_ids:
                ex.add(oid)
            memory["excluded_obj_ids"] = sorted(ex)
            state.selected_obj_id = None
            state.awaiting_choice = True
            state.awaiting_confirmation = False
        elif user_content in labels:
            set_selected_by_label(user_content)
            state.awaiting_choice = False
            state.awaiting_confirmation = False
        elif user_content.isdigit():
            idx = int(user_content) - 1
            if int(user_content) == none_index:
                ex = set(memory.get("excluded_obj_ids") or [])
                for oid in obj_ids:
                    ex.add(oid)
                memory["excluded_obj_ids"] = sorted(ex)
                state.selected_obj_id = None
                state.awaiting_choice = True
                state.awaiting_confirmation = False
            else:
                if 0 <= idx < len(labels):
                    set_selected_by_label(labels[idx])
                state.awaiting_choice = False
                state.awaiting_confirmation = False
    elif t == "confirm":
        obj_id = ctx.get("obj_id")
        action = str(ctx.get("action") or "").upper()
        if user_content.upper() == "YES" and isinstance(obj_id, str):
            state.pending_action_obj_id = obj_id
            state.selected_obj_id = obj_id
            if action in {"APPROACH", "ALIGN_YAW"}:
                state.pending_mode = action
        else:
            state.pending_action_obj_id = None
            state.pending_mode = None
            state.selected_obj_id = None
            state.awaiting_anything_else = True
        state.awaiting_confirmation = False
    elif t == "help":
        obj_id = ctx.get("obj_id")
        if user_content.upper() == "YES" and isinstance(obj_id, str):
            state.pending_action_obj_id = obj_id
            state.selected_obj_id = obj_id
            state.pending_mode = "ALIGN_YAW"
        else:
            state.pending_action_obj_id = None
            state.pending_mode = None
            state.selected_obj_id = None
            state.awaiting_anything_else = True
        state.awaiting_help = False
    elif t == "anything_else":
        if user_content.upper() == "YES":
            state.awaiting_mode_select = True
            state.awaiting_anything_else = False
        else:
            reset_conversation_only()
            auto_continue = False
    elif t == "mode_select":
        uc = user_content.strip().upper()
        if uc in {"APPROACH", "ALIGN_YAW"}:
            state.pending_mode = uc
        elif user_content == "1":
            state.pending_mode = "APPROACH"
        elif user_content == "2":
            state.pending_mode = "ALIGN_YAW"
        state.awaiting_mode_select = False
        state.awaiting_choice = True
    elif t == "terminal_ack":
        reset_conversation_only()
        auto_continue = False

    state.last_prompt_context = None
    update_candidates_cb()
    return auto_continue


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="Show debug UI windows (e.g., logs).")
    ap.add_argument("--backend", choices=["oracle", "hf"], default="oracle")
    ap.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path/HF id of a merged standalone model directory (recommended).",
    )
    # Backward compatible aliases (deprecated).
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="DEPRECATED: use --model_path")
    ap.add_argument("--merged_model_path", type=str, default=None, help="DEPRECATED: use --model_path")
    ap.add_argument("--adapter_path", type=str, default=None, help="DEPRECATED: adapters are not supported; use a merged model.")
    ap.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--planner", type=str, default="curobo")
    # NOTE: `AppLauncher.add_app_launcher_args(ap)` already defines:
    #   --headless, --enable_cameras, --device
    # Do not add them here, otherwise argparse raises duplicate-field errors.
    ap.add_argument("--candidate_max_dist", type=int, default=2)
    ap.add_argument("--workspace_min_xyz", type=float, nargs=3, default=None)
    ap.add_argument("--workspace_max_xyz", type=float, nargs=3, default=None)
    ap.add_argument("--pregrasp_offset_m", type=float, default=0.08)
    ap.add_argument("--grasp_depth_m", type=float, default=0.03)
    ap.add_argument("--lift_height_m", type=float, default=0.08)
    ap.add_argument("--align_steps", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_objects", type=int, default=4)
    ap.add_argument("--spawn_min", type=float, nargs=3, default=[0.2, -0.3, 0.9])
    ap.add_argument("--spawn_max", type=float, nargs=3, default=[0.60, 0.45, 1.05])
    ap.add_argument("--min_distance", type=float, default=0.1)
    ap.add_argument("--scale_min", type=float, default=None)
    ap.add_argument("--scale_max", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--ee_link", type=str, default="j2n6s300_end_effector")
    ap.add_argument("--speed", type=float, default=0.7)
    ap.add_argument("--rot_speed", type=float, default=2.0)
    AppLauncher.add_app_launcher_args(ap)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Launch Kit
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac/Omni may preload a non-package module named `environments`, which breaks
    # `import environments.<...>` later. If that happens, evict it so our local package wins.
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]

    # Imports that require an active Kit app (i.e., provide `omni.*`) MUST be deferred
    # until after AppLauncher has started.
    import carb  # noqa: E402
    import isaaclab.sim as sim_utils  # noqa: E402
    from environments.reach_to_grasp_VLA.config import (  # noqa: E402
        DEFAULT_SCENE,
        DEFAULT_CAMERA,
        DEFAULT_TOP_DOWN_CAMERA,
    )
    from environments.reach_to_grasp_VLA.utils import design_scene  # noqa: E402
    from environments.utils.camera import create_topdown_camera  # noqa: E402
    from environments.utils.object_loader import (  # noqa: E402
        ObjectLoader,
        ObjectLoaderConfig,
        SpawnBounds,
    )
    from environments.utils.physix import (  # noqa: E402
        PhysicsConfig,
        apply_to_simulation_cfg,
        object_loader_kwargs_from_physix,
    )
    from motion_generation.planners import PlannerContext  # noqa: E402

    carb_settings = carb.settings.get_settings()
    carb_settings.set_bool("/isaaclab/cameras_enabled", bool(args.enable_cameras))

    # Scene setup
    phys = PhysicsConfig(device=str(args.device))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]
    if bool(args.enable_cameras):
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)

    # Spawn objects
    prim_paths: List[str] = []
    id_to_label: Dict[str, str] = {}
    if not getattr(args, "no_objects", False):
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"
        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=[ycb_dir],
            bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
            min_distance=float(args.min_distance),
            uniform_scale_range=(args.scale_min, args.scale_max) if args.scale_min and args.scale_max else None,
            **phys_loader_kwargs,
        )
        loader = ObjectLoader(loader_cfg)
        prim_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
        try:
            prim_to_label = loader.get_last_spawn_labels()
            id_to_label = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            id_to_label = {}

    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    # Controller and input
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(args.ee_link),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(args.speed),
        workspace_min=(0.20, -0.45, 0.01) if args.workspace_min_xyz is None else tuple(args.workspace_min_xyz),
        workspace_max=(0.6, 0.45, 0.35) if args.workspace_max_xyz is None else tuple(args.workspace_max_xyz),
        log_ee_pos=False,
        log_ee_frame="base",
        log_every_n_steps=50,
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.reset(robot)
    mode_manager = ModeManager(initial_mode="translate")
    controller.set_mode("translate")

    mux_input = CommandMuxInputProvider()
    teleop_provider = None
    if not args.headless:
        teleop_provider = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(args.rot_speed) * sim.get_physics_dt(),
        )
        mux_input.set_base(teleop_provider)
        translate_fn, rotate_fn, gripper_fn = mode_manager.get_mode_callbacks()
        teleop_provider.add_mode_callbacks(translate_fn, rotate_fn, gripper_fn)
    controller.set_input_provider(mux_input)

    # Tracker + extractor
    tracker = ObjectsTracker(prim_paths=prim_paths)
    table_z_base = DEFAULT_SCENE.table_translation[2] - DEFAULT_SCENE.robot_base_height
    extractor = InputExtractor(
        ExtractorConfig(
            workspace_min_xyz=tuple(ctrl_cfg.safety_cfg.workspace_min),
            workspace_max_xyz=tuple(ctrl_cfg.safety_cfg.workspace_max),
            table_z=float(table_z_base),
            candidate_max_dist=int(args.candidate_max_dist),
        )
    )

    # Backend
    if args.backend == "oracle":
        backend = OracleBackend()
    else:
        from llm.inference import InferenceConfig

        model_path = args.model_path or args.merged_model_path or args.model_name
        if args.adapter_path:
            raise SystemExit("adapter_path is deprecated and not supported. Please pass a merged model via --model_path.")

        hf_cfg = InferenceConfig(
            model_path=str(model_path),
            use_4bit=bool(args.use_4bit),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            seed=int(args.seed),
            deterministic=False,
        )
        backend = HFBackend(hf_cfg)

    # Planner / executor
    cfg_dir = str((ROOT / "motion_generation" / "planners" / "planners_config").resolve())
    urdf_path = str((Path(cfg_dir) / "cuRobo" / "kinovaJacoJ2N6S300.urdf").resolve())
    planner_ctx = PlannerContext(
        base_frame="base_link",
        ee_link_name=str(args.ee_link),
        urdf_path=urdf_path,
        config_dir=cfg_dir,
    )
    ee_body_id = int(robot.find_bodies([args.ee_link])[0][0])
    executor = ActionExecutor(
        cfg=ExecutorConfig(
            pregrasp_offset_m=float(args.pregrasp_offset_m),
            grasp_depth_m=float(args.grasp_depth_m),
            lift_height_m=float(args.lift_height_m),
            align_steps=int(args.align_steps),
        ),
        controller=controller,
        mux_input=mux_input,
        teleop_provider=teleop_provider,
        planner_kind=str(args.planner),
        planner_ctx=planner_ctx,
        ee_body_id=ee_body_id,
        device=str(sim.device),
        step_pos_m=float(ctrl_cfg.linear_speed_mps) * float(sim.get_physics_dt()),
    )

    # Prime extractor with the initial pose so gripper_hist is populated.
    try:
        init_objs = tracker.snapshot()
    except Exception:
        init_objs = []
    try:
        extractor.build_from_robot(robot, ee_body_id, init_objs, user_state=extractor.memory.get("user_state"))
    except Exception:
        pass

    # UI is optional (disabled in --headless). Import omni.ui wrapper lazily to avoid
    # importing omni.ui in non-Kit contexts.
    AssistUI = None
    if not bool(args.headless):
        from copilot_demo.ui_omni import AssistUI as _AssistUI  # noqa: E402

        AssistUI = _AssistUI

    ui = None

    def update_candidates():
        extractor._update_candidates(extractor.last_objects_b)

    def ask_assistance() -> None:
        objs = []
        try:
            objs = tracker.snapshot()
            for o in objs:
                if o.id in id_to_label:
                    o.label = id_to_label[o.id]
        except Exception:
            objs = []
        extractor.memory["user_state"] = {"mode": _mode_for_oracle(mode_manager.current_mode.value)}
        input_blob = extractor.build_from_robot(robot, ee_body_id, objs, user_state=extractor.memory.get("user_state"))
        try:
            tool_call = backend.predict(input_blob)
            validate_tool_call(tool_call)
        except Exception as e:
            if ui:
                ui.set_status(f"Model error: {e}")
            return

        # Track last tool calls
        memory = extractor.memory
        memory.setdefault("last_tool_calls", [])
        memory["last_tool_calls"].append(tool_call["tool"])
        memory["last_tool_calls"] = memory["last_tool_calls"][-3:]

        if ui:
            ui.set_status(f"{tool_call['tool']}")
        if tool_call["tool"] == "INTERACT":
            text = tool_call["args"]["text"]
            choices: Sequence[str] = tool_call["args"]["choices"]
            if ui:
                ui.set_status(text)
                ui.set_choices(choices)
            memory["past_dialogs"].append({"role": "assistant", "content": text})
            memory["n_interactions"] = int(memory.get("n_interactions", 0)) + 1
            memory["last_prompt"] = {"kind": tool_call["args"]["kind"], "text": text, "choices": list(choices)}
            return

        if tool_call["tool"] in {"APPROACH", "ALIGN_YAW"}:
            executor.execute(
                tool_call,
                objects_b=extractor.last_objects_b,
                robot=robot,
                gripper_yaw_bin=extractor.gripper_hist[-1]["yaw"] if extractor.gripper_hist else None,
            )
            memory["last_action"] = {"tool": tool_call["tool"], "obj": tool_call["args"]["obj"]}
            memory["n_interactions"] = int(memory.get("n_interactions", 0)) + 1
        update_candidates()

    def on_choice(choice_str: str) -> None:
        memory = extractor.memory
        user_content = _choice_to_user_content(choice_str)
        memory["past_dialogs"].append({"role": "user", "content": user_content})
        auto_continue = True
        if isinstance(backend, OracleBackend):
            if backend.state is None:
                backend._ensure_state(memory.get("last_action", {}).get("obj", "obj0"))
            auto_continue = _apply_oracle_user_reply(
                user_content,
                memory=memory,
                state=backend.state,
                objects=extractor.last_objects_b,
                update_candidates_cb=update_candidates,
            )
        else:
            semantic = _strip_choice_label(choice_str).strip().lower()
            if semantic == "none of them":
                last_prompt = memory.get("last_prompt") or {}
                _apply_none_exclusion(memory, last_prompt, extractor.last_objects_b)
        if ui:
            ui.set_choices([])
        update_candidates()
        if auto_continue:
            ask_assistance()

    def on_mode_change(mode: str) -> None:
        # Accept either UI strings ("translate"/"rotate"/"gripper") or legacy labels.
        m = str(mode).lower().strip()
        if m == "translation":
            m = "translate"
        if m == "rotation":
            m = "rotate"
        if m not in {"translate", "rotate", "gripper"}:
            return
        mode_manager.switch_to(m)  # controller + UI will be updated via the mode_manager callback below
        extractor.memory["user_state"] = {"mode": _mode_for_oracle(m)}

    def on_reset() -> None:
        extractor.reset()
        if ui:
            ui.set_choices([])
            ui.set_status("Reset.")

    if AssistUI is not None:
        ui = AssistUI(
            on_ask=ask_assistance,
            on_reset=on_reset,
            on_choice=on_choice,
            on_mode_change=on_mode_change,
            show_logs=bool(getattr(args, "debug", False)),
            initial_mode=str(mode_manager.current_mode.value),
            enabled=True,
        )
        try:
            ui.set_bounds_hint(ctrl_cfg.safety_cfg.workspace_min, ctrl_cfg.safety_cfg.workspace_max)
        except Exception:
            pass

    # Keep controller + UI in sync when mode changes via keyboard callbacks (I/O/P) or UI buttons.
    def _on_mode_changed(new_mode) -> None:
        try:
            controller.set_mode(new_mode.value)
        except Exception:
            pass
        if ui:
            try:
                ui.set_mode(new_mode.value)
            except Exception:
                pass
        extractor.memory["user_state"] = {"mode": _mode_for_oracle(str(new_mode.value))}

    mode_manager.set_mode_change_callback(_on_mode_changed)
    # Apply initial mode state to UI/controller.
    _on_mode_changed(mode_manager.current_mode)

    dt = sim.get_physics_dt()
    while simulation_app.is_running():
        executor.tick(robot)
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()

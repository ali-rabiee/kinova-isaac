# `environments/`

Scene-environment packages for the Kinova Jaco2 demos.

## Layout

- `base.py` — shared `BaseSceneEnv`, `SceneConfig`, `CameraConfig`, `TopDownCameraConfig`, and the `design_scene` helper used by every concrete environment.
- `ycb_reach_to_grasp/` — reach-to-grasp scene with YCB objects loaded from Isaac Nucleus. Provides `YCBReachToGraspEnv` plus module-level `DEFAULT_SCENE` / `DEFAULT_CAMERA` / `DEFAULT_TOP_DOWN_CAMERA`. Top-down camera is opt-in via `env.attach_top_down_camera()`.
- `cubes/` — block-stacking scene with uniform colored cubes. Provides `CubesEnv`, the `BOX_COLORS` palette, and per-prim helpers (`label_for_prim`, `read_prim_world_yaw_rad`, `read_prim_height_m`).
- `utils/object_loader.py` — generic object loader (USD or `box` mode).
- `utils/physix.py` — physics configuration (sim dt, substeps, friction, etc.).
- `utils/camera/` — top-down camera prim creation.

## Using an environment

```python
from environments.ycb_reach_to_grasp import YCBReachToGraspEnv

env = YCBReachToGraspEnv(device="cuda:0")
sim = env.build_simulation()
env.set_default_camera_view()        # GUI only
env.design_scene()                   # ground / light / table / robot
env.attach_top_down_camera()         # optional
loader = env.build_object_loader(
    spawn_min=(0.30, -0.30, 0.85),
    spawn_max=(0.55,  0.30, 0.92),
    min_distance=0.10,
)
prim_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=4)
env.reset()
```

```python
from environments.cubes import CubesEnv

env = CubesEnv(device="cuda:0", box_size=0.08)
sim = env.build_simulation()
env.design_scene()
loader = env.build_object_loader(spawn_min=(0.30, -0.30, 0.90),
                                 spawn_max=(0.55,  0.30, 0.95),
                                 min_distance=0.22)
spawned = loader.spawn(parent_prim_path="/World/Origin1", num_objects=3)
env.reset()
for p in spawned:
    label, color, idx = env.label_for_prim(p)
```

## Smoke-test demos

```bash
./IsaacLab/isaaclab.sh -p kinova-isaac/environments/ycb_reach_to_grasp/demo.py --device cuda
./IsaacLab/isaaclab.sh -p kinova-isaac/environments/cubes/demo.py --device cuda --num-objects 3
```

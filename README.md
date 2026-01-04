# kinova-isaac

Kinova Jaco2 (J2N6S300) **simulation demos, controllers, and motion-generation** built on top of `IsaacLab/` (and optionally cuRobo).

This folder is intended to be run using **Isaac Lab’s launcher** so you get the correct Isaac Sim Python/runtime:

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --device cuda
```

## What’s inside

- `demo.py`: Main interactive demo (cartesian velocity jog) + optional object spawning + logging.
- `controllers/`: Modular cartesian velocity jog controller + keyboard input + safety/modes.
- `motion_generation/`: Grasp-loop demo + planner adapters (`scripted`, `rmpflow`, `curobo`, `lula`) + grasp pose providers (`obb`, `replicator`).
- `data_collection/`: Episode runner that logs trajectories/objects for repeated grasp attempts.
- `environments/`: Scene construction (`reach_to_grasp`) and object loading utilities.
- `assist/`: Logging utilities and “assist” orchestration components (currently used mainly for logging/tracking).

## Copilot demo (IsaacSim)

- Interactive (oracle backend): `python -m copilot_demo.demo_isaacsim --backend oracle --planner curobo --enable_cameras`
- HF backend example: `python -m copilot_demo.demo_isaacsim --backend hf --model_name Qwen/Qwen2.5-7B-Instruct --adapter_path grasp-copilot/models/adapter --planner curobo`
- Headless tests: `pytest -q` (set `ISAACSIM_AVAILABLE=1` to run the opt-in IsaacSim check)

## Prerequisites

- **Isaac Sim + Isaac Lab**: `kinova-isaac` imports `isaaclab.*` and expects to run under an Isaac Sim runtime.
- **GPU**: Most workflows are meant for `--device cuda` (but `--device cpu` can work for basic checks).
- **Nucleus assets (optional)**: Some demos spawn YCB objects from Isaac Nucleus. If you don’t have access, use `--no-objects` where supported.
- **Python Packages**: `curobo` (planning), `numpy` & `Pillow` (cameras/logging).

## Run demos

### 1) Main demo: cartesian jog + optional object spawning (`kinova-isaac/demo.py`)

GUI (keyboard teleop):

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --device cuda
```

Headless (no keyboard control; useful for smoke tests):

```bash
./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --headless --device cuda
```

Useful flags (see `kinova-isaac/scripts/cli.py`):

- `--no-objects`: disable object spawning
- `--num-objects N`: number of objects
- `--spawn-min x y z` / `--spawn-max x y z`: spawn bounds
- `--speed`: jog linear speed
- `--print-ee`: print end-effector position

### 2) Controller-only demo (`kinova-isaac/controllers/demo.py`)

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/controllers/demo.py --device cuda
```

### 3) Environment smoke demo (`kinova-isaac/environments/reach_to_grasp/demo.py`)

This runs a simple loop that resets and perturbs joint targets.

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/environments/reach_to_grasp/demo.py --device cuda
```

### 4) Motion generation grasp loop (`kinova-isaac/motion_generation/demo.py`)

Scripted planner (default) + OBB grasp provider:

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/motion_generation/demo.py --device cuda --planner scripted --grasp obb --num-episodes 3
```

Headless:

```bash
./IsaacLab/isaaclab.sh -p kinova-isaac/motion_generation/demo.py --headless --device cuda --planner scripted --num-episodes 3
```

Key options (see `kinova-isaac/motion_generation/cli.py`):

- `--planner`: `scripted | rmpflow | curobo | lula`
- `--grasp`: `obb | replicator`
- `--num-episodes`, `--num-objects`
- `--spawn-min/--spawn-max`, `--min-distance`, `--scale-min/--scale-max`
- `--pregrasp`, `--lift`, `--tolerance`

### 5) Data collection (`kinova-isaac/data_collection/demo.py`)

Runs multiple episodes and logs results under `--logs-root` (default `logs/assist`):

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/data_collection/demo.py --headless --device cuda --planner scripted --num-episodes 10 --logs-root kinova-isaac/logs/assist
```

### 6) Data Collection Test (Uniform Boxes + cuRobo)

Test command for VLA data collection with uniform boxes:

```bash
python -m data_collection.collect_data \
  --profile vla_v1 \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner curobo_v2 \
  --device cuda:0 \
  --enable_cameras \
  --log-rate-hz 5 \
  --num-episodes 10 \
  --spawn-mode box \
  --planner-speed-mps 0.4 \
  --planner-waypoint-max-seg-m 0.01 \
  --target-selection first \
  --max-steps-per-episode 8000 \
  --grasp-depth -0.07 \
  --close-if-within-m 0.005 \
  --min-distance 0.20 \
  --domain-rand \
  --domain-rand-seed 0
```

**Key Arguments:**
- `--spawn-mode box`: Spawn uniform cubes instead of random objects.
- `--box-size`: Side length of the cubes (m).
- `--curobo-world-from-scene`: Sync simulation obstacles to planner world.
- `--enable_cameras`: Record camera images.
- `--log-rate-hz`: Tick/image logging rate. In `vla_v1` this is also the intended **policy rate**.
- `--domain-rand`: Enable per-episode domain randomization (camera + lighting). Off by default.
- `--workspace-min-z`: Minimum allowed EE height (negative allows reaching surface).

### 7) Data Collection (Random Objects)

To run with random YCB objects instead of boxes, use `--spawn-mode usd` (or omit the arg as it is default):

```bash
python -m data_collection.collect_data \
  --profile vla_v1 \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner curobo_v2 \
  --device cuda:0 \
  --enable_cameras \
  --log-rate-hz 5 \
  --render-rate-hz 60 \
  --curobo-world-from-scene \
  --num-episodes 10 \
  --workspace-min-z -0.02 \
  --spawn-mode usd
```

### Customization Guide

Common arguments for tuning the data collection process:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--num-episodes` | `10` | Total number of grasp attempts to run. |
| `--headless` | (flag) | Run without GUI (faster, good for batch collection). |
| `--num-objects` | `5` | Number of objects to spawn per episode. |
| `--planner` | `curobo_v2` | Planner backend (`curobo_v2`, `scripted`, `rmpflow`). |
| `--log-rate-hz` | `5` | Tick/image logging rate (and intended policy rate for VLA training). |
| `--grasp-depth` | `-0.05` | Grasp offset relative to object top (m). Increase magnitude to grasp deeper. |
| `--lift` | `0.15` | Height to lift object after grasping (m). |
| `--workspace-min-z` | `0.0` | Safety floor for EE. Set negative (e.g. `-0.02`) if robot can't reach table surface. |
| `--spawn-mode` | `usd` | `usd` for random YCB objects, `box` for uniform cubes. |
| `--box-size` | `0.05` | Size of cubes when using `--spawn-mode box`. |
| `--domain-rand` | off | Enable per-episode domain randomization (camera + lighting). |
| `--domain-rand-seed` | (none) | Optional RNG seed for reproducible randomization. |
| `--domain-rand-camera-xy-m` | `0.02` | Uniform XY jitter (m) for the top-down camera. |
| `--domain-rand-camera-z-m` | `0.10` | Uniform Z jitter (m) for the top-down camera height. |
| `--domain-rand-camera-yaw-deg` | `20.0` | Uniform yaw jitter (deg) for the top-down camera. |
| `--domain-rand-camera-pitch-deg` | `0.0` | Uniform pitch jitter (deg) for the top-down camera (tilt). |
| `--domain-rand-camera-roll-deg` | `0.0` | Uniform roll jitter (deg) for the top-down camera (tilt). |
| `--domain-rand-camera-fov-deg` | `5.0` | Uniform FOV jitter (deg) for the top-down camera. |
| `--domain-rand-light-intensity-mult-min/max` | `0.5/1.5` | Dome light intensity multiplier range. |
| `--domain-rand-light-color-jitter` | `0.15` | Dome light RGB jitter per channel (0..1). |

## Logs

- `kinova-isaac/logs/assist/`: JSONL logs and metadata emitted by the demos/data-collection scripts.

## Notes / gotchas

- **Keyboard teleop requires GUI**: run without `--headless` to use keyboard input.
- **Object spawning depends on asset availability**: if spawning fails, rerun with `--no-objects` to isolate issues.

## VLA data collection (recommended command)

Uniform boxes + cuRobo planner + cameras + domain randomization, with **box locations reshuffled every episode**:

```bash
python -m data_collection.collect_data \
  --profile vla_v1 \
  --env reach_to_grasp_VLA \
  --control planner \
  --planner curobo_v2 \
  --device cuda:0 \
  --enable_cameras \
  --log-rate-hz 5 \
  --num-episodes 10 \
  --spawn-mode box \
  --planner-speed-mps 0.4 \
  --planner-waypoint-max-seg-m 0.01 \
  --target-selection first \
  --max-steps-per-episode 8000 \
  --grasp-depth -0.07 \
  --close-if-within-m 0.005 \
  --min-distance 0.20 \
  --domain-rand \
  --domain-rand-seed 0 \
  --respawn-every-n-episodes 1
```


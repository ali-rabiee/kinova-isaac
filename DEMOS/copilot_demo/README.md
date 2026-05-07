## Copilot Demo (Isaac Sim / IsaacLab)

Small Isaac Sim demo that wires **grasp-copilot** style “tool calls” into a Kinova Jaco2 scene:

- **Extractor**: reads robot + object state and builds a discrete `grasp-copilot` input blob.
- **Backend**: produces one tool call (`INTERACT`, `APPROACH`, `ALIGN_YAW`) using either:
  - **oracle** (rule-based, from `grasp-copilot/data_generator/oracle.py`)
  - **hf** (HuggingFace model via `grasp-copilot/llm/inference.py`)
- **Executor**: turns `APPROACH` / `ALIGN_YAW` into motion via the planner + waypoint follower.
- **UI**: an `omni.ui` panel to request assistance + answer `INTERACT` prompts.

### Prerequisites

- You should run inside the **Isaac Sim runtime** (IsaacLab environment).
- This repo expects the **`grasp-copilot/`** folder to exist next to the repo root (used for `data_generator` + optional `llm`).

### Run (recommended)

From repo root (uses IsaacLab’s python):

```bash
cd /home/ali/github/ali-rabiee
./IsaacLab/isaaclab.sh -p -m copilot_demo.demo_isaacsim \
  --backend oracle \
  --planner curobo \
  --enable_cameras \
  --device cuda:0
```

### Run (direct python)

If you prefer running directly (make sure you activate your env first):

```bash
cd /home/ali/github/ali-rabiee/kinova-isaac
conda activate kinova
python copilot_demo/copilot_demo/demo_isaacsim.py --backend oracle --planner curobo --enable_cameras --device cuda:0
```

### Debug UI (separate log window)

Add `--debug` to show the **“Grasp Copilot Logs”** window:

```bash
python copilot_demo/copilot_demo/demo_isaacsim.py --debug --backend oracle --planner curobo --enable_cameras --device cuda:0
```

### Controls

- **Ask assistance**: runs the backend once and either:
  - executes motion (`APPROACH` / `ALIGN_YAW`), or
  - shows an `INTERACT` prompt with choice buttons.
- **Mode switching**
  - **UI buttons**: Translate / Rotate / Gripper (highlights current mode).
  - **Keyboard**:
    - `I` / `i` → translate
    - `O` / `o` → rotate
    - `P` / `p` → gripper

### Backends

- **Oracle** (fast, no model required):

```bash
python copilot_demo/copilot_demo/demo_isaacsim.py --backend oracle --planner curobo
```

- **HF backend** (model-driven):

```bash
python copilot_demo/copilot_demo/demo_isaacsim.py \
  --backend hf \
  --model_path /home/ali/github/ali-rabiee/grasp-copilot/models/checkpoint-3000 \
  --planner curobo
```

### Tests (no Isaac Sim required for most)

```bash
cd /home/ali/github/ali-rabiee/kinova-isaac
pytest -q copilot_demo/tests
```

Opt-in IsaacSim-dependent check:

```bash
ISAACSIM_AVAILABLE=1 pytest -q copilot_demo/tests/test_headless_opt_in.py
```

### Troubleshooting

- **`attempted relative import with no known parent package`**
  - Run with `python -m copilot_demo.demo_isaacsim` or use the IsaacLab launcher command above.
- **Objects float/sink / weird collisions**
  - Collision proxies are **off by default** now (`PhysicsConfig.use_collision_proxies = False`).
  - If you enable proxies, expect simplified contact behavior.



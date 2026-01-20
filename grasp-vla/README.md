### xVLA (Stage-2) rollout in Isaac Sim (real-time)

This folder contains an IsaacLab rollout script that runs the **xVLA Stage-2** policy closed-loop in Isaac Sim:

- **Observations**: top-down RGB + robot state (`--state-mode joints8` by default)
- **Actions**: 7D controller command applied every physics step

### Run the updated Stage-2 model (GUI, with policy tracing)

Activate the conda env you use for IsaacLab, then run:

```bash
conda activate riften

/home/kye/IsaacLab/isaaclab.sh -p /home/kye/Desktop/Depo/Code/kinova-isaac/grasp-vla/rollout_xvla_isaac.py \
  --device cuda:0 --enable_cameras \
  --lerobot-src /home/kye/Desktop/Depo/Code/Grasp-VLA/lerobot/src \
  --model-dir /home/kye/Desktop/Depo/Code/kinova-isaac/grasp-vla/models/xvla/stage2 \
  --seed 0 \
  --policy-hz 5 \
  --state-mode joints8 \
  --init-joints 0.0072 2.2704 4.5114 0.2286 5.0707 1.4764 \
  --action-scale 5 \
  --action-ema 0.2 \
  --num-objects 4 --box-size 0.05 \
  --spawn-min 0.25 -0.40 0.825 \
  --spawn-max 0.60 0.40 0.825 \
  --min-distance 0.25 \
  --target-index 1 \
  --instruction "Pick up the red box." \
  --trace-policy --trace-policy-every 1 \
  --max-seconds 120
```

Notes:
- Box IDs are spawned as `Obj_01..Obj_04` with colors matching the dataset convention: **red, blue, yellow, purple**.
- For an in-distribution evaluation (exact object poses from a recorded demo), add:
  - `--layout-raw-episode /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/raw_data/session_YYYYMMDD_HHMMSS/episode_0000`
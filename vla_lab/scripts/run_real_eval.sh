#!/usr/bin/env bash
# Placeholder wrapper for real-robot eval (fill in ROS launch + policy server).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${CFG:-${REPO_ROOT}/vla_lab/configs/eval_real.yaml}"

echo "[run_real_eval] Template only — implement your ROS bringup, then invoke your policy loop."
echo "[run_real_eval] Config: ${CFG}"
echo "Suggested flow:"
echo "  1) Start camera + robot drivers"
echo "  2) Start LeRobot policy server (optional) or use vla_lab.smolvla_bridge.policy_wrapper in-process"
echo "  3) Call vla_lab.real_robot.kinova_bridge.KinovaPolicyServer.step() under a logging harness"
exit 0

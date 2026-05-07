from __future__ import annotations

import os
import pytest


if os.environ.get("ISAACSIM_AVAILABLE") != "1":
    pytest.skip("ISAACSIM_AVAILABLE!=1; skipping IsaacSim-dependent test", allow_module_level=True)


def test_oracle_backend_runs_with_dummy_blob():
    from data_generator.oracle import validate_tool_call  # type: ignore
    from copilot_demo.backends import OracleBackend

    backend = OracleBackend()
    input_blob = {
        "objects": [{"id": "o1", "label": "mug", "cell": "B2", "yaw": "N", "is_held": False}],
        "gripper_hist": [{"cell": "B1", "yaw": "N", "z": "HIGH"}] * 6,
        "memory": {
            "n_interactions": 0,
            "past_dialogs": [],
            "candidates": ["o1"],
            "last_tool_calls": [],
            "excluded_obj_ids": [],
            "last_action": {},
        },
        "user_state": {"mode": "translation"},
    }
    tool = backend.predict(input_blob)
    validate_tool_call(tool)


"""Assistant backends (oracle + HF) that emit validated tool calls."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from data_generator.oracle import OracleState, oracle_decide_tool, validate_tool_call  # type: ignore
from llm.inference import InferenceConfig  # type: ignore

INSTRUCTION = (
    "Given the robot observation and dialog context, infer the user's intent and "
    "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
    "If the tool is INTERACT, you must output at most 5 choices total."
)


def _strip_choice_label(choice: str) -> str:
    parts = choice.split(")", 1)
    return parts[1].strip() if len(parts) == 2 else choice.strip()


def _sanitize_tool_call(out: Dict[str, Any]) -> Dict[str, Any]:
    """Strip extra keys before validation, mirroring gui_assist_demo semantics."""
    if not isinstance(out, dict):
        return out
    tool = out.get("tool")
    args = out.get("args")
    if tool == "INTERACT" and isinstance(args, dict):
        return {"tool": "INTERACT", "args": {k: args.get(k) for k in ("kind", "text", "choices") if k in args}}
    if tool in {"APPROACH", "ALIGN_YAW"} and isinstance(args, dict):
        return {"tool": tool, "args": {"obj": args.get("obj")}}
    return out


class OracleBackend:
    """Wrapper around oracle_decide_tool that maintains OracleState."""

    def __init__(self) -> None:
        self.state: Optional[OracleState] = None

    def reset(self) -> None:
        self.state = None

    def _ensure_state(self, intended_obj_id: str) -> OracleState:
        if self.state is None:
            self.state = OracleState(intended_obj_id=intended_obj_id)
        return self.state

    def predict(self, input_blob: Dict[str, Any]) -> Dict[str, Any]:
        objects = input_blob.get("objects") or []
        gripper_hist = input_blob.get("gripper_hist") or []
        memory = input_blob.get("memory") or {}
        user_state = input_blob.get("user_state")
        intended = None
        if objects:
            # Choose the closest candidate to current gripper cell as the "intent".
            try:
                cur_cell = gripper_hist[-1]["cell"]
            except Exception:
                cur_cell = None
            best = None
            best_dist = 1e9
            for o in objects:
                try:
                    if cur_cell is None:
                        best = o["id"]
                        break
                    d = abs(int(o.get("_tmp_rank", 0)))  # optional injected rank
                    # If no rank, fall back to manhattan distance if available.
                except Exception:
                    d = None
                if d is None:
                    try:
                        from data_generator import grid as gridlib  # type: ignore

                        d = gridlib.manhattan(cur_cell, o["cell"])
                    except Exception:
                        d = 0
                if d < best_dist:
                    best_dist = d
                    best = o.get("id", "obj0")
            intended = best
        state = self._ensure_state(intended or "obj0")
        tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state=user_state)
        validate_tool_call(tool)
        return tool


class HFBackend:
    """
    HuggingFace backend that keeps the model loaded across GUI interactions.
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self._loaded = False
        self._model = None
        self._tok = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from llm.inference import _load_model_and_tokenizer  # type: ignore

        self._model, self._tok = _load_model_and_tokenizer(self.cfg)
        self._loaded = True

    def predict(self, input_blob: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._model is not None and self._tok is not None
        prompt = f"{INSTRUCTION}\n\nInput:\n{json.dumps(input_blob, ensure_ascii=False)}"
        from llm.inference import _build_messages, _generate_once, json_loads_strict  # type: ignore

        messages = _build_messages(prompt)
        raw1 = _generate_once(self._model, self._tok, messages, self.cfg)
        try:
            out = json_loads_strict(raw1)
        except Exception:
            repair_messages = _build_messages("Return ONLY valid JSON for the previous answer.\n\nPrevious answer:\n" + raw1)
            raw2 = _generate_once(self._model, self._tok, repair_messages, self.cfg)
            out = json_loads_strict(raw2)

        out = _sanitize_tool_call(out)
        validate_tool_call(out)
        return out


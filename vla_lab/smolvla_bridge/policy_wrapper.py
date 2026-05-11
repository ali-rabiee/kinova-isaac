"""Load SmolVLA (LeRobot) and produce (T, 7) action chunks for `PolicyInputProvider`.

Gripper channel is filled with 0 (this bridge exports 6D actions without gripper).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .action_obs_contract import CAMERA_KEYS, STATE_KEY


class SmolVLAIsaacPolicy:
    """Wraps LeRobot SmolVLA + preprocessor; unnormalizes actions with dataset stats."""

    def __init__(
        self,
        *,
        policy_path: str | Path,
        dataset_root: str | Path,
        device: torch.device,
    ) -> None:
        try:
            from lerobot.datasets.utils import load_stats
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SmolVLA inference requires `lerobot`. Install:\n"
                "  pip install -r vla_lab/requirements-smolvla.txt"
            ) from exc

        policy_path = str(policy_path)
        ds_root = Path(dataset_root)
        if not ds_root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset root not found: {ds_root}")

        stats_path = ds_root / "meta" / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Expected dataset stats at {stats_path}. "
                "Pass the same --out-dir you used for convert_kinova_to_lerobot."
            )
        dataset_stats = load_stats(ds_root)
        if not dataset_stats or "action" not in dataset_stats:
            raise ValueError(f"Invalid or empty stats in {stats_path}")

        act_stats = dataset_stats["action"]
        self._action_mean = torch.as_tensor(act_stats["mean"], dtype=torch.float32, device=device)
        self._action_std = torch.as_tensor(act_stats["std"], dtype=torch.float32, device=device)

        self._policy = SmolVLAPolicy.from_pretrained(policy_path)
        self._policy.to(device)
        self._policy.eval()

        pre, _ = make_pre_post_processors(
            self._policy.config,
            pretrained_path=policy_path,
            dataset_stats=dataset_stats,
        )
        self._preprocessor = pre
        self.device = device

    def reset(self) -> None:
        try:
            self._policy.reset()
        except Exception:
            pass

    @torch.no_grad()
    def predict_chunk_phys(
        self,
        *,
        rgb_chw_float: torch.Tensor,
        state6: torch.Tensor,
        instruction: str,
    ) -> torch.Tensor:
        """Return (T, 7) float32 on self.device: 6D EE delta + gripper (0)."""

        # rgb_chw_float: (3,H,W) in [0,1]
        img = rgb_chw_float.unsqueeze(0).to(self.device, dtype=torch.float32)
        st = state6.flatten()[:6].float().unsqueeze(0).to(self.device)

        obs: dict[str, Any] = {STATE_KEY: st}
        for k in CAMERA_KEYS:
            obs[k] = img.clone()
        obs["task"] = [instruction]

        batch = self._preprocessor(obs)
        actions = self._policy.predict_action_chunk(batch)
        actions = actions[..., : self._action_mean.shape[0]]
        phys = actions * self._action_std + self._action_mean
        phys = phys.squeeze(0)
        g = torch.zeros((phys.shape[0], 1), device=self.device, dtype=torch.float32)
        return torch.cat([phys[:, :6], g], dim=-1)

    @torch.no_grad()
    def predict_chunk_phys_k(
        self,
        *,
        rgb_chw_float: torch.Tensor,
        state6: torch.Tensor,
        instruction: str,
        k: int,
        base_seed: int = 1337,
    ) -> torch.Tensor:
        """Draw K candidates (K, T, 7) using different RNG seeds (flow sampling noise)."""

        k = max(1, int(k))
        out: List[torch.Tensor] = []
        for i in range(k):
            try:
                torch.manual_seed(int(base_seed) + int(i))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(base_seed) + int(i))
            except Exception:
                pass
            try:
                self.reset()
            except Exception:
                pass
            out.append(
                self.predict_chunk_phys(rgb_chw_float=rgb_chw_float, state6=state6, instruction=instruction)
            )
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def predict_chunk_phys_ttc(
        self,
        *,
        rgb_chw_float: torch.Tensor,
        state6: torch.Tensor,
        instruction: str,
        ttc_cfg_dict: Dict[str, Any],
        latency_log: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """TTC + gating (twin / noisy-pair) matching `vla_lab.ttc.TTCPipeline` semantics."""

        from vla_lab.ttc import TTCConfig, _normalize_gating_label, select_from_candidates
        from vla_lab.ttc_methods.entropy_gate import twin_sample_uncertainty, uncertainty_gate_effective_k

        cfg = TTCConfig.from_dict(ttc_cfg_dict)
        k_max = max(1, int(cfg.k_action_samples))
        t0 = time.time()
        forward_ms = 0.0
        score_ms = 0.0
        uncertainty: Optional[float] = None
        gated = False
        gate_expanded = False
        k_eff = k_max
        gmode = _normalize_gating_label(str(cfg.gating))

        def _log_record(*, k_eff_i: int, score_ms_i: float) -> None:
            if latency_log is None:
                return
            total_ms = (time.time() - t0) * 1000.0
            latency_log.append(
                {
                    "forward_ms": float(forward_ms),
                    "score_ms": float(score_ms_i),
                    "total_ms": float(total_ms),
                    "k_eff": int(k_eff_i),
                    "k_max": int(k_max),
                    "gated": bool(gated),
                    "gate_expanded": bool(gate_expanded),
                    "uncertainty": uncertainty,
                }
            )

        if k_max > 1 and gmode == "twin_uncertainty":
            t_fw0 = time.time()
            try:
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
            except Exception:
                pass
            self.reset()
            a0 = self.predict_chunk_phys(rgb_chw_float=rgb_chw_float, state6=state6, instruction=instruction)
            # Second probe: low noise (LeRobot uses internal sampling; approximate via fresh seed).
            try:
                torch.manual_seed(1)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(1)
            except Exception:
                pass
            self.reset()
            a1 = self.predict_chunk_phys(rgb_chw_float=rgb_chw_float, state6=state6, instruction=instruction)
            forward_ms += (time.time() - t_fw0) * 1000.0
            uncertainty = float(twin_sample_uncertainty(a0, a1).item())
            k_eff = uncertainty_gate_effective_k(
                uncertainty,
                threshold=float(cfg.gating_threshold),
                k_when_uncertain=k_max,
                k_when_certain=1,
            )
            gated = True
            gate_expanded = k_eff > 1
            if k_eff == 1:
                _log_record(k_eff_i=1, score_ms_i=max(0.0, (time.time() - t0) * 1000.0 - forward_ms))
                return a0

        elif k_max > 1 and gmode == "noisy_pair":
            t_fw0 = time.time()
            try:
                torch.manual_seed(100)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(100)
            except Exception:
                pass
            self.reset()
            a0 = self.predict_chunk_phys(rgb_chw_float=rgb_chw_float, state6=state6, instruction=instruction)
            try:
                torch.manual_seed(101)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(101)
            except Exception:
                pass
            self.reset()
            a1 = self.predict_chunk_phys(rgb_chw_float=rgb_chw_float, state6=state6, instruction=instruction)
            forward_ms += (time.time() - t_fw0) * 1000.0
            uncertainty = float(twin_sample_uncertainty(a0, a1).item())
            k_eff = uncertainty_gate_effective_k(
                uncertainty,
                threshold=float(cfg.gating_threshold),
                k_when_uncertain=k_max,
                k_when_certain=1,
            )
            gated = True
            gate_expanded = k_eff > 1
            if k_eff == 1:
                _log_record(k_eff_i=1, score_ms_i=max(0.0, (time.time() - t0) * 1000.0 - forward_ms))
                return a0

        t_fw1 = time.time()
        if k_eff > 1 and cfg.include_deterministic_candidate:
            try:
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
            except Exception:
                pass
            self.reset()
            a_det = self.predict_chunk_phys(rgb_chw_float=rgb_chw_float, state6=state6, instruction=instruction)
            rest = self.predict_chunk_phys_k(
                rgb_chw_float=rgb_chw_float,
                state6=state6,
                instruction=instruction,
                k=int(k_eff - 1),
                base_seed=2048,
            )
            candidates = torch.cat([a_det.unsqueeze(0), rest], dim=0)
        else:
            candidates = self.predict_chunk_phys_k(
                rgb_chw_float=rgb_chw_float,
                state6=state6,
                instruction=instruction,
                k=int(k_eff),
                base_seed=2048,
            )
        forward_ms += (time.time() - t_fw1) * 1000.0

        t_sc0 = time.time()
        best, _scores = select_from_candidates(candidates, cfg.selection)
        score_ms += (time.time() - t_sc0) * 1000.0
        gate_expanded = gate_expanded or (gated and int(candidates.shape[0]) > 1)
        _log_record(k_eff_i=int(candidates.shape[0]), score_ms_i=score_ms)
        return best

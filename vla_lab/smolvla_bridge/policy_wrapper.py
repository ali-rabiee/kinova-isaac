"""Load SmolVLA (LeRobot) and produce (T, 7) action chunks for `PolicyInputProvider`.

Gripper channel is filled with 0 (this bridge exports 6D actions without gripper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

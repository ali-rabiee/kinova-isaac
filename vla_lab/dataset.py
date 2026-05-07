"""PyTorch dataset over `data_collection.collect_data --profile vla_v1` outputs.

Expected on-disk layout (produced by `vla_v1` profile in planner mode):

    <logs_root>/session_YYYYMMDD_HHMMSS/
        episode_0000/
            metadata.json
            instruction.json     # {"language_command": "...", "target_label": ...}
            ticks.jsonl          # one JSON record per logged tick
            events.jsonl
            images/image_000000.png, image_000001.png, ...
        episode_0001/
            ...
        episode_0002/
            ...

The collect_data logger stringifies floats with 4 decimal places via
`_format_numbers(...)` in `data_collection/core/logger.py`, so this loader
recursively converts numeric strings back to floats while parsing.

Each dataset sample corresponds to one *training-tick* (an observation + the
chunk of next-T actions to take from it). The action is read from
`policy.action_from_prev` of the *next* T ticks: action[t] is the delta
applied between tick (t-1) and tick (t), so given observation o_t we predict
[a_{t+1}, ..., a_{t+T}].
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # PIL is already used by `vla_v1.py` to write images, so reading is fine.
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vla_lab.dataset requires Pillow (PIL). Install with `pip install Pillow`."
    ) from exc


_NUM_PAT = re.compile(r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")


def _to_float_maybe(value: Any) -> Any:
    """Recursively convert stringified numbers in a JSON record back to floats.

    The session logger writes floats as fixed-decimal strings (see
    `_format_numbers` in data_collection/core/logger.py). This helper undoes that
    so downstream code sees real numbers.
    """

    if isinstance(value, str) and _NUM_PAT.match(value):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_to_float_maybe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_float_maybe(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Episode discovery + parsing
# ---------------------------------------------------------------------------


@dataclass
class TickRecord:
    """Lightweight in-memory representation of one tick row used for training."""

    tick_idx: int
    image_rel: Optional[str]
    ee_pos_b: Tuple[float, float, float]
    ee_quat_b: Tuple[float, float, float, float]
    gripper_state: str  # "open" / "close"
    # Action-from-prev is None for tick 0 of each episode.
    action_from_prev: Optional[Tuple[float, float, float, float, float, float, int]]


@dataclass
class EpisodeRecord:
    folder: Path
    instruction: str
    target_label: str
    ticks: List[TickRecord]


def _parse_instruction(folder: Path) -> Tuple[str, str]:
    """Read instruction.json. Returns (language_command, target_label)."""

    p = folder / "instruction.json"
    if not p.exists():
        return "", ""
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return "", ""
    return str(obj.get("language_command", "")), str(obj.get("target_label", ""))


def _parse_ticks(ticks_path: Path) -> List[TickRecord]:
    out: List[TickRecord] = []
    with ticks_path.open("r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = _to_float_maybe(raw)

            try:
                tick_idx = int(rec.get("tick_idx", line_idx))
            except Exception:
                tick_idx = line_idx

            image_rel = None
            try:
                image_rel = rec.get("image", {}).get("path")
            except Exception:
                image_rel = None

            ee_pos = tuple(rec["robot"]["ee_pose_b"]["position_m"])  # type: ignore[index]
            ee_quat = tuple(rec["robot"]["ee_pose_b"]["orientation_wxyz"])  # type: ignore[index]
            grip_state = str(rec["robot"]["gripper"]["state"])  # type: ignore[index]

            action: Optional[Tuple[float, float, float, float, float, float, int]] = None
            try:
                policy = rec.get("policy", {})
                afp = policy.get("action_from_prev")
                if isinstance(afp, dict):
                    dpos = list(afp["ee_delta_pos_b"])
                    drot = list(afp["ee_delta_rotvec_b"])
                    g = int(afp.get("gripper_action", 0))
                    action = (
                        float(dpos[0]),
                        float(dpos[1]),
                        float(dpos[2]),
                        float(drot[0]),
                        float(drot[1]),
                        float(drot[2]),
                        int(g),
                    )
            except Exception:
                action = None

            out.append(
                TickRecord(
                    tick_idx=tick_idx,
                    image_rel=image_rel,
                    ee_pos_b=tuple(float(v) for v in ee_pos),  # type: ignore[arg-type]
                    ee_quat_b=tuple(float(v) for v in ee_quat),  # type: ignore[arg-type]
                    gripper_state=grip_state,
                    action_from_prev=action,
                )
            )
    return out


def discover_episodes(roots: Sequence[Path]) -> List[EpisodeRecord]:
    """Find all `episode_XXXX/` folders under any of the given session roots."""

    episodes: List[EpisodeRecord] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue

        # Layouts:
        # - `session_ts/episode_####/` — session may hold an empty top-level `ticks.jsonl` (still scan `episode_*`).
        # - `logs/data_collection/` with multiple `session_*`
        # - single episode folder with real `ticks.jsonl` only
        ep_dirs: List[Path] = []
        direct_eps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
        if direct_eps:
            ep_dirs = direct_eps
        elif any(p.is_dir() and p.name.startswith("session_") for p in root.iterdir()):
            for sd in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_")):
                ep_dirs.extend(
                    sorted(p for p in sd.iterdir() if p.is_dir() and p.name.startswith("episode_"))
                )
        elif (root / "ticks.jsonl").exists():
            ep_dirs = [root]

        for ep_dir in ep_dirs:
            ticks_path = ep_dir / "ticks.jsonl"
            if not ticks_path.exists():
                continue
            ticks = _parse_ticks(ticks_path)
            if not ticks:
                continue
            instr, target_label = _parse_instruction(ep_dir)
            episodes.append(
                EpisodeRecord(
                    folder=ep_dir,
                    instruction=instr,
                    target_label=target_label,
                    ticks=ticks,
                )
            )

    return episodes


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@dataclass
class TinyTokenizer:
    """Minimal whitespace tokenizer with a learned vocabulary.

    Saved alongside the model checkpoint so eval can reconstruct the same
    token IDs.
    """

    word_to_id: Dict[str, int] = field(default_factory=dict)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    max_len: int = 24

    @classmethod
    def build_from_corpus(
        cls,
        sentences: Sequence[str],
        max_len: int = 24,
        min_count: int = 1,
        max_vocab: int = 1024,
    ) -> "TinyTokenizer":
        from collections import Counter

        counts: Counter[str] = Counter()
        for s in sentences:
            for tok in cls._normalize(s):
                counts[tok] += 1

        word_to_id: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        for tok, c in counts.most_common(max_vocab - len(word_to_id)):
            if c < min_count:
                continue
            if tok not in word_to_id:
                word_to_id[tok] = len(word_to_id)
        return cls(word_to_id=word_to_id, max_len=max_len)

    @staticmethod
    def _normalize(text: str) -> List[str]:
        text = text.strip().lower()
        # Replace any non-alphanumeric character with spaces, then split.
        out: List[str] = []
        cur: List[str] = []
        for ch in text:
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (token_ids[max_len], attention_mask[max_len])."""

        toks = self._normalize(text)
        ids = [self.bos_id]
        for tok in toks:
            ids.append(self.word_to_id.get(tok, self.unk_id))
            if len(ids) >= self.max_len - 1:
                break
        ids.append(self.eos_id)

        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad = self.max_len - len(ids)
            ids = ids + [self.pad_id] * pad
            attn = attn + [0] * pad

        return (
            torch.tensor(ids[: self.max_len], dtype=torch.long),
            torch.tensor(attn[: self.max_len], dtype=torch.long),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_to_id": self.word_to_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "max_len": self.max_len,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TinyTokenizer":
        return cls(
            word_to_id={str(k): int(v) for k, v in d.get("word_to_id", {}).items()},
            pad_id=int(d.get("pad_id", 0)),
            unk_id=int(d.get("unk_id", 1)),
            bos_id=int(d.get("bos_id", 2)),
            eos_id=int(d.get("eos_id", 3)),
            max_len=int(d.get("max_len", 24)),
        )


# ---------------------------------------------------------------------------
# Action / state utilities
# ---------------------------------------------------------------------------


def _zero_action() -> Tuple[float, float, float, float, float, float, int]:
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


@dataclass
class ActionStats:
    """Per-dimension mean/std for the 6 continuous action dims (delta pose).

    Gripper (index 6) is left untransformed (-1/0/+1 -> -1/0/+1).
    """

    mean: torch.Tensor  # shape (6,)
    std: torch.Tensor  # shape (6,)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": [float(v) for v in self.mean.tolist()],
            "std": [float(v) for v in self.std.tolist()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "ActionStats":
        return cls(
            mean=torch.tensor(d["mean"], dtype=torch.float32),
            std=torch.tensor(d["std"], dtype=torch.float32),
        )

    def normalize(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Normalize the 6 continuous dims of an (T, 7) action chunk in place."""
        out = action_chunk.clone()
        out[..., :6] = (out[..., :6] - self.mean) / (self.std + 1e-6)
        return out

    def denormalize(self, action_chunk: torch.Tensor) -> torch.Tensor:
        out = action_chunk.clone()
        out[..., :6] = out[..., :6] * (self.std + 1e-6) + self.mean
        return out


def compute_action_stats(episodes: Sequence[EpisodeRecord]) -> ActionStats:
    vals: List[List[float]] = []
    for ep in episodes:
        for t in ep.ticks:
            if t.action_from_prev is None:
                continue
            vals.append([float(x) for x in t.action_from_prev[:6]])
    if not vals:
        return ActionStats(
            mean=torch.zeros(6, dtype=torch.float32),
            std=torch.ones(6, dtype=torch.float32),
        )
    arr = torch.tensor(vals, dtype=torch.float32)
    mean = arr.mean(dim=0)
    std = arr.std(dim=0).clamp_min(1e-3)
    return ActionStats(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    image_size: int = 224
    chunk_len: int = 8
    state_dim: int = 4  # [ee_x, ee_y, ee_z, gripper_open=1.0/0.0]
    action_dim: int = 7  # [dx, dy, dz, drx, dry, drz, gripper]
    drop_no_image: bool = True
    # If True (default), use torchvision.io.read_image when available (often faster than PIL).
    fast_image_io: bool = True


class KinovaSessionDataset(Dataset):
    """Dataset of (image, state, language_tokens, action_chunk, mask) samples.

    This dataset is *frame-indexed*: each sample corresponds to one observed
    tick `t` of an episode, with the target being the chunk of next
    `chunk_len` actions [a_{t+1}, ..., a_{t+chunk_len}].

    Frames whose t+1 is past the end of the episode are still kept; the
    `mask` field marks which timesteps of the chunk are real.
    """

    def __init__(
        self,
        episodes: List[EpisodeRecord],
        tokenizer: TinyTokenizer,
        cfg: DatasetConfig,
        action_stats: Optional[ActionStats] = None,
        train: bool = True,
    ) -> None:
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.action_stats = action_stats
        self.train = train
        self._fast_io = bool(cfg.fast_image_io)
        self._tv_read_image = None
        self._tv_interpolate = None
        if self._fast_io:
            try:
                from torchvision.io import read_image as _read_image  # type: ignore

                from torch.nn.functional import interpolate as _interpolate

                self._tv_read_image = _read_image
                self._tv_interpolate = _interpolate
            except Exception:
                self._fast_io = False

        self._index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            n = len(ep.ticks)
            if n <= 1:
                continue
            for t in range(n - 1):
                tick = ep.ticks[t]
                if cfg.drop_no_image and not tick.image_rel:
                    continue
                self._index.append((ep_idx, t))

        if not self._index:
            raise RuntimeError(
                "KinovaSessionDataset: no usable training frames found. Did you collect "
                "data with `--enable_cameras` so images are written to disk?"
            )

    def __len__(self) -> int:
        return len(self._index)

    def _load_image(self, ep: EpisodeRecord, rel: Optional[str]) -> torch.Tensor:
        H = W = self.cfg.image_size
        if rel is None:
            return torch.zeros(3, H, W, dtype=torch.float32)
        path = ep.folder / rel
        if not path.exists():
            return torch.zeros(3, H, W, dtype=torch.float32)
        if self._fast_io and self._tv_read_image is not None and self._tv_interpolate is not None:
            try:
                t8 = self._tv_read_image(str(path))  # CHW uint8
                if t8.dim() != 3 or t8.shape[0] == 1:
                    raise ValueError("unexpected image shape")
                if t8.shape[0] == 4:
                    t8 = t8[:3]
                t = t8.float() / 255.0
                if t.shape[1] != H or t.shape[2] != W:
                    t = self._tv_interpolate(
                        t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze(0)
                return t.contiguous()
            except Exception:
                pass
        try:
            img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8, copy=True)
        except Exception:
            return torch.zeros(3, H, W, dtype=torch.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return t

    def _state(self, tick: TickRecord) -> torch.Tensor:
        # 4D state: [ee_x, ee_y, ee_z, gripper_open]
        gx = 1.0 if str(tick.gripper_state).lower().startswith("o") else 0.0
        st = torch.tensor(
            [
                float(tick.ee_pos_b[0]),
                float(tick.ee_pos_b[1]),
                float(tick.ee_pos_b[2]),
                gx,
            ],
            dtype=torch.float32,
        )
        # We could append orientation; keep state_dim=4 for the tiny baseline.
        if self.cfg.state_dim != 4:
            pad = torch.zeros(max(0, self.cfg.state_dim - 4), dtype=torch.float32)
            st = torch.cat([st, pad])[: self.cfg.state_dim]
        return st

    def _action_chunk(
        self, ep: EpisodeRecord, t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.cfg.chunk_len
        out = torch.zeros(T, self.cfg.action_dim, dtype=torch.float32)
        mask = torch.zeros(T, dtype=torch.float32)
        n = len(ep.ticks)
        for i in range(T):
            j = t + 1 + i
            if j >= n:
                break
            a = ep.ticks[j].action_from_prev or _zero_action()
            for k in range(min(self.cfg.action_dim, 7)):
                out[i, k] = float(a[k])
            mask[i] = 1.0
        return out, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self._index[idx]
        ep = self.episodes[ep_idx]
        tick = ep.ticks[t]

        image = self._load_image(ep, tick.image_rel)
        state = self._state(tick)
        actions, mask = self._action_chunk(ep, t)
        ids, attn = self.tokenizer.encode(ep.instruction)

        if self.action_stats is not None:
            actions = self.action_stats.normalize(actions)

        return {
            "image": image,
            "state": state,
            "lang_ids": ids,
            "lang_mask": attn,
            "actions": actions,
            "action_mask": mask,
        }


def split_episodes(
    episodes: List[EpisodeRecord],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[EpisodeRecord], List[EpisodeRecord]]:
    """Episode-level train/val split (avoids leaking ticks of the same episode)."""

    if not episodes:
        return [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    if val_fraction <= 0 or len(episodes) <= 1:
        n_val = 0
    else:
        n_val = int(math.floor(len(episodes) * val_fraction))
        # Always keep at least one episode for training.
        n_val = max(0, min(n_val, len(episodes) - 1))
    val = [episodes[i] for i in idx[:n_val]]
    train = [episodes[i] for i in idx[n_val:]]
    return train, val

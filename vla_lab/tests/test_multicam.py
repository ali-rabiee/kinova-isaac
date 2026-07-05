"""Offline tests for the multi-camera model/dataset path (2026-07 wrist camera).

These need torch (+PIL for the dataset round-trip) and are skipped cleanly when
unavailable, keeping run_tests.sh green on torch-free machines.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _cfg(**kw):
    from vla_lab.models import TinyVLAConfig

    base = dict(
        image_size=64, embed_dim=32, vis_channels=(8, 16, 16, 32), lang_heads=2,
        decoder_heads=2, vocab_size=32, max_lang_len=8,
    )
    base.update(kw)
    return TinyVLAConfig(**base)


def _dummy(cfg, b=2):
    img = torch.rand(b, 3, cfg.image_size, cfg.image_size)
    st = torch.rand(b, cfg.state_dim)
    ids = torch.randint(0, cfg.vocab_size, (b, cfg.max_lang_len))
    mask = torch.ones(b, cfg.max_lang_len, dtype=torch.long)
    return img, st, ids, mask


def test_single_camera_state_dict_unchanged() -> None:
    """Old checkpoints must load strict=True into new code."""
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.models import TinyVLA, TinyVLAConfig

    cfg = _cfg()
    m = TinyVLA(cfg)
    assert cfg.cameras == ("overhead",)
    assert "camera_embed" not in m.state_dict()
    # Config dict without the new keys (as stored by old checkpoints) round-trips.
    old = {k: v for k, v in cfg.to_dict().items() if k not in ("cameras", "camera_dropout")}
    m2 = TinyVLA(TinyVLAConfig.from_dict(old))
    m2.load_state_dict(m.state_dict(), strict=True)


def test_two_camera_forward_shapes_and_param_delta() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.models import TinyVLA

    cfg1, cfg2 = _cfg(), _cfg(cameras=("overhead", "wrist"), camera_dropout=0.25)
    m1, m2 = TinyVLA(cfg1), TinyVLA(cfg2)
    img, st, ids, mask = _dummy(cfg2)
    out = m2(img, st, ids, mask, image_wrist=torch.rand_like(img))
    n_tok = (cfg2.image_size // 16) ** 2
    assert out.actions.shape == (2, cfg2.chunk_len, cfg2.action_dim)
    assert out.bottleneck.shape[1] == 2 * n_tok + cfg2.max_lang_len + 1
    assert out.features.shape[1] == n_tok, "features must stay first-stream-only (DINOv2 contract)"
    assert m2.num_parameters() == m1.num_parameters() + 2 * cfg2.embed_dim


def test_camera_present_masks_stream_pixels() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.models import TinyVLA

    cfg = _cfg(cameras=("overhead", "wrist"))
    m = TinyVLA(cfg)
    m.eval()
    img, st, ids, mask = _dummy(cfg)
    wr = torch.rand_like(img)
    keep_oh = torch.tensor([[1.0, 0.0]] * 2)
    with torch.no_grad():
        a_full = m(img, st, ids, mask, image_wrist=wr, camera_present=torch.ones(2, 2)).actions
        a_oh = m(img, st, ids, mask, image_wrist=wr, camera_present=keep_oh).actions
        a_oh2 = m(img, st, ids, mask, image_wrist=torch.zeros_like(wr), camera_present=keep_oh).actions
    assert not torch.allclose(a_full, a_oh), "masking a stream must change the output"
    assert torch.allclose(a_oh, a_oh2, atol=1e-5), "a masked stream must ignore its pixels"


def test_wrist_only_model_and_missing_stream_zeros() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.models import TinyVLA

    cfg = _cfg(cameras=("wrist",))
    m = TinyVLA(cfg)
    img, st, ids, mask = _dummy(cfg)
    out = m(image=None, state=st, lang_ids=ids, lang_mask=mask, image_wrist=img)
    assert out.actions.shape == (2, cfg.chunk_len, cfg.action_dim)


def _write_fake_session(root: Path, *, with_wrist: bool, n_eps: int = 2, n_ticks: int = 6) -> None:
    from PIL import Image
    import numpy as np

    for epi in range(n_eps):
        ep = root / f"episode_{epi:04d}"
        (ep / "images").mkdir(parents=True, exist_ok=True)
        (ep / "instruction.json").write_text(json.dumps({"language_command": "pick up the red box.", "target_label": "red box 1"}))
        (ep / "episode_summary.json").write_text(json.dumps({"success": True, "truncated": False}))
        (ep / "metadata.json").write_text(json.dumps({"config": {"log_rate_hz": 15}}))
        with (ep / "ticks.jsonl").open("w") as f:
            for t in range(n_ticks):
                Image.fromarray(np.full((16, 16, 3), 40 * t % 255, np.uint8)).save(ep / "images" / f"image_{t:06d}.png")
                rec = {
                    "tick_idx": t,
                    "robot": {
                        "ee_pose_b": {"position_m": ["0.4", "0.1", "0.2"], "orientation_wxyz": ["1", "0", "0", "0"]},
                        "gripper": {"state": "open"},
                    },
                    "policy": {"action_from_prev": None if t == 0 else {
                        "dt_s": "0.0667", "ee_delta_pos_b": ["0.01", "0.0", "0.0"],
                        "ee_delta_rotvec_b": ["0", "0", "0"], "gripper_action": 0}},
                    "image": {"path": f"images/image_{t:06d}.png"},
                }
                if with_wrist:
                    Image.fromarray(np.full((16, 16, 3), 30 * t % 255, np.uint8)).save(ep / "images" / f"wrist_{t:06d}.png")
                    rec["image_wrist"] = {"path": f"images/wrist_{t:06d}.png"}
                f.write(json.dumps(rec) + "\n")


def test_dataset_camera_sets_and_missing_wrist_error() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.dataset import DatasetConfig, KinovaSessionDataset, TinyTokenizer, discover_episodes

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "session_x"
        _write_fake_session(root, with_wrist=True)
        eps = discover_episodes([root], success_only=True)
        tok = TinyTokenizer.build_from_corpus([e.instruction for e in eps])
        for cams, keys in [
            (("overhead",), {"image"}),
            (("wrist",), {"image_wrist"}),
            (("overhead", "wrist"), {"image", "image_wrist"}),
        ]:
            ds = KinovaSessionDataset(eps, tok, DatasetConfig(image_size=32, cameras=cams))
            s = ds[0]
            assert keys.issubset(s.keys())
            assert not (({"image", "image_wrist"} - keys) & set(s.keys()))
            for k in keys:
                assert tuple(s[k].shape) == (3, 32, 32)
            assert tuple(s["camera_present"].shape) == (len(cams),)
            assert float(s["camera_present"].min()) == 1.0

        root2 = Path(td) / "session_nowrist"
        _write_fake_session(root2, with_wrist=False)
        eps2 = discover_episodes([root2], success_only=True)
        try:
            KinovaSessionDataset(eps2, tok, DatasetConfig(image_size=32, cameras=("wrist",)))
            raise AssertionError("expected RuntimeError for wrist-on-legacy-session")
        except RuntimeError as e:
            assert "collect_v4" in str(e)


def test_ttc_pipeline_passes_wrist_through() -> None:
    if not _HAS_TORCH:
        print("SKIP (torch unavailable)")
        return
    from vla_lab.models import TinyVLA
    from vla_lab.ttc import TTCConfig, TTCPipeline

    cfg = _cfg(cameras=("overhead", "wrist"))
    m = TinyVLA(cfg)
    m.eval()
    pipe = TTCPipeline(m, TTCConfig(k_action_samples=2, noise_std=0.05, gating="twin_uncertainty"))
    img = torch.rand(3, 64, 64)
    wr = torch.rand(3, 64, 64)
    st = torch.rand(cfg.state_dim)
    ids = torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,))
    mask = torch.ones(cfg.max_lang_len, dtype=torch.long)
    chunk = pipe.predict_action_chunk(img, st, ids, mask, image_wrist=wr, camera_present=torch.ones(2))
    assert tuple(chunk.shape) == (cfg.chunk_len, cfg.action_dim)
    pipe.close()


if __name__ == "__main__":
    from vla_lab.tests import run_namespace

    raise SystemExit(1 if run_namespace(dict(globals()), label="test_multicam") else 0)

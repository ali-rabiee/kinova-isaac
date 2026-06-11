"""Minimal PyTorch GPU smoketest: random data + 10-neuron MLP.

Verifies that PyTorch can allocate tensors, run forward/backward passes, and
train on the selected device (CUDA when available).

Usage:
    python -m vla_lab.unity_smoketest
    python -m vla_lab.unity_smoketest --device cuda:0 --epochs 50
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyMLP(nn.Module):
    """Single hidden layer with exactly 10 neurons."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_random_dataset(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
    w = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    b = rng.standard_normal((output_dim,)).astype(np.float32)
    y = x @ w + b + 0.05 * rng.standard_normal((n_samples, output_dim)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch GPU smoketest (10-neuron MLP)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g. cuda:0, cpu)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    _seed_everything(args.seed)
    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device}) but torch.cuda.is_available() is False")
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = None

    x, y = _make_random_dataset(args.n_samples, args.input_dim, args.output_dim, args.seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = TinyMLP(args.input_dim, args.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"device: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"hidden neurons: 10 | samples: {args.n_samples} | epochs: {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch {epoch:3d}/{args.epochs}  loss={epoch_loss / n_batches:.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    model.eval()
    with torch.no_grad():
        sample_x = x[:4].to(device)
        sample_pred = model(sample_x).cpu()
        print("sample predictions (first 4 rows):")
        for i in range(sample_x.shape[0]):
            print(f"  pred={sample_pred[i].tolist()}  target={y[i].tolist()}")

    print("smoketest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""profile_training.py — diagnose training speed on MPS/CUDA/CPU.

Runs a single epoch of InceptionTime1D on a synthetic DeepBeat-sized dataset
(100 000 × 800) and reports time breakdown for:
  - Data loading (shuffle + slice + device transfer)
  - Forward pass
  - Backward pass
  - Optimizer step
  - loss.item() sync (the MPS→CPU sync overhead)

Also sweeps batch_size = [64, 128, 256, 512] so you can see the impact.

Usage
-----
    .venv/bin/python scripts/profile_training.py
    .venv/bin/python scripts/profile_training.py --n-samples 20000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nstad_bench.models.inception_time import InceptionTime1D


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    """Wait for all pending device ops to complete."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Profile one epoch
# ─────────────────────────────────────────────────────────────────────────────

def profile_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    measure_item: bool = True,
    n_batches_profile: int = 50,
) -> dict[str, float]:
    """Time one epoch (or first n_batches_profile batches) component-by-component."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    t_load = t_fwd = t_bwd = t_opt = t_item = 0.0
    n = 0

    loader_iter = iter(loader)
    _sync(device)

    for _ in range(min(n_batches_profile, len(loader))):
        # ── Data loading + device transfer ────────────────────────────────────
        t0 = time.perf_counter()
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            break
        xb = xb.to(device)
        yb = yb.to(device)
        _sync(device)
        t_load += time.perf_counter() - t0

        # ── Forward ───────────────────────────────────────────────────────────
        _sync(device)
        t0 = time.perf_counter()
        logits = model(xb)
        loss = criterion(logits, yb)
        _sync(device)
        t_fwd += time.perf_counter() - t0

        # ── Backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        _sync(device)
        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        t_bwd += time.perf_counter() - t0

        # ── Optimizer step ────────────────────────────────────────────────────
        _sync(device)
        t0 = time.perf_counter()
        optimizer.step()
        _sync(device)
        t_opt += time.perf_counter() - t0

        # ── loss.item() — MPS→CPU sync ────────────────────────────────────────
        if measure_item:
            _sync(device)
            t0 = time.perf_counter()
            _ = float(loss.item())
            _sync(device)
            t_item += time.perf_counter() - t0

        n += 1

    return {
        "n_batches":  n,
        "t_load_ms":  1000 * t_load  / n,
        "t_fwd_ms":   1000 * t_fwd   / n,
        "t_bwd_ms":   1000 * t_bwd   / n,
        "t_opt_ms":   1000 * t_opt   / n,
        "t_item_ms":  1000 * t_item  / n,
        "t_total_ms": 1000 * (t_load + t_fwd + t_bwd + t_opt + t_item) / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep batch sizes
# ─────────────────────────────────────────────────────────────────────────────

def sweep(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_sizes: list[int],
    n_batches_profile: int,
) -> None:
    dataset = TensorDataset(X, y)

    print(f"\n{'batch_size':>12}  {'load':>8}  {'fwd':>8}  {'bwd':>8}  "
          f"{'opt':>8}  {'item()':>8}  {'total/batch':>12}  "
          f"{'est epoch min':>14}")
    print("-" * 90)

    n_total = len(X)
    for bs in batch_sizes:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        model = InceptionTime1D(in_channels=1, nb_filters=32, depth=4).to(device)

        # warm-up (2 batches, not measured)
        it = iter(loader)
        for _ in range(2):
            xb, yb = next(it)
            _ = model(xb.to(device))

        r = profile_epoch(model, loader, device,
                          n_batches_profile=n_batches_profile)

        batches_per_epoch = n_total / bs
        est_epoch_s = r["t_total_ms"] / 1000 * batches_per_epoch
        print(
            f"{bs:>12}  "
            f"{r['t_load_ms']:>7.1f}ms  "
            f"{r['t_fwd_ms']:>7.1f}ms  "
            f"{r['t_bwd_ms']:>7.1f}ms  "
            f"{r['t_opt_ms']:>7.1f}ms  "
            f"{r['t_item_ms']:>7.1f}ms  "
            f"{r['t_total_ms']:>11.1f}ms  "
            f"{est_epoch_s/60:>13.1f} min"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Source dataset size (default 100 000 = DeepBeat default)")
    parser.add_argument("--seq-len",   type=int, default=800,
                        help="Sequence length (default 800 = DeepBeat)")
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Batches to profile per batch_size (default 50)")
    args = parser.parse_args()

    device = _get_device()
    print(f"Device          : {device}")
    print(f"n_samples       : {args.n_samples:,}")
    print(f"seq_len         : {args.seq_len}")
    print(f"Batches profiled: {args.n_batches} per batch_size")

    # Synthetic data — same dtype and shape as DeepBeat after RawSignal
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((args.n_samples, args.seq_len)).astype(np.float32)
    y_np = rng.integers(0, 2, size=args.n_samples, dtype=np.int64)
    # _preprocess: (N, T) → (N, 1, T)
    X_t = torch.from_numpy(X_np[:, None, :])
    y_t = torch.from_numpy(y_np)

    sweep(X_t, y_t, device,
          batch_sizes=[64, 128, 256, 512, 1024],
          n_batches_profile=args.n_batches)

    print()
    print("Notes:")
    print("  load   = shuffle + slice TensorDataset + .to(device)")
    print("  item() = loss.item() — forces MPS→CPU sync every batch")
    print("  est epoch min = extrapolated to full dataset")
    print()
    print("Fix candidates:")
    print("  1. Increase batch_size (see table above)")
    print("  2. Remove loss.item() from inner loop (accumulate, sync once per epoch)")
    print("  3. Both combined")


if __name__ == "__main__":
    main()

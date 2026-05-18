"""extract_deepbeat_subset.py — save a small DeepBeat NPZ for Kaggle upload.

The full train.npz is ~18 GB (2.8M windows × 800 samples × float64).
This script extracts max_per_class windows from train.npz and all of test.npz,
then saves them as a compact float32 NPZ suitable for Kaggle upload (~130 MB).

Usage
-----
    .venv/bin/python scripts/extract_deepbeat_subset.py
    .venv/bin/python scripts/extract_deepbeat_subset.py --max-per-class 10000
    .venv/bin/python scripts/extract_deepbeat_subset.py --out ~/Desktop/deepbeat_10k.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = Path.home() / ".nstad_bench" / "data" / "deepbeat"
DEFAULT_OUT      = DEFAULT_DATA_DIR / "deepbeat_10k.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=10_000)
    parser.add_argument("--data-dir",      type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out",           type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed",          type=int,  default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load source (train.npz) ───────────────────────────────────────────────
    train_path = args.data_dir / "train.npz"
    print(f"Loading {train_path} …  (may take a few minutes for large files)")
    train = np.load(train_path)

    signal = train["signal"]           # (N, 800, 1)  float64
    rhythm = train["rhythm"]           # (N, 2)       float64  [SR, AF] one-hot

    # col0=SR, col1=AF
    label = rhythm.argmax(axis=1).astype(np.int64)   # 0=SR, 1=AF

    # Drop NaN/Inf
    finite_mask = np.isfinite(signal[:, :, 0]).all(axis=1)
    signal, label = signal[finite_mask], label[finite_mask]

    # Balance and cap
    X_s_list, y_s_list = [], []
    for cls in [0, 1]:
        idx = np.where(label == cls)[0]
        if len(idx) > args.max_per_class:
            idx = rng.choice(idx, args.max_per_class, replace=False)
        X_s_list.append(signal[idx, :, 0])
        y_s_list.append(label[idx])

    X_s = np.concatenate(X_s_list, axis=0).astype(np.float32)   # (2*mpc, 800)
    y_s = np.concatenate(y_s_list, axis=0)
    print(f"  source: {len(X_s)} windows  "
          f"SR={int((y_s==0).sum())}  AF={int((y_s==1).sum())}")

    # ── Load target (test.npz) ────────────────────────────────────────────────
    test_path = args.data_dir / "test.npz"
    print(f"Loading {test_path} …")
    test = np.load(test_path)

    sig_t = test["signal"]
    lab_t = test["rhythm"].argmax(axis=1).astype(np.int64)
    fin_t = np.isfinite(sig_t[:, :, 0]).all(axis=1)
    X_t = sig_t[fin_t, :, 0].astype(np.float32)
    y_t = lab_t[fin_t]
    print(f"  target: {len(X_t)} windows  "
          f"SR={int((y_t==0).sum())}  AF={int((y_t==1).sum())}")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X_source=X_s, y_source=y_s,
        X_target=X_t, y_target=y_t,
    )
    size_mb = args.out.stat().st_size / 1024**2
    print(f"\nSaved → {args.out}  ({size_mb:.0f} MB)")
    print("Upload this file to Kaggle as a dataset.")


if __name__ == "__main__":
    main()

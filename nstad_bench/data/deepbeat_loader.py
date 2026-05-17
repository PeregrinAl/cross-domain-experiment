"""DeepBeat PPG dataset loader — inter-patient domain split.

Dataset layout (Synapse syn21985690)
-------------------------------------
Three NumPy archives, each storing the following arrays:

    signal     (N, 800, 1)  float64  PPG waveform, normalised to [0, 1]
    rhythm     (N, 2)       float64  one-hot: col0=SR (normal), col1=AF
    qa_label   (N, 3)       float32  signal quality (3 classes, not used here)
    parameters (N, 3)       object   [timestamp, session_id, patient_id]

Patient split (non-overlapping between train and test)
------------------------------------------------------
    train.npz      patients   1–137  (137 patients, N ≈ 2 803 934)
    validate.npz   patients 138–153  (16 patients,  N ≈   518 782)
    test.npz       patients 146–167  (22 patients,  N ≈    17 617)
                   └ 8 patients overlap with validate; 14 unique to test

Domain adaptation split used here
----------------------------------
    Source → train.npz    (patients   1–137, ~balanced SR/AF: 54.6 / 45.4 %)
    Target → test.npz     (patients 146–167, natural imbalance: 76 % SR / 24 % AF)

The 22 test patients have no overlap with the 137 training patients, giving a
clean inter-patient domain shift.  The different class ratios (balanced train vs
imbalanced test) are an intentional part of the shift — they reflect real-world
prevalence variation between patient populations.

Class balancing in the loader
------------------------------
The source is capped to ``max_per_class`` windows per class (default 50 000)
to keep training manageable without losing label diversity.  The target is
**not** rebalanced so that evaluation metrics reflect the natural distribution.

Signal
------
800-sample PPG windows at 125 Hz (6.4 s).  Only channel 0 of the (N, 800, 1)
array is used → shape (N, 800), single channel.

Usage
-----
::

    from nstad_bench.data.deepbeat_loader import deepbeat_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("deepbeat_patient_split", deepbeat_loader())
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

#: Default data root — respects ``NSTAD_DATA_ROOT`` env var.
def _default_root() -> Path:
    env = os.environ.get("NSTAD_DATA_ROOT")
    if env:
        return Path(env) / "deepbeat"
    return Path.home() / ".nstad_bench" / "data" / "deepbeat"


def deepbeat_loader(
    data_root: str | Path | None = None,
    *,
    max_per_class: int = 50_000,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a loader for the DeepBeat inter-patient domain split.

    Parameters
    ----------
    data_root:
        Directory containing ``train.npz`` and ``test.npz``.
        Default: ``~/.nstad_bench/data/deepbeat/``
        (or ``$NSTAD_DATA_ROOT/deepbeat/`` if the env var is set).
    max_per_class:
        Maximum number of windows per class in the **source** domain
        (``train.npz``).  The source contains ~1.5M SR and ~1.3M AF windows;
        capping to 50 000 each reduces peak RAM from ~35 GB to ~600 MB while
        keeping full class diversity.  The **target** (``test.npz``) is loaded
        without capping (~17 k windows total).
    seed:
        Random seed for sub-sampling the source domain.

    Returns
    -------
    Callable
        Zero-argument function returning
        ``(X_source, y_source, X_target, y_target)``.

        - ``X_*``  — float32, shape ``(N, 800)``
        - ``y_*``  — int64,   shape ``(N,)``,  0 = SR (normal), 1 = AF
    """
    root = Path(data_root) if data_root else _default_root()

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_path = root / "train.npz"
        test_path  = root / "test.npz"
        for p in (train_path, test_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"DeepBeat file not found: {p}\n"
                    "Download with:  SYNAPSE_TOKEN=<token> nstad-download deepbeat"
                )

        rng = np.random.default_rng(seed)

        # ── Source: train.npz (patients 1–137) ───────────────────────────
        log.info("Loading DeepBeat source (train.npz) …")
        tr = np.load(train_path, allow_pickle=True)
        X_s_all = tr["signal"][:, :, 0].astype(np.float32)   # (N, 800)
        y_s_all = tr["rhythm"][:, 1].astype(np.int64)          # 1=AF, 0=SR

        # Drop windows that contain NaN or Inf (0.15% of train.npz are fully
        # NaN rows from corrupted recording sessions).
        finite_mask = np.isfinite(X_s_all).all(axis=1)
        n_dropped = int((~finite_mask).sum())
        if n_dropped:
            log.warning(
                "  dropped %d source windows containing NaN/Inf (%.2f%%)",
                n_dropped, 100 * n_dropped / len(X_s_all),
            )
            X_s_all = X_s_all[finite_mask]
            y_s_all = y_s_all[finite_mask]

        # Cap each class independently
        idx_s = _balanced_subsample(y_s_all, max_per_class, rng)
        X_s = X_s_all[idx_s]
        y_s = y_s_all[idx_s]
        log.info(
            "  source: %d windows  SR=%d  AF=%d",
            len(X_s), int((y_s == 0).sum()), int((y_s == 1).sum()),
        )

        # ── Target: test.npz (patients 146–167) ──────────────────────────
        log.info("Loading DeepBeat target (test.npz) …")
        te = np.load(test_path, allow_pickle=True)
        X_t = te["signal"][:, :, 0].astype(np.float32)
        y_t = te["rhythm"][:, 1].astype(np.int64)
        # Shuffle only (no cap — target should reflect natural distribution)
        perm = rng.permutation(len(X_t))
        X_t, y_t = X_t[perm], y_t[perm]
        log.info(
            "  target: %d windows  SR=%d (%.0f%%)  AF=%d (%.0f%%)",
            len(X_t),
            int((y_t == 0).sum()), 100 * (y_t == 0).mean(),
            int((y_t == 1).sum()), 100 * (y_t == 1).mean(),
        )

        return X_s, y_s, X_t, y_t

    return _load


def _balanced_subsample(
    y: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return shuffled indices with at most *max_per_class* samples per label."""
    idx_parts = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        n   = min(len(idx), max_per_class)
        idx_parts.append(rng.choice(idx, n, replace=False))
    combined = np.concatenate(idx_parts)
    rng.shuffle(combined)
    return combined

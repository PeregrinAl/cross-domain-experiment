"""CWRU Bearing Dataset loader — cross-load 0 HP → 3 HP domain split.

Domain split (per task spec)
----------------------------
    Source → 0 HP  (Normal + all defect sizes at 0 HP)
    Target → 3 HP  (Normal + all defect sizes at 3 HP)

Cross-load is a *strong* cross-condition shift: the bearing dynamics change
with mechanical load even when the defect geometry is identical, which moves
the resonance bands the model must rely on for fault detection.

Binary label: 0 = healthy (Normal), 1 = faulty (any fault type).

Signal
------
Drive-end accelerometer (``DE_time`` key in the .mat file), 12 kHz.
Windows of 1024 samples with stride 512 (50 % overlap).
Per-window z-score normalisation applied.

Data path
---------
Resolved in priority order (see ``nstad_bench.data._paths``):

1. *data_root* argument (overrides everything; pass this on Kaggle where
   the dataset slug subdirectory does not match the canonical name)
2. ``$DATA_ROOT/cwru/``
3. ``$NSTAD_DATA_ROOT/cwru/``
4. ``~/.nstad_bench/data/cwru/``

Filename convention
-------------------
The loader recognises both the canonical layout used by ``nstad-download``
(``Normal_0HP.mat``, ``B007_DE_0HP.mat``, …) and a recursive .mat scan that
groups by HP suffix.  This makes it work both locally and on the Kaggle
dataset ``brjapon/cwru-bearing-datasets`` once the user surfaces ``.mat``
files anywhere under the resolved root.

Usage
-----
::

    from nstad_bench.data.cwru_loader import cwru_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("cwru_load_0_3", cwru_loader())

    # Kaggle override: dataset slug rarely matches "cwru"
    register_dataset(
        "cwru_load_0_3",
        cwru_loader(data_root="/kaggle/input/cwru-bearing-datasets"),
    )
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from nstad_bench.data._paths import resolve_data_root

log = logging.getLogger(__name__)

WIN: int = 1024
STRIDE: int = 512

_NORMAL_PAT = re.compile(r"(?i)normal")
# Old canonical format: Normal_0HP.mat, B007_DE_0HP.mat
_HP_PAT_OLD = re.compile(r"(?i)_(\d+)\s*hp")
# brjapon/Kaggle format: Time_Normal_1_098.mat, B007_1_123.mat, OR007_6_1_136.mat
# HP load is always the second-to-last underscore token (file ID is last).
_HP_PAT_NEW = re.compile(r"_(\d+)_\d+$")
# esraakhaled/Kaggle format: B007_3.mat, Normal_0.mat, OR007@6_3.mat
# HP load is the single trailing underscore token (no file-ID suffix).
# Tried last so it doesn't shadow the two-token formats above.
_HP_PAT_TRAIL = re.compile(r"_(\d+)$")


def _default_root() -> Path:
    return resolve_data_root("cwru")


def _read_de_signal(path: Path) -> np.ndarray | None:
    """Return the DE_time channel from a .mat file, or ``None`` if absent.

    Files without a ``*_DE_time`` key (fan-end-only files, accelerometer
    summary files, etc.) are silently ignored — callers treat ``None`` as
    "skip this file".
    """
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required for the CWRU loader. Install with: pip install scipy"
        ) from exc

    try:
        mat = sio.loadmat(str(path))
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return None

    key = next((k for k in mat if "DE_time" in k), None)
    if key is None:
        return None
    sig = np.asarray(mat[key]).squeeze().astype(np.float32)
    return sig if sig.ndim == 1 and sig.size >= WIN else None


def _slice_windows(sig: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Return ``(N_windows, win)`` z-scored windows from a 1-D signal."""
    if sig.size < win:
        return np.empty((0, win), dtype=np.float32)
    n = 1 + (sig.size - win) // stride
    starts = np.arange(n) * stride
    out = np.stack([sig[s: s + win] for s in starts]).astype(np.float32)
    mu = out.mean(axis=1, keepdims=True)
    sigma = out.std(axis=1, keepdims=True) + 1e-8
    return (out - mu) / sigma


def _discover_files(root: Path, hp: str) -> tuple[list[Path], list[Path]]:
    """Return ``(normal_files, fault_files)`` matching load *hp* under *root*.

    Two filename conventions are recognised:
    - Old canonical: ``Normal_0HP.mat``, ``B007_DE_0HP.mat``
    - Kaggle brjapon: ``Time_Normal_1_098.mat``, ``B007_1_123.mat``,
      ``OR007_6_1_136.mat`` (HP is always second-to-last ``_``-token)

    Files are classified as Normal if the stem contains ``"normal"``; all
    other matching files are treated as fault recordings.  Search is
    recursive so Kaggle's nested directory layouts work without any
    pre-processing.
    """
    normal: list[Path] = []
    fault: list[Path] = []
    target_hp = str(hp)
    for p in sorted(root.rglob("*.mat")):
        m = (_HP_PAT_OLD.search(p.stem)
             or _HP_PAT_NEW.search(p.stem)
             or _HP_PAT_TRAIL.search(p.stem))
        if not m or m.group(1) != target_hp:
            continue
        if _NORMAL_PAT.search(p.stem):
            normal.append(p)
        else:
            fault.append(p)
    return normal, fault


def _load_domain(
    root: Path,
    hp: str,
    win: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one HP domain → ``(X, y)``."""
    normal_files, fault_files = _discover_files(root, hp)
    if not normal_files and not fault_files:
        raise FileNotFoundError(
            f"No CWRU .mat files found for HP={hp} under {root}.\n"
            "Expected filenames in one of two formats:\n"
            "  canonical: Normal_0HP.mat, B007_DE_0HP.mat\n"
            "  brjapon/Kaggle: Time_Normal_0_098.mat, B007_0_123.mat\n"
            "Set data_root / DATA_ROOT / NSTAD_DATA_ROOT to your CWRU directory."
        )

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for tag, files, label in (
        ("normal", normal_files, 0),
        ("fault", fault_files, 1),
    ):
        for p in files:
            sig = _read_de_signal(p)
            if sig is None:
                continue
            wins = _slice_windows(sig, win, stride)
            X_parts.append(wins)
            y_parts.append(np.full(len(wins), label, dtype=np.int64))
            log.debug("  %s  %s → %d windows", tag, p.name, len(wins))

    if not X_parts:
        raise FileNotFoundError(
            f"CWRU files for HP={hp} under {root} contain no usable DE_time data."
        )
    return np.concatenate(X_parts), np.concatenate(y_parts)


def cwru_loader(
    data_root: str | Path | None = None,
    *,
    source_hp: str = "0",
    target_hp: str = "3",
    win: int = WIN,
    stride: int = STRIDE,
    balance: bool = True,
    max_per_class: int | None = None,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a zero-arg loader for the CWRU cross-load domain split.

    Parameters
    ----------
    data_root :
        Directory containing the CWRU ``.mat`` files (searched recursively).
        Pass the Kaggle slug path directly here — env-var resolution is
        skipped when this argument is given.
    source_hp, target_hp :
        Load levels (string, in HP) used for source and target domains.
        Default is the 0 → 3 HP shift from the task spec.
    win, stride :
        Window length and step in samples (defaults: 1024 / 512 at 12 kHz).
    balance :
        Downsample the majority class to match the minority class per domain.
    max_per_class :
        Hard cap on per-class window count per domain (after balancing).
    seed :
        Random seed for subsampling and shuffling.

    Returns
    -------
    Callable
        Zero-argument function returning ``(X_s, y_s, X_t, y_t)``.

        - ``X_*`` — float32, shape ``(N, win)``
        - ``y_*`` — int64,   shape ``(N,)``,  0 = healthy, 1 = faulty
    """
    root = Path(data_root) if data_root else _default_root()

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        log.info("CWRU cross-load %sHP → %sHP, root=%s", source_hp, target_hp, root)
        X_s, y_s = _load_domain(root, source_hp, win, stride)
        X_t, y_t = _load_domain(root, target_hp, win, stride)

        from nstad_bench.data.deepbeat_loader import _balanced_subsample

        def _apply(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_h = int((y == 0).sum())
            n_f = int((y == 1).sum())
            cap = min(n_h, n_f) if (balance and n_h and n_f) else None
            if max_per_class is not None:
                cap = min(cap, max_per_class) if cap is not None else max_per_class
            if cap is not None:
                idx = _balanced_subsample(y, cap, rng)
                X, y = X[idx], y[idx]
            perm = rng.permutation(len(X))
            return X[perm], y[perm]

        X_s, y_s = _apply(X_s, y_s)
        X_t, y_t = _apply(X_t, y_t)

        log.info(
            "CWRU %sHP→%sHP — source: %d (h=%d, f=%d)  target: %d (h=%d, f=%d)",
            source_hp, target_hp,
            len(X_s), int((y_s == 0).sum()), int((y_s == 1).sum()),
            len(X_t), int((y_t == 0).sum()), int((y_t == 1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load


__all__ = ["cwru_loader"]

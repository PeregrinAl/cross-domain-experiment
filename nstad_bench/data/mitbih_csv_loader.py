"""MIT-BIH loader for the Kaggle mondejar CSV format.

Used as a fallback when WFDB (.hea/.dat/.atr) files are unavailable — e.g.
on the Kaggle dataset ``mondejar/mitbih-database`` which ships pre-segmented
CSV files instead of raw PhysioNet recordings.

CSV format (mondejar)
---------------------
``mitbih_train.csv`` and ``mitbih_test.csv``, no header, 188 columns:

    columns 0–186 : 187-sample beat waveform (float, already normalised)
    column    187 : class label  0=N  1=S  2=V  3=F  4=Q

Binary mapping (AAMI N-vs-rest):
    0 → 0  (Normal)
    1/2/3/4 → 1  (arrhythmia)

Domain split:
    Source → train CSV  (~87 k beats, DS1 proxy)
    Target → test  CSV  (~21 k beats, DS2 proxy)

Note: this split is not the canonical inter-patient DS1/DS2 partition, but
it is a reasonable proxy when WFDB files are unavailable.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from nstad_bench.data._paths import resolve_data_root

log = logging.getLogger(__name__)

WIN_CSV: int = 187   # beat length in the mondejar CSV


def _default_root() -> Path:
    return resolve_data_root("mitbih")


def _load_csv_split(csv_path: Path, max_per_class: int, rng: np.random.Generator):
    """Read one CSV file and return balanced (X, y)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for the CSV loader.") from exc

    df = pd.read_csv(csv_path, header=None)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    raw_labels = df.iloc[:, -1].values.astype(int)
    y_all = (raw_labels != 0).astype(np.int64)   # binary: 0=N, 1=arrhythmia

    idx_n   = np.where(y_all == 0)[0]
    idx_arr = np.where(y_all == 1)[0]

    n_take = min(max_per_class, len(idx_n), len(idx_arr))
    chosen_n   = rng.choice(idx_n,   n_take, replace=False)
    chosen_arr = rng.choice(idx_arr, n_take, replace=False)

    idx = np.concatenate([chosen_n, chosen_arr])
    perm = rng.permutation(len(idx))
    idx = idx[perm]

    log.debug(
        "  %s: %d N + %d arr → sampled %d each  (total %d)",
        csv_path.name, len(idx_n), len(idx_arr), n_take, len(idx),
    )
    return X_all[idx], y_all[idx]


def mitbih_csv_loader(
    data_root: str | Path | None = None,
    *,
    max_per_class: int = 3_000,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a loader for the mondejar MIT-BIH CSV dataset.

    Parameters
    ----------
    data_root :
        Directory containing ``mitbih_train.csv`` and ``mitbih_test.csv``.
    max_per_class :
        Maximum beats per class per split (balanced subsampling).
    seed :
        Random seed for reproducible subsampling.

    Returns
    -------
    Callable
        Zero-argument function returning ``(X_source, y_source, X_target, y_target)``.

        - ``X_*`` — float32, shape ``(N, 187)``
        - ``y_*`` — int64,   shape ``(N,)``,   0 = N, 1 = arrhythmia
    """
    root = Path(data_root) if data_root else _default_root()

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_csv = root / "mitbih_train.csv"
        test_csv  = root / "mitbih_test.csv"

        for p in (train_csv, test_csv):
            if not p.exists():
                raise FileNotFoundError(
                    f"MIT-BIH CSV file not found: {p}\n"
                    "Expected mitbih_train.csv and mitbih_test.csv under data_root.\n"
                    f"data_root resolved to: {root}"
                )

        rng = np.random.default_rng(seed)
        log.info("MIT-BIH CSV loader — root=%s  max_per_class=%d", root, max_per_class)

        X_s, y_s = _load_csv_split(train_csv, max_per_class, rng)
        X_t, y_t = _load_csv_split(test_csv,  max_per_class, rng)

        log.info(
            "MIT-BIH CSV — source: %d (N=%d, A=%d)  target: %d (N=%d, A=%d)",
            len(X_s), int((y_s == 0).sum()), int((y_s == 1).sum()),
            len(X_t), int((y_t == 0).sum()), int((y_t == 1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load


__all__ = ["mitbih_csv_loader"]

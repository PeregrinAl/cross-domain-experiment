"""CWRU Bearing Dataset loader — cross-severity domain split.

Domain split
------------
Both domains use the **same load condition (0 HP / 1797 RPM)**.
The domain axis is **defect size**:

    Source → Normal_0HP  +  small defects  0.007" (B007, IR007, OR007@*)
    Target → Normal_0HP  +  larger defects 0.014" (B014, IR014, OR014@6)
                                       and 0.021" (B021, IR021, OR021@*)

Rationale
---------
The previous cross-load split (0 HP → 3 HP) was found to be **trivially easy**
(model AUC ≈ 1.0, kurtosis-only logistic regression AUC ≈ 0.97) because the
healthy/faulty contrast is dominated by impulsivity — a feature that transfers
perfectly across load conditions.

The cross-severity split asks a harder question: can a model trained to detect
*small* defects (0.007") detect *larger* defects (0.014"–0.021")?  Larger
defects produce stronger impulses (higher kurtosis) than small ones, so the
source distribution (small fault) is *inside* the target distribution (large
fault).  The shift direction is non-trivial for a model trained only on source.

Binary label: 0 = healthy bearing (Normal), 1 = faulty bearing (any fault type).

Defect sizes used
-----------------
Source faults (0.007" — small):
    B007_DE, IR007_DE, OR007@3_DE, OR007@6_DE, OR007@12_DE  (5 files)

Target faults (0.014" and 0.021" — larger):
    B014_DE, IR014_DE, OR014@6_DE                            (3 files)
    B021_DE, IR021_DE, OR021@3_DE, OR021@6_DE, OR021@12_DE  (5 files)

Normal:
    Normal_0HP is shared by both source and target — the healthy baseline is
    identical in both conditions because only the defect size varies.

Note: 0.028" defects (B028, IR028) are excluded — they are an extreme severity
not included in the 007/014/021 benchmark convention and would only make the
target easier.

Signal
------
Drive-end accelerometer (``DE_time`` key in the .mat file).
Windows of 1024 samples at 12 kHz.  Per-window z-score normalisation applied.

Task difficulty (cross-severity)
---------------------------------
Unlike the cross-load split (AUC ≈ 1.0 trivially), the cross-severity split
is expected to be substantially harder:

* Small-defect (0.007") fault signals have kurtosis ≈ 1–4 — only slightly
  above Gaussian; a source-trained model may not generalise to the larger
  impulses of 0.021" defects.
* The decision boundary shifts between domains: the model must learn
  *shape/periodicity* features rather than pure impulsivity magnitude.

Target SourceOnly AUC is expected to fall into the 0.75–0.90 range, making
this a genuinely interesting domain-adaptation benchmark.

Data path
---------
Resolved in priority order:

1. *data_root* argument (if given)
2. ``$NSTAD_DATA_ROOT/cwru/``  environment variable
3. ``~/.nstad_bench/data/cwru/``  default

Download with::

    nstad-download cwru --subset 12k_drive_end

Usage
-----
::

    from nstad_bench.data.cwru_loader import cwru_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("cwru_severity", cwru_loader())

    # Custom data path
    register_dataset("cwru_severity", cwru_loader(data_root="/path/to/cwru"))
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

WIN = 1024   # samples per window

# ── Cross-severity split definition ──────────────────────────────────────────

_HP = "0"   # fixed load condition for both domains

# Source: small defects (0.007")
_SOURCE_FAULT_STEMS: list[str] = [
    "B007_DE",
    "IR007_DE",
    "OR007@3_DE", "OR007@6_DE", "OR007@12_DE",
]

# Target: larger defects (0.014" + 0.021")
_TARGET_FAULT_STEMS: list[str] = [
    "B014_DE", "IR014_DE", "OR014@6_DE",
    "B021_DE", "IR021_DE",
    "OR021@3_DE", "OR021@6_DE", "OR021@12_DE",
]

_NORMAL_STEM = "Normal"


def _default_root() -> Path:
    env = os.environ.get("NSTAD_DATA_ROOT")
    if env:
        return Path(env) / "cwru"
    return Path.home() / ".nstad_bench" / "data" / "cwru"


def _windows(path: Path) -> np.ndarray:
    """Load the DE_time channel from a .mat file and slice into windows."""
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required for the CWRU loader.  "
            "Install it with:  pip install scipy"
        ) from exc

    mat = sio.loadmat(str(path))
    key = next((k for k in mat if "DE_time" in k), None)
    if key is None:
        raise KeyError(f"No 'DE_time' key found in {path}.  Keys: {list(mat.keys())}")

    sig = mat[key].squeeze().astype(np.float32)
    n   = len(sig) // WIN
    arr = sig[:n * WIN].reshape(n, WIN)

    # Per-window z-score
    mu    = arr.mean(axis=1, keepdims=True)
    sigma = arr.std(axis=1, keepdims=True) + 1e-8
    return (arr - mu) / sigma


def _load_split(
    root: Path,
    fault_stems: list[str],
    hp: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Normal + fault windows for one domain.

    Returns
    -------
    X : float32 (N, 1024)
    y : int64   (N,)  — 0 = healthy, 1 = faulty
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    # ── Normal ────────────────────────────────────────────────────────────────
    p = root / f"{_NORMAL_STEM}_{hp}HP.mat"
    if not p.exists():
        raise FileNotFoundError(
            f"CWRU file not found: {p}\n"
            "Download with:  nstad-download cwru --subset 12k_drive_end\n"
            "Or set data_root / NSTAD_DATA_ROOT to your CWRU directory."
        )
    wins = _windows(p)
    X_parts.append(wins)
    y_parts.append(np.zeros(len(wins), dtype=np.int64))
    log.debug("  loaded %s: %d windows (label=0)", p.name, len(wins))

    # ── Faults ────────────────────────────────────────────────────────────────
    missing: list[str] = []
    for stem in fault_stems:
        p = root / f"{stem}_{hp}HP.mat"
        if not p.exists():
            missing.append(p.name)
            log.debug("  skip missing: %s", p.name)
            continue
        wins = _windows(p)
        X_parts.append(wins)
        y_parts.append(np.ones(len(wins), dtype=np.int64))
        log.debug("  loaded %s: %d windows (label=1)", p.name, len(wins))

    if missing:
        log.warning(
            "%d files not found — download the full 12k_drive_end subset.\n"
            "Missing: %s",
            len(missing), ", ".join(missing),
        )

    return np.concatenate(X_parts), np.concatenate(y_parts)


def cwru_loader(
    data_root: str | Path | None = None,
    *,
    balance: bool = True,
    max_per_class: int | None = None,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a loader for the CWRU cross-severity domain split.

    Both domains are recorded at 0 HP.  The domain shift is **defect size**:

    * Source — small defects (0.007"): B007, IR007, OR007@3/6/12 + Normal
    * Target — larger defects (0.014" + 0.021"): B014, IR014, OR014@6,
                                                  B021, IR021, OR021@3/6/12
                                                  + Normal (same file)

    Parameters
    ----------
    data_root:
        Directory containing the ``.mat`` files.
        Resolved from ``NSTAD_DATA_ROOT`` env var or ``~/.nstad_bench/data/cwru/``
        if not given.
    balance:
        If ``True`` (default), downsample the majority class to match the
        minority class count in each domain.  The normal class (238 windows)
        is the bottleneck; fault classes are downsampled to match it.
    max_per_class:
        If set, cap each class to this many windows per domain after balancing.
    seed:
        Random seed for sub-sampling.

    Returns
    -------
    Callable
        Zero-argument function returning
        ``(X_source, y_source, X_target, y_target)``.

        - ``X_*``  — float32, shape ``(N, 1024)``
        - ``y_*``  — int64,   shape ``(N,)``,  0 = healthy, 1 = faulty
    """
    root = Path(data_root) if data_root else _default_root()

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        log.info("CWRU cross-severity — loading source (0.007\") …")
        X_s, y_s = _load_split(root, _SOURCE_FAULT_STEMS, _HP)

        log.info("CWRU cross-severity — loading target (0.014\" + 0.021\") …")
        X_t, y_t = _load_split(root, _TARGET_FAULT_STEMS, _HP)

        # ── Balance and/or cap ────────────────────────────────────────────────
        from nstad_bench.data.deepbeat_loader import _balanced_subsample

        def _apply_cap(X: np.ndarray, y: np.ndarray, cap: int) -> tuple:
            idx = _balanced_subsample(y, cap, rng)
            return X[idx], y[idx]

        for tag in ("source", "target"):
            X, y = (X_s, y_s) if tag == "source" else (X_t, y_t)
            n_healthy = int((y == 0).sum())
            n_fault   = int((y == 1).sum())
            can_balance = balance and n_healthy > 0 and n_fault > 0
            cap = min(n_healthy, n_fault) if can_balance else None
            if max_per_class is not None:
                cap = min(cap, max_per_class) if cap is not None else max_per_class
            if cap is not None:
                X, y = _apply_cap(X, y, cap)
                log.debug(
                    "  %s balanced: healthy %d→%d, fault %d→%d",
                    tag, n_healthy, int((y==0).sum()), n_fault, int((y==1).sum()),
                )
            if tag == "source":
                X_s, y_s = X, y
            else:
                X_t, y_t = X, y

        # ── Shuffle ───────────────────────────────────────────────────────────
        p_s = rng.permutation(len(X_s))
        X_s, y_s = X_s[p_s], y_s[p_s]
        p_t = rng.permutation(len(X_t))
        X_t, y_t = X_t[p_t], y_t[p_t]

        log.info(
            "CWRU severity 007→014+021 — "
            "source: %d (healthy=%d, fault=%d)  "
            "target: %d (healthy=%d, fault=%d)",
            len(X_s), int((y_s==0).sum()), int((y_s==1).sum()),
            len(X_t), int((y_t==0).sum()), int((y_t==1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load

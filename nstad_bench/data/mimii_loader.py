"""MIMII (Malfunctioning Industrial Machine Investigation) loader.

Domain split — synthetic, by machine ID
---------------------------------------
The MIMII-2019 corpus ships ~1000 sound clips for each of four machines
(pump, valve, fan, slider) recorded at 16 kHz from four different machine
units per machine type (id_00, id_02, id_04, id_06).  MIMII-DG (DCASE 2022)
adds an explicit source/target labelling for domain generalisation, but is
not on Kaggle — only on Zenodo.

For the Kaggle-only path we treat **machine ID as the domain**: train on one
unit (default ``id_00``), evaluate on a different unit selected by ``seed``
(rotating through ``id_02``, ``id_04``, ``id_06``).  The cross-unit shift is
driven by manufacturing tolerances and individual wear patterns, mirroring
the same kind of "same machine type, different physical unit" shift that
MIMII-DG annotates explicitly.

Binary task — anomaly detection (per-window):
    0 = normal       (clips from .../normal/)
    1 = abnormal     (clips from .../abnormal/)

Signal
------
Mono audio (channel 0 of the original 8-mic array) @ 16 kHz.  Each 10-second
clip is sliced into non-overlapping 1-second windows (``WIN = 16_000``
samples).  Per-window z-score normalisation is applied.

Dataset layout
--------------
Compatible with the MIMII-2019 PhysioNet/Zenodo layout::

    <root>/<snr>_dB/<machine>/id_XX/<normal|abnormal>/NNNNNNNN.wav

The loader searches recursively for matching ``id_XX/{normal,abnormal}/*.wav``
files, so any nested Kaggle layout works once you point ``data_root`` at the
slug path.

Data path
---------
Resolved in priority order (see ``nstad_bench.data._paths``):

1. *data_root* argument (overrides everything)
2. ``$DATA_ROOT/mimii/``
3. ``$NSTAD_DATA_ROOT/mimii/``
4. ``~/.nstad_bench/data/mimii/``

Usage
-----
::

    from nstad_bench.data.mimii_loader import mimii_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset(
        "mimii_pump_id0_id6",
        mimii_loader(machine="pump", source_id="id_00", target_id="id_06"),
    )

    # Kaggle override:
    register_dataset(
        "mimii_pump_id0_id6",
        mimii_loader(
            data_root="/kaggle/input/mimii-pump-sound-dataset",
            machine="pump",
        ),
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

FS: int = 16_000             # MIMII sampling rate
WIN_SEC: float = 1.0         # window length in seconds
WIN: int = int(FS * WIN_SEC) # 16_000 samples
CLIP_SEC: float = 10.0       # nominal MIMII clip duration

MACHINE_TYPES: tuple[str, ...] = ("pump", "valve", "fan", "slider")

# Target-domain rotation pool — the four unit IDs MIMII ships.  Source defaults
# to id_00 (the development unit); seed picks the target from the remaining
# three so source != target by construction.
DEFAULT_SOURCE_ID: str = "id_00"
DEFAULT_TARGET_POOL: tuple[str, ...] = ("id_02", "id_04", "id_06")

# Path-component regexes used to classify a WAV file inside the search tree.
_ID_PAT     = re.compile(r"(?:^|[/_])(id_\d+)(?:[/_]|$)")
_NORMAL_PAT = re.compile(r"(?:^|/)normal(?:/|$)", re.IGNORECASE)
_ABNORM_PAT = re.compile(r"(?:^|/)abnormal(?:/|$)", re.IGNORECASE)


def _default_root() -> Path:
    return resolve_data_root("mimii")


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(
    root: Path,
    machine: str,
    unit_id: str,
) -> tuple[list[Path], list[Path]]:
    """Return ``(normal_wavs, abnormal_wavs)`` for ``(machine, unit_id)``.

    A file is included only when its **path** contains the machine name
    (case-insensitive), the unit ID, and either ``/normal/`` or ``/abnormal/``.
    Classification by full-path components — not just filename — keeps the
    search robust against arbitrary nesting under the Kaggle slug.
    """
    normals: list[Path] = []
    abnorms: list[Path] = []
    machine_lc = machine.lower()
    for p in sorted(root.rglob("*.wav")):
        text = str(p.as_posix()).lower()
        if machine_lc not in text:
            continue
        m_id = _ID_PAT.search(text)
        if not m_id or m_id.group(1) != unit_id:
            continue
        if _NORMAL_PAT.search(text):
            normals.append(p)
        elif _ABNORM_PAT.search(text):
            abnorms.append(p)
    return normals, abnorms


# ─────────────────────────────────────────────────────────────────────────────
# WAV reading and windowizing
# ─────────────────────────────────────────────────────────────────────────────

def _read_wav(path: Path, target_fs: int = FS) -> np.ndarray:
    """Return the first channel of *path* as float32 in [-1, 1] @ *target_fs*.

    MIMII recordings are 8-channel 16 kHz; we use channel 0 for the single-
    channel benchmark.  Resampling is only done if the source rate differs
    from *target_fs* (rare for MIMII but possible for Kaggle re-uploads).
    """
    from scipy.io import wavfile

    fs, data = wavfile.read(str(path))
    if data.ndim == 2:
        data = data[:, 0]
    data = data.astype(np.float32)
    # int16 → [-1, 1]; float WAVs pass through unchanged.
    if np.issubdtype(np.dtype(data.dtype), np.floating):
        sig = data
    else:
        sig = data / float(np.iinfo(np.int16).max)
    if fs != target_fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(fs, target_fs)
        sig = resample_poly(sig, target_fs // g, fs // g).astype(np.float32)
    return sig


def _windowize(sig: np.ndarray, win: int = WIN) -> np.ndarray:
    """Slice *sig* into non-overlapping ``win``-sample windows, z-score per window.

    Returns ``(n_windows, win)`` float32 — empty array if ``sig`` is shorter
    than one window.
    """
    n = sig.size // win
    if n == 0:
        return np.empty((0, win), dtype=np.float32)
    X = sig[: n * win].reshape(n, win).astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Domain loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_domain(
    root: Path,
    machine: str,
    unit_id: str,
    win: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one ``(machine, unit_id)`` domain into ``(X, y)``."""
    normal_files, abnorm_files = _discover_files(root, machine, unit_id)
    if not normal_files and not abnorm_files:
        raise FileNotFoundError(
            f"No MIMII WAVs found for machine={machine!r}, unit={unit_id!r} "
            f"under {root}.\n"
            "Expected layout: <root>/<snr>_dB/<machine>/<id_XX>/<normal|abnormal>/*.wav\n"
            "Set data_root / DATA_ROOT / NSTAD_DATA_ROOT accordingly."
        )

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for tag, files, label in (
        ("normal",   normal_files, 0),
        ("abnormal", abnorm_files, 1),
    ):
        for p in files:
            try:
                sig = _read_wav(p)
            except Exception as exc:
                log.warning("  unreadable WAV %s: %s", p.name, exc)
                continue
            wins = _windowize(sig, win)
            if wins.size:
                X_parts.append(wins)
                y_parts.append(np.full(len(wins), label, dtype=np.int64))
                log.debug("  %s  %s → %d windows", tag, p.name, len(wins))

    if not X_parts:
        return np.empty((0, win), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public loader factory
# ─────────────────────────────────────────────────────────────────────────────

def mimii_loader(
    data_root: str | Path | None = None,
    *,
    machine: str = "pump",
    source_id: str = DEFAULT_SOURCE_ID,
    target_id: str | None = None,
    target_pool: tuple[str, ...] = DEFAULT_TARGET_POOL,
    win: int = WIN,
    balance: bool = True,
    max_per_class: int | None = None,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a zero-arg loader for one MIMII cross-unit domain pair.

    Parameters
    ----------
    data_root :
        Directory containing the MIMII tree (searched recursively).  Pass the
        Kaggle slug path directly here.
    machine :
        Machine type — one of ``MACHINE_TYPES`` (default ``"pump"``).
    source_id :
        Unit ID used for the source domain (default ``"id_00"``).
    target_id :
        Unit ID used for the target domain.  When ``None`` (default), the
        target is rotated through ``target_pool`` by ``seed``.  Pass an
        explicit string to fix the target across seeds (e.g. for a worst-case
        evaluation on ``"id_06"``).
    target_pool :
        Pool of candidate target unit IDs when ``target_id is None``.  Default
        ``("id_02", "id_04", "id_06")`` — every unit but the source.
    win :
        Window length in samples (default 16_000 = 1 s @ 16 kHz).
    balance :
        Downsample the majority class within each domain.  MIMII is heavily
        normal-skewed (~9:1 normal:abnormal in clip counts; even more skewed
        after window expansion since normal clips outnumber abnormals).
        Balancing is the default to avoid degenerate "all-normal" classifiers.
    max_per_class :
        Hard cap on per-class window count per domain (after balancing).
        Useful to keep loading time bounded — a full MIMII machine can be
        ~10 GB on disk.
    seed :
        Random seed for subsampling, shuffling, and target-pool rotation.

    Returns
    -------
    Callable
        Zero-argument function returning ``(X_s, y_s, X_t, y_t)``.

        - ``X_*`` — float32, shape ``(N, win)``
        - ``y_*`` — int64,   shape ``(N,)``,  0 = normal, 1 = abnormal
    """
    if machine not in MACHINE_TYPES:
        raise ValueError(
            f"Unknown MIMII machine {machine!r}. Expected one of {MACHINE_TYPES}."
        )
    root = Path(data_root) if data_root else _default_root()
    resolved_target = target_id or target_pool[seed % len(target_pool)]
    if resolved_target == source_id:
        raise ValueError(
            f"Source and target unit IDs are identical ({source_id!r}). "
            "Pick a different target_pool or pass target_id explicitly."
        )

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        log.info(
            "MIMII machine=%s  source=%s  target=%s  root=%s",
            machine, source_id, resolved_target, root,
        )
        X_s, y_s = _load_domain(root, machine, source_id, win)
        X_t, y_t = _load_domain(root, machine, resolved_target, win)

        from nstad_bench.data.deepbeat_loader import _balanced_subsample

        def _apply(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_norm = int((y == 0).sum())
            n_anom = int((y == 1).sum())
            cap = min(n_norm, n_anom) if (balance and n_norm and n_anom) else None
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
            "MIMII %s %s→%s — source: %d (norm=%d, anom=%d)  target: %d (norm=%d, anom=%d)",
            machine, source_id, resolved_target,
            len(X_s), int((y_s == 0).sum()), int((y_s == 1).sum()),
            len(X_t), int((y_t == 0).sum()), int((y_t == 1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load


__all__ = [
    "mimii_loader",
    "MACHINE_TYPES",
    "DEFAULT_SOURCE_ID",
    "DEFAULT_TARGET_POOL",
]

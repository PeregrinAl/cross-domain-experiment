"""Sleep-EDF database loader — cross-subject Wake vs. Sleep domain split.

Domain split (per task spec)
----------------------------
1. Pick *n_subjects* subjects deterministically from the seed.
2. Hold out one of them as the **target**; the rest are the **source**.

Different seeds yield different (source-subjects, target-subject) splits,
giving an honest cross-subject shift with a built-in variance estimate.

Binary task — Wake vs. Sleep:
    0 = Wake          (Sleep stage W)
    1 = Sleep         (Sleep stage 1, 2, 3/4, R; AASM consolidated)

Signal
------
EEG Fpz-Cz channel @ 100 Hz, 30-second epochs (3000 samples per window).
Per-epoch z-score normalisation applied.

Dataset layout
--------------
Compatible with the Kaggle dataset
``wzy20210331/sleep-edf-database-expanded-1-0-0`` and the PhysioNet
release.  Files come in pairs::

    SC4001E0-PSG.edf          polysomnography signals
    SC4001EC-Hypnogram.edf    sleep-stage annotations

The loader recurses under the resolved data root looking for
``*-PSG.edf`` files and matches each against the adjacent
``*-Hypnogram.edf`` by the 6-character subject/night prefix.

Data path
---------
1. *data_root* argument (overrides everything)
2. ``$DATA_ROOT/sleep-edf/``
3. ``$NSTAD_DATA_ROOT/sleep-edf/``
4. ``~/.nstad_bench/data/sleep-edf/``

Usage
-----
::

    from nstad_bench.data.sleep_edf_loader import sleep_edf_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("sleep_edf_loso", sleep_edf_loader(seed=0))

    # Kaggle: the slug directory rarely matches "sleep-edf"
    register_dataset(
        "sleep_edf_loso",
        sleep_edf_loader(
            data_root="/kaggle/input/sleep-edf-database-expanded-1-0-0",
            seed=0,
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

FS: int = 100         # target sampling rate (Hz)
EPOCH_SEC: int = 30   # AASM scoring epoch length
WIN: int = FS * EPOCH_SEC   # 3000 samples per epoch

# Hypnogram annotation strings → binary label (0 = wake, 1 = sleep).
# Stage 4 was merged into Stage 3 by AASM in 2007 — included here for older recordings.
_STAGE_MAP: dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 1,
    "Sleep stage 3": 1,
    "Sleep stage 4": 1,
    "Sleep stage R": 1,
}

# Subject/night prefix used to pair PSG ↔ Hypnogram. Format: "SC4001" / "ST7011".
_PREFIX_RE = re.compile(r"^([A-Za-z]{2}\d{4})")

# Channel name candidates in priority order (different recordings label it differently).
_FPZ_CZ_CANDIDATES: tuple[str, ...] = (
    "EEG Fpz-Cz", "Fpz-Cz", "EEG Fpz", "FpzCz",
)


def _default_root() -> Path:
    return resolve_data_root("sleep-edf")


def _index_subjects(root: Path) -> dict[str, tuple[Path, Path]]:
    """Return ``{subject_prefix: (psg_path, hypno_path)}`` for every pair under *root*.

    A pair is included only when both files exist.  Subjects without a
    hypnogram (e.g. corrupted records) are skipped.
    """
    psg_files = {
        m.group(1): p
        for p in sorted(root.rglob("*-PSG.edf"))
        if (m := _PREFIX_RE.match(p.name))
    }
    hyp_files = {
        m.group(1): p
        for p in sorted(root.rglob("*-Hypnogram.edf"))
        if (m := _PREFIX_RE.match(p.name))
    }
    return {k: (psg_files[k], hyp_files[k]) for k in psg_files if k in hyp_files}


def _read_subject(psg: Path, hyp: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for one subject from its PSG + hypnogram EDF pair.

    Epochs scored as ``"Sleep stage ?"`` or ``"Movement time"`` are dropped
    silently — they carry no usable label.  ``X`` is float32 ``(N, WIN)``;
    ``y`` is int64 ``(N,)`` with the Wake / Sleep binary label.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required for the Sleep-EDF loader. Install with: pip install mne"
        ) from exc

    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR", stim_channel=None)
    ann = mne.read_annotations(hyp)
    raw.set_annotations(ann, emit_warning=False)

    ch_name = next((c for c in _FPZ_CZ_CANDIDATES if c in raw.ch_names), None)
    if ch_name is None:
        raise KeyError(
            f"No Fpz-Cz channel found in {psg.name}. "
            f"Channels present: {raw.ch_names}"
        )
    raw.pick([ch_name])
    if int(round(raw.info["sfreq"])) != FS:
        raw.resample(FS, verbose="ERROR")

    sig = raw.get_data()[0].astype(np.float32)   # (n_samples,)
    sfreq = raw.info["sfreq"]

    X_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    for onset, duration, description in zip(ann.onset, ann.duration, ann.description):
        label = _STAGE_MAP.get(str(description))
        if label is None:
            continue
        # An annotation typically spans many 30-s epochs; chop it into WIN-sized pieces.
        n_epochs = int(round(duration / EPOCH_SEC))
        start_sample = int(round(onset * sfreq))
        for k in range(n_epochs):
            s = start_sample + k * WIN
            e = s + WIN
            if e > sig.size:
                break
            X_parts.append(sig[s:e])
            y_parts.append(label)

    if not X_parts:
        return np.empty((0, WIN), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = np.stack(X_parts).astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - mu) / sigma
    y = np.asarray(y_parts, dtype=np.int64)
    return X, y


def _stack_subjects(
    subjects: list[str],
    index: dict[str, tuple[Path, Path]],
) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sid in subjects:
        psg, hyp = index[sid]
        try:
            Xi, yi = _read_subject(psg, hyp)
        except Exception as exc:
            log.warning("Skipping subject %s (%s): %s", sid, psg.name, exc)
            continue
        if Xi.size:
            Xs.append(Xi)
            ys.append(yi)
            log.debug("  %s: %d epochs (W=%d, S=%d)", sid, len(Xi), int((yi == 0).sum()), int((yi == 1).sum()))
    if not Xs:
        return np.empty((0, WIN), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(Xs), np.concatenate(ys)


def sleep_edf_loader(
    data_root: str | Path | None = None,
    *,
    n_subjects: int = 10,
    seed: int = 0,
    balance: bool = True,
    max_per_class: int | None = None,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a zero-arg loader for the cross-subject Sleep-EDF split.

    Parameters
    ----------
    data_root :
        Sleep-EDF directory (searched recursively for ``*-PSG.edf`` /
        ``*-Hypnogram.edf`` pairs).
    n_subjects :
        Total subjects to draw from the database.  One of them becomes the
        target, the rest become source.  Default 10 per task spec.
    seed :
        Random seed used for **both** subject selection and target choice.
        Different seeds → different (source, target) splits — this is how
        the benchmark provides variance estimates across 3 seeds.
    balance :
        If ``True``, downsample the majority class within each domain to
        match the minority class count.  Sleep epochs vastly outnumber
        Wake epochs in healthy recordings, so balancing is on by default.
    max_per_class :
        Optional hard cap on per-class epoch count per domain (after balancing).

    Returns
    -------
    Callable
        Zero-argument function returning ``(X_s, y_s, X_t, y_t)``.

        - ``X_*`` — float32, shape ``(N, 3000)``
        - ``y_*`` — int64,   shape ``(N,)``,  0 = Wake, 1 = Sleep
    """
    root = Path(data_root) if data_root else _default_root()

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        index = _index_subjects(root)
        if not index:
            raise FileNotFoundError(
                f"No Sleep-EDF PSG/Hypnogram pairs found under {root}.\n"
                "Expected files: *-PSG.edf paired with *-Hypnogram.edf.\n"
                "Set data_root / DATA_ROOT / NSTAD_DATA_ROOT accordingly."
            )

        available = sorted(index.keys())
        if len(available) < n_subjects:
            log.warning(
                "Only %d subjects available, requested %d — using all of them.",
                len(available), n_subjects,
            )
            chosen = available
        else:
            chosen_idx = rng.choice(len(available), n_subjects, replace=False)
            chosen = [available[i] for i in sorted(chosen_idx)]

        # Pick one of the chosen subjects as target; remaining become source.
        target_pos = int(rng.integers(len(chosen)))
        target = [chosen[target_pos]]
        source = [s for i, s in enumerate(chosen) if i != target_pos]

        log.info("Sleep-EDF cross-subject split (seed=%d)", seed)
        log.info("  Source (%d subjects): %s", len(source), source)
        log.info("  Target (%d subject):  %s", len(target), target)

        X_s, y_s = _stack_subjects(source, index)
        X_t, y_t = _stack_subjects(target, index)

        from nstad_bench.data.deepbeat_loader import _balanced_subsample

        def _apply(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_w = int((y == 0).sum())
            n_s = int((y == 1).sum())
            cap = min(n_w, n_s) if (balance and n_w and n_s) else None
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
            "Sleep-EDF — source: %d (W=%d, S=%d)  target: %d (W=%d, S=%d)",
            len(X_s), int((y_s == 0).sum()), int((y_s == 1).sum()),
            len(X_t), int((y_t == 0).sum()), int((y_t == 1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load


__all__ = ["sleep_edf_loader"]

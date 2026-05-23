"""CHB-MIT Scalp EEG Database loader — leave-one-subject-out seizure detection.

Domain split (per task spec)
----------------------------
LOSO across paediatric patients from the CHB-MIT database.  For a given
*test_patient*, the source domain is the union of the remaining patients
in *train_patients*; the target domain is the single held-out patient.
This is the standard cross-subject protocol used in the seizure-detection
literature.

Binary task — seizure detection (per-window):
    0 = interictal   (no annotated seizure overlaps the window)
    1 = ictal        (window overlaps at least one annotated seizure interval)

Signal
------
Single bipolar channel (default ``F7-T7``) sampled @ 256 Hz.  Each EDF is
bandpass-filtered 0.5-40 Hz (Butterworth order 4, zero-phase ``filtfilt``)
before being chopped into non-overlapping 2-second windows (512 samples).
Per-window z-score normalisation is applied.

Dataset layout
--------------
Compatible with the Kaggle dataset
``shajinrp/seizure-eeg-chb-mit-and-siena-scalp`` and the PhysioNet
release.  Files for each patient live under a ``chbXX/`` directory::

    chb01/chb01_01.edf
    chb01/chb01_03.edf
    chb01/chb01-summary.txt        seizure annotations for this patient
    ...

``chbXX-summary.txt`` lists, for every EDF in the directory, the
seizure intervals in seconds from the start of that file::

    File Name: chb01_03.edf
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

Some patients use the alternate keys ``Seizure 1 Start Time`` /
``Seizure 1 End Time`` when more than one seizure occurs in a file;
both forms are parsed.

Data path
---------
Resolved in priority order (see ``nstad_bench.data._paths``):

1. *data_root* argument (overrides everything; pass this on Kaggle where
   the dataset slug subdirectory does not match the canonical name)
2. ``$DATA_ROOT/chbmit/``
3. ``$NSTAD_DATA_ROOT/chbmit/``
4. ``~/.nstad_bench/data/chbmit/``

The loader searches *recursively* under the resolved root, so any nested
layout (``chbmit/chb01/...`` or ``shajinrp/chbmit/chb01/...``) works.

Usage
-----
::

    from nstad_bench.data.chbmit_loader import chbmit_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset(
        "chbmit_loso_chb01",
        chbmit_loader(test_patient="chb01"),
    )

    # Kaggle override
    register_dataset(
        "chbmit_loso_chb01",
        chbmit_loader(
            data_root="/kaggle/input/seizure-eeg-chb-mit-and-siena-scalp",
            test_patient="chb01",
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

FS: int = 256              # CHB-MIT sampling rate
WIN_SEC: float = 2.0       # window length in seconds
WIN: int = int(FS * WIN_SEC)   # 512 samples

# Default patient cohort — diverse age / seizure-profile mix per the task plan.
DEFAULT_PATIENTS: tuple[str, ...] = (
    "chb01", "chb03", "chb05", "chb08", "chb10", "chb16",
)

# Channel name candidates (case-insensitive).  CHB-MIT lists channels as
# bipolar pairs; F7-T7 is the standard frontal-temporal channel in the
# seizure-detection literature.  Fallbacks cover recordings that label the
# pair slightly differently (with or without "EEG " prefix, spaces, etc.).
_CHANNEL_CANDIDATES: tuple[str, ...] = (
    "F7-T7", "FP1-F7", "T7-P7", "F8-T8",
)

# Default bandpass for paediatric seizure detection (Hz).
BANDPASS_LO: float = 0.5
BANDPASS_HI: float = 40.0


def _default_root() -> Path:
    return resolve_data_root("chbmit")


# ─────────────────────────────────────────────────────────────────────────────
# Seizure-annotation parser
# ─────────────────────────────────────────────────────────────────────────────

_FILE_RE  = re.compile(r"^File Name:\s*(\S+\.edf)", re.IGNORECASE)
_NSEIZ_RE = re.compile(r"^Number of Seizures in File:\s*(\d+)", re.IGNORECASE)
# Matches both "Seizure Start Time: 2996 seconds" and
# "Seizure 1 Start Time: 2996 seconds" (multi-seizure files).
_START_RE = re.compile(
    r"^Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds",
    re.IGNORECASE,
)
_END_RE   = re.compile(
    r"^Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds",
    re.IGNORECASE,
)


def _parse_summary(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse a CHB-MIT ``chbXX-summary.txt`` into ``{edf_filename: [(start, end), ...]}``.

    Seconds are integer offsets from the start of each EDF.  Files with zero
    seizures still receive an entry mapped to an empty list, so callers can
    distinguish "interictal-only file" from "file not listed in summary".
    """
    annotations: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    pending_start: int | None = None

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_file = _FILE_RE.match(line)
        if m_file:
            current_file = m_file.group(1)
            annotations.setdefault(current_file, [])
            pending_start = None
            continue

        if current_file is None:
            continue

        m_nseiz = _NSEIZ_RE.match(line)
        if m_nseiz:
            # Reset (already initialised by the File Name match above).
            continue

        m_start = _START_RE.match(line)
        if m_start:
            pending_start = int(m_start.group(1))
            continue

        m_end = _END_RE.match(line)
        if m_end and pending_start is not None:
            annotations[current_file].append((pending_start, int(m_end.group(1))))
            pending_start = None

    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# Signal preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(sig: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _select_channel(raw, candidates: tuple[str, ...]) -> str:
    """Return the first channel from *candidates* present in *raw*, case-insensitive."""
    available = {c.upper(): c for c in raw.ch_names}
    for name in candidates:
        if name.upper() in available:
            return available[name.upper()]
    raise KeyError(
        f"None of {candidates} found in EDF channels: {raw.ch_names[:10]}..."
    )


def _read_edf(path: Path, channel_candidates: tuple[str, ...]) -> tuple[np.ndarray, int]:
    """Return ``(signal, sfreq)`` for the selected channel from *path*.

    Raises FileNotFoundError-style errors with context if the EDF is unreadable
    or lacks any of the candidate channels.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required for the CHB-MIT loader. Install with: pip install mne"
        ) from exc

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR", stim_channel=None)
    ch_name = _select_channel(raw, channel_candidates)
    raw.pick([ch_name])
    sfreq = int(round(raw.info["sfreq"]))
    if sfreq != FS:
        raw.resample(FS, verbose="ERROR")
        sfreq = FS
    return raw.get_data()[0].astype(np.float32), sfreq


def _windowize(
    sig: np.ndarray,
    seizures: list[tuple[int, int]],
    fs: int,
    win: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice *sig* into non-overlapping windows of *win* samples.

    A window is labelled ``1`` (ictal) if it overlaps any annotated seizure
    interval by at least one sample, else ``0`` (interictal).  Per-window
    z-score normalisation is applied.

    Returns ``(X, y)`` with shapes ``(N, win)`` / ``(N,)``.
    """
    n = sig.size // win
    if n == 0:
        return np.empty((0, win), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = sig[: n * win].reshape(n, win).astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - mu) / sigma

    y = np.zeros(n, dtype=np.int64)
    for s_sec, e_sec in seizures:
        s_smp = s_sec * fs
        e_smp = e_sec * fs
        # Window k spans samples [k*win, (k+1)*win); flag any overlap.
        first = max(0, s_smp // win)
        last  = min(n - 1, (e_smp - 1) // win) if e_smp > 0 else -1
        if last >= first:
            y[first:last + 1] = 1
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level loading
# ─────────────────────────────────────────────────────────────────────────────

def _index_patient(
    root: Path,
    patient: str,
) -> tuple[Path, dict[str, list[tuple[int, int]]]]:
    """Return ``(patient_dir, annotations)`` for *patient*.

    Recursively searches *root* for the patient directory containing
    ``<patient>-summary.txt``.  Raises ``FileNotFoundError`` with a helpful
    message if not found — the typical cause is the Kaggle slug path being
    different from the canonical layout.
    """
    summary_name = f"{patient}-summary.txt"
    matches = list(root.rglob(summary_name))
    if not matches:
        raise FileNotFoundError(
            f"No {summary_name} found under {root}.\n"
            f"Expected layout: <root>/{patient}/{summary_name} (and matching .edf files).\n"
            "Set data_root / DATA_ROOT / NSTAD_DATA_ROOT to the CHB-MIT directory."
        )
    summary_path = matches[0]
    return summary_path.parent, _parse_summary(summary_path)


def _read_patient(
    root: Path,
    patient: str,
    *,
    channel_candidates: tuple[str, ...] = _CHANNEL_CANDIDATES,
    bp_lo: float = BANDPASS_LO,
    bp_hi: float = BANDPASS_HI,
    win: int = WIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all EDFs for *patient* into a single ``(X, y)`` pair."""
    patient_dir, annotations = _index_patient(root, patient)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for edf_name, seizures in sorted(annotations.items()):
        edf_path = patient_dir / edf_name
        if not edf_path.exists():
            # Some Kaggle dumps drop a few EDFs; skip silently rather than abort.
            log.debug("  missing EDF (skip): %s", edf_path)
            continue
        try:
            sig, fs = _read_edf(edf_path, channel_candidates)
        except Exception as exc:
            log.warning("  unreadable EDF %s: %s", edf_path.name, exc)
            continue

        sig = _bandpass(sig, fs, bp_lo, bp_hi)
        Xi, yi = _windowize(sig, seizures, fs, win)
        if Xi.size:
            X_parts.append(Xi)
            y_parts.append(yi)
            log.debug(
                "  %s: %d wins (ictal=%d, inter=%d)",
                edf_name, len(Xi), int(yi.sum()), int((yi == 0).sum()),
            )

    if not X_parts:
        return np.empty((0, win), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def _stack_patients(
    patients: list[str],
    root: Path,
    *,
    channel_candidates: tuple[str, ...] = _CHANNEL_CANDIDATES,
    bp_lo: float = BANDPASS_LO,
    bp_hi: float = BANDPASS_HI,
    win: int = WIN,
) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for pid in patients:
        try:
            Xi, yi = _read_patient(
                root, pid,
                channel_candidates=channel_candidates,
                bp_lo=bp_lo, bp_hi=bp_hi, win=win,
            )
        except FileNotFoundError as exc:
            log.warning("Skipping patient %s: %s", pid, exc)
            continue
        if Xi.size:
            Xs.append(Xi)
            ys.append(yi)
            log.info(
                "  %s: %d windows (ictal=%d, inter=%d)",
                pid, len(Xi), int(yi.sum()), int((yi == 0).sum()),
            )
    if not Xs:
        return np.empty((0, win), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(Xs), np.concatenate(ys)


# ─────────────────────────────────────────────────────────────────────────────
# Public loader factory
# ─────────────────────────────────────────────────────────────────────────────

def chbmit_loader(
    data_root: str | Path | None = None,
    *,
    test_patient: str = "chb01",
    train_patients: tuple[str, ...] | None = None,
    channel_candidates: tuple[str, ...] = _CHANNEL_CANDIDATES,
    bp_lo: float = BANDPASS_LO,
    bp_hi: float = BANDPASS_HI,
    win: int = WIN,
    balance: bool = True,
    max_per_class: int | None = None,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a zero-arg loader for the CHB-MIT LOSO seizure-detection split.

    Parameters
    ----------
    data_root :
        Directory containing CHB-MIT patient subdirectories (searched
        recursively).  Pass the Kaggle slug path directly here — env-var
        resolution is skipped when this argument is given.
    test_patient :
        The held-out target patient (e.g. ``"chb01"``).  All windows from this
        patient go to the target domain.
    train_patients :
        Source-domain patients.  Defaults to ``DEFAULT_PATIENTS`` minus
        *test_patient*.  Pass an explicit tuple to use a different cohort.
    channel_candidates :
        Channel names tried in order; the first one present in each EDF is
        used.  Default: F7-T7, FP1-F7, T7-P7, F8-T8.
    bp_lo, bp_hi :
        Bandpass cut-off frequencies (Hz).  Default 0.5-40 — standard for
        paediatric seizure detection.
    win :
        Window length in samples.  Default 512 (= 2 s @ 256 Hz).
    balance :
        Downsample the majority class (interictal) to match the minority
        (ictal) per domain.  Source-only models on heavily imbalanced data
        collapse to the majority class; balancing is the standard recipe
        in the CHB-MIT literature.
    max_per_class :
        Optional hard cap on per-class window count per domain (after balancing).
    seed :
        Random seed for subsampling and shuffling.

    Returns
    -------
    Callable
        Zero-argument function returning ``(X_s, y_s, X_t, y_t)``.

        - ``X_*`` — float32, shape ``(N, win)``
        - ``y_*`` — int64,   shape ``(N,)``,  0 = interictal, 1 = ictal
    """
    root = Path(data_root) if data_root else _default_root()
    if train_patients is None:
        train_patients = tuple(p for p in DEFAULT_PATIENTS if p != test_patient)

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        log.info(
            "CHB-MIT LOSO: source=%s  target=%s  root=%s",
            list(train_patients), test_patient, root,
        )

        X_s, y_s = _stack_patients(
            list(train_patients), root,
            channel_candidates=channel_candidates,
            bp_lo=bp_lo, bp_hi=bp_hi, win=win,
        )
        X_t, y_t = _stack_patients(
            [test_patient], root,
            channel_candidates=channel_candidates,
            bp_lo=bp_lo, bp_hi=bp_hi, win=win,
        )

        if X_s.size == 0:
            raise FileNotFoundError(
                f"No usable source data for CHB-MIT (train_patients={train_patients}). "
                f"Check that the Kaggle slug path resolves to a directory containing "
                f"chbXX-summary.txt files."
            )

        from nstad_bench.data.deepbeat_loader import _balanced_subsample

        def _apply(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_inter = int((y == 0).sum())
            n_ictal = int((y == 1).sum())
            cap = min(n_inter, n_ictal) if (balance and n_inter and n_ictal) else None
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
            "CHB-MIT LOSO — source: %d (inter=%d, ictal=%d)  target: %d (inter=%d, ictal=%d)",
            len(X_s), int((y_s == 0).sum()), int((y_s == 1).sum()),
            len(X_t), int((y_t == 0).sum()), int((y_t == 1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load


__all__ = ["chbmit_loader", "DEFAULT_PATIENTS"]

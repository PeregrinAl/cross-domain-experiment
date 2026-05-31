"""MIT-BIH Arrhythmia Database loader — inter-patient DS1 → DS2 split.

Domain split (AAMI EC57 recommended partition)
----------------------------------------------
    Source → DS1  (22 records: 101, 106, 108, …, 230)
    Target → DS2  (22 records: 100, 103, 105, …, 234)

Binary label — AAMI EC57 N-vs-rest:
    0 = N superclass  (N, L, R, e, j — normal + bundle-branch blocks)
    1 = non-N         (A, a, J, S, V, E, F, /, f, Q — all ectopic/unknown)

Signal
------
Lead II (channel 0), 360 Hz.  Windows of 280 samples (≈0.78 s) centred on each
annotated R-peak.  Per-beat z-score normalisation applied.

Sampling strategy
-----------------
The arrhythmia class is unevenly distributed across DS1 records: a handful of
records (109, 118, 207, 124) contribute the majority of arrhythmia beats, while
others (101: 3, 108: 21, 112: 2, 230: 1) have fewer than 50 arrhythmia beats
and would be under-represented to <10 windows after random pool sampling.

This loader uses **stratified per-record sampling**:

1. Filter: only include a record in the arrhythmia pool if it has ≥
   *min_beats_per_record* arrhythmia beats (default 50).
2. Allocate: distribute *max_per_class* windows across remaining records using
   an iterative equal-share algorithm — records with fewer beats than their
   equal share contribute all their beats; the saved budget is redistributed
   to the rest.
3. Sample: draw the allocated number of beats at random from each record.

The same procedure is applied to the normal class (all DS1 records have ≥ 50
normal beats so none are filtered out).

Result (cap=3000)
-----------------
    Normal     min=166  median=167  max=167  (18 records, 3 000 total)
    Arrhythmia min= 53  median=227  max=228  (16 records, 3 000 total)

Compare with the naive random-pool approach (same cap):
    Normal     min=107  median=160  max=274
    Arrhythmia min=  7  median= 99  max=627   ← 7 records with < 50 beats

Usage
-----
::

    from nstad_bench.data.mitbih_loader import mitbih_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("mitbih_ds1_ds2", mitbih_loader())
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from nstad_bench.data._paths import resolve_data_root

log = logging.getLogger(__name__)

# ── AAMI EC57 inter-patient partition ────────────────────────────────────────

DS1: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)
DS2: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)

#: AAMI N superclass — Normal beat and bundle-branch blocks → label 0.
NORMAL_SYMBOLS: frozenset[str] = frozenset({"N", "L", "R", "e", "j"})

#: All non-N AAMI beat types → label 1.
#: S superclass (supraventricular ectopic): A, a, J, S
#: V superclass (ventricular ectopic):      V, E
#: F superclass (fusion):                   F
#: Q superclass (unknown/unclassifiable):   /, f, Q
ARRHYTHMIA_SYMBOLS: frozenset[str] = frozenset({"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"})

WIN: int = 140   # half-window → 280 samples total (≈0.78 s at 360 Hz)


def _default_root() -> Path:
    return resolve_data_root("mitbih")


# ── Core segmentation ─────────────────────────────────────────────────────────

def _segment_records_mondejar(
    records: tuple[str, ...],
    root: Path,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Segment beats from the mondejar/mitbih-database Kaggle per-record CSV format.

    File layout expected in *root*::

        {rec}.csv              — signal, CSV with header row; MLII in the column
                                 whose name contains "MLII" (e.g. ``"'MLII'"``).
                                 Fallback: 3rd column (index 2), matching the
                                 typical 4-column layout: elapsed, sample#, MLII, V5.
        {rec}annotations.txt   — beat annotations in WFDB text format::

                                     Time       Sample #  Type  Sub  Chan  Num
                                     0:00.261     94       N      0    0    0
                                     …

                                 Lines where the sample-# field is not an integer
                                 (header, rhythm labels, etc.) are silently skipped.

    Each beat is a z-score normalised float32 array of shape ``(2*WIN,)`` (280 samples).
    Records missing either file are skipped with a warning.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for the mondejar MIT-BIH format.") from exc

    pool: dict[str, dict[str, list]] = {}
    for rec in records:
        pool[rec] = {"normal": [], "arr": []}
        sig_path = root / f"{rec}.csv"
        ann_path = root / f"{rec}annotations.txt"

        if not sig_path.exists() or not ann_path.exists():
            log.warning("Record %s: %s not found — skipping", rec,
                        "signal CSV" if not sig_path.exists() else "annotations")
            continue

        try:
            df = pd.read_csv(sig_path)
            # Column whose name contains "MLII" (may be stored as "'MLII'" with
            # embedded single quotes from the original WFDB export).
            mlii_col = next((c for c in df.columns if "MLII" in c), None)
            if mlii_col is None:
                # Fallback: 3rd column (index 2) matches typical 4-col layout
                mlii_col = df.columns[2] if len(df.columns) >= 3 else df.columns[0]
                log.debug("Record %s: no 'MLII' column found, using %r", rec, mlii_col)
            s = df[mlii_col].to_numpy().astype(np.float32)
            s = (s - s.mean()) / (s.std() + 1e-8)
        except Exception as exc:
            log.warning("Could not read signal %s: %s — skipping", sig_path.name, exc)
            continue

        try:
            with open(ann_path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    # Detect whether the first field is a time string (e.g. "0:00.261")
                    # or a raw sample number.  WFDB text annotations always have
                    # the time first, then sample #, then beat type.
                    if ":" in parts[0]:          # time-prefixed: Time Sample# Type …
                        if len(parts) < 3:
                            continue
                        try:
                            idx = int(parts[1])
                            sym = parts[2]
                        except ValueError:
                            continue
                    else:                         # bare: Sample# Type …
                        try:
                            idx = int(parts[0])
                            sym = parts[1]
                        except ValueError:
                            continue

                    if idx < WIN or idx + WIN >= len(s):
                        continue
                    beat = s[idx - WIN: idx + WIN]
                    if sym in ARRHYTHMIA_SYMBOLS:
                        pool[rec]["arr"].append(beat)
                    elif sym in NORMAL_SYMBOLS:
                        pool[rec]["normal"].append(beat)
        except Exception as exc:
            log.warning("Could not read annotations %s: %s — skipping", ann_path.name, exc)

    return pool


def _segment_records(
    records: tuple[str, ...],
    root: Path,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Segment beats from *records* and return ``{record: {class: [beats]}}``.

    Each beat is a z-score normalised float32 array of shape ``(180,)``.
    Records that cannot be read are skipped with a warning.
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "wfdb is required for the MIT-BIH loader.  "
            "Install it with:  pip install wfdb"
        ) from exc

    pool: dict[str, dict[str, list]] = {}
    for rec in records:
        pool[rec] = {"normal": [], "arr": []}
        try:
            sig, _ = wfdb.rdsamp(str(root / rec), channels=[0])
            ann    = wfdb.rdann(str(root / rec), "atr")
        except Exception as exc:
            log.warning("Could not read record %s: %s — skipping", rec, exc)
            continue

        s = sig[:, 0].astype(np.float32)
        s = (s - s.mean()) / (s.std() + 1e-8)

        for idx, sym in zip(ann.sample, ann.symbol):
            if idx < WIN or idx + WIN >= len(s):
                continue
            beat = s[idx - WIN: idx + WIN]
            if sym in ARRHYTHMIA_SYMBOLS:
                pool[rec]["arr"].append(beat)
            elif sym in NORMAL_SYMBOLS:
                pool[rec]["normal"].append(beat)

    return pool


# ── Stratified allocation ─────────────────────────────────────────────────────

def _stratified_allocation(
    counts: dict[str, int],
    total_target: int,
) -> dict[str, int]:
    """Distribute *total_target* beats across records by iterative equal-share.

    Records with fewer beats than their equal share contribute everything they
    have; the saved budget is redistributed to the remaining records.

    Parameters
    ----------
    counts:
        ``{record: available_beat_count}`` — only records to sample from.
    total_target:
        Desired total number of beats.

    Returns
    -------
    dict
        ``{record: n_beats_to_sample}`` — values ≤ counts[record].
    """
    if not counts:
        return {}

    remaining = min(total_target, sum(counts.values()))
    sorted_recs = sorted(counts.items(), key=lambda x: x[1])
    allocation: dict[str, int] = {}

    for i, (rec, n) in enumerate(sorted_recs):
        n_left    = len(sorted_recs) - i
        equal_share = remaining // n_left
        take = min(n, equal_share)
        allocation[rec] = take
        remaining -= take

    # Distribute any remaining beats (due to integer division) to highest-count
    # records that still have headroom.
    leftover = total_target - sum(allocation.values())
    for rec, _ in reversed(sorted_recs):
        if leftover <= 0:
            break
        headroom = counts[rec] - allocation[rec]
        add = min(headroom, leftover)
        allocation[rec] += add
        leftover -= add

    return allocation


def _sample_from_allocation(
    pool: dict[str, dict[str, list[np.ndarray]]],
    allocation: dict[str, int],
    class_key: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw beats per the allocation dict and return ``(X, y_record)``."""
    X_parts: list[np.ndarray] = []
    pid_parts: list[np.ndarray] = []

    for rec, n_take in allocation.items():
        beats = pool[rec][class_key]
        idx   = rng.choice(len(beats), n_take, replace=False)
        X_parts.append(np.stack([beats[i] for i in idx]))
        pid_parts.append(np.full(n_take, rec, dtype=object))

    if not X_parts:
        return np.empty((0, 2 * WIN), dtype=np.float32), np.empty(0, dtype=object)

    return np.concatenate(X_parts), np.concatenate(pid_parts)


# ── Public loader factory ─────────────────────────────────────────────────────

def mitbih_loader(
    data_root: str | Path | None = None,
    *,
    max_per_class: int = 3_000,
    min_beats_per_record: int = 50,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a loader for the MIT-BIH DS1 → DS2 inter-patient split.

    Parameters
    ----------
    data_root:
        Directory containing the MITDB .hea/.dat/.atr files.
        Resolved from ``NSTAD_DATA_ROOT`` env var or
        ``~/.nstad_bench/data/mitbih/`` if not given.
    max_per_class:
        Target number of beats per class (normal / arrhythmia) per domain.
        Beats are allocated across records by stratified equal-share sampling.
    min_beats_per_record:
        Records with fewer than this many beats of a given class are excluded
        from that class's pool.  Prevents <50-beat records from being
        under-represented to near-zero windows after sampling.

        Default 50.  At ``max_per_class=3000`` this excludes 4 DS1 records
        from the arrhythmia pool (101: 3, 108: 21, 112: 2, 230: 1 beats).

    seed:
        Random seed for reproducible beat sampling.

    Returns
    -------
    Callable
        Zero-argument function returning
        ``(X_source, y_source, X_target, y_target)``.

        - ``X_*``  — float32, shape ``(N, 280)``
        - ``y_*``  — int64,   shape ``(N,)``,  0 = N-superclass, 1 = non-N

    Notes
    -----
    ``X_source`` comes from DS1, ``X_target`` from DS2.  For source, stratified
    allocation is applied per class separately.  For target, stratified
    allocation uses DS2 records, with the same ``min_beats_per_record`` filter.
    """
    root = Path(data_root) if data_root else _default_root()
    # Capture the partition lists at factory time so that callers who
    # temporarily monkey-patch DS1/DS2 for testing see the right values
    # when they eventually invoke the returned callable.
    _ds1: tuple[str, ...] = DS1
    _ds2: tuple[str, ...] = DS2

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        # Auto-detect storage format:
        #   mondejar per-record CSV  → {rec}.csv + {rec}annotations.txt
        #   WFDB (PhysioNet)         → {rec}.dat / {rec}.hea / {rec}.atr
        # Probe with the first DS1 record (101 is always in DS1).
        if (root / f"{_ds1[0]}.csv").exists():
            log.info("MIT-BIH: detected mondejar per-record CSV format at %s", root)
            _seg = _segment_records_mondejar
        else:
            log.info("MIT-BIH: using WFDB reader at %s", root)
            _seg = _segment_records

        log.info("Segmenting DS1 (source) …")
        src_pool = _seg(_ds1, root)
        log.info("Segmenting DS2 (target) …")
        tgt_pool = _seg(_ds2, root)

        def _build_split(
            pool: dict, records: tuple[str, ...]
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return (X, y) for one split using stratified per-class sampling."""
            # counts per record, filtered by min_beats_per_record
            norm_counts = {
                r: len(pool[r]["normal"])
                for r in records
                if len(pool[r]["normal"]) >= min_beats_per_record
            }
            arr_counts = {
                r: len(pool[r]["arr"])
                for r in records
                if len(pool[r]["arr"]) >= min_beats_per_record
            }

            excl_norm = [r for r in records if 0 < len(pool[r]["normal"]) < min_beats_per_record]
            excl_arr  = [r for r in records if 0 < len(pool[r]["arr"])    < min_beats_per_record]
            if excl_norm:
                log.debug("  Excluded from normal pool (<%d beats): %s", min_beats_per_record, excl_norm)
            if excl_arr:
                log.debug("  Excluded from arrhythmia pool (<%d beats): %s", min_beats_per_record, excl_arr)

            alloc_norm = _stratified_allocation(norm_counts, max_per_class)
            alloc_arr  = _stratified_allocation(arr_counts,  max_per_class)

            X_norm, _ = _sample_from_allocation(pool, alloc_norm, "normal", rng)
            X_arr,  _ = _sample_from_allocation(pool, alloc_arr,  "arr",    rng)

            X = np.concatenate([X_norm, X_arr])
            y = np.concatenate([
                np.zeros(len(X_norm), dtype=np.int64),
                np.ones( len(X_arr),  dtype=np.int64),
            ])
            perm = rng.permutation(len(X))

            _log_alloc("normal",     alloc_norm, norm_counts)
            _log_alloc("arrhythmia", alloc_arr,  arr_counts)

            return X[perm], y[perm]

        X_s, y_s = _build_split(src_pool, _ds1)
        X_t, y_t = _build_split(tgt_pool, _ds2)

        if len(X_s) == 0 or len(X_t) == 0:
            raise FileNotFoundError(
                f"No MIT-BIH beats loaded from {root}.\n"
                "Supported formats (auto-detected):\n"
                "  WFDB          — .hea/.dat/.atr files for records 100–234\n"
                "  mondejar CSV  — {rec}.csv + {rec}annotations.txt per record\n"
                "  pre-segmented — mitbih_train.csv + mitbih_test.csv\n"
                "Check KAGGLE_MITBIH_DIR points to the directory that contains "
                "those files directly (not a subdirectory of it)."
            )

        log.info(
            "MIT-BIH DS1→DS2 — source: %d  (N=%d, A=%d)  "
            "target: %d  (N=%d, A=%d)",
            len(X_s), int((y_s==0).sum()), int((y_s==1).sum()),
            len(X_t), int((y_t==0).sum()), int((y_t==1).sum()),
        )
        return X_s, y_s, X_t, y_t

    return _load


def _log_alloc(
    class_name: str,
    allocation: dict[str, int],
    counts: dict[str, int],
) -> None:
    if not allocation:
        return
    vals = list(allocation.values())
    log.debug(
        "  %s allocation — records=%d  min=%d  median=%.0f  max=%d  total=%d",
        class_name, len(vals), min(vals),
        float(np.median(vals)), max(vals), sum(vals),
    )
    for rec, n in sorted(allocation.items(), key=lambda x: x[1]):
        log.debug("    %s: %d / %d", rec, n, counts[rec])

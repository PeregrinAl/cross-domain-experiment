"""Tests for nstad_bench.data.mitbih_loader.

Uses a synthetic wfdb fixture: two records with known beat counts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

pytestmark = pytest.mark.skipif(not HAS_WFDB, reason="wfdb not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic wfdb fixture
# ─────────────────────────────────────────────────────────────────────────────

WIN = 140    # half-window used by the loader

def _write_record(
    root: Path,
    rec_id: str,
    n_normal: int,
    n_arr: int,
    rng: np.random.Generator,
) -> None:
    """Write a synthetic WFDB record with *n_normal* N-beats and *n_arr* V-beats."""
    n_beats   = n_normal + n_arr
    beat_gap  = 300      # samples between R-peaks
    sig_len   = (n_beats + 2) * beat_gap

    # Scale to int16 range so wfdb ADC conversion stays in bounds.
    sig_f32 = rng.standard_normal(sig_len).astype(np.float32)
    sig_d   = (sig_f32 * 200).clip(-32768, 32767).astype(np.int16)

    samples = []
    symbols = []
    for i in range(n_beats):
        idx = (i + 1) * beat_gap
        samples.append(idx)
        symbols.append("N" if i < n_normal else "V")

    wfdb.wrsamp(
        rec_id,
        fs=360,
        units=["mV"],
        sig_name=["MLII"],
        d_signal=sig_d.reshape(-1, 1),
        fmt=["16"],
        adc_gain=[200.0],
        baseline=[0],
        write_dir=str(root),
    )
    wfdb.wrann(
        rec_id,
        "atr",
        sample=np.array(samples),
        symbol=symbols,
        write_dir=str(root),
    )


@pytest.fixture(scope="module")
def mitbih_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("mitbih")
    rng  = np.random.default_rng(0)
    # DS1: two records
    #   reca: 200 normal, 80 arrhythmia  (both ≥ 50)
    #   recb: 150 normal,  5 arrhythmia  (arrhythmia < 50 → excluded)
    _write_record(root, "reca", 200, 80,  rng)
    _write_record(root, "recb", 150,  5,  rng)
    # DS2: one record
    _write_record(root, "recc", 100, 60,  rng)
    return root


def _loader(root, ds1=("reca", "recb"), ds2=("recc",), **kw):
    """Build a loader with patched DS1/DS2 pointing to the synthetic records."""
    import nstad_bench.data.mitbih_loader as m
    orig_ds1, orig_ds2 = m.DS1, m.DS2
    m.DS1, m.DS2 = ds1, ds2
    try:
        fn = m.mitbih_loader(data_root=root, **kw)
    finally:
        m.DS1, m.DS2 = orig_ds1, orig_ds2
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:
    def test_returns_four(self, mitbih_root):
        assert len(_loader(mitbih_root)()) == 4

    def test_shapes(self, mitbih_root):
        X_s, y_s, X_t, y_t = _loader(mitbih_root)()
        assert X_s.ndim == 2 and X_s.shape[1] == 280
        assert X_t.ndim == 2 and X_t.shape[1] == 280
        assert X_s.shape[0] == y_s.shape[0]
        assert X_t.shape[0] == y_t.shape[0]

    def test_dtypes(self, mitbih_root):
        X_s, y_s, X_t, y_t = _loader(mitbih_root)()
        assert X_s.dtype == np.float32
        assert y_s.dtype == np.int64

    def test_binary_labels(self, mitbih_root):
        _, y_s, _, y_t = _loader(mitbih_root)()
        assert set(np.unique(y_s)).issubset({0, 1})
        assert set(np.unique(y_t)).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampling — min_beats filter
# ─────────────────────────────────────────────────────────────────────────────

class TestStratification:
    def test_low_beat_record_excluded_from_arr(self, mitbih_root):
        """recb has only 5 arrhythmia beats → excluded; source arr total = 80."""
        _, y_s, _, _ = _loader(
            mitbih_root,
            max_per_class=999,
            min_beats_per_record=50,
        )()
        assert int((y_s == 1).sum()) == 80   # only reca's 80 arrhythmia beats

    def test_low_beat_record_included_when_filter_lowered(self, mitbih_root):
        """With min_beats_per_record=1, recb's 5 arrhythmia beats are included."""
        _, y_s, _, _ = _loader(
            mitbih_root,
            max_per_class=999,
            min_beats_per_record=1,
        )()
        # reca(80) + recb(5) = 85, capped at min(200+150, 80+5, 999) = 85
        assert int((y_s == 1).sum()) == 85

    def test_cap_respected(self, mitbih_root):
        cap = 40
        _, y_s, _, _ = _loader(
            mitbih_root,
            max_per_class=cap,
            min_beats_per_record=1,
        )()
        assert int((y_s == 0).sum()) <= cap
        assert int((y_s == 1).sum()) <= cap

    def test_allocation_does_not_exceed_available(self, mitbih_root):
        """No record can contribute more beats than it has."""
        # This is tested implicitly: if it did, wfdb indexing would error.
        X_s, y_s, _, _ = _loader(
            mitbih_root,
            max_per_class=999,
            min_beats_per_record=1,
        )()
        assert len(y_s) > 0


# ─────────────────────────────────────────────────────────────────────────────
# _stratified_allocation unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStratifiedAllocation:
    def _alloc(self, counts, total):
        from nstad_bench.data.mitbih_loader import _stratified_allocation
        return _stratified_allocation(counts, total)

    def test_total_equals_target(self):
        counts = {"a": 100, "b": 200, "c": 300}
        alloc  = self._alloc(counts, 150)
        assert sum(alloc.values()) == 150

    def test_does_not_exceed_available(self):
        counts = {"a": 10, "b": 500}
        alloc  = self._alloc(counts, 300)
        for rec, n in alloc.items():
            assert n <= counts[rec]

    def test_small_record_gets_all(self):
        """A record with fewer beats than its share contributes everything."""
        counts = {"small": 20, "big": 1000}
        alloc  = self._alloc(counts, 200)
        assert alloc["small"] == 20

    def test_equal_distribution_when_all_large(self):
        counts = {"a": 1000, "b": 1000, "c": 1000}
        alloc  = self._alloc(counts, 300)
        assert alloc == {"a": 100, "b": 100, "c": 100}

    def test_empty_counts_returns_empty(self):
        assert self._alloc({}, 100) == {}

    def test_target_larger_than_total_available(self):
        """When target > sum(counts), take everything."""
        counts = {"a": 50, "b": 30}
        alloc  = self._alloc(counts, 9999)
        assert alloc["a"] == 50
        assert alloc["b"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed(self, mitbih_root):
        import nstad_bench.data.mitbih_loader as m
        orig_ds1, orig_ds2 = m.DS1, m.DS2
        m.DS1, m.DS2 = ("reca", "recb"), ("recc",)
        try:
            a = m.mitbih_loader(data_root=mitbih_root, seed=3)
            b = m.mitbih_loader(data_root=mitbih_root, seed=3)
            np.testing.assert_array_equal(a()[0], b()[0])
        finally:
            m.DS1, m.DS2 = orig_ds1, orig_ds2


# ─────────────────────────────────────────────────────────────────────────────
# Missing data
# ─────────────────────────────────────────────────────────────────────────────

class TestMissing:
    def test_missing_record_skipped(self, tmp_path):
        """A missing record should log a warning and be skipped gracefully."""
        import nstad_bench.data.mitbih_loader as m
        orig_ds1, orig_ds2 = m.DS1, m.DS2
        m.DS1 = ("nonexistent_record",)
        m.DS2 = ("nonexistent_record2",)
        try:
            loader = m.mitbih_loader(data_root=tmp_path, max_per_class=10)
            X_s, y_s, X_t, y_t = loader()
            # no records loaded → empty arrays
            assert len(y_s) == 0
            assert len(y_t) == 0
        finally:
            m.DS1, m.DS2 = orig_ds1, orig_ds2

"""Tests for nstad_bench.data.deepbeat_loader.

Uses a tiny synthetic .npz fixture that matches the real DeepBeat schema:
  signal     (N, 800, 1)  float32
  rhythm     (N, 2)       float64  one-hot [SR, AF]
  parameters (N, 3)       object   [timestamp, session_id, patient_id]
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_N_TRAIN = 300   # windows per class in synthetic train
_N_TEST  = 40

def _make_npz(path: Path, n_sr: int, n_af: int, patient_start: int) -> None:
    rng = np.random.default_rng(42)
    n   = n_sr + n_af

    signal = rng.standard_normal((n, 800, 1)).astype(np.float32)

    rhythm = np.zeros((n, 2), dtype=np.float64)
    rhythm[:n_sr, 0] = 1.0   # SR
    rhythm[n_sr:, 1] = 1.0   # AF

    # parameters: [timestamp, session_id, patient_id]
    import pandas as pd
    ts = pd.Timestamp("2020-01-01")
    params = np.empty((n, 3), dtype=object)
    params[:, 0] = ts
    params[:, 1] = " a"
    params[:, 2] = np.repeat(
        np.arange(patient_start, patient_start + 5), n // 5 + 1
    )[:n]

    np.savez(path, signal=signal, rhythm=rhythm,
             qa_label=np.zeros((n, 3), dtype=np.float32),
             parameters=params)


@pytest.fixture(scope="module")
def deepbeat_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("deepbeat")
    _make_npz(root / "train.npz", n_sr=_N_TRAIN, n_af=_N_TRAIN, patient_start=1)
    _make_npz(root / "test.npz",  n_sr=_N_TEST,  n_af=_N_TEST,  patient_start=100)
    return root


def _loader(root, **kw):
    from nstad_bench.data.deepbeat_loader import deepbeat_loader
    return deepbeat_loader(data_root=root, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:
    def test_returns_four_arrays(self, deepbeat_root):
        assert len(_loader(deepbeat_root)()) == 4

    def test_shapes(self, deepbeat_root):
        X_s, y_s, X_t, y_t = _loader(deepbeat_root)()
        assert X_s.ndim == 2 and X_s.shape[1] == 800
        assert X_t.ndim == 2 and X_t.shape[1] == 800
        assert X_s.shape[0] == y_s.shape[0]
        assert X_t.shape[0] == y_t.shape[0]

    def test_dtypes(self, deepbeat_root):
        X_s, y_s, X_t, y_t = _loader(deepbeat_root)()
        assert X_s.dtype == np.float32
        assert X_t.dtype == np.float32
        assert y_s.dtype == np.int64
        assert y_t.dtype == np.int64

    def test_binary_labels(self, deepbeat_root):
        _, y_s, _, y_t = _loader(deepbeat_root)()
        assert set(np.unique(y_s)).issubset({0, 1})
        assert set(np.unique(y_t)).issubset({0, 1})

    def test_both_classes_present(self, deepbeat_root):
        _, y_s, _, y_t = _loader(deepbeat_root)()
        assert 0 in y_s and 1 in y_s
        assert 0 in y_t and 1 in y_t


# ─────────────────────────────────────────────────────────────────────────────
# Source vs target counts
# ─────────────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_source_uses_train(self, deepbeat_root):
        """Source total ≤ 2 × max_per_class (from train.npz)."""
        _, y_s, _, _ = _loader(deepbeat_root, max_per_class=50)()
        assert len(y_s) <= 100

    def test_target_uses_test(self, deepbeat_root):
        """Target contains all test windows (no cap on target)."""
        _, _, _, y_t = _loader(deepbeat_root)()
        assert len(y_t) == _N_TEST * 2   # n_sr + n_af from test.npz

    def test_source_capped(self, deepbeat_root):
        cap = 10
        _, y_s, _, _ = _loader(deepbeat_root, max_per_class=cap)()
        assert (y_s == 0).sum() <= cap
        assert (y_s == 1).sum() <= cap

    def test_target_not_capped(self, deepbeat_root):
        """Target is never sub-sampled — natural distribution preserved."""
        _, _, _, y_t = _loader(deepbeat_root, max_per_class=5)()
        assert len(y_t) == _N_TEST * 2


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed(self, deepbeat_root):
        from nstad_bench.data.deepbeat_loader import deepbeat_loader
        a = deepbeat_loader(data_root=deepbeat_root, seed=7, max_per_class=20)
        b = deepbeat_loader(data_root=deepbeat_root, seed=7, max_per_class=20)
        np.testing.assert_array_equal(a()[0], b()[0])

    def test_different_seeds_differ(self, deepbeat_root):
        from nstad_bench.data.deepbeat_loader import deepbeat_loader
        a = deepbeat_loader(data_root=deepbeat_root, seed=0,  max_per_class=20)
        b = deepbeat_loader(data_root=deepbeat_root, seed=99, max_per_class=20)
        assert not np.array_equal(a()[0], b()[0])


# ─────────────────────────────────────────────────────────────────────────────
# Missing data
# ─────────────────────────────────────────────────────────────────────────────

class TestMissing:
    def test_missing_train_raises(self, tmp_path):
        from nstad_bench.data.deepbeat_loader import deepbeat_loader
        # only test.npz present
        _make_npz(tmp_path / "test.npz", 20, 20, 100)
        with pytest.raises(FileNotFoundError, match="train.npz"):
            deepbeat_loader(data_root=tmp_path)()

    def test_missing_test_raises(self, tmp_path):
        from nstad_bench.data.deepbeat_loader import deepbeat_loader
        _make_npz(tmp_path / "train.npz", 20, 20, 1)
        with pytest.raises(FileNotFoundError, match="test.npz"):
            deepbeat_loader(data_root=tmp_path)()


# ─────────────────────────────────────────────────────────────────────────────
# _balanced_subsample helper
# ─────────────────────────────────────────────────────────────────────────────

class TestBalancedSubsample:
    def test_caps_each_class(self):
        from nstad_bench.data.deepbeat_loader import _balanced_subsample
        rng = np.random.default_rng(0)
        y = np.array([0]*100 + [1]*200)
        idx = _balanced_subsample(y, max_per_class=30, rng=rng)
        y_sub = y[idx]
        assert (y_sub == 0).sum() == 30
        assert (y_sub == 1).sum() == 30

    def test_no_cap_when_under_limit(self):
        from nstad_bench.data.deepbeat_loader import _balanced_subsample
        rng = np.random.default_rng(0)
        y = np.array([0]*10 + [1]*10)
        idx = _balanced_subsample(y, max_per_class=50, rng=rng)
        assert len(idx) == 20

    def test_output_is_shuffled(self):
        from nstad_bench.data.deepbeat_loader import _balanced_subsample
        rng = np.random.default_rng(0)
        y = np.array([0]*50 + [1]*50)
        idx = _balanced_subsample(y, max_per_class=50, rng=rng)
        # should not be purely [0,0,...,1,1,...] after shuffle
        labels = y[idx]
        assert not (np.all(labels[:50] == 0) and np.all(labels[50:] == 1))

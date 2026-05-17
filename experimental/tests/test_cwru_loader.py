"""Tests for nstad_bench.data.cwru_loader (cross-severity split).

Fixture layout
--------------
    cwru_root/
        Normal_0HP.mat          ← shared healthy baseline
        B007_DE_0HP.mat  }
        IR007_DE_0HP.mat }      source faults (0.007")
        OR007@6_DE_0HP.mat }
        B014_DE_0HP.mat  }
        IR014_DE_0HP.mat }      target faults (0.014")
        OR014@6_DE_0HP.mat }
        B021_DE_0HP.mat  }
        IR021_DE_0HP.mat }      target faults (0.021")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")

WIN    = 1024
_N_WINS = 8   # windows per synthetic file

# Source fault stems used in the fixture (subset of full list)
_SRC_STEMS = ["B007_DE", "IR007_DE", "OR007@6_DE"]
# Target fault stems used in the fixture
_TGT_STEMS = ["B014_DE", "IR014_DE", "OR014@6_DE", "B021_DE", "IR021_DE"]


def _write_mat(path: Path, n_windows: int = _N_WINS) -> None:
    rng = np.random.default_rng(int(path.stem.encode("utf-8").hex(), 16) % 2**31)
    de_time = rng.standard_normal(n_windows * WIN).astype(np.float64)
    sio.savemat(str(path), {"DE_time": de_time})


@pytest.fixture(scope="module")
def cwru_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cwru_severity")
    _write_mat(root / "Normal_0HP.mat")
    for stem in _SRC_STEMS:
        _write_mat(root / f"{stem}_0HP.mat")
    for stem in _TGT_STEMS:
        _write_mat(root / f"{stem}_0HP.mat")
    return root


@pytest.fixture(scope="module")
def cwru_root_no_target_faults(tmp_path_factory) -> Path:
    """Fixture with only Normal + source faults — no target fault files."""
    root = tmp_path_factory.mktemp("cwru_no_tgt")
    _write_mat(root / "Normal_0HP.mat")
    for stem in _SRC_STEMS:
        _write_mat(root / f"{stem}_0HP.mat")
    return root


def _loader(root, **kw):
    from nstad_bench.data.cwru_loader import cwru_loader
    return cwru_loader(data_root=root, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:
    def test_returns_four(self, cwru_root):
        assert len(_loader(cwru_root)()) == 4

    def test_shapes(self, cwru_root):
        X_s, y_s, X_t, y_t = _loader(cwru_root)()
        assert X_s.ndim == 2 and X_s.shape[1] == WIN
        assert X_t.ndim == 2 and X_t.shape[1] == WIN
        assert X_s.shape[0] == y_s.shape[0]
        assert X_t.shape[0] == y_t.shape[0]

    def test_dtypes(self, cwru_root):
        X_s, y_s, X_t, y_t = _loader(cwru_root)()
        assert X_s.dtype == np.float32
        assert y_s.dtype == np.int64

    def test_binary_labels(self, cwru_root):
        _, y_s, _, y_t = _loader(cwru_root)()
        assert set(np.unique(y_s)).issubset({0, 1})
        assert set(np.unique(y_t)).issubset({0, 1})

    def test_both_classes_source(self, cwru_root):
        _, y_s, _, _ = _loader(cwru_root)()
        assert 0 in y_s and 1 in y_s

    def test_both_classes_target(self, cwru_root):
        _, _, _, y_t = _loader(cwru_root)()
        assert 0 in y_t and 1 in y_t


# ─────────────────────────────────────────────────────────────────────────────
# Label correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestLabels:
    def test_normal_is_label_zero(self, cwru_root):
        X_s, y_s, _, _ = _loader(cwru_root)()
        assert (y_s == 0).sum() >= 1

    def test_source_fault_is_label_one(self, cwru_root):
        # balance=False to see raw counts: 3 source fault files × _N_WINS each
        _, y_s, _, _ = _loader(cwru_root, balance=False)()
        assert (y_s == 1).sum() >= _N_WINS * len(_SRC_STEMS)

    def test_target_fault_is_label_one(self, cwru_root):
        _, _, _, y_t = _loader(cwru_root, balance=False)()
        assert (y_t == 1).sum() >= _N_WINS * len(_TGT_STEMS)

    def test_source_and_target_share_normal(self, cwru_root):
        """Both domains load from the same Normal_0HP file."""
        # After balancing, both should have the same healthy window count
        # (the normal file has _N_WINS windows which sets the balance cap).
        _, y_s, _, y_t = _loader(cwru_root, balance=True)()
        assert int((y_s == 0).sum()) == int((y_t == 0).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Balancing
# ─────────────────────────────────────────────────────────────────────────────

class TestBalance:
    def test_balance_true_equalises_classes(self, cwru_root):
        _, y_s, _, y_t = _loader(cwru_root, balance=True)()
        assert (y_s == 0).sum() == (y_s == 1).sum()
        assert (y_t == 0).sum() == (y_t == 1).sum()

    def test_balance_false_preserves_imbalance(self, cwru_root):
        """With balance=False, fault windows outnumber healthy in both domains."""
        _, y_s, _, y_t = _loader(cwru_root, balance=False)()
        assert (y_s == 1).sum() > (y_s == 0).sum()
        assert (y_t == 1).sum() > (y_t == 0).sum()

    def test_balance_skipped_when_target_faults_absent(self, cwru_root_no_target_faults):
        """If target fault files are missing, balance=True must not empty the domain."""
        loader = _loader(cwru_root_no_target_faults, balance=True)
        _, _, _, y_t = loader()
        # target has only Normal — label 0 — balance skipped, domain not emptied
        assert len(y_t) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNorm:
    def test_per_window_zscore(self, cwru_root):
        X_s, _, _, _ = _loader(cwru_root)()
        means = X_s.mean(axis=1)
        stds  = X_s.std(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(stds,  1.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Missing files
# ─────────────────────────────────────────────────────────────────────────────

class TestMissing:
    def test_missing_normal_raises(self, tmp_path):
        from nstad_bench.data.cwru_loader import cwru_loader
        with pytest.raises(FileNotFoundError, match="Normal_0HP"):
            cwru_loader(data_root=tmp_path)()

    def test_missing_target_faults_warns_but_succeeds(self, cwru_root_no_target_faults):
        """Target fault files absent → loader succeeds, target has only Normal."""
        loader = _loader(cwru_root_no_target_faults, balance=False)
        # must not raise
        X_s, y_s, X_t, y_t = loader()
        assert set(np.unique(y_t)) == {0}   # only Normal in target


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed(self, cwru_root):
        from nstad_bench.data.cwru_loader import cwru_loader
        a = cwru_loader(data_root=cwru_root, seed=3)
        b = cwru_loader(data_root=cwru_root, seed=3)
        np.testing.assert_array_equal(a()[0], b()[0])

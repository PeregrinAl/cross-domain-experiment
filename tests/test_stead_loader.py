"""Tests for nstad_bench.data.stead_loader.

All tests use an in-memory synthetic HDF5 fixture that matches the real STEAD
schema (shape (6000, 3) per waveform, ``network_code`` attribute).  No real
data download is required.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic HDF5 fixtures
# ─────────────────────────────────────────────────────────────────────────────

_NETWORKS_EQ = {
    "CI": 120,   # SoCal — dominates source
    "AZ": 40,    # SoCal
    "NC": 60,    # NorCal — target
    "BK": 20,    # NorCal
    "NN": 15,    # Intermountain West — excluded
}
# Noise pool must be ≥ 2 × max(src_eq, tgt_eq) so that after a 50/50 split
# neither half bottlenecks the earthquake count.
# src_eq = 160, tgt_eq = 80  → need ≥ 2 × 160 = 320 noise traces.
_N_NOISE = 400


def _write_eq_chunk(path: Path) -> None:
    """Write a synthetic earthquake HDF5 chunk with known network counts."""
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as fh:
        grp = fh.create_group("data")
        idx = 0
        for net, count in _NETWORKS_EQ.items():
            for _ in range(count):
                key = f"ev_{idx:06d}"
                ds = grp.create_dataset(key, data=rng.standard_normal((6000, 3)).astype(np.float32))
                ds.attrs["network_code"] = net.encode()
                ds.attrs["trace_category"] = b"earthquake_local"
                idx += 1


def _write_noise_chunk(path: Path) -> None:
    """Write a synthetic noise HDF5 chunk (no network_code filter needed)."""
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as fh:
        grp = fh.create_group("data")
        for i in range(_N_NOISE):
            key = f"no_{i:06d}"
            ds = grp.create_dataset(key, data=rng.standard_normal((6000, 3)).astype(np.float32))
            ds.attrs["network_code"] = b"CI"   # noise labelled under CI but ignored
            ds.attrs["trace_category"] = b"noise"


@pytest.fixture(scope="module")
def stead_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("stead_data")
    _write_eq_chunk(root / "chunk2.hdf5")
    _write_noise_chunk(root / "chunk1.hdf5")
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(root, **kwargs):
    from nstad_bench.data.stead_loader import stead_nc_sc_loader
    return stead_nc_sc_loader(data_root=root, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — basic contract
# ─────────────────────────────────────────────────────────────────────────────

class TestLoaderContract:

    def test_returns_four_arrays(self, stead_root):
        loader = _make_loader(stead_root)
        result = loader()
        assert len(result) == 4

    def test_shapes_consistent(self, stead_root):
        X_s, y_s, X_t, y_t = _make_loader(stead_root)()
        assert X_s.ndim == 2
        assert X_s.shape[1] == 6000
        assert X_s.shape[0] == y_s.shape[0]
        assert X_t.shape[0] == y_t.shape[0]

    def test_dtypes(self, stead_root):
        X_s, y_s, X_t, y_t = _make_loader(stead_root)()
        assert X_s.dtype == np.float32
        assert X_t.dtype == np.float32
        assert y_s.dtype == np.int64
        assert y_t.dtype == np.int64

    def test_labels_binary(self, stead_root):
        _, y_s, _, y_t = _make_loader(stead_root)()
        assert set(np.unique(y_s)).issubset({0, 1})
        assert set(np.unique(y_t)).issubset({0, 1})

    def test_both_classes_in_source(self, stead_root):
        _, y_s, _, _ = _make_loader(stead_root)()
        assert 0 in y_s and 1 in y_s, "Source must contain both eq (1) and noise (0)"

    def test_both_classes_in_target(self, stead_root):
        _, _, _, y_t = _make_loader(stead_root)()
        assert 0 in y_t and 1 in y_t, "Target must contain both eq (1) and noise (0)"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — geographic split correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestGeographicSplit:

    def test_source_earthquake_count(self, stead_root):
        """Source earthquakes come from CI+AZ only (120+40 = 160 traces)."""
        X_s, y_s, _, _ = _make_loader(stead_root, max_per_class=999)()
        src_eq_expected = _NETWORKS_EQ["CI"] + _NETWORKS_EQ["AZ"]
        src_eq_actual = int((y_s == 1).sum())
        assert src_eq_actual == src_eq_expected

    def test_target_earthquake_count(self, stead_root):
        """Target earthquakes come from NC+BK only (60+20 = 80 traces)."""
        _, _, _, y_t = _make_loader(stead_root, max_per_class=999)()
        tgt_eq_expected = _NETWORKS_EQ["NC"] + _NETWORKS_EQ["BK"]
        tgt_eq_actual = int((y_t == 1).sum())
        assert tgt_eq_actual == tgt_eq_expected

    def test_excluded_network_absent(self, stead_root):
        """NN (Intermountain West) must appear in neither domain."""
        # We can't inspect network_code from the returned arrays directly, but
        # the total earthquake count must equal SoCal + NorCal only.
        X_s, y_s, X_t, y_t = _make_loader(stead_root, max_per_class=999)()
        total_eq = int((y_s == 1).sum()) + int((y_t == 1).sum())
        expected_total = sum(
            v for k, v in _NETWORKS_EQ.items() if k in {"CI", "AZ", "NC", "BK"}
        )
        assert total_eq == expected_total

    def test_noise_split_between_domains(self, stead_root):
        """Both source and target must contain noise traces."""
        _, y_s, _, y_t = _make_loader(stead_root, max_per_class=999)()
        assert (y_s == 0).sum() > 0
        assert (y_t == 0).sum() > 0

    def test_source_target_disjoint_earthquakes(self, stead_root):
        """Source earthquake count + target earthquake count = total available eq."""
        X_s, y_s, X_t, y_t = _make_loader(stead_root, max_per_class=999)()
        src_eq = int((y_s == 1).sum())
        tgt_eq = int((y_t == 1).sum())
        available = sum(
            v for k, v in _NETWORKS_EQ.items() if k in {"CI", "AZ", "NC", "BK"}
        )
        assert src_eq + tgt_eq == available


# ─────────────────────────────────────────────────────────────────────────────
# Tests — max_per_class capping
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxPerClass:

    def test_eq_capped_in_source(self, stead_root):
        cap = 50
        _, y_s, _, _ = _make_loader(stead_root, max_per_class=cap)()
        assert (y_s == 1).sum() <= cap

    def test_eq_capped_in_target(self, stead_root):
        cap = 30
        _, _, _, y_t = _make_loader(stead_root, max_per_class=cap)()
        assert (y_t == 1).sum() <= cap

    def test_classes_balanced_eq_more_than_noise(self, tmp_path_factory):
        """When noise < earthquakes, eq is capped DOWN to noise count."""
        # Build a tiny fixture with very few noise traces so the balancing
        # code must cut the earthquake side.
        root = tmp_path_factory.mktemp("stead_unbal")
        rng  = np.random.default_rng(3)
        with __import__("h5py").File(root / "chunk2.hdf5", "w") as fh:
            grp = fh.create_group("data")
            for i in range(40):  # 40 earthquakes
                ds = grp.create_dataset(f"ev_{i:04d}",
                    data=rng.standard_normal((6000, 3)).astype(np.float32))
                ds.attrs["network_code"] = b"CI"
                ds.attrs["trace_category"] = b"earthquake_local"
        with __import__("h5py").File(root / "chunk1.hdf5", "w") as fh:
            grp = fh.create_group("data")
            for i in range(10):  # only 10 noise traces → each split half gets 5
                ds = grp.create_dataset(f"no_{i:04d}",
                    data=rng.standard_normal((6000, 3)).astype(np.float32))
                ds.attrs["trace_category"] = b"noise"

        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        loader = stead_nc_sc_loader(
            data_root=root,
            source_networks=frozenset({"CI"}),
            target_networks=frozenset({"CI"}),  # same network → both get all 40 eq
            max_per_class=999,
        )
        _, y_s, _, _ = loader()
        # After 50/50 noise split: 5 noise per domain → eq capped to 5
        assert int((y_s == 0).sum()) == int((y_s == 1).sum())

    def test_classes_balanced_after_large_cap(self, stead_root):
        """With abundant noise (large fixture), eq and noise counts must match."""
        _, y_s, _, y_t = _make_loader(stead_root, max_per_class=999)()
        assert int((y_s == 0).sum()) == int((y_s == 1).sum())
        assert int((y_t == 0).sum()) == int((y_t == 1).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Tests — normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalisation:

    def test_z_score_applied(self, stead_root):
        """Each trace should have near-zero mean and unit variance after z-score."""
        X_s, _, _, _ = _make_loader(stead_root, max_per_class=20)()
        # Per-trace mean and std
        means = X_s.mean(axis=1)
        stds  = X_s.std(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-4)
        np.testing.assert_allclose(stds,  1.0, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_output(self, stead_root):
        """Two loaders with the same seed must return identical arrays."""
        # lru_cache makes the SAME loader instance always return the same result;
        # use two *different* factory calls (fresh lru_cache each time).
        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        loader_a = stead_nc_sc_loader(data_root=stead_root, seed=1, max_per_class=40)
        loader_b = stead_nc_sc_loader(data_root=stead_root, seed=1, max_per_class=40)
        Xa, ya, _, _ = loader_a()
        Xb, yb, _, _ = loader_b()
        np.testing.assert_array_equal(Xa, Xb)
        np.testing.assert_array_equal(ya, yb)

    def test_different_seeds_may_differ(self, stead_root):
        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        # Use a cap that actually triggers sub-sampling (>available traces = no diff)
        loader_a = stead_nc_sc_loader(data_root=stead_root, seed=0, max_per_class=50)
        loader_b = stead_nc_sc_loader(data_root=stead_root, seed=99, max_per_class=50)
        Xa, _, _, _ = loader_a()
        Xb, _, _, _ = loader_b()
        # With high probability the random permutation differs
        # (not guaranteed for tiny arrays, but seed=0 vs seed=99 should differ)
        assert not np.array_equal(Xa, Xb) or len(Xa) < 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests — missing data
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingData:

    def test_missing_chunk_raises(self, tmp_path):
        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        loader = stead_nc_sc_loader(data_root=tmp_path)
        with pytest.raises(FileNotFoundError, match="chunk2.hdf5"):
            loader()


# ─────────────────────────────────────────────────────────────────────────────
# Tests — custom network groups
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomNetworks:

    def test_custom_source_only_ci(self, stead_root):
        """Custom source = CI only."""
        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        loader = stead_nc_sc_loader(
            data_root=stead_root,
            source_networks=frozenset({"CI"}),
            target_networks=frozenset({"NC"}),
            max_per_class=999,
        )
        _, y_s, _, _ = loader()
        assert (y_s == 1).sum() == _NETWORKS_EQ["CI"]

    def test_empty_target_network_returns_zero_eq(self, stead_root):
        """Target set with no matching waveforms → zero earthquake traces."""
        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        loader = stead_nc_sc_loader(
            data_root=stead_root,
            source_networks=frozenset({"CI"}),
            target_networks=frozenset({"XX"}),  # no such network in fixture
            max_per_class=999,
        )
        _, _, _, y_t = loader()
        assert (y_t == 1).sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — _attr_str helper
# ─────────────────────────────────────────────────────────────────────────────

class TestAttrStr:

    def test_bytes(self):
        from nstad_bench.data.stead_loader import _attr_str
        assert _attr_str(b"CI") == "CI"

    def test_str(self):
        from nstad_bench.data.stead_loader import _attr_str
        assert _attr_str("NC") == "NC"

    def test_numpy_bytes_scalar(self):
        from nstad_bench.data.stead_loader import _attr_str
        val = np.bytes_(b"BK")
        assert _attr_str(val) == "BK"

    def test_numpy_str_scalar(self):
        from nstad_bench.data.stead_loader import _attr_str
        val = np.str_("AZ")
        assert _attr_str(val) == "AZ"

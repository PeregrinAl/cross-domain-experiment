"""Tests for nstad_bench.data.cwru_loader — the cross-load 0HP→3HP variant.

Does not require real CWRU .mat files: ``_read_de_signal`` is monkey-patched
to return synthetic vibration traces from filename patterns under a tmp dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nstad_bench.data import cwru_loader as cwru_mod


def _make_fake_root(tmp_path: Path) -> Path:
    """Create empty .mat files using brjapon/Kaggle naming convention."""
    root = tmp_path / "cwru"
    root.mkdir()
    for stem in (
        # 0 HP — brjapon format: <Type>_<HP>_<FileID>
        "Time_Normal_0_097", "B007_0_122", "IR007_0_109", "OR007_6_0_135",
        # 3 HP
        "Time_Normal_3_262", "B007_3_237", "IR007_3_274", "OR007_6_3_250",
        # noise file with no load pattern — must be ignored
        "fan_end_summary",
    ):
        (root / f"{stem}.mat").touch()
    return root


def _fake_reader(rng_seed_from_path):
    """Returns a fake _read_de_signal that derives its signal from the filename."""
    def _read(path: Path) -> np.ndarray:
        # Healthy signals are tiny gaussians; fault signals add periodic impulses.
        rng = np.random.default_rng(hash(path.stem) % (2 ** 32))
        n = 8192
        sig = rng.normal(0, 1, n).astype(np.float32)
        if "Normal" not in path.stem:
            impulse_idx = np.arange(50, n, 200)
            sig[impulse_idx] += 5.0   # fault impulses
        return sig
    return _read


@pytest.fixture()
def patched_reader(monkeypatch):
    monkeypatch.setattr(cwru_mod, "_read_de_signal", _fake_reader(None))
    yield


def test_discovers_files_by_hp_suffix(patched_reader, tmp_path):
    root = _make_fake_root(tmp_path)
    normal_0, fault_0 = cwru_mod._discover_files(root, "0")
    normal_3, fault_3 = cwru_mod._discover_files(root, "3")
    assert {p.stem for p in normal_0} == {"Time_Normal_0_097"}
    assert {p.stem for p in normal_3} == {"Time_Normal_3_262"}
    assert len(fault_0) == 3 and len(fault_3) == 3
    # fan_end_summary has no load pattern — must be absent from all lists
    all_found = normal_0 + fault_0 + normal_3 + fault_3
    assert all("fan_end_summary" not in p.stem for p in all_found)


def test_loader_returns_correct_shapes(patched_reader, tmp_path):
    root = _make_fake_root(tmp_path)
    loader = cwru_mod.cwru_loader(data_root=root, seed=0, max_per_class=20)
    X_s, y_s, X_t, y_t = loader()
    # Windows are 1024 samples
    assert X_s.shape[1] == cwru_mod.WIN == 1024
    assert X_t.shape[1] == 1024
    assert X_s.dtype == np.float32
    assert y_s.dtype == np.int64
    # Both classes present in each domain
    assert set(np.unique(y_s)) == {0, 1}
    assert set(np.unique(y_t)) == {0, 1}
    # Per-domain balance respected
    assert (y_s == 0).sum() == (y_s == 1).sum()
    assert (y_t == 0).sum() == (y_t == 1).sum()


def test_explicit_data_root_overrides_env(monkeypatch, patched_reader, tmp_path):
    root = _make_fake_root(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "nonexistent"))
    monkeypatch.setenv("NSTAD_DATA_ROOT", str(tmp_path / "also-nonexistent"))
    # Should still find the files because data_root is explicit
    loader = cwru_mod.cwru_loader(data_root=root, seed=0, max_per_class=10)
    X_s, *_ = loader()
    assert X_s.shape[0] > 0


def test_missing_root_raises_helpful_error(patched_reader, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    loader = cwru_mod.cwru_loader(data_root=empty, seed=0)
    with pytest.raises(FileNotFoundError, match="No CWRU .mat files found"):
        loader()


def test_slice_windows_z_score_normalises():
    sig = np.linspace(0, 100, 5000, dtype=np.float32)
    wins = cwru_mod._slice_windows(sig, win=1024, stride=512)
    assert wins.shape[0] == 1 + (5000 - 1024) // 512
    assert wins.shape[1] == 1024
    # Each window has mean ≈ 0 and std ≈ 1
    np.testing.assert_allclose(wins.mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(wins.std(axis=1), 1, atol=1e-3)


def test_stride_overlap_matches_spec():
    """Windows should overlap by 512 samples per the task spec."""
    sig = np.zeros(2048, dtype=np.float32)
    sig[0] = 1.0     # mark sample 0
    sig[512] = 1.0   # mark sample 512 — second window's first sample if stride=512
    wins = cwru_mod._slice_windows(sig, win=1024, stride=512)
    assert wins.shape[0] == 3   # starts at 0, 512, 1024

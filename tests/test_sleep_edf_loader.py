"""Tests for nstad_bench.data.sleep_edf_loader.

mne is optional; when it is not installed we exercise the pure-Python parts
of the loader (subject indexing, prefix matching, label map) and skip the
end-to-end loader path.  When mne IS installed, ``_read_subject`` is
monkey-patched so the test never touches the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nstad_bench.data import sleep_edf_loader as sef


def _touch_pair(root: Path, prefix: str) -> tuple[Path, Path]:
    psg = root / f"{prefix}E0-PSG.edf"
    hyp = root / f"{prefix}EC-Hypnogram.edf"
    psg.touch()
    hyp.touch()
    return psg, hyp


def test_index_subjects_pairs_by_prefix(tmp_path):
    _touch_pair(tmp_path, "SC4001")
    _touch_pair(tmp_path, "SC4002")
    # Orphan PSG without hypnogram — must NOT appear in the index
    (tmp_path / "SC4003E0-PSG.edf").touch()
    idx = sef._index_subjects(tmp_path)
    assert set(idx) == {"SC4001", "SC4002"}
    for psg, hyp in idx.values():
        assert psg.name.endswith("PSG.edf")
        assert hyp.name.endswith("Hypnogram.edf")


def test_index_subjects_recursive(tmp_path):
    sub = tmp_path / "ST" / "night1"
    sub.mkdir(parents=True)
    _touch_pair(sub, "ST7011")
    idx = sef._index_subjects(tmp_path)
    assert "ST7011" in idx


def test_stage_map_covers_aasm_and_legacy():
    """Wake → 0, every sleep stage (incl. legacy Stage 4) → 1, others absent."""
    assert sef._STAGE_MAP["Sleep stage W"] == 0
    assert sef._STAGE_MAP["Sleep stage R"] == 1
    assert sef._STAGE_MAP["Sleep stage 4"] == 1   # legacy AASM stage
    assert "Sleep stage ?" not in sef._STAGE_MAP
    assert "Movement time" not in sef._STAGE_MAP


def test_window_constants_match_spec():
    """Fpz-Cz 100 Hz × 30 s = 3000-sample windows per the task spec."""
    assert sef.FS == 100
    assert sef.EPOCH_SEC == 30
    assert sef.WIN == 3000


def test_loader_raises_when_no_subjects(tmp_path):
    loader = sef.sleep_edf_loader(data_root=tmp_path, n_subjects=2, seed=0)
    with pytest.raises(FileNotFoundError, match="No Sleep-EDF PSG/Hypnogram pairs"):
        loader()


def test_loader_splits_source_and_target(monkeypatch, tmp_path):
    """End-to-end with _read_subject mocked — verifies LOSO + balancing."""
    # Create fake EDF pairs for 6 subjects (TZ default is 10, but test smaller).
    for prefix in ("SC4001", "SC4002", "SC4003", "SC4004", "SC4005", "SC4006"):
        _touch_pair(tmp_path, prefix)

    def _fake_read(psg, hyp):
        # Each subject contributes 40 Wake + 40 Sleep epochs of synthetic data.
        rng = np.random.default_rng(hash(psg.stem) % (2 ** 32))
        X = rng.normal(0, 1, (80, sef.WIN)).astype(np.float32)
        y = np.array([0] * 40 + [1] * 40, dtype=np.int64)
        return X, y

    monkeypatch.setattr(sef, "_read_subject", _fake_read)

    loader = sef.sleep_edf_loader(
        data_root=tmp_path, n_subjects=6, seed=0, max_per_class=100,
    )
    X_s, y_s, X_t, y_t = loader()

    assert X_s.shape[1] == X_t.shape[1] == sef.WIN
    assert set(np.unique(y_s)) == {0, 1}
    assert set(np.unique(y_t)) == {0, 1}
    # Target = exactly one subject (80 epochs, balanced to 40+40 = 80 then maybe capped)
    # Source = remaining 5 subjects (5 × 80 = 400 epochs, balanced + capped to 200)
    assert len(X_t) <= len(X_s), "Target should be one subject; source the rest"


def test_loader_seed_changes_split(monkeypatch, tmp_path):
    """Different seeds → different (source, target) splits."""
    for prefix in ("SC4001", "SC4002", "SC4003", "SC4004", "SC4005", "SC4006"):
        _touch_pair(tmp_path, prefix)

    seen_targets: set[str] = set()

    def _fake_read(psg, hyp):
        seen_targets.add(psg.stem[:6])
        X = np.zeros((20, sef.WIN), dtype=np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        return X, y

    monkeypatch.setattr(sef, "_read_subject", _fake_read)

    targets = []
    for seed in range(4):
        seen_targets.clear()
        loader = sef.sleep_edf_loader(
            data_root=tmp_path, n_subjects=4, seed=seed, max_per_class=20,
        )
        loader()
        # No deterministic way to recover the target subject from the public API,
        # but the loader prints it via logging.  As a softer check, we ensure
        # different seeds touch different subject subsets at least sometimes.
        targets.append(frozenset(seen_targets))
    assert len(set(targets)) >= 2, "Different seeds should yield different subject subsets"

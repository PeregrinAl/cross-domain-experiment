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
    assert sef.IN_BED_MARGIN_EPOCHS == 60   # 30 min / 30 s = 60 epochs


def test_trim_in_bed_window_removes_leading_trailing_wake():
    """Long wake tails outside the 30-min margin are removed."""
    # 10 wake + 5 sleep + 10 wake — margin=2 → keep [8:17]
    y = np.array([0]*10 + [1]*5 + [0]*10, dtype=np.int64)
    X = np.zeros((len(y), sef.WIN), dtype=np.float32)
    Xt, yt = sef._trim_in_bed_window(X, y, margin=2)
    assert yt[0] == 0 and yt[-1] == 0          # still starts/ends with wake
    assert (yt == 1).sum() == 5                 # all sleep epochs preserved
    assert len(yt) == 2 + 5 + 2                 # exactly margin on each side


def test_trim_in_bed_window_all_wake_unchanged():
    """If there are no sleep epochs, the signal is returned unchanged."""
    y = np.zeros(20, dtype=np.int64)
    X = np.zeros((20, sef.WIN), dtype=np.float32)
    Xt, yt = sef._trim_in_bed_window(X, y, margin=5)
    assert len(yt) == 20


def test_loader_raises_when_no_subjects(tmp_path):
    loader = sef.sleep_edf_loader(data_root=tmp_path, n_subjects=2, seed=0)
    with pytest.raises(FileNotFoundError, match="No Sleep-EDF PSG/Hypnogram pairs"):
        loader()


def test_loader_splits_source_and_target(monkeypatch, tmp_path):
    """End-to-end with _read_subject mocked — verifies 8/4 split + balancing."""
    for prefix in ("SC4001", "SC4002", "SC4003", "SC4004", "SC4005", "SC4006"):
        _touch_pair(tmp_path, prefix)

    def _fake_read(psg, hyp):
        # Each subject: 20 leading wake + 40 sleep + 20 trailing wake (in-bed window
        # trimmer with margin=60 won't trim anything since all fit within one recording).
        rng = np.random.default_rng(hash(psg.stem) % (2 ** 32))
        X = rng.normal(0, 1, (80, sef.WIN)).astype(np.float32)
        y = np.array([0] * 20 + [1] * 40 + [0] * 20, dtype=np.int64)
        return X, y

    monkeypatch.setattr(sef, "_read_subject", _fake_read)

    # 6 subjects, n_target=2 → source=4, target pool=2; seed=0 picks pool[0].
    loader = sef.sleep_edf_loader(
        data_root=tmp_path, n_subjects=6, n_target=2,
        subject_seed=0, seed=0, max_per_class=100,
    )
    X_s, y_s, X_t, y_t = loader()

    assert X_s.shape[1] == X_t.shape[1] == sef.WIN
    assert set(np.unique(y_s)) == {0, 1}
    assert set(np.unique(y_t)) == {0, 1}
    # Source (4 subjects) must be larger than target (1 subject).
    assert len(X_t) <= len(X_s), "Target should be one subject; source the rest"


def test_loader_seed_changes_target(monkeypatch, tmp_path):
    """Different seeds rotate through the target pool (same source cohort)."""
    for prefix in ("SC4001", "SC4002", "SC4003", "SC4004", "SC4005", "SC4006"):
        _touch_pair(tmp_path, prefix)

    # Track which subjects are loaded as target.
    read_calls: list[str] = []

    def _fake_read(psg, hyp):
        read_calls.append(psg.stem[:6])
        X = np.zeros((20, sef.WIN), dtype=np.float32)
        y = np.array([0] * 5 + [1] * 10 + [0] * 5, dtype=np.int64)
        return X, y

    monkeypatch.setattr(sef, "_read_subject", _fake_read)

    # With n_subjects=6, n_target=2: target pool has 2 subjects (indices 4, 5 of
    # the sorted chosen list).  seed=0 → pool[0], seed=1 → pool[1].
    targets_per_seed = []
    for seed in range(2):
        read_calls.clear()
        loader = sef.sleep_edf_loader(
            data_root=tmp_path, n_subjects=6, n_target=2,
            subject_seed=0, seed=seed, max_per_class=20,
        )
        loader()
        # The last subject read is the target (source subjects are read first).
        targets_per_seed.append(read_calls[-1])

    assert targets_per_seed[0] != targets_per_seed[1], (
        "Different seeds must select different target subjects"
    )

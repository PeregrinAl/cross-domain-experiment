"""Smoke test for run_one_config.py — synthetic data, no filesystem deps.

What this verifies
------------------
1. The CLI parses the spec'd args and dispatches correctly.
2. The compatibility validator exits cleanly (code 2) on the documented
   incompatible combinations — statistical model × deep DA, neural model ×
   shallow DA, incompatible (φ, θ) pair.
3. A successful run appends exactly one row to the CSV with the full
   schema, plausible numeric values, and the right CLI key strings.
4. A second invocation of the same config overwrites the previous row
   rather than duplicating it.
5. `gain_over_source` is filled in for non-source-only configs when a
   matching source-only row already lives in the CSV.

What this does NOT verify
-------------------------
* Real data loaders (those are exercised on Kaggle).
* Deep branch training math (that is left to its existing unit tests —
  the wrapper just forwards to the same registered classes).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data — separable two-Gaussian binary problem with a mean shift
# between source and target.  Long enough for log-stft / cwt-morlet defaults.
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    T = 256
    n_per = 40
    t = np.linspace(0, 4 * np.pi, T, dtype=np.float32)
    X0_s = rng.normal(0.0, 0.2, (n_per, T)).astype(np.float32)
    X1_s = (np.sin(t)[None, :] + rng.normal(0.0, 0.2, (n_per, T))).astype(np.float32)
    X_s = np.vstack([X0_s, X1_s])
    y_s = np.array([0] * n_per + [1] * n_per, dtype=np.int64)

    X0_t = rng.normal(0.25, 0.2, (n_per, T)).astype(np.float32)
    X1_t = (np.sin(t)[None, :] + 0.25 + rng.normal(0.0, 0.2, (n_per, T))).astype(np.float32)
    X_t = np.vstack([X0_t, X1_t])
    y_t = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    return X_s, y_s, X_t, y_t


@pytest.fixture()
def patched_loaders(monkeypatch):
    """Replace each dataset loader factory with one that returns synthetic arrays."""
    from nstad_bench.data import mitbih_loader as mitbih_mod
    from nstad_bench.data import sleep_edf_loader as sleep_mod
    from nstad_bench.data import cwru_loader as cwru_mod

    def _factory(seed: int = 0, **_kwargs):
        return lambda: _make_synthetic(seed)

    monkeypatch.setattr(mitbih_mod, "mitbih_loader",
                        lambda data_root=None, seed=0, **kw: _factory(seed))
    monkeypatch.setattr(sleep_mod, "sleep_edf_loader",
                        lambda data_root=None, seed=0, **kw: _factory(seed))
    monkeypatch.setattr(cwru_mod, "cwru_loader",
                        lambda data_root=None, seed=0, **kw: _factory(seed))
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility validator — must exit cleanly with code 2 on documented mismatches
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    # stat model × deep DA
    ["--dataset", "mitbih", "--representation", "raw",
     "--model", "logreg", "--da-method", "codats", "--seed", "0"],
    # neural model × shallow DA (other than source-only)
    ["--dataset", "mitbih", "--representation", "raw",
     "--model", "inception-time", "--da-method", "coral", "--seed", "0"],
    # incompatible (φ, θ): raw × resnet2d
    ["--dataset", "mitbih", "--representation", "raw",
     "--model", "resnet2d", "--da-method", "source-only", "--seed", "0"],
])
def test_incompatible_exits_code_2(argv, tmp_path):
    """Each documented incompatibility should cause sys.exit with code 2."""
    import run_one_config as roc

    out = tmp_path / "results.csv"
    argv = argv + ["--output-csv", str(out)]
    with pytest.raises(SystemExit) as exc:
        roc.main(argv)
    assert exc.value.code == 2 or "INCOMPATIBLE" in str(exc.value)
    assert not out.exists(), "No CSV row should be written on a compatibility skip"


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — stat branch
# ─────────────────────────────────────────────────────────────────────────────

def test_stat_source_only_appends_row(tmp_path, patched_loaders):
    import run_one_config as roc

    out = tmp_path / "results.csv"
    argv = [
        "--dataset", "mitbih", "--representation", "raw",
        "--model", "logreg", "--da-method", "source-only",
        "--seed", "0", "--output-csv", str(out),
    ]
    rc = roc.main(argv)
    assert rc == 0
    assert out.exists()

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1
    row = rows[0]

    assert row["dataset"] == "mitbih"
    assert row["representation"] == "raw"
    assert row["model"] == "logreg"
    assert row["da_method"] == "source-only"
    assert int(row["seed"]) == 0
    # gain over source-only IS 0 by definition
    assert float(row["gain_over_source"]) == 0.0
    # Sanity: AUCs in [0, 1], delta defined
    for key in ("roc_auc", "pr_auc", "in_domain_auc", "cross_domain_auc"):
        assert 0.0 <= float(row[key]) <= 1.0, f"{key}={row[key]}"
    assert abs(float(row["delta_auc"]) -
               (float(row["in_domain_auc"]) - float(row["cross_domain_auc"]))) < 1e-6
    assert float(row["training_time_sec"]) > 0
    assert row["timestamp"].endswith("Z")


def test_rerun_overwrites_previous_row(tmp_path, patched_loaders):
    import run_one_config as roc

    out = tmp_path / "results.csv"
    argv = [
        "--dataset", "cwru", "--representation", "raw",
        "--model", "random-forest", "--da-method", "source-only",
        "--seed", "1", "--output-csv", str(out),
    ]
    roc.main(argv)
    roc.main(argv)   # second invocation with identical args
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1, "Same-key reruns must overwrite, not duplicate"


def test_gain_over_source_resolves_from_csv(tmp_path, patched_loaders):
    import run_one_config as roc

    out = tmp_path / "results.csv"
    # 1) source-only baseline first
    roc.main([
        "--dataset", "sleep-edf", "--representation", "raw",
        "--model", "logreg", "--da-method", "source-only",
        "--seed", "0", "--output-csv", str(out),
    ])
    # 2) coral on the same dataset / repr / model / seed
    roc.main([
        "--dataset", "sleep-edf", "--representation", "raw",
        "--model", "logreg", "--da-method", "coral",
        "--seed", "0", "--output-csv", str(out),
    ])

    rows = list(csv.DictReader(open(out)))
    by_method = {r["da_method"]: r for r in rows}
    so = by_method["source-only"]
    coral = by_method["coral"]
    expected_gain = float(coral["cross_domain_auc"]) - float(so["cross_domain_auc"])
    assert abs(float(coral["gain_over_source"]) - expected_gain) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — deep branch.  Skipped when torch is unavailable.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def fast_deep_defaults(monkeypatch):
    """Shrink deep training so the smoke test stays under a few seconds."""
    pytest.importorskip("torch")
    import run_one_config as roc
    fast = {
        "InceptionTime1D": {"epochs": 2, "lr": 1e-3, "batch_size": 16,
                            "nb_filters": 4, "depth": 1},
        "PatchTST":        {"epochs": 2, "lr": 1e-3, "batch_size": 16,
                            "d_model": 16, "n_heads": 2, "n_layers": 1},
        "ResNet18_2D":     {"epochs": 2, "lr": 1e-3, "batch_size": 16},
    }
    for k, v in fast.items():
        monkeypatch.setitem(roc._DEFAULT_MODEL_CFG, k, v)
    # Also shrink DA hyperparams so any deep DA loop runs in seconds.
    fast_da = {
        "MK_MMD":  {"n_epochs": 1, "lr": 1e-3, "lambda_mmd": 1.0, "batch_size": 16},
        "CoDATS":  {"n_epochs": 1, "lr": 1e-3, "lr_disc": 1e-3,
                    "lambda_domain": 1.0, "batch_size": 16},
    }
    for k, v in fast_da.items():
        monkeypatch.setitem(roc._DEFAULT_DA_HP, k, v)
    yield


def test_deep_source_only_runs(tmp_path, patched_loaders, fast_deep_defaults):
    import run_one_config as roc

    out = tmp_path / "results.csv"
    rc = roc.main([
        "--dataset", "mitbih", "--representation", "raw",
        "--model", "inception-time", "--da-method", "source-only",
        "--seed", "0", "--output-csv", str(out),
    ])
    assert rc == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1
    assert rows[0]["model"] == "inception-time"
    assert 0.0 <= float(rows[0]["cross_domain_auc"]) <= 1.0

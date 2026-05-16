"""End-to-end smoke test for the experiment runner.

Checks
------
1.  Minimal config (1 ds × 1 φ × 1 θ × 1 ψ × 1 seed) runs without error.
2.  Parquet round-trip: written file reads back with identical dtypes and values.
3.  Reproducibility: running the same config twice gives bit-identical results.
4.  Compatibility mask: an incompatible (φ, θ) pair in the config is silently
    skipped — no error, and no row for that pair in the output.
5.  Seed column: every result row carries the integer seed.

Usage
-----
    .venv/bin/python scripts/smoke_runner_e2e.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nstad_bench.experiments.runner import (
    RESULT_COLS,
    register_dataset,
    run_experiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_e2e")

# ── Synthetic dataset ─────────────────────────────────────────────────────────
# Pre-generate arrays once so every call to the loader returns the SAME data.
# Using a stateful generator inside the loader would advance its state on each
# call and silently produce different training sets across runs.
N, T = 120, 128

_X_s = np.random.default_rng(0).standard_normal((N, T)).astype(np.float32)
_y_s = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.int64)
_X_t = np.random.default_rng(1).standard_normal((N, T)).astype(np.float32)
_y_t = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.int64)

def _make_smoke_dataset():
    return _X_s, _y_s, _X_t, _y_t

register_dataset("smoke_ds", _make_smoke_dataset)

# ─────────────────────────────────────────────────────────────────────────────

CONFIG      = ROOT / "configs" / "smoke_minimal.yaml"
CONFIG_ROOT = ROOT / "configs"

PASS = "  ✓"
FAIL = "  ✗"

def banner(msg: str) -> None:
    log.info("─" * 60)
    log.info(msg)
    log.info("─" * 60)

# ── Check 1 + 2: Run end-to-end, measure time, verify Parquet round-trip ──────

banner("CHECK 1+2  end-to-end run + Parquet round-trip")

with tempfile.TemporaryDirectory() as tmp:
    # Override output_dir inside the config at runtime via monkeypatching yaml
    import yaml  # noqa: PLC0415

    with open(CONFIG) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["output_dir"] = tmp

    tmp_cfg = Path(tmp) / "smoke_minimal.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg_dict, f)

    t0 = time.perf_counter()
    df1: pd.DataFrame = run_experiment(tmp_cfg, config_root=CONFIG_ROOT)
    elapsed = time.perf_counter() - t0
    log.info("Run 1 completed in %.1f s — %d rows", elapsed, len(df1))

    # ── Check 1: schema ────────────────────────────────────────────────────────
    missing = [c for c in RESULT_COLS if c not in df1.columns]
    if missing:
        log.error(FAIL + " Schema missing columns: %s", missing)
        sys.exit(1)
    log.info(PASS + " Schema OK: %s", RESULT_COLS)

    if df1.empty:
        log.error(FAIL + " DataFrame is empty — expected rows")
        sys.exit(1)
    log.info(PASS + " %d result rows produced", len(df1))

    # Expected metrics
    expected_metrics = {"roc_auc", "pr_auc", "source_roc_auc", "delta_auc", "gain"}
    got_metrics      = set(df1["metric_name"].unique())
    if not expected_metrics.issubset(got_metrics):
        log.error(FAIL + " Missing metrics: %s", expected_metrics - got_metrics)
        sys.exit(1)
    log.info(PASS + " All expected metrics present: %s", sorted(got_metrics))

    # ── Check 2: Parquet round-trip ────────────────────────────────────────────
    parquet_path = Path(tmp) / "smoke_minimal.parquet"
    if not parquet_path.exists():
        log.error(FAIL + " Parquet file not created at %s", parquet_path)
        sys.exit(1)

    df_rt = pd.read_parquet(parquet_path, engine="pyarrow")

    # Column set
    if set(df_rt.columns) != set(RESULT_COLS):
        log.error(FAIL + " Parquet columns mismatch: %s", df_rt.columns.tolist())
        sys.exit(1)
    log.info(PASS + " Parquet column set matches RESULT_COLS")

    # Shape
    if df_rt.shape != df1.shape:
        log.error(FAIL + " Shape mismatch: in-memory %s vs parquet %s",
                  df1.shape, df_rt.shape)
        sys.exit(1)
    log.info(PASS + " Shape preserved: %s", df1.shape)

    # Values (float tolerance for NaN-safe comparison)
    df1_s  = df1[RESULT_COLS].sort_values(RESULT_COLS[:6] + ["metric_name"]).reset_index(drop=True)
    df_rt_s = df_rt[RESULT_COLS].sort_values(RESULT_COLS[:6] + ["metric_name"]).reset_index(drop=True)

    str_cols  = ["config_hash", "dataset", "phi", "theta", "psi", "metric_name"]
    num_cols  = ["metric_value", "metric_ci_lower", "metric_ci_upper"]

    ok = True
    for c in str_cols:
        if not (df1_s[c] == df_rt_s[c]).all():
            log.error(FAIL + " String column %r mismatch after round-trip", c)
            ok = False
    for c in num_cols:
        a, b = df1_s[c].values, df_rt_s[c].values
        # NaN must appear in same positions
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            log.error(FAIL + " NaN pattern mismatch in column %r", c)
            ok = False
        else:
            mask = ~np.isnan(a)
            if mask.any() and not np.allclose(a[mask], b[mask], rtol=1e-9, atol=0):
                log.error(FAIL + " Float value mismatch in column %r", c)
                ok = False
    if not ok:
        sys.exit(1)
    log.info(PASS + " Parquet round-trip: values identical (NaN-safe, rtol=1e-9)")

    # ── Check 3: Reproducibility — run again with same config ──────────────────
    banner("CHECK 3  Reproducibility (second run, same config)")
    df2: pd.DataFrame = run_experiment(tmp_cfg, config_root=CONFIG_ROOT)

    df2_s = df2[RESULT_COLS].sort_values(RESULT_COLS[:6] + ["metric_name"]).reset_index(drop=True)

    ok = True
    for c in str_cols:
        if not (df1_s[c] == df2_s[c]).all():
            log.error(FAIL + " Reproducibility: column %r differs between runs", c)
            ok = False
    for c in num_cols:
        a, b = df1_s[c].values, df2_s[c].values
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            log.error(FAIL + " Reproducibility: NaN pattern differs in column %r", c)
            ok = False
        else:
            mask = ~np.isnan(a)
            if mask.any() and not np.allclose(a[mask], b[mask], rtol=1e-9, atol=0):
                log.error(FAIL + " Reproducibility: values differ in column %r", c)
                ok = False
    if not ok:
        sys.exit(1)
    log.info(PASS + " Both runs produce bit-identical results")


# ── Check 4: Compatibility mask — incompatible pair skipped ───────────────────

banner("CHECK 4  Compatibility mask (incompatible pair silently skipped)")

with tempfile.TemporaryDirectory() as tmp:
    with open(CONFIG) as f:
        cfg_dict = yaml.safe_load(f)

    # Add an incompatible pair: LogSTFT × InceptionTime1D = false in compat.yaml
    cfg_dict["output_dir"]        = tmp
    cfg_dict["representations"]   = ["RawSignal", "LogSTFT"]
    cfg_dict["models"]            = {
        "InceptionTime1D": {
            "epochs": 5, "lr": 1e-3, "batch_size": 32,
            "nb_filters": 16, "depth": 2,
        }
    }

    tmp_cfg = Path(tmp) / "compat_test.yaml"
    tmp_cfg.write_text(yaml.dump(cfg_dict))

    df_compat = run_experiment(tmp_cfg, config_root=CONFIG_ROOT)

    # Only RawSignal × InceptionTime1D should appear — LogSTFT is incompatible
    phi_vals = df_compat["phi"].unique().tolist()
    if "LogSTFT" in phi_vals:
        log.error(FAIL + " LogSTFT appeared in results — compat mask broken! %s", phi_vals)
        sys.exit(1)
    if "RawSignal" not in phi_vals:
        log.error(FAIL + " RawSignal missing from results: %s", phi_vals)
        sys.exit(1)
    log.info(PASS + " Incompatible LogSTFT×InceptionTime1D skipped; only %s ran", phi_vals)


# ── Check 5: Seed column ──────────────────────────────────────────────────────

banner("CHECK 5  seed column present and correct in every row")

with tempfile.TemporaryDirectory() as tmp:
    with open(CONFIG) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["output_dir"] = tmp
    cfg_dict["random_search"]["seeds"] = [7, 13]   # two distinct seeds

    tmp_cfg = Path(tmp) / "seeds_test.yaml"
    tmp_cfg.write_text(yaml.dump(cfg_dict))

    df_seeds = run_experiment(tmp_cfg, config_root=CONFIG_ROOT)

    # seed column must be integer
    if not pd.api.types.is_integer_dtype(df_seeds["seed"]):
        log.error(FAIL + " seed column dtype is %s, expected int", df_seeds["seed"].dtype)
        sys.exit(1)
    log.info(PASS + " seed column dtype: %s", df_seeds["seed"].dtype)

    # No NaNs
    if df_seeds["seed"].isna().any():
        log.error(FAIL + " NaN values found in seed column")
        sys.exit(1)
    log.info(PASS + " No NaN in seed column")

    seen_seeds = set(df_seeds["seed"].unique())
    if seen_seeds != {7, 13}:
        log.error(FAIL + " Expected seeds {7, 13}, got %s", seen_seeds)
        sys.exit(1)
    log.info(PASS + " Both seeds present in output: %s", sorted(seen_seeds))

    # Every row has a seed value (none missing)
    if len(df_seeds) == 0:
        log.error(FAIL + " Empty result for multi-seed run")
        sys.exit(1)
    log.info(PASS + " %d rows, all carry seed", len(df_seeds))

# ── Summary ───────────────────────────────────────────────────────────────────

banner("ALL CHECKS PASSED")
log.info("  Check 1: end-to-end run completes in %.1f s", elapsed)
log.info("  Check 2: Parquet round-trip — lossless")
log.info("  Check 3: Reproducibility — bit-identical across runs")
log.info("  Check 4: Compatibility mask — incompatible pairs silently skipped")
log.info("  Check 5: seed column — integer, no NaN, correct values")

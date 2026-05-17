"""Dry run — verifies both benchmark loaders + one pass through the runner.

What it checks
--------------
1. Each loader returns (X_s, y_s, X_t, y_t) with correct dtypes and shapes.
2. Both classes present in every split.
3. Window widths match expected values (MIT-BIH 280, DeepBeat 800).
4. One minimal runner pass per dataset (SourceOnly, InceptionTime1D, 3 epochs).

Usage
-----
    .venv/bin/python scripts/dry_run.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,          # suppress loader INFO during dry run
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dry_run")
logging.getLogger("dry_run").setLevel(logging.INFO)

PASS = "  ✓"
FAIL = "  ✗"
failures: list[str] = []

def ok(msg: str) -> None:
    log.info(PASS + "  " + msg)

def fail(msg: str) -> None:
    log.error(FAIL + "  " + msg)
    failures.append(msg)

def section(title: str) -> None:
    log.info("")
    log.info("─" * 60)
    log.info("  %s", title)
    log.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check one split
# ─────────────────────────────────────────────────────────────────────────────

def check_split(
    name: str,
    X_s: np.ndarray, y_s: np.ndarray,
    X_t: np.ndarray, y_t: np.ndarray,
    expected_width: int,
) -> None:
    for split, X, y in [("source", X_s, y_s), ("target", X_t, y_t)]:
        tag = f"{name}/{split}"

        # shape consistency
        if X.ndim != 2:
            fail(f"{tag}: X.ndim={X.ndim}, expected 2")
        elif X.shape[1] != expected_width:
            fail(f"{tag}: X.shape[1]={X.shape[1]}, expected {expected_width}")
        else:
            ok(f"{tag}: shape {X.shape}")

        if X.shape[0] != y.shape[0]:
            fail(f"{tag}: X rows {X.shape[0]} ≠ y len {y.shape[0]}")

        # dtypes
        if X.dtype != np.float32:
            fail(f"{tag}: X dtype={X.dtype}, expected float32")
        if y.dtype != np.int64:
            fail(f"{tag}: y dtype={y.dtype}, expected int64")

        # both classes present
        classes = set(np.unique(y).tolist())
        if classes != {0, 1}:
            fail(f"{tag}: labels={classes}, expected {{0, 1}}")
        else:
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            ok(f"{tag}: labels {{0, 1}}  counts  0→{n0}  1→{n1}  "
               f"ratio {n1/n0:.2f}x")

        # no NaN/Inf in X
        if not np.isfinite(X).all():
            fail(f"{tag}: X contains NaN or Inf")
        else:
            ok(f"{tag}: X fully finite")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MIT-BIH
# ─────────────────────────────────────────────────────────────────────────────

section("MIT-BIH  DS1→DS2  (cap=300/class)")

from nstad_bench.data.mitbih_loader import mitbih_loader
from nstad_bench.experiments.runner import register_dataset

try:
    loader_mitbih = mitbih_loader(max_per_class=300, seed=0)
    X_s, y_s, X_t, y_t = loader_mitbih()
    check_split("mitbih", X_s, y_s, X_t, y_t, expected_width=280)
except Exception as exc:
    fail(f"mitbih loader raised: {exc}")
    X_s = y_s = X_t = y_t = None

# ─────────────────────────────────────────────────────────────────────────────
# 2. DeepBeat
# ─────────────────────────────────────────────────────────────────────────────

section("DeepBeat  train→test  (cap=500/class source)")

from nstad_bench.data.deepbeat_loader import deepbeat_loader

# cap=500 used for BOTH the data check and the runner pass below so the
# lru_cache on _load() is shared — train.npz is read only once.
loader_deep = None
try:
    loader_deep = deepbeat_loader(max_per_class=500, seed=0)
    X_s_d, y_s_d, X_t_d, y_t_d = loader_deep()
    check_split("deepbeat", X_s_d, y_s_d, X_t_d, y_t_d, expected_width=800)

    # Source should be ~balanced; target keeps natural distribution
    n0s, n1s = int((y_s_d == 0).sum()), int((y_s_d == 1).sum())
    ok(f"deepbeat/source: SR={n0s}  AF={n1s}  (balanced by cap)")
    n0t, n1t = int((y_t_d == 0).sum()), int((y_t_d == 1).sum())
    ok(f"deepbeat/target: SR={n0t}  AF={n1t}  natural distribution "
       f"(SR {100*n0t/(n0t+n1t):.0f}%)")
except Exception as exc:
    fail(f"deepbeat loader raised: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Minimal runner pass — one dataset per loader
# ─────────────────────────────────────────────────────────────────────────────

section("Runner  —  one SourceOnly pass per dataset (3 epochs)")

MINI_CFG = {
    "experiment_name": "dry_run",
    "n_bootstrap": 20,
    "screening":     {"top_k": 1, "metric": "delta_auc"},
    "random_search": {"n_trials": 1, "seeds": [0], "base_seed": 0},
    "representations": {"RawSignal": {}},
    "models": {
        "InceptionTime1D": {
            "epochs": 3, "lr": 1e-3, "batch_size": 32,
            "nb_filters": 16, "depth": 2,
        }
    },
    "adaptation_methods": {"SourceOnly": {}},
}

from nstad_bench.experiments.runner import run_experiment

for ds_name, loader_fn in [
    ("mitbih_ds1_ds2",         mitbih_loader(max_per_class=200, seed=0)),
    # Reuse the already-loaded loader_deep (lru_cache hit — no second disk read).
    *([("deepbeat_patient_split", loader_deep)] if loader_deep is not None else []),
]:
    register_dataset(ds_name, loader_fn)
    cfg = {**MINI_CFG, "datasets": [ds_name]}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg["output_dir"] = tmp
            tmp_cfg = Path(tmp) / "dry_run.yaml"
            tmp_cfg.write_text(yaml.dump(cfg))
            df = run_experiment(tmp_cfg, config_root=ROOT / "configs")

        if df.empty:
            fail(f"runner/{ds_name}: returned empty DataFrame")
        else:
            auc_rows = df[df["metric_name"] == "roc_auc"]
            auc_val  = auc_rows["metric_value"].iloc[0] if len(auc_rows) else float("nan")
            ok(f"runner/{ds_name}: {len(df)} result rows  "
               f"SourceOnly ROC-AUC={auc_val:.4f}")
    except Exception as exc:
        fail(f"runner/{ds_name}: raised {type(exc).__name__}: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

log.info("")
log.info("=" * 60)
if failures:
    log.error("  DRY RUN FAILED  —  %d issue(s):", len(failures))
    for f_ in failures:
        log.error("    • %s", f_)
    sys.exit(1)
else:
    log.info("  DRY RUN PASSED  —  all checks green")
log.info("=" * 60)

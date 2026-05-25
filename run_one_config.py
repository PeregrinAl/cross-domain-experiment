"""run_one_config.py — single (dataset, repr, model, da-method, seed) runner.

Designed for the Kaggle notebook workflow: each invocation runs one cell
of the experiment matrix and appends exactly one row to the output CSV.
The script is fully restart-safe — re-running the same arguments overwrites
the matching row, and other configs in the CSV are preserved untouched.

CLI
---
::

    python run_one_config.py \\
        --dataset {mitbih,sleep-edf,cwru,chbmit,mimii} \\
        --representation {raw,log-stft,cwt-morlet} \\
        --model {random-forest,logreg,gbm,inception-time,patch-tst,resnet2d} \\
        --da-method {source-only,coral,subspace-alignment,codats,deep-coral,mk-mmd} \\
        --seed N \\
        --output-csv path/to/results.csv

Compatibility validation (early exit, non-zero code, no traceback)
------------------------------------------------------------------
1. Statistical models (logreg / random-forest / gbm) must be paired with
   *shallow* DA methods (source-only / coral / subspace-alignment).
2. Neural models (inception-time / patch-tst / resnet2d) must be paired
   with *deep* DA methods (source-only / codats / m2n2 / mk-mmd).
3. Each (representation, model) pair must be listed as ``true`` in the
   relevant ``configs/compatibility.yaml`` (resp. statistical/).

Output row schema
-----------------
``dataset, representation, model, da_method, seed,
roc_auc, pr_auc, f1,
in_domain_auc, cross_domain_auc, delta_auc, gain_over_source,
training_time_sec, timestamp``

``gain_over_source`` is computed at write-time by reading any existing
source-only row for the same ``(dataset, representation, model, seed)``
already present in the CSV.  Source-only rows themselves write ``0.0``.

Environment
-----------
* ``DATA_ROOT`` — defaults to ``/kaggle/input`` (set by the Kaggle notebook).
* ``OUTPUT_ROOT`` — defaults to ``/kaggle/working``.  Used only for the
  default output CSV path; pass ``--output-csv`` to override.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure the repo root is on sys.path when invoked from anywhere (Kaggle, CWD, etc.)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("run_one_config")

# ─────────────────────────────────────────────────────────────────────────────
# CLI ↔ internal name maps
# ─────────────────────────────────────────────────────────────────────────────

DATASETS: dict[str, str] = {
    "mitbih":    "mitbih_ds1_ds2",
    "sleep-edf": "sleep_edf_loso",
    "cwru":      "cwru_load_0_3",
    "chbmit":    "chbmit_loso",
    "mimii":     "mimii_pump_cross_unit",
}

REPRESENTATIONS: dict[str, str] = {
    "raw":        "RawSignal",
    "log-stft":   "LogSTFT",
    "cwt-morlet": "CWT_Morlet",
}

# Models split by branch
STAT_MODELS: dict[str, str] = {
    "logreg":        "LogReg",
    "random-forest": "RandomForest",
    "gbm":           "GBM",
}
NEURAL_MODELS: dict[str, str] = {
    "inception-time": "InceptionTime1D",
    "patch-tst":      "PatchTST",
    "resnet2d":       "ResNet18_2D",
}
MODELS: dict[str, str] = {**STAT_MODELS, **NEURAL_MODELS}

# DA methods split by branch
SHALLOW_DA: dict[str, str] = {
    "source-only":        "SourceOnly",
    "coral":              "CORAL",
    "subspace-alignment": "SubspaceAlignment",
}
DEEP_DA: dict[str, str] = {
    "source-only": "SourceOnly",
    "deep-coral":  "DeepCORAL",
    "codats":      "CoDATS",
    "mk-mmd":      "MK_MMD",
}
DA_METHODS: dict[str, set[str]] = {
    # CLI key → set of allowed branches ("stat", "deep")
    "source-only":        {"stat", "deep"},
    "coral":              {"stat"},
    "subspace-alignment": {"stat"},
    "deep-coral":         {"deep"},
    "codats":             {"deep"},
    "mk-mmd":             {"deep"},
}

# Combos that are mathematically valid but computationally prohibitive at the
# given feature dimensionality.  Shallow CORAL and SubspaceAlignment are O(D^3)
# in feature dim (covariance/SVD on a D×D matrix), which is fine for low-D
# representations but ~15 min per fit on MIMII raw (D = 16000 samples per 1s
# window).  We skip these the same way as INCOMPATIBLE (exit 2 → driver SKIP).
#
# Use log-stft + CORAL on MIMII instead — same alignment principle, ~500-d
# features, runs in seconds.  For deep DA on MIMII raw, all four neural methods
# (source-only / deep-coral / codats / mk-mmd) work on the 128-d projector
# embedding regardless of input dimensionality.
_PROHIBITIVE_COMBOS: set[tuple[str, str, str]] = {
    # MIMII raw: D = 16 000 (16 kHz × 1 s window) → ~15 min per fit
    ("mimii", "raw", "coral"),
    ("mimii", "raw", "subspace-alignment"),
    # MIMII log-stft: D ≈ 8 250 (33 bins × 250 frames) → still ~3-5 min per fit.
    # 3 seeds → 10-15 min for one (model, da) combo.  Skip to keep wall time
    # bounded; CWRU + CHB-MIT × log-stft × CORAL (D ≤ 528) cover the
    # "spectral repr × shallow alignment" comparison.
    ("mimii", "log-stft", "coral"),
    ("mimii", "log-stft", "subspace-alignment"),
    # MIMII × cwt-morlet: (32, 16000) scalogram OOMs ResNet2D on T4 even at
    # batch=8 — a single BatchNorm op tries to allocate ~7.8 GiB.  The
    # long-seq guard in _build_neural_model is not sufficient for this combo;
    # skip all four DA methods.  CWRU/CHB-MIT × cwt-morlet × ResNet2D cover
    # the "wavelet repr × deep 2-D" comparison at tractable T.
    ("mimii", "cwt-morlet", "source-only"),
    ("mimii", "cwt-morlet", "deep-coral"),
    ("mimii", "cwt-morlet", "codats"),
    ("mimii", "cwt-morlet", "mk-mmd"),
}

# Default hyperparameters per architecture / DA method (mid-point of search spaces).
# Picking fixed defaults here means the CSV row reflects a deterministic
# (model, da_method) — no random HP-sampling, so seeds vary only training noise.
_DEFAULT_MODEL_CFG: dict[str, dict[str, Any]] = {
    # Neural
    "InceptionTime1D": {"epochs": 30, "lr": 1e-3, "batch_size": 64,
                        "nb_filters": 32, "depth": 6},
    "PatchTST":        {"epochs": 30, "lr": 5e-4, "batch_size": 32,
                        "d_model": 128, "n_heads": 4, "n_layers": 3},
    "ResNet18_2D":     {"epochs": 30, "lr": 5e-4, "batch_size": 32},
    # Statistical
    "LogReg":       {"C": 1.0, "max_iter": 300, "solver": "saga"},
    "RandomForest": {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 1, "n_jobs": -1},
    "GBM":          {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
}

_DEFAULT_DA_HP: dict[str, dict[str, Any]] = {
    # Neural DA
    "MK_MMD":    {"n_epochs": 15, "lr": 1e-4, "lambda_mmd": 1.0, "batch_size": 64},
    "CoDATS":    {"n_epochs": 15, "lr": 1e-4, "lr_disc": 1e-3,
                  "lambda_domain": 1.0, "batch_size": 64},
    "DeepCORAL": {"n_epochs": 15, "lr": 1e-4, "lambda_coral": 1.0, "batch_size": 64},
    # Statistical DA
    "CORAL":             {"lambda_reg": 1e-3, "align_mean": True},
    "SubspaceAlignment": {"n_components": 30},
    "SourceOnly":        {},
}

_REPR_PARAMS: dict[str, dict[str, Any]] = {
    "RawSignal":  {},
    "LogSTFT":    {"n_fft": 64, "hop_length": 64},
    "CWT_Morlet": {"n_scales": 32},
}

CSV_COLUMNS: list[str] = [
    "dataset", "representation", "model", "da_method", "seed",
    "roc_auc", "pr_auc", "f1",
    "in_domain_auc", "cross_domain_auc", "delta_auc", "gain_over_source",
    "training_time_sec", "timestamp",
]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch  # noqa: WPS433
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass   # torch is optional for stat-only runs


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility validation
# ─────────────────────────────────────────────────────────────────────────────

def _load_compat(branch: str) -> dict[str, dict[str, bool]]:
    """Load the repr × model compatibility matrix for *branch* ("stat" / "deep")."""
    if branch == "stat":
        path = ROOT / "configs" / "statistical" / "compatibility.yaml"
    else:
        path = ROOT / "configs" / "compatibility.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["compatibility"]


def _validate(args: argparse.Namespace) -> tuple[str, str, str, str, str]:
    """Return (branch, ds_key, phi, theta, psi) after validating compatibility.

    On invalid combinations, prints a one-line message to stderr and exits
    with code 2 — the Kaggle notebook treats this as "skip this config" and
    moves on.  No traceback, no checkpoint pollution.
    """
    ds_key = DATASETS[args.dataset]
    phi    = REPRESENTATIONS[args.representation]
    theta  = MODELS[args.model]
    is_stat = args.model in STAT_MODELS
    branch = "stat" if is_stat else "deep"

    allowed_branches = DA_METHODS[args.da_method]
    if branch not in allowed_branches:
        other = "stat" if branch == "deep" else "deep"
        sys.exit(
            f"INCOMPATIBLE: model={args.model!r} ({branch} branch) cannot run "
            f"with da-method={args.da_method!r} ({other} branch). Skipping."
        )

    if (args.dataset, args.representation, args.da_method) in _PROHIBITIVE_COMBOS:
        sys.exit(
            f"PROHIBITIVE: {args.dataset}×{args.representation}×{args.da_method} "
            f"is computationally infeasible (shallow CORAL/SubspaceAlignment are "
            f"O(D^3); MIMII raw D=16000 → ~15 min per fit). Use "
            f"{args.dataset}×log-stft for shallow DA, or {args.dataset}×raw with "
            f"deep-coral/codats/mk-mmd. Skipping."
        )

    if branch == "stat":
        psi = ({"source-only": "SourceOnly"} | SHALLOW_DA)[args.da_method]
    else:
        psi = ({"source-only": "SourceOnly"} | DEEP_DA)[args.da_method]

    compat = _load_compat(branch)
    if not compat.get(phi, {}).get(theta, False):
        sys.exit(
            f"INCOMPATIBLE: representation={args.representation!r} ({phi}) "
            f"and model={args.model!r} ({theta}) are marked incompatible in "
            f"configs{'/statistical' if branch == 'stat' else ''}/compatibility.yaml. "
            "Skipping."
        )

    return branch, ds_key, phi, theta, psi


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registration
# ─────────────────────────────────────────────────────────────────────────────

def _register_dataset(cli_name: str, ds_key: str, seed: int) -> None:
    """Lazy-register the requested dataset with the shared registry.

    Honors ``DATA_ROOT`` via the loader factories; the Kaggle notebook may
    additionally pass an explicit slug path through the env vars
    ``KAGGLE_MITBIH_DIR`` / ``KAGGLE_SLEEP_EDF_DIR`` / ``KAGGLE_CWRU_DIR`` /
    ``KAGGLE_CHBMIT_DIR`` / ``KAGGLE_MIMII_DIR`` when the Kaggle dataset
    slug differs from the canonical subdir name.
    """
    from nstad_bench.experiments.runner import register_dataset

    if cli_name == "mitbih":
        override = os.environ.get("KAGGLE_MITBIH_DIR")
        root = Path(override) if override else None
        # Auto-detect format: CSV (mondejar Kaggle dataset) vs WFDB (PhysioNet raw)
        # mondejar dataset has CSV either directly in root or in mitbih_database/ subdir
        def _find_csv_root(r: Path | None) -> Path | None:
            if r is None:
                return None
            for candidate in [r, r / "mitbih_database"]:
                if (candidate / "mitbih_train.csv").exists():
                    return candidate
            return None
        _csv_root = _find_csv_root(root)
        if _csv_root:
            from nstad_bench.data.mitbih_csv_loader import mitbih_csv_loader
            log.info("MIT-BIH: using CSV loader (mondejar format) from %s", _csv_root)
            register_dataset(ds_key, mitbih_csv_loader(data_root=_csv_root, seed=seed))
        else:
            from nstad_bench.data.mitbih_loader import mitbih_loader
            log.info("MIT-BIH: using WFDB loader (PhysioNet format)")
            register_dataset(ds_key, mitbih_loader(data_root=override, seed=seed))
    elif cli_name == "sleep-edf":
        from nstad_bench.data.sleep_edf_loader import sleep_edf_loader
        override = os.environ.get("KAGGLE_SLEEP_EDF_DIR")
        register_dataset(ds_key, sleep_edf_loader(data_root=override, seed=seed))
    elif cli_name == "cwru":
        from nstad_bench.data.cwru_loader import cwru_loader
        override = os.environ.get("KAGGLE_CWRU_DIR")
        register_dataset(ds_key, cwru_loader(data_root=override, seed=seed))
    elif cli_name == "chbmit":
        from nstad_bench.data.chbmit_loader import chbmit_loader, DEFAULT_PATIENTS
        # Rotate the LOSO test patient through DEFAULT_PATIENTS by seed —
        # mirrors the Sleep-EDF "seed picks fold from target pool" convention
        # so each seed evaluates a different held-out patient.
        test_patient = DEFAULT_PATIENTS[seed % len(DEFAULT_PATIENTS)]
        override = os.environ.get("KAGGLE_CHBMIT_DIR")
        log.info("CHB-MIT: LOSO fold for seed=%d → test_patient=%s", seed, test_patient)
        register_dataset(
            ds_key,
            chbmit_loader(data_root=override, test_patient=test_patient, seed=seed),
        )
    elif cli_name == "mimii":
        from nstad_bench.data.mimii_loader import mimii_loader
        # Default machine type is pump (the only one universally available on
        # Kaggle via senaca/mimii-pump-sound-dataset).  Override via
        # KAGGLE_MIMII_MACHINE if you've also uploaded valve / fan / slider.
        machine = os.environ.get("KAGGLE_MIMII_MACHINE", "pump")
        override = os.environ.get("KAGGLE_MIMII_DIR")
        log.info("MIMII: machine=%s (seed=%d rotates target unit)", machine, seed)
        register_dataset(
            ds_key,
            mimii_loader(data_root=override, machine=machine, seed=seed),
        )
    else:   # pragma: no cover — argparse choices already restrict this
        raise ValueError(f"Unknown dataset CLI key: {cli_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Single execution — shared between branches
# ─────────────────────────────────────────────────────────────────────────────

def _build_repr(phi: str):
    from nstad_bench.representations import CWT_Morlet, LogSTFT, RawSignal
    registry = {"RawSignal": RawSignal, "LogSTFT": LogSTFT, "CWT_Morlet": CWT_Morlet}
    return registry[phi](**_REPR_PARAMS[phi])


def _build_neural_model(theta: str, X_repr: np.ndarray):
    """Build a neural model, auto-shrinking capacity for long input sequences.

    For 1-D inputs with T > 8000 samples (e.g. MIMII raw, T=16000), the default
    InceptionTime1D / PatchTST configs OOM on a 15 GB T4 GPU — activation memory
    scales as ``batch_size × channels × T``.  We auto-shrink batch_size, filters,
    depth (InceptionTime) or d_model, n_layers (PatchTST) so the model fits.

    The threshold T=8000 is conservative: CHB-MIT (T=512), CWRU (T=1024), and
    MIT-BIH (T=180) all stay below it and use full-capacity defaults; only
    MIMII raw (T=16000) triggers the shrink.
    """
    from nstad_bench.models import InceptionTime1D, PatchTST, ResNet18_2D
    cfg = _DEFAULT_MODEL_CFG[theta].copy()

    # Long-sequence guard for 1-D models on raw audio-rate input.
    seq_len = int(X_repr.shape[-1])
    if X_repr.ndim == 2 and seq_len > 8000:
        if theta == "InceptionTime1D":
            cfg["batch_size"] = min(cfg["batch_size"], 16)
            cfg["nb_filters"] = min(cfg.get("nb_filters", 32), 16)
            cfg["depth"]      = min(cfg.get("depth", 6), 4)
            log.info("  long-seq guard: InceptionTime1D → batch=16, nb_filters=16, depth=4")
        elif theta == "PatchTST":
            cfg["batch_size"] = min(cfg["batch_size"], 8)
            cfg["d_model"]    = min(cfg.get("d_model", 128), 64)
            cfg["n_layers"]   = min(cfg.get("n_layers", 3), 2)
            log.info("  long-seq guard: PatchTST → batch=8, d_model=64, n_layers=2")

    # Long-time-dim guard for 2-D representations on ResNet2D.
    # CWT-Morlet on MIMII raw windows: scalogram shape (32, 16000) creates >1 GB
    # intermediate activations per BN layer at default batch=32, OOMs on T4.
    # Drop to batch=8 (~130 MB/activation → ~1.3 GB total).  Wall-time/epoch
    # roughly unchanged: 4x fewer steps per epoch × 4x more steps = same.
    elif X_repr.ndim == 3 and X_repr.shape[-1] > 4000:
        if theta == "ResNet18_2D":
            cfg["batch_size"] = min(cfg["batch_size"], 8)
            log.info(
                "  long-seq guard: ResNet18_2D wide-T (T=%d) → batch=8",
                X_repr.shape[-1],
            )

    train_keys = {"epochs", "lr", "batch_size"}
    ctor = {k: v for k, v in cfg.items() if k not in train_keys}
    if theta == "InceptionTime1D":
        in_ch = 1 if X_repr.ndim == 2 else X_repr.shape[1]
        return InceptionTime1D(in_channels=in_ch, **ctor), cfg
    if theta == "ResNet18_2D":
        return ResNet18_2D(in_channels=1, **ctor), cfg
    if theta == "PatchTST":
        return PatchTST(in_channels=1, seq_len=seq_len, **ctor), cfg
    raise ValueError(f"Unknown neural model: {theta!r}")


def _build_stat_model(theta: str):
    from nstad_bench.models.statistical import GBM, LogReg, RandomForest
    registry = {"LogReg": LogReg, "RandomForest": RandomForest, "GBM": GBM}
    return registry[theta](**_DEFAULT_MODEL_CFG[theta])


def _build_neural_adapt(psi: str, X_s: np.ndarray, y_s: np.ndarray):
    """Build a deep DA adapter, shrinking batch_size for long sequences.

    Mirrors the long-sequence guard in ``_build_neural_model``: deep-DA fine-
    tuning loops do the same backbone+projector forward pass as fit(), so
    activation memory scales the same way.  Drop batch_size from 64 → 16
    when T > 8000 to keep T4 GPU memory bounded.
    """
    from nstad_bench.adaptation import CoDATS, DeepCORAL, MK_MMD, SourceOnly
    hp = _DEFAULT_DA_HP[psi].copy()

    if X_s.ndim == 2 and X_s.shape[-1] > 8000:
        hp["batch_size"] = min(hp.get("batch_size", 64), 16)
        log.info("  long-seq guard: %s → batch=%d", psi, hp["batch_size"])
    elif X_s.ndim == 3 and X_s.shape[-1] > 4000:
        hp["batch_size"] = min(hp.get("batch_size", 64), 8)
        log.info("  long-seq guard: %s wide-T → batch=%d", psi, hp["batch_size"])

    if psi == "SourceOnly":
        return SourceOnly()
    if psi == "MK_MMD":
        return MK_MMD(X_s, y_s, **hp)
    if psi == "CoDATS":
        return CoDATS(X_s, y_s, **hp)
    if psi == "DeepCORAL":
        return DeepCORAL(X_s, y_s, **hp)
    raise ValueError(f"Unknown neural DA method: {psi!r}")


def _build_stat_adapt(psi: str, X_s: np.ndarray, y_s: np.ndarray):
    from nstad_bench.adaptation.statistical import (
        CORAL, SourceOnly, SubspaceAlignment,
    )
    hp = _DEFAULT_DA_HP[psi]
    if psi == "SourceOnly":
        return SourceOnly()
    if psi == "CORAL":
        return CORAL(X_s, y_s, **hp)
    if psi == "SubspaceAlignment":
        return SubspaceAlignment(X_s, y_s, **hp)
    raise ValueError(f"Unknown statistical DA method: {psi!r}")


def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _execute(branch: str, ds_key: str, phi: str, theta: str, psi: str, seed: int) -> dict[str, float]:
    """Run one full (φ, θ, ψ) pipeline and return metric dict.

    The pipeline matches the long-format runner ``_run_single`` for parity
    with existing MIT-BIH results, but emits a flat one-row summary instead
    of the long-format multi-metric table.
    """
    from sklearn.metrics import f1_score

    from nstad_bench.experiments.runner import _DATA_REGISTRY, _stratified_val_split
    from nstad_bench.metrics.scores import best_threshold, pr_auc, roc_auc

    loader = _DATA_REGISTRY[ds_key]
    X_s_raw, y_s, X_t_raw, y_t = loader()
    log.info("Data loaded: source=%s target=%s", X_s_raw.shape, X_t_raw.shape)

    repr_obj = _build_repr(phi)
    repr_obj.fit(X_s_raw)
    X_s = repr_obj.transform(X_s_raw).astype(np.float32)
    X_t = repr_obj.transform(X_t_raw).astype(np.float32)
    log.info("Repr %s applied: %s → %s", phi, X_s_raw.shape, X_s.shape)

    t0 = time.time()
    if branch == "deep":
        import torch  # noqa: WPS433
        model, cfg = _build_neural_model(theta, X_s)
        device = _get_device()
        model = model.to(device)
        log.info("Training %s on %s …", theta, device)
        model.fit(
            X_s, y_s,
            epochs=int(cfg.get("epochs", 30)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=int(cfg.get("batch_size", 64)),
        )
    else:
        model = _build_stat_model(theta)
        log.info("Fitting %s …", theta)
        model.fit(X_s, y_s)

    # Source val split — identical RNG to the long-format runner so the held-out
    # source pool is the same one used for early-stopping and threshold selection.
    _, _, X_val, y_val = _stratified_val_split(X_s, y_s)
    in_domain_auc = float(roc_auc(y_val, model.predict_proba(X_val)[:, 1]))

    if branch == "deep":
        adapt = _build_neural_adapt(psi, X_s, y_s)
    else:
        adapt = _build_stat_adapt(psi, X_s, y_s)
    adapted = adapt.adapt(model, X_t)

    val_score = adapted.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, val_score)
    tgt_score = adapted.predict_proba(X_t)[:, 1]

    cross_domain_auc = float(roc_auc(y_t, tgt_score))
    pr_auc_val = float(pr_auc(y_t, tgt_score))
    f1_val = float(f1_score(y_t, (tgt_score >= threshold).astype(int), zero_division=0))
    training_time_sec = time.time() - t0
    delta_auc = in_domain_auc - cross_domain_auc

    log.info(
        "DONE: in=%.4f tgt=%.4f Δ=%+.4f f1=%.4f t=%.1fs",
        in_domain_auc, cross_domain_auc, delta_auc, f1_val, training_time_sec,
    )

    return {
        "roc_auc":           cross_domain_auc,   # primary metric is target ROC-AUC
        "pr_auc":            pr_auc_val,
        "f1":                f1_val,
        "in_domain_auc":     in_domain_auc,
        "cross_domain_auc":  cross_domain_auc,
        "delta_auc":         delta_auc,
        "training_time_sec": training_time_sec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV append + gain_over_source resolution
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_source_only_auc(csv_path: Path, args: argparse.Namespace) -> float | None:
    """Return the cross_domain_auc for the matching source-only row, if any."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row["dataset"] == args.dataset
                        and row["representation"] == args.representation
                        and row["model"] == args.model
                        and row["da_method"] == "source-only"
                        and int(row["seed"]) == int(args.seed)):
                    return float(row["cross_domain_auc"])
    except Exception as exc:
        log.warning("Could not read existing CSV %s: %s", csv_path, exc)
    return None


def _strip_existing_row(csv_path: Path, args: argparse.Namespace) -> None:
    """Remove any prior row matching the exact run keys (re-run scenario).

    Re-runs are explicit: invoking the script with the same arguments produces
    a single, authoritative row.  Without this, the CSV would accumulate stale
    rows from failed earlier attempts.
    """
    if not csv_path.exists():
        return
    keep: list[dict[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("dataset") == args.dataset
                    and row.get("representation") == args.representation
                    and row.get("model") == args.model
                    and row.get("da_method") == args.da_method
                    and row.get("seed") == str(args.seed)):
                continue
            keep.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(keep)


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one (dataset, repr, model, da-method, seed) config.",
    )
    p.add_argument("--dataset",        required=True, choices=sorted(DATASETS))
    p.add_argument("--representation", required=True, choices=sorted(REPRESENTATIONS))
    p.add_argument("--model",          required=True, choices=sorted(MODELS))
    p.add_argument("--da-method",      required=True, choices=sorted(DA_METHODS))
    p.add_argument("--seed",           required=True, type=int)
    p.add_argument("--output-csv",     default=None,
                   help="Path to the append-mode CSV. Defaults to $OUTPUT_ROOT/results.csv.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def _default_output_csv() -> Path:
    from nstad_bench.data._paths import resolve_output_root
    return resolve_output_root() / "results.csv"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    branch, ds_key, phi, theta, psi = _validate(args)
    _set_seeds(args.seed)
    _register_dataset(args.dataset, ds_key, args.seed)

    log.info(
        "Config: ds=%s repr=%s model=%s da=%s seed=%d  [%s branch]",
        args.dataset, args.representation, args.model, args.da_method, args.seed, branch,
    )

    metrics = _execute(branch, ds_key, phi, theta, psi, args.seed)

    csv_path = Path(args.output_csv) if args.output_csv else _default_output_csv()
    if args.da_method == "source-only":
        gain = 0.0
    else:
        so_auc = _lookup_source_only_auc(csv_path, args)
        gain = (metrics["cross_domain_auc"] - so_auc) if so_auc is not None else float("nan")

    row = {
        "dataset":           args.dataset,
        "representation":    args.representation,
        "model":             args.model,
        "da_method":         args.da_method,
        "seed":              args.seed,
        "roc_auc":           f"{metrics['roc_auc']:.6f}",
        "pr_auc":            f"{metrics['pr_auc']:.6f}",
        "f1":                f"{metrics['f1']:.6f}",
        "in_domain_auc":     f"{metrics['in_domain_auc']:.6f}",
        "cross_domain_auc":  f"{metrics['cross_domain_auc']:.6f}",
        "delta_auc":         f"{metrics['delta_auc']:.6f}",
        "gain_over_source":  f"{gain:.6f}" if not (isinstance(gain, float) and np.isnan(gain)) else "",
        "training_time_sec": f"{metrics['training_time_sec']:.4f}",
        "timestamp":         datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    _strip_existing_row(csv_path, args)
    _append_row(csv_path, row)
    log.info("Row appended → %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

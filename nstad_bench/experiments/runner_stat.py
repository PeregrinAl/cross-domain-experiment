"""Two-stage experiment runner for the *statistical* branch of nstad_bench.

Parallel of :mod:`nstad_bench.experiments.runner` that drives
:class:`nstad_bench.models.statistical.base.StatModel` and
:class:`nstad_bench.adaptation.statistical.base.BaseStatAdaptation`
implementations instead of the neural ones.

Reuses the shared infrastructure
--------------------------------
The neural runner already contains a lot of code that is independent of
the model/adaptation registries — config hashing, HP sampling, the
result schema, top-K screening, gain rows, checkpointing, save helpers.
This module imports all of that from
:mod:`nstad_bench.experiments.runner` so the two branches stay in sync.

The dataset registry is also shared — call
:func:`nstad_bench.experiments.runner.register_dataset` once per loader
and either runner can pick it up.

What is replaced
----------------
  * Model / adaptation / representation registries → statistical classes.
  * ``_build_model`` / ``_build_adapt`` → constructor switches that
    forward all YAML kwargs straight to the underlying sklearn-style
    estimator (no ``in_channels`` / ``seq_len`` auto-detection,
    no ``epochs`` / ``lr`` / ``batch_size`` training keys).
  * ``_run_single`` → numpy-only pipeline (no ``torch.manual_seed``,
    no ``model.to(device)``, no batched inference).
  * Default compatibility-matrix path → ``configs/statistical/``.

Public entry point
------------------
:func:`run_experiment_stat` — same signature and behaviour as
:func:`nstad_bench.experiments.runner.run_experiment`, including
checkpoint/resume semantics and the ``stage1_only`` / ``dry_run`` flags.
The output Parquet schema is identical, so both branches' results can
be concatenated and analysed by the same :mod:`nstad_bench.analysis`
pipeline.

Quick start
-----------
::

    from nstad_bench.experiments.runner import register_dataset
    from nstad_bench.experiments.runner_stat import run_experiment_stat

    register_dataset("mitbih_ds1_ds2", mitbih_loader())
    df = run_experiment_stat("configs/statistical/mitbih.yaml")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nstad_bench.adaptation.statistical import (
    CORAL,
    KMM,
    ImportanceReweighting,
    SourceOnly,
    SubspaceAlignment,
)
from nstad_bench.experiments.runner import (
    RESULT_COLS,
    RunConfig,
    _add_gain_rows,
    _adaptation_configs,
    _load_checkpoint,
    _load_compat_mask,
    _row,
    _save,
    _save_checkpoint,
    _screening_configs,
    _select_top_k,
    _stratified_val_split,
    _DATA_REGISTRY,
    register_dataset,
)
from nstad_bench.metrics.bootstrap import bootstrap_ci
from nstad_bench.metrics.scores import (
    best_threshold,
    mcc,
    pr_auc,
    roc_auc,
)
from nstad_bench.models.statistical import (
    GBM,
    SVM,
    LogReg,
    RandomForest,
    StatTestClassifier,
)
from nstad_bench.representations import CWT_Morlet, LogSTFT, RawSignal

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical-branch registries
# ─────────────────────────────────────────────────────────────────────────────

_REPR_REGISTRY: dict[str, type] = {
    "RawSignal":  RawSignal,
    "LogSTFT":    LogSTFT,
    "CWT_Morlet": CWT_Morlet,
}

_MODEL_REGISTRY: dict[str, type] = {
    "LogReg":             LogReg,
    "RandomForest":       RandomForest,
    "GBM":                GBM,
    "SVM":                SVM,
    "StatTestClassifier": StatTestClassifier,
}

_ADAPT_REGISTRY: dict[str, type] = {
    "SourceOnly":            SourceOnly,
    "CORAL":                 CORAL,
    "SubspaceAlignment":     SubspaceAlignment,
    "ImportanceReweighting": ImportanceReweighting,
    "KMM":                   KMM,
}


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(
    theta: str,
    X_repr: np.ndarray,  # noqa: ARG001 — kept for parity with the neural runner
    model_cfg: dict[str, Any],
) -> Any:
    """Construct a statistical model instance from *theta* and YAML kwargs.

    Statistical models receive every YAML key as a constructor kwarg —
    there are no auto-determined params (unlike ``in_channels`` /
    ``seq_len`` in the neural runner) and no training-only keys
    (epochs / lr / batch_size).
    """
    cls = _MODEL_REGISTRY[theta]
    return cls(**model_cfg)


def _build_adapt(
    psi: str,
    hp: dict[str, Any],
    X_s_repr: np.ndarray,
    y_s: np.ndarray,
) -> Any:
    """Construct a statistical adaptation method.

    Convention (mirrors the neural runner):
      - ``SourceOnly``: no args
      - everything else: ``(X_source, y_source, **hp)``
    """
    cls = _ADAPT_REGISTRY[psi]
    if psi == "SourceOnly":
        return cls()
    if psi in ("CORAL", "SubspaceAlignment", "ImportanceReweighting", "KMM"):
        return cls(X_s_repr, y_s, **hp)
    raise ValueError(
        f"No builder defined for psi={psi!r}.  "
        "Register it in _ADAPT_REGISTRY and _build_adapt()."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-run execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_single(
    cfg: RunConfig,
    model_train_cfg: dict[str, dict[str, Any]],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    """Execute one (φ, θ, ψ, dataset, seed, hp_trial) run on the
    statistical branch.

    Same threshold-selection / metric-bootstrapping protocol as the
    neural ``_run_single``, but without any torch / device handling.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    log.debug("  → %s", cfg)

    # ── Load data ────────────────────────────────────────────────────────────
    loader = _DATA_REGISTRY.get(cfg.dataset)
    if loader is None:
        raise KeyError(
            f"Dataset {cfg.dataset!r} is not registered. "
            "Call nstad_bench.experiments.runner.register_dataset() first."
        )
    X_s_raw, y_s, X_t_raw, y_t = loader()
    log.debug(
        "    data: source=%s target=%s", X_s_raw.shape, X_t_raw.shape
    )

    # ── Reproducibility ──────────────────────────────────────────────────────
    np.random.seed(cfg.seed)

    # ── Representation (fitted on source only) ───────────────────────────────
    repr_obj = _REPR_REGISTRY[cfg.phi]()
    repr_obj.fit(X_s_raw)
    X_s = repr_obj.transform(X_s_raw).astype(np.float32)
    X_t = repr_obj.transform(X_t_raw).astype(np.float32)
    log.debug("    repr %s: %s → %s", cfg.phi, X_s_raw.shape, X_s.shape)

    # ── Train source model ───────────────────────────────────────────────────
    m_cfg = model_train_cfg.get(cfg.theta, {})
    model = _build_model(cfg.theta, X_s, m_cfg)
    model.fit(X_s, y_s)

    # Re-derive the same val split used by the neural branch (same RNG seed)
    # so threshold selection lives on a held-out source-val pool.
    _, _, X_val, y_val = _stratified_val_split(X_s, y_s)

    src_auc = roc_auc(y_s, model.predict_proba(X_s)[:, 1])

    # ── Adapt ────────────────────────────────────────────────────────────────
    adapt_obj = _build_adapt(cfg.psi, cfg.hp_dict, X_s, y_s)
    adapted   = adapt_obj.adapt(model, X_t)

    # ── Threshold selection on source-val (with the *adapted* model) ─────────
    val_score = adapted.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, val_score)

    val_pred  = (val_score >= threshold).astype(int)
    src_val_f1 = float(f1_score(y_val, val_pred, zero_division=0.0))

    # ── Target metrics with bootstrap CI ─────────────────────────────────────
    tgt_score = adapted.predict_proba(X_t)[:, 1]

    ci_roc = bootstrap_ci(
        roc_auc, y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed
    )
    ci_pr = bootstrap_ci(
        pr_auc, y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed
    )
    d_auc = float(src_auc - ci_roc.estimate)

    def _f1_at(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(f1_score(yt, (ys >= threshold).astype(int),
                              zero_division=0.0))

    def _prec_at(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(precision_score(yt, (ys >= threshold).astype(int),
                                     zero_division=0.0))

    def _rec_at(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(recall_score(yt, (ys >= threshold).astype(int),
                                  zero_division=0.0))

    def _mcc_at(yt: np.ndarray, ys: np.ndarray) -> float:
        return mcc(yt, ys, threshold=threshold)

    ci_f1   = bootstrap_ci(_f1_at,   y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed)
    ci_prec = bootstrap_ci(_prec_at, y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed)
    ci_rec  = bootstrap_ci(_rec_at,  y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed)
    ci_mcc  = bootstrap_ci(_mcc_at,  y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed)

    d_f1 = src_val_f1 - ci_f1.estimate

    log.info(
        "  %-60s  roc=%.4f  f1=%.4f  Δauc=%+.4f  Δf1=%+.4f  t=%.3f",
        str(cfg), ci_roc.estimate, ci_f1.estimate, d_auc, d_f1, threshold,
    )

    return [
        _row(cfg, "roc_auc",        ci_roc.estimate, ci_roc.lower, ci_roc.upper),
        _row(cfg, "pr_auc",         ci_pr.estimate,  ci_pr.lower,  ci_pr.upper),
        _row(cfg, "source_roc_auc", src_auc),
        _row(cfg, "delta_auc",      d_auc),
        _row(cfg, "f1",             ci_f1.estimate,   ci_f1.lower,   ci_f1.upper),
        _row(cfg, "precision",      ci_prec.estimate, ci_prec.lower, ci_prec.upper),
        _row(cfg, "recall",         ci_rec.estimate,  ci_rec.lower,  ci_rec.upper),
        _row(cfg, "mcc",            ci_mcc.estimate,  ci_mcc.lower,  ci_mcc.upper),
        _row(cfg, "source_val_f1",  src_val_f1),
        _row(cfg, "delta_f1",       d_f1),
        _row(cfg, "threshold",      threshold),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_stat(
    config_path: str | Path,
    *,
    config_root: str | Path | None = None,
    dry_run: bool = False,
    stage1_only: bool = False,
) -> pd.DataFrame:
    """Load a YAML config and run the two-stage statistical experiment.

    Same semantics as :func:`nstad_bench.experiments.runner.run_experiment`
    (checkpoint/resume, ``stage1_only``, ``dry_run``, fault isolation).
    The only differences are the model/adaptation registries and the
    default ``config_root``.

    Parameters
    ----------
    config_path :
        Path to the experiment YAML.
    config_root :
        Directory containing ``compatibility.yaml``.  Defaults to
        ``<repo_root>/configs/statistical/`` for this runner (mirrors
        the neural runner's default of ``<repo_root>/configs/``).
    dry_run :
        Expand and log the run queue without executing any run.
    stage1_only :
        Run only the SourceOnly screening stage and stop.
    """
    config_path = Path(config_path)
    if config_root is None:
        # Default: <repo_root>/configs/statistical
        # runner_stat.py lives at nstad_bench/experiments/runner_stat.py
        # → parents[3] is <repo_root>
        config_root = Path(__file__).resolve().parents[2] / "configs" / "statistical"
    config_root = Path(config_root)

    with open(config_path) as f:
        exp_cfg = yaml.safe_load(f)

    exp_name    = exp_cfg["experiment_name"]
    n_bootstrap = int(exp_cfg.get("n_bootstrap", 200))
    output_dir  = Path(exp_cfg.get("output_dir", "results"))
    output_path = output_dir / f"{exp_name}.parquet"
    base_seed   = int(exp_cfg["random_search"].get("base_seed", 0))
    top_k       = int(exp_cfg["screening"]["top_k"])
    s_metric    = exp_cfg["screening"].get("metric", "delta_auc")
    model_train_cfg: dict[str, dict[str, Any]] = exp_cfg.get("models", {})

    compat = _load_compat_mask(config_root)

    s1_ckpt_path = output_dir / f"{exp_name}.s1.parquet"
    s2_ckpt_path = output_dir / f"{exp_name}.s2.parquet"

    log.info("=" * 70)
    log.info("Experiment : %s  [statistical branch]", exp_name)
    log.info("Output     : %s", output_path)
    log.info("n_bootstrap: %d", n_bootstrap)
    log.info("=" * 70)

    # ── Stage 1: Screening ───────────────────────────────────────────────────
    log.info("--- Stage 1: Screening (SourceOnly) ---")
    s1_cfgs = _screening_configs(exp_cfg, compat)

    if dry_run:
        log.info(
            "DRY RUN — Stage 1: %d configs  (Stage 2 TBD after screening)",
            len(s1_cfgs),
        )
        return pd.DataFrame(columns=RESULT_COLS)

    s1_rows, s1_done_hashes = _load_checkpoint(s1_ckpt_path)
    if s1_done_hashes:
        log.info(
            "Resume: Stage 1 checkpoint found — %d/%d configs already done",
            len(s1_done_hashes), len(s1_cfgs),
        )

    for i, cfg in enumerate(s1_cfgs, 1):
        if cfg.config_hash in s1_done_hashes:
            log.info("[S1 %d/%d] SKIP (checkpoint) %s", i, len(s1_cfgs), cfg)
            continue
        log.info("[S1 %d/%d] %s", i, len(s1_cfgs), cfg)
        try:
            new_rows = _run_single(cfg, model_train_cfg, n_bootstrap)
            s1_rows.extend(new_rows)
            s1_done_hashes.add(cfg.config_hash)
            _save_checkpoint(s1_rows, s1_ckpt_path)
        except Exception as exc:
            log.error("FAILED [S1 %d/%d] %s — %s", i, len(s1_cfgs), cfg, exc,
                      exc_info=True)

    top_pairs = _select_top_k(s1_rows, top_k, s_metric)

    if stage1_only:
        df = pd.DataFrame(s1_rows)[RESULT_COLS]
        s1_results_path = output_dir / f"{exp_name}.s1_results.parquet"
        _save(df, output_dir, s1_results_path)
        log.info(
            "=== Stage 1 complete: %d rows saved to %s ===",
            len(df), s1_results_path,
        )
        log.info(
            "    Checkpoint kept at %s — Stage 2 will resume from here.",
            s1_ckpt_path,
        )
        if not top_pairs:
            log.warning("No (φ, θ) pairs survived screening.")
        return df

    if not top_pairs:
        log.warning("No (φ, θ) pairs survived screening; skipping Stage 2.")
        df = pd.DataFrame(s1_rows)[RESULT_COLS]
        _save(df, output_dir, output_path)
        s1_ckpt_path.unlink(missing_ok=True)
        return df

    # ── Stage 2: Adaptation ──────────────────────────────────────────────────
    log.info("--- Stage 2: Adaptation (top-%d pairs × all ψ) ---", top_k)
    s2_cfgs = _adaptation_configs(exp_cfg, compat, top_pairs, base_seed)

    s2_rows, s2_done_hashes = _load_checkpoint(s2_ckpt_path)
    if s2_done_hashes:
        log.info(
            "Resume: Stage 2 checkpoint found — %d/%d configs already done",
            len(s2_done_hashes), len(s2_cfgs),
        )

    for i, cfg in enumerate(s2_cfgs, 1):
        if cfg.config_hash in s2_done_hashes:
            log.info("[S2 %d/%d] SKIP (checkpoint) %s", i, len(s2_cfgs), cfg)
            continue
        log.info("[S2 %d/%d] %s", i, len(s2_cfgs), cfg)
        try:
            new_rows = _run_single(cfg, model_train_cfg, n_bootstrap)
            s2_rows.extend(new_rows)
            s2_done_hashes.add(cfg.config_hash)
            _save_checkpoint(s2_rows, s2_ckpt_path)
        except Exception as exc:
            log.error("FAILED [S2 %d/%d] %s — %s", i, len(s2_cfgs), cfg, exc,
                      exc_info=True)

    all_rows = s1_rows + s2_rows
    df = pd.DataFrame(all_rows)[RESULT_COLS]
    df = _add_gain_rows(df)

    _save(df, output_dir, output_path)
    log.info(
        "=== Done: %d rows (%d unique config_hashes) saved to %s ===",
        len(df),
        df["config_hash"].nunique(),
        output_path,
    )

    s1_ckpt_path.unlink(missing_ok=True)
    s2_ckpt_path.unlink(missing_ok=True)
    log.info("Checkpoints removed (run completed successfully)")

    return df


__all__ = [
    "register_dataset",
    "run_experiment_stat",
    "RunConfig",
    "RESULT_COLS",
]

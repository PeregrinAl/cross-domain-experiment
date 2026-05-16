"""Two-stage experiment runner for the nstad_bench benchmark.

Pipeline
--------
Stage 1 — Screening
    Run every valid (φ, θ) pair with SourceOnly across all datasets and seeds.
    Rank pairs by mean ΔAUC = AUC_source − AUC_target (largest gap = most
    interesting adaptation target).  Select the top-K pairs.

Stage 2 — Adaptation
    Run the top-K (φ, θ) pairs with every adaptation method ψ.
    For methods with a hyperparameter space, perform random search over
    *n_trials* independent HP draws, each evaluated with all configured seeds.

Output schema (long Parquet, one row per run × metric)
------------------------------------------------------
config_hash       16-char SHA-256 prefix that uniquely identifies the run.
dataset           Dataset name (matches the registered loader key).
phi               Representation name (e.g. "RawSignal").
theta             Model name (e.g. "InceptionTime1D").
psi               Adaptation method name (e.g. "MK_MMD").
seed              Integer random seed used for model init and adaptation.
metric_name       One of: roc_auc, pr_auc, source_roc_auc, delta_auc, gain.
metric_value      Point estimate.
metric_ci_lower   Bootstrap 95 % CI lower bound (NaN for scalar metrics).
metric_ci_upper   Bootstrap 95 % CI upper bound (NaN for scalar metrics).

Usage
-----
::

    # 1. Register one or more dataset loaders.
    from nstad_bench.experiments.runner import register_dataset
    register_dataset("my_dataset", lambda: (X_s, y_s, X_t, y_t))

    # 2. Run the experiment.
    from nstad_bench.experiments.runner import run_experiment
    df = run_experiment("configs/my_experiment.yaml")

YAML config format
------------------
See ``configs/experiment_example.yaml`` for a fully annotated example.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from nstad_bench.adaptation import CoDATS, M2N2, MK_MMD, SourceOnly
from nstad_bench.metrics.bootstrap import bootstrap_ci
from nstad_bench.metrics.scores import pr_auc, roc_auc
from nstad_bench.models import InceptionTime1D, PatchTST, ResNet18_2D
from nstad_bench.representations import CARLA_SSL, CWT_Morlet, LogSTFT, RawSignal

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Public dataset registry
# ─────────────────────────────────────────────────────────────────────────────

_DATA_REGISTRY: dict[
    str,
    Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
] = {}


def register_dataset(
    name: str,
    loader: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Register a dataset loader with the experiment runner.

    Parameters
    ----------
    name :
        Identifier used in YAML configs under the ``datasets`` key, e.g.
        ``"mitbih_ds1_ds2"``.
    loader :
        Zero-argument callable that returns
        ``(X_source, y_source, X_target, y_target)`` as float32 / int64
        numpy arrays.  Called once per run; caching is the caller's
        responsibility (e.g. wrap with ``functools.lru_cache``).
    """
    _DATA_REGISTRY[name] = loader
    log.debug("Registered dataset %r", name)


# ─────────────────────────────────────────────────────────────────────────────
# Internal registries (not user-facing)
# ─────────────────────────────────────────────────────────────────────────────

_REPR_REGISTRY: dict[str, type] = {
    "RawSignal":  RawSignal,
    "LogSTFT":    LogSTFT,
    "CWT_Morlet": CWT_Morlet,
    "CARLA_SSL":  CARLA_SSL,
}

_MODEL_REGISTRY: dict[str, type] = {
    "InceptionTime1D": InceptionTime1D,
    "ResNet18_2D":     ResNet18_2D,
    "PatchTST":        PatchTST,
}

_ADAPT_REGISTRY: dict[str, type] = {
    "SourceOnly": SourceOnly,
    "MK_MMD":     MK_MMD,
    "CoDATS":     CoDATS,
    "M2N2":       M2N2,
}

# Constructor params that are auto-determined from data; ignored if present in YAML.
_AUTO_PARAMS: dict[str, set[str]] = {
    "InceptionTime1D": {"in_channels"},
    "ResNet18_2D":     {"in_channels"},
    "PatchTST":        {"in_channels", "seq_len"},
}
# Params that control training, not model architecture.
_TRAIN_KEYS = frozenset({"epochs", "lr", "batch_size"})


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility mask
# ─────────────────────────────────────────────────────────────────────────────

def _load_compat_mask(config_root: Path) -> dict[str, dict[str, bool]]:
    path = config_root / "compatibility.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["compatibility"]


def _is_compatible(phi: str, theta: str, mask: dict[str, dict[str, bool]]) -> bool:
    return bool(mask.get(phi, {}).get(theta, False))


# ─────────────────────────────────────────────────────────────────────────────
# Config hash
# ─────────────────────────────────────────────────────────────────────────────

def _config_hash(d: dict[str, Any]) -> str:
    """Return the first 16 hex characters of the SHA-256 of canonical JSON."""
    canon = json.dumps(d, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_hp(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sample one HP configuration from a space specification.

    Spec types accepted in the YAML ``adaptation_methods.<psi>`` dict
    ----------------------------------------------------------------
    ``{type: log_float, low: …, high: …}``  log-uniform float in [low, high]
    ``{type: float,     low: …, high: …}``  uniform float in [low, high]
    ``{type: int,       low: …, high: …}``  uniform integer in [low, high]
    ``{type: choice,    choices: […]}``      uniform choice from list
    Any other value                          fixed (passed through unchanged)
    """
    sampled: dict[str, Any] = {}
    for key, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            sampled[key] = spec          # fixed value
            continue
        t = spec["type"]
        if t == "log_float":
            lo, hi = np.log10(spec["low"]), np.log10(spec["high"])
            sampled[key] = float(10.0 ** rng.uniform(lo, hi))
        elif t == "float":
            sampled[key] = float(rng.uniform(spec["low"], spec["high"]))
        elif t == "int":
            sampled[key] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif t == "choice":
            choices = spec["choices"]
            sampled[key] = choices[int(rng.integers(len(choices)))]
        else:
            raise ValueError(
                f"Unknown HP spec type {t!r} for key {key!r}. "
                "Supported: log_float, float, int, choice."
            )
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Run descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    """Immutable descriptor for one model-training + adaptation run.

    ``hp_params`` is stored as a sorted tuple of ``(key, value)`` pairs so
    the dataclass remains hashable even when HP values are lists or floats.
    """

    dataset:   str
    phi:       str    # representation
    theta:     str    # model
    psi:       str    # adaptation method
    seed:      int
    hp_trial:  int    # 0-based HP trial index (0 = only trial for SourceOnly)
    hp_params: tuple  # sorted ((key, value), …) — reconstruct with .hp_dict
    stage:     str    # "screening" | "adaptation"

    @property
    def hp_dict(self) -> dict[str, Any]:
        return dict(self.hp_params)

    @property
    def config_hash(self) -> str:
        return _config_hash({
            "dataset":  self.dataset,
            "phi":      self.phi,
            "theta":    self.theta,
            "psi":      self.psi,
            "seed":     self.seed,
            "hp_trial": self.hp_trial,
            "hp":       self.hp_dict,
        })

    def __str__(self) -> str:
        return (
            f"[{self.stage}] {self.phi}×{self.theta}×{self.psi}"
            f"  ds={self.dataset}  seed={self.seed}  trial={self.hp_trial}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result schema helpers
# ─────────────────────────────────────────────────────────────────────────────

RESULT_COLS = [
    "config_hash",
    "dataset",
    "phi",
    "theta",
    "psi",
    "seed",
    "metric_name",
    "metric_value",
    "metric_ci_lower",
    "metric_ci_upper",
]


def _row(
    cfg: RunConfig,
    metric_name: str,
    value: float,
    lower: float = float("nan"),
    upper: float = float("nan"),
) -> dict[str, Any]:
    return {
        "config_hash":     cfg.config_hash,
        "dataset":         cfg.dataset,
        "phi":             cfg.phi,
        "theta":           cfg.theta,
        "psi":             cfg.psi,
        "seed":            cfg.seed,
        "metric_name":     metric_name,
        "metric_value":    float(value),
        "metric_ci_lower": float(lower),
        "metric_ci_upper": float(upper),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model / adaptation builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(
    theta: str,
    X_repr: np.ndarray,
    model_cfg: dict[str, Any],
) -> Any:
    """Construct the right model instance from *theta* and data shape.

    Parameters
    ----------
    theta :
        Model name (key into ``_MODEL_REGISTRY``).
    X_repr :
        Representative array of shape ``(N, T)`` or ``(N, F, T)``.
    model_cfg :
        Per-theta dict from the experiment YAML (may contain training keys
        such as ``epochs``, ``lr``, ``batch_size``, and constructor keys such
        as ``nb_filters``, ``depth``, etc.).
    """
    cls   = _MODEL_REGISTRY[theta]
    extra = {
        k: v for k, v in model_cfg.items()
        if k not in _TRAIN_KEYS | _AUTO_PARAMS.get(theta, set())
    }
    if theta == "InceptionTime1D":
        in_ch = 1 if X_repr.ndim == 2 else X_repr.shape[1]
        return cls(in_channels=in_ch, **extra)
    elif theta == "ResNet18_2D":
        return cls(in_channels=1, **extra)
    elif theta == "PatchTST":
        return cls(in_channels=1, seq_len=int(X_repr.shape[-1]), **extra)
    else:
        return cls(**extra)


def _build_adapt(
    psi: str,
    hp: dict[str, Any],
    X_s_repr: np.ndarray,
    y_s: np.ndarray,
) -> Any:
    """Construct the adaptation method with the given HP dict."""
    cls = _ADAPT_REGISTRY[psi]
    if psi == "SourceOnly":
        return cls()
    elif psi in ("MK_MMD", "CoDATS"):
        return cls(X_s_repr, y_s, **hp)
    elif psi == "M2N2":
        return cls(**hp)
    else:
        raise ValueError(
            f"No builder defined for psi={psi!r}. "
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
    """Execute one (φ, θ, ψ, dataset, seed, hp_trial) run.

    Returns
    -------
    list[dict]
        One dict per metric (roc_auc, pr_auc, source_roc_auc, delta_auc).
        The ``gain`` metric is added post-hoc by ``_add_gain_rows``.
    """
    import torch  # local import to keep module importable without torch

    log.debug("  → %s", cfg)

    # ── Load data ─────────────────────────────────────────────────────────────
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

    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Representation (fitted on source only) ────────────────────────────────
    repr_obj = _REPR_REGISTRY[cfg.phi]()
    repr_obj.fit(X_s_raw)
    X_s = repr_obj.transform(X_s_raw).astype(np.float32)
    X_t = repr_obj.transform(X_t_raw).astype(np.float32)
    log.debug("    repr %s: %s → %s", cfg.phi, X_s_raw.shape, X_s.shape)

    # ── Train source model ────────────────────────────────────────────────────
    m_cfg = model_train_cfg.get(cfg.theta, {})
    model = _build_model(cfg.theta, X_s, m_cfg)
    model.fit(
        X_s, y_s,
        epochs=int(m_cfg.get("epochs", 30)),
        lr=float(m_cfg.get("lr", 1e-3)),
        batch_size=int(m_cfg.get("batch_size", 64)),
    )

    # Source AUC (on training data — proxy for source performance)
    src_auc = roc_auc(y_s, model.predict_proba(X_s)[:, 1])

    # ── Adapt ─────────────────────────────────────────────────────────────────
    adapt_obj = _build_adapt(cfg.psi, cfg.hp_dict, X_s, y_s)
    adapted   = adapt_obj.adapt(model, X_t)

    # ── Target metrics with bootstrap CI ─────────────────────────────────────
    tgt_score = adapted.predict_proba(X_t)[:, 1]

    ci_roc = bootstrap_ci(
        roc_auc, y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed
    )
    ci_pr = bootstrap_ci(
        pr_auc, y_t, tgt_score, n_bootstrap=n_bootstrap, seed=cfg.seed
    )
    d_auc = float(src_auc - ci_roc.estimate)

    log.info(
        "  %-60s  roc=%.4f [%.4f,%.4f]  Δauc=%+.4f",
        str(cfg), ci_roc.estimate, ci_roc.lower, ci_roc.upper, d_auc,
    )

    return [
        _row(cfg, "roc_auc",        ci_roc.estimate, ci_roc.lower, ci_roc.upper),
        _row(cfg, "pr_auc",         ci_pr.estimate,  ci_pr.lower,  ci_pr.upper),
        _row(cfg, "source_roc_auc", src_auc),
        _row(cfg, "delta_auc",      d_auc),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Config generation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_representations(raw: Any) -> list[str]:
    """Accept either a list of strings or a dict {name: params}."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        return list(raw.keys())
    raise ValueError(f"'representations' must be a list or dict, got {type(raw).__name__}")


def _screening_configs(
    exp_cfg: dict[str, Any],
    compat: dict[str, dict[str, bool]],
) -> list[RunConfig]:
    """All valid (φ, θ, dataset, seed) × SourceOnly for stage 1."""
    seeds    = exp_cfg["random_search"]["seeds"]
    datasets = exp_cfg["datasets"]
    phis     = _parse_representations(exp_cfg["representations"])
    thetas   = list(exp_cfg["models"].keys())

    configs: list[RunConfig] = []
    for phi, theta, dataset, seed in product(phis, thetas, datasets, seeds):
        if not _is_compatible(phi, theta, compat):
            log.debug("  skip incompatible: %s × %s", phi, theta)
            continue
        configs.append(RunConfig(
            dataset=dataset, phi=phi, theta=theta,
            psi="SourceOnly", seed=int(seed),
            hp_trial=0, hp_params=(),
            stage="screening",
        ))
    log.info("Stage 1 — %d screening configs", len(configs))
    return configs


def _adaptation_configs(
    exp_cfg: dict[str, Any],
    compat: dict[str, dict[str, bool]],
    top_pairs: list[tuple[str, str]],
    base_seed: int = 0,
) -> list[RunConfig]:
    """Top-K (φ, θ) × every ψ (with HP trials) × datasets × seeds."""
    seeds      = exp_cfg["random_search"]["seeds"]
    n_trials   = int(exp_cfg["random_search"]["n_trials"])
    datasets   = exp_cfg["datasets"]
    psi_spaces = exp_cfg["adaptation_methods"]

    configs: list[RunConfig] = []
    for (phi, theta), psi, dataset, seed in product(
        top_pairs, psi_spaces.keys(), datasets, seeds
    ):
        space = psi_spaces[psi] or {}
        if not space:
            # No HP space (e.g. SourceOnly) — single trial
            configs.append(RunConfig(
                dataset=dataset, phi=phi, theta=theta,
                psi=psi, seed=int(seed),
                hp_trial=0, hp_params=(),
                stage="adaptation",
            ))
        else:
            for trial in range(n_trials):
                # Deterministic RNG per (psi, trial, base_seed) — independent of seed
                rng_seed = int(
                    hashlib.sha256(
                        f"{psi}_{trial}_{base_seed}".encode()
                    ).hexdigest()[:8],
                    16,
                ) % (2 ** 31)
                hp = _sample_hp(space, np.random.default_rng(rng_seed))
                configs.append(RunConfig(
                    dataset=dataset, phi=phi, theta=theta,
                    psi=psi, seed=int(seed),
                    hp_trial=trial,
                    hp_params=tuple(sorted(hp.items())),
                    stage="adaptation",
                ))
    log.info("Stage 2 — %d adaptation configs", len(configs))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Screening — select top-K (φ, θ) pairs
# ─────────────────────────────────────────────────────────────────────────────

def _select_top_k(
    rows: list[dict[str, Any]],
    k: int,
    metric: str = "delta_auc",
) -> list[tuple[str, str]]:
    """Rank (φ, θ) pairs by mean *metric* and return the top-K.

    For ``delta_auc`` (positive = domain gap): higher → more interesting.
    For ``roc_auc`` / ``source_roc_auc``: higher → better representation.
    """
    df  = pd.DataFrame(rows)
    sub = df[df["metric_name"] == metric]

    if sub.empty:
        all_pairs = list({(r["phi"], r["theta"]) for r in rows})
        log.warning(
            "No rows for screening metric %r; returning all %d pairs.",
            metric, len(all_pairs),
        )
        return all_pairs[:k]

    ranked = (
        sub.groupby(["phi", "theta"])["metric_value"]
        .mean()
        .sort_values(ascending=False)
    )
    top = [pair for pair in ranked.index[:k]]
    log.info(
        "Top-%d (φ, θ) pairs by mean %s:\n%s",
        k, metric, ranked.head(k).to_string(),
    )
    return top


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc gain computation
# ─────────────────────────────────────────────────────────────────────────────

def _add_gain_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``gain = roc_auc_adapted − roc_auc_source_only`` rows.

    For each (φ, θ, dataset, seed), the SourceOnly roc_auc is used as the
    baseline.  For SourceOnly rows themselves, gain = 0.

    CI columns are left as NaN because the gain mixes two random variables
    whose joint distribution is not straightforward to bootstrap here.
    """
    roc = df[df["metric_name"] == "roc_auc"].copy()

    # Mean source-only roc_auc per (phi, theta, dataset, seed)
    ref = (
        roc[roc["psi"] == "SourceOnly"]
        .groupby(["phi", "theta", "dataset", "seed"])["metric_value"]
        .mean()
        .rename("so_auc")
        .reset_index()
    )
    if ref.empty:
        log.warning("No SourceOnly roc_auc rows found; skipping gain computation.")
        return df

    merged = roc.merge(ref, on=["phi", "theta", "dataset", "seed"], how="left")
    merged = merged.dropna(subset=["so_auc"])

    gain_rows = [
        {
            "config_hash":     r["config_hash"],
            "dataset":         r["dataset"],
            "phi":             r["phi"],
            "theta":           r["theta"],
            "psi":             r["psi"],
            "seed":            r["seed"],
            "metric_name":     "gain",
            "metric_value":    float(r["metric_value"] - r["so_auc"]),
            "metric_ci_lower": float("nan"),
            "metric_ci_upper": float("nan"),
        }
        for _, r in merged.iterrows()
    ]

    if gain_rows:
        df = pd.concat([df, pd.DataFrame(gain_rows)], ignore_index=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, output_dir: Path, output_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df[RESULT_COLS].to_parquet(output_path, index=False, engine="pyarrow")
    log.info("Saved %d rows → %s", len(df), output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config_path: str | Path,
    *,
    config_root: str | Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Load a YAML config and run the full two-stage experiment pipeline.

    Parameters
    ----------
    config_path :
        Path to the experiment YAML file.
    config_root :
        Directory that contains ``compatibility.yaml``.  Defaults to
        ``<repo_root>/configs/``.
    dry_run :
        If ``True``, expand and log the run queue without executing any run.
        Useful for estimating total compute before committing.
        Returns an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format results table (schema: ``RESULT_COLS``).
        Also written to ``<output_dir>/<experiment_name>.parquet``.

    Notes
    -----
    Failed runs are logged at ERROR level and skipped; the runner continues
    with remaining configs so a single exception never aborts the experiment.
    """
    config_path = Path(config_path)
    if config_root is None:
        # Default: <repo_root>/configs  (runner.py lives 3 levels deep)
        config_root = Path(__file__).resolve().parents[3] / "configs"
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

    log.info("=" * 70)
    log.info("Experiment : %s", exp_name)
    log.info("Output     : %s", output_path)
    log.info("n_bootstrap: %d", n_bootstrap)
    log.info("=" * 70)

    # ── Stage 1: Screening ────────────────────────────────────────────────────
    log.info("--- Stage 1: Screening (SourceOnly) ---")
    s1_cfgs = _screening_configs(exp_cfg, compat)

    if dry_run:
        s2_cfgs_estimate = len(s1_cfgs)  # upper bound before top-K is known
        log.info(
            "DRY RUN — Stage 1: %d configs  (Stage 2 TBD after screening)",
            s2_cfgs_estimate,
        )
        return pd.DataFrame(columns=RESULT_COLS)

    s1_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(s1_cfgs, 1):
        log.info("[S1 %d/%d] %s", i, len(s1_cfgs), cfg)
        try:
            s1_rows.extend(_run_single(cfg, model_train_cfg, n_bootstrap))
        except Exception as exc:
            log.error("FAILED [S1 %d/%d] %s — %s", i, len(s1_cfgs), cfg, exc,
                      exc_info=True)

    # ── Select top-K ─────────────────────────────────────────────────────────
    top_pairs = _select_top_k(s1_rows, top_k, s_metric)
    if not top_pairs:
        log.warning("No (φ, θ) pairs survived screening; skipping Stage 2.")
        df = pd.DataFrame(s1_rows, columns=RESULT_COLS)
        _save(df, output_dir, output_path)
        return df

    # ── Stage 2: Adaptation ───────────────────────────────────────────────────
    log.info("--- Stage 2: Adaptation (top-%d pairs × all ψ) ---", top_k)
    s2_cfgs = _adaptation_configs(exp_cfg, compat, top_pairs, base_seed)

    s2_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(s2_cfgs, 1):
        log.info("[S2 %d/%d] %s", i, len(s2_cfgs), cfg)
        try:
            s2_rows.extend(_run_single(cfg, model_train_cfg, n_bootstrap))
        except Exception as exc:
            log.error("FAILED [S2 %d/%d] %s — %s", i, len(s2_cfgs), cfg, exc,
                      exc_info=True)

    # ── Assemble, annotate, save ──────────────────────────────────────────────
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
    return df


__all__ = ["register_dataset", "run_experiment", "RunConfig", "RESULT_COLS"]

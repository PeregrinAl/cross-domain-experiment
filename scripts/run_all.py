"""run_all.py — run the full benchmark on both datasets.

Sequentially executes the two-stage pipeline (screening → adaptation) for
MIT-BIH DS1→DS2 and DeepBeat inter-patient split, writing Parquet results to
``results/``.

Usage
-----
    .venv/bin/python scripts/run_all.py [--dry-run] [--config-root CONFIGS_DIR]

Options
-------
--dry-run       Run 3-epoch smoke test instead of full configs (fast verification).
--config-root   Path to directory containing mitbih.yaml and deepbeat.yaml.
                Defaults to ``configs/`` relative to the repo root.

Output
------
Results are written to::

    results/mitbih_<timestamp>.parquet
    results/deepbeat_<timestamp>.parquet

Each run adds a new timestamped file — existing results are never overwritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")

BENCHMARK_DATASETS = [
    ("mitbih_ds1_ds2",       "mitbih.yaml"),
    ("deepbeat_patient_split", "deepbeat.yaml"),
]

# Minimal config used for --dry-run (3 epochs, small model)
_DRY_CFG = {
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


def _run_dataset(
    ds_name: str,
    config_path: Path,
    dry_run: bool,
) -> bool:
    """Run one dataset. Returns True on success."""
    from nstad_bench.experiments.runner import run_experiment

    log.info("━" * 60)
    log.info("  Dataset: %s", ds_name)
    log.info("  Config:  %s", config_path)
    if dry_run:
        log.info("  Mode:    DRY RUN (3 epochs)")
    log.info("━" * 60)

    try:
        if dry_run:
            # Load the real config to get experiment_name / output_dir,
            # then override everything else with the minimal dry-run settings.
            base = yaml.safe_load(config_path.read_text())
            cfg = {
                **_DRY_CFG,
                "experiment_name": base.get("experiment_name", ds_name) + "_dryrun",
                "output_dir":      base.get("output_dir", "results/"),
                "datasets":        [ds_name],
            }
            with tempfile.TemporaryDirectory() as tmp:
                tmp_cfg = Path(tmp) / "dry.yaml"
                tmp_cfg.write_text(yaml.dump(cfg))
                df = run_experiment(tmp_cfg, config_root=config_path.parent)
        else:
            df = run_experiment(config_path, config_root=config_path.parent)

        if df.empty:
            log.error("  ✗  %s: runner returned empty DataFrame", ds_name)
            return False

        auc_rows = df[df["metric_name"] == "roc_auc"]
        auc_val  = auc_rows["metric_value"].iloc[0] if len(auc_rows) else float("nan")
        log.info("  ✓  %s: %d rows  SourceOnly ROC-AUC=%.4f", ds_name, len(df), auc_val)
        return True

    except Exception as exc:
        log.exception("  ✗  %s: %s: %s", ds_name, type(exc).__name__, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full nstad_bench benchmark")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="3-epoch smoke test (fast verification, no real results)",
    )
    parser.add_argument(
        "--config-root", type=Path, default=ROOT / "configs",
        help="Directory containing *.yaml experiment configs",
    )
    args = parser.parse_args()

    config_root: Path = args.config_root
    results: dict[str, bool] = {}

    # Register loaders before running
    from nstad_bench.data.mitbih_loader import mitbih_loader
    from nstad_bench.data.deepbeat_loader import deepbeat_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("mitbih_ds1_ds2",         mitbih_loader())
    register_dataset("deepbeat_patient_split",  deepbeat_loader())

    for ds_name, config_file in BENCHMARK_DATASETS:
        config_path = config_root / config_file
        if not config_path.exists():
            log.error("Config not found: %s — skipping %s", config_path, ds_name)
            results[ds_name] = False
            continue
        results[ds_name] = _run_dataset(ds_name, config_path, dry_run=args.dry_run)

    # Summary
    log.info("")
    log.info("=" * 60)
    n_ok  = sum(results.values())
    n_all = len(results)
    if n_ok == n_all:
        log.info("  ALL %d DATASETS COMPLETED SUCCESSFULLY", n_all)
    else:
        log.error("  %d / %d DATASETS FAILED:", n_all - n_ok, n_all)
        for ds, ok in results.items():
            if not ok:
                log.error("    • %s", ds)
        sys.exit(1)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

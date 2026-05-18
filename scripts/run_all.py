"""run_all.py — run the full benchmark on both datasets.

Sequentially executes the two-stage pipeline (screening → adaptation) for
MIT-BIH DS1→DS2 and DeepBeat inter-patient split, writing Parquet results to
``results/``.

Usage
-----
    # Foreground (blocks terminal):
    .venv/bin/python scripts/run_all.py

    # Background with persistent log (recommended for long runs / SSH):
    mkdir -p logs
    nohup .venv/bin/python scripts/run_all.py \
        > logs/full_run_$(date +%Y%m%d_%H%M).log 2>&1 &
    echo "PID $!"

    # Or let the script manage the log file itself (--log-dir):
    .venv/bin/python scripts/run_all.py --log-dir logs &
    # → logs/run_20260518_1430.log  (stdout still echoed to terminal)

Options
-------
--dry-run       Run 3-epoch smoke test instead of full configs (fast verification).
--config-root   Path to directory containing mitbih.yaml and deepbeat.yaml.
                Defaults to ``configs/`` relative to the repo root.
--log-dir       Directory for the auto-named log file.  When given, all output
                goes to both the log file AND stdout (tee behaviour).
                Auto-named as ``run_YYYYMMDD_HHMM[_dry].log``.
--log-file      Explicit log file path (overrides --log-dir auto-naming).

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
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup  (called after args are parsed so we know the log path)
# ─────────────────────────────────────────────────────────────────────────────

_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_logging(log_file: Path | None) -> None:
    """Configure root logger: always stream to stdout; optionally tee to file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        handlers.append(fh)

    logging.basicConfig(
        level=logging.INFO,
        format=_FMT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,   # override any earlier basicConfig calls from imports
    )

    if log_file is not None:
        # Announce the log path on both streams before anything else is logged.
        print(f"Logging to: {log_file}", flush=True)


log = logging.getLogger("run_all")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_DATASETS = [
    ("mitbih_ds1_ds2",         "mitbih.yaml"),
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

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_dataset(
    ds_name: str,
    config_path: Path,
    dry_run: bool,
    stage1_only: bool = False,
) -> bool:
    """Run one dataset through the pipeline.  Returns True on success."""
    from nstad_bench.experiments.runner import run_experiment

    if dry_run:
        mode = "DRY RUN (3 epochs)"
    elif stage1_only:
        mode = "STAGE 1 ONLY (screening)"
    else:
        mode = "FULL"

    log.info("━" * 60)
    log.info("  Dataset : %s", ds_name)
    log.info("  Config  : %s", config_path)
    log.info("  Mode    : %s", mode)
    log.info("━" * 60)

    try:
        if dry_run:
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
            df = run_experiment(
                config_path,
                config_root=config_path.parent,
                stage1_only=stage1_only,
            )

        if df.empty:
            log.error("  ✗  %s: runner returned empty DataFrame", ds_name)
            return False

        auc_rows = df[df["metric_name"] == "roc_auc"]
        auc_val  = auc_rows["metric_value"].iloc[0] if len(auc_rows) else float("nan")
        log.info("  ✓  %s: %d rows  SourceOnly ROC-AUC=%.4f", ds_name, len(df), auc_val)
        return True

    except Exception:
        log.exception("  ✗  %s raised an exception", ds_name)
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full nstad_bench benchmark (MIT-BIH + DeepBeat).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="3-epoch smoke test — fast verification, no real results written",
    )
    parser.add_argument(
        "--stage1-only", action="store_true",
        help=(
            "Run only Stage 1 (SourceOnly screening) and stop. "
            "Results saved to results/<name>.s1_results.parquet. "
            "Checkpoint kept so Stage 2 can resume without re-running Stage 1."
        ),
    )
    parser.add_argument(
        "--config-root", type=Path, default=ROOT / "configs",
        metavar="DIR",
        help="Directory containing mitbih.yaml and deepbeat.yaml (default: configs/)",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        metavar="DIR",
        help="Write an auto-named log file here in addition to stdout",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        metavar="FILE",
        help="Explicit log file path (overrides --log-dir)",
    )
    args = parser.parse_args()

    # Resolve log file path
    log_file: Path | None = None
    if args.log_file:
        log_file = args.log_file
    elif args.log_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        if args.dry_run:
            suffix = "_dry"
        elif args.stage1_only:
            suffix = "_s1"
        else:
            suffix = ""
        log_file = args.log_dir / f"run_{stamp}{suffix}.log"

    _setup_logging(log_file)

    # Header
    log.info("=" * 60)
    log.info("  nstad_bench  run_all.py")
    log.info("  Started : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if log_file:
        log.info("  Log file: %s", log_file)
    if args.dry_run:
        mode_label = "DRY RUN"
    elif args.stage1_only:
        mode_label = "STAGE 1 ONLY"
    else:
        mode_label = "FULL"
    log.info("  Mode    : %s", mode_label)
    log.info("=" * 60)

    # Register loaders
    from nstad_bench.data.mitbih_loader import mitbih_loader
    from nstad_bench.data.deepbeat_loader import deepbeat_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("mitbih_ds1_ds2",         mitbih_loader())
    # Cap DeepBeat source to 10 000 per class (20 000 total).
    # Default is 50 000 per class (100 000 total), which makes each training
    # epoch ~5× slower than MIT-BIH and yields ~37 h for Stage 1 on CPU/MPS.
    # 10 000 per class still gives 3× more gradient updates per epoch than
    # MIT-BIH and is ample for InceptionTime1D / PatchTST convergence.
    register_dataset("deepbeat_patient_split",  deepbeat_loader(max_per_class=10_000))

    # Run each dataset
    config_root: Path = args.config_root
    results: dict[str, bool] = {}

    for ds_name, config_file in BENCHMARK_DATASETS:
        config_path = config_root / config_file
        if not config_path.exists():
            log.error("Config not found: %s — skipping %s", config_path, ds_name)
            results[ds_name] = False
            continue
        results[ds_name] = _run_dataset(
            ds_name, config_path,
            dry_run=args.dry_run,
            stage1_only=args.stage1_only,
        )

    # Summary
    n_ok  = sum(results.values())
    n_all = len(results)
    log.info("")
    log.info("=" * 60)
    log.info("  Finished: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if n_ok == n_all:
        log.info("  ALL %d DATASETS COMPLETED SUCCESSFULLY", n_all)
    else:
        log.error("  %d / %d DATASETS FAILED:", n_all - n_ok, n_all)
        for ds, ok in results.items():
            if not ok:
                log.error("    • %s", ds)
        sys.exit(1)
    log.info("=" * 60)
    if log_file:
        # Final reminder of where the log lives (useful when running in bg)
        print(f"\nLog written to: {log_file}", flush=True)


if __name__ == "__main__":
    main()

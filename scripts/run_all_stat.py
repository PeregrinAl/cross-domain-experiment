"""run_all_stat.py — run the full *statistical* benchmark on both datasets.

Parallel of :mod:`scripts.run_all` that drives the statistical-branch
runner (:func:`nstad_bench.experiments.runner_stat.run_experiment_stat`)
over the configs at ``configs/statistical/``.  Same CLI, same flags,
same logging — only the imported runner and the default ``config-root``
change.

Usage
-----
    .venv/bin/python scripts/run_all_stat.py
    .venv/bin/python scripts/run_all_stat.py --dry-run
    .venv/bin/python scripts/run_all_stat.py --stage1-only
    .venv/bin/python scripts/run_all_stat.py --log-dir logs &

Output
------
    results/statistical/mitbih_stat.parquet
    results/statistical/deepbeat_stat.parquet
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

_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        handlers.append(fh)
    logging.basicConfig(
        level=logging.INFO, format=_FMT, datefmt=_DATEFMT,
        handlers=handlers, force=True,
    )
    if log_file is not None:
        print(f"Logging to: {log_file}", flush=True)


log = logging.getLogger("run_all_stat")

# Datasets to run — same two as the neural ``run_all.py``.
BENCHMARK_DATASETS = [
    ("mitbih_ds1_ds2",         "mitbih.yaml"),
    ("deepbeat_patient_split", "deepbeat.yaml"),
]

# Minimal config for --dry-run.  Uses LogReg (fast, no neural keys).
_DRY_CFG = {
    "n_bootstrap": 20,
    "screening":     {"top_k": 1, "metric": "delta_auc"},
    "random_search": {"n_trials": 1, "seeds": [0], "base_seed": 0},
    "representations": {"RawSignal": {}},
    "models": {"LogReg": {"C": 1.0, "max_iter": 200, "random_state": 0}},
    "adaptation_methods": {"SourceOnly": {}},
}


def _run_dataset(
    ds_name: str,
    config_path: Path,
    dry_run: bool,
    stage1_only: bool = False,
) -> bool:
    from nstad_bench.experiments.runner_stat import run_experiment_stat

    if dry_run:
        mode = "DRY RUN"
    elif stage1_only:
        mode = "STAGE 1 ONLY (screening)"
    else:
        mode = "FULL"

    log.info("━" * 60)
    log.info("  Dataset : %s", ds_name)
    log.info("  Config  : %s", config_path)
    log.info("  Mode    : %s  [statistical branch]", mode)
    log.info("━" * 60)

    try:
        if dry_run:
            base = yaml.safe_load(config_path.read_text())
            cfg = {
                **_DRY_CFG,
                "experiment_name": base.get("experiment_name", ds_name) + "_dryrun",
                "output_dir":      base.get("output_dir", "results/statistical/"),
                "datasets":        [ds_name],
            }
            with tempfile.TemporaryDirectory() as tmp:
                tmp_cfg = Path(tmp) / "dry.yaml"
                tmp_cfg.write_text(yaml.dump(cfg))
                df = run_experiment_stat(tmp_cfg, config_root=config_path.parent)
        else:
            df = run_experiment_stat(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full nstad_bench *statistical* benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke-test with LogReg + SourceOnly only")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Run only Stage 1 (SourceOnly screening) and stop")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                        help="Subset of dataset keys to run (default: all "
                             "configured).  Example: --datasets mitbih_ds1_ds2")
    parser.add_argument("--config-root", type=Path,
                        default=ROOT / "configs" / "statistical",
                        metavar="DIR",
                        help="Directory with statistical configs "
                             "(default: configs/statistical/)")
    parser.add_argument("--log-dir", type=Path, default=None, metavar="DIR")
    parser.add_argument("--log-file", type=Path, default=None, metavar="FILE")
    args = parser.parse_args()

    log_file: Path | None = None
    if args.log_file:
        log_file = args.log_file
    elif args.log_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = "_dry" if args.dry_run else ("_s1" if args.stage1_only else "")
        log_file = args.log_dir / f"run_stat_{stamp}{suffix}.log"

    _setup_logging(log_file)

    log.info("=" * 60)
    log.info("  nstad_bench  run_all_stat.py")
    log.info("  Started : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if log_file:
        log.info("  Log file: %s", log_file)
    log.info("  Mode    : %s", "DRY RUN" if args.dry_run
                              else "STAGE 1 ONLY" if args.stage1_only else "FULL")
    log.info("=" * 60)

    # Filter BENCHMARK_DATASETS by --datasets if provided.
    if args.datasets:
        wanted = set(args.datasets)
        selected = [(n, c) for n, c in BENCHMARK_DATASETS if n in wanted]
        unknown = wanted - {n for n, _ in BENCHMARK_DATASETS}
        if unknown:
            log.error("Unknown dataset name(s): %s", sorted(unknown))
            log.error("Available: %s", [n for n, _ in BENCHMARK_DATASETS])
            sys.exit(2)
    else:
        selected = list(BENCHMARK_DATASETS)
    selected_names = {n for n, _ in selected}

    from nstad_bench.experiments.runner import register_dataset

    if "mitbih_ds1_ds2" in selected_names:
        from nstad_bench.data.mitbih_loader import mitbih_loader
        register_dataset("mitbih_ds1_ds2", mitbih_loader())
    if "deepbeat_patient_split" in selected_names:
        # Same DeepBeat cap as the neural ``run_all.py`` for parity.
        from nstad_bench.data.deepbeat_loader import deepbeat_loader
        register_dataset("deepbeat_patient_split", deepbeat_loader(max_per_class=10_000))

    config_root: Path = args.config_root
    results: dict[str, bool] = {}
    for ds_name, config_file in selected:
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

    n_ok, n_all = sum(results.values()), len(results)
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
        print(f"\nLog written to: {log_file}", flush=True)


if __name__ == "__main__":
    main()

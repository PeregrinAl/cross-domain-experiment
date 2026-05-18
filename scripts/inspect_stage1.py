"""inspect_stage1.py — analyse Stage 1 screening results before running Stage 2.

Reads the ``results/<name>.s1_results.parquet`` files produced by
``run_all.py --stage1-only`` and prints a sanity-check report covering:

1. Per-pair ΔAUC table (mean ± std across seeds, ranked).
2. Top-2 pairs per dataset — diversity check (not the same model twice).
3. Source-only ROC-AUC vs expected ranges:
       MIT-BIH  0.85–0.92
       DeepBeat 0.65–0.80
4. Bootstrap CI width check (95 % CI should be < 0.10).

Usage
-----
    # After run_all.py --stage1-only finishes:
    .venv/bin/python scripts/inspect_stage1.py

    # Custom results dir:
    .venv/bin/python scripts/inspect_stage1.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Expected ranges per dataset
# ─────────────────────────────────────────────────────────────────────────────

_AUC_RANGES: dict[str, tuple[float, float]] = {
    "mitbih_ds1_ds2":         (0.85, 0.92),
    "deepbeat_patient_split": (0.65, 0.80),
}

_CI_WIDTH_WARN = 0.10   # flag CI wider than this

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  ✓"
WARN = "  ⚠"
FAIL = "  ✗"
_issues: list[str] = []


def _ok(msg: str) -> None:
    print(PASS, msg)


def _warn(msg: str) -> None:
    print(WARN, msg)
    _issues.append(msg)


def _fail(msg: str) -> None:
    print(FAIL, msg)
    _issues.append(msg)


def _section(title: str) -> None:
    print()
    print("─" * 70)
    print(" ", title)
    print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_dataset(df: pd.DataFrame, ds_name: str, top_k: int = 2) -> None:
    _section(f"Dataset: {ds_name}  ({len(df)} rows)")

    roc  = df[df["metric_name"] == "roc_auc"].copy()
    dauc = df[df["metric_name"] == "delta_auc"].copy()

    # ── 1. ΔAUC table ─────────────────────────────────────────────────────────
    print()
    print("  ΔAUC per (φ, θ) pair  [mean ± std across seeds, ranked ↓]")
    print()

    dauc_stats = (
        dauc.groupby(["phi", "theta"])["metric_value"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    col_w = max(len(r) for r in dauc_stats["phi"] + "×" + dauc_stats["theta"]) + 2
    print(f"  {'pair':<{col_w}}  {'mean ΔAUC':>10}  {'std':>8}  {'n seeds':>7}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*8}  {'-'*7}")
    for _, row in dauc_stats.iterrows():
        pair = f"{row['phi']}×{row['theta']}"
        print(
            f"  {pair:<{col_w}}  {row['mean']:>+10.4f}  "
            f"{row['std']:>8.4f}  {int(row['count']):>7}"
        )

    # ── 2. Top-K pairs — diversity check ──────────────────────────────────────
    print()
    top_pairs = dauc_stats.head(top_k)[["phi", "theta"]].values.tolist()
    top_thetas = [t for _, t in top_pairs]
    top_phis   = [p for p, _ in top_pairs]

    print(f"  Top-{top_k} pairs (will enter Stage 2):")
    for phi, theta in top_pairs:
        mean_d = dauc_stats.loc[
            (dauc_stats["phi"] == phi) & (dauc_stats["theta"] == theta), "mean"
        ].values[0]
        print(f"    {phi}×{theta}  (mean ΔAUC={mean_d:+.4f})")

    if len(set(top_thetas)) == 1:
        _warn(
            f"{ds_name}: top-{top_k} pairs use the same model "
            f"({top_thetas[0]}). Consider whether this is expected."
        )
    else:
        _ok(f"{ds_name}: top-{top_k} pairs use different models → diverse")

    if len(set(top_phis)) == 1:
        _warn(
            f"{ds_name}: top-{top_k} pairs use the same representation "
            f"({top_phis[0]})."
        )
    else:
        _ok(f"{ds_name}: top-{top_k} pairs use different representations → diverse")

    # ── 3. Source-only ROC-AUC range check ────────────────────────────────────
    lo, hi = _AUC_RANGES.get(ds_name, (0.0, 1.0))
    auc_vals = roc["metric_value"]
    mean_auc = float(auc_vals.mean())
    std_auc  = float(auc_vals.std())

    print()
    print(f"  Source-only ROC-AUC:  mean={mean_auc:.4f}  std={std_auc:.4f}  "
          f"[min={auc_vals.min():.4f}, max={auc_vals.max():.4f}]")
    print(f"  Expected range: [{lo:.2f}, {hi:.2f}]")

    if lo <= mean_auc <= hi:
        _ok(f"{ds_name}: mean AUC={mean_auc:.4f} within expected [{lo:.2f},{hi:.2f}]")
    elif mean_auc < lo:
        _fail(
            f"{ds_name}: mean AUC={mean_auc:.4f} BELOW expected lower bound {lo:.2f}. "
            "Check loader / class balance / n_epochs."
        )
    else:
        _warn(
            f"{ds_name}: mean AUC={mean_auc:.4f} ABOVE expected upper bound {hi:.2f}. "
            "Possible data leakage or overly easy task."
        )

    # ── 4. Bootstrap CI width ──────────────────────────────────────────────────
    roc_with_ci = roc.dropna(subset=["metric_ci_lower", "metric_ci_upper"])
    if roc_with_ci.empty:
        _warn(f"{ds_name}: no CI values found in roc_auc rows.")
    else:
        ci_widths = roc_with_ci["metric_ci_upper"] - roc_with_ci["metric_ci_lower"]
        mean_w = float(ci_widths.mean())
        max_w  = float(ci_widths.max())
        print()
        print(f"  Bootstrap CI width (95%): mean={mean_w:.4f}  max={max_w:.4f}  "
              f"(threshold < {_CI_WIDTH_WARN:.2f})")
        if max_w >= _CI_WIDTH_WARN:
            _warn(
                f"{ds_name}: max CI width={max_w:.4f} ≥ {_CI_WIDTH_WARN}. "
                "Consider increasing n_bootstrap or dataset size."
            )
        else:
            _ok(f"{ds_name}: all CI widths < {_CI_WIDTH_WARN:.2f} → reasonable")

    # ── 5. Per-pair AUC breakdown (detail) ────────────────────────────────────
    print()
    print("  Per-pair ROC-AUC breakdown:")
    auc_per_pair = (
        roc.groupby(["phi", "theta"])["metric_value"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    for _, row in auc_per_pair.iterrows():
        pair = f"{row['phi']}×{row['theta']}"
        print(f"    {pair:<45}  AUC={row['mean']:.4f} ± {row['std']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Stage 1 screening results before running Stage 2.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        metavar="DIR",
        help="Directory containing *.s1_results.parquet files (default: results/)",
    )
    parser.add_argument(
        "--top-k", type=int, default=2,
        metavar="K",
        help="Number of top pairs to highlight (should match config top_k, default 2)",
    )
    args = parser.parse_args()

    s1_files = sorted(args.results_dir.glob("*.s1_results.parquet"))
    if not s1_files:
        print(f"No *.s1_results.parquet files found in {args.results_dir}")
        print("Run:  .venv/bin/python scripts/run_all.py --stage1-only --log-dir logs")
        sys.exit(1)

    print("=" * 70)
    print("  nstad_bench  Stage 1 Inspection Report")
    print(f"  Files: {[f.name for f in s1_files]}")
    print("=" * 70)

    all_dfs: list[pd.DataFrame] = []
    for fpath in s1_files:
        df = pd.read_parquet(fpath)
        all_dfs.append(df)

        # Infer dataset name from rows
        datasets_in_file = df["dataset"].unique().tolist()
        for ds in datasets_in_file:
            _analyse_dataset(df[df["dataset"] == ds], ds, top_k=args.top_k)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if not _issues:
        print("  ALL CHECKS PASSED — safe to proceed to Stage 2")
        print()
        print("  Run Stage 2:")
        print("    mkdir -p logs")
        print("    nohup .venv/bin/python scripts/run_all.py --log-dir logs \\")
        print("        > /dev/null 2>&1 &")
        print("    echo \"PID $!\"")
    else:
        print(f"  {len(_issues)} ISSUE(S) FOUND — review before Stage 2:")
        for issue in _issues:
            print(f"    • {issue}")
    print("=" * 70)

    if any(i.startswith(FAIL.strip()) for i in _issues):
        sys.exit(1)


if __name__ == "__main__":
    main()

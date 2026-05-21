"""aggregate.py — summarise CSV results and run three-factor ANOVA.

Inputs
------
One or more append-mode CSV files written by ``run_one_config.py``.  By
default the script scans ``$OUTPUT_ROOT`` (or ``results/`` if unset) for
every ``*.csv`` and concatenates them.  Schema must match
``run_one_config.CSV_COLUMNS``.

What it produces
----------------
* A per-config summary table (mean ± std across seeds).
* A "best" table showing the highest ``gain_over_source`` and the lowest
  ``delta_auc`` for each dataset.
* Three-factor ANOVA tables (representation × model × da_method) on the
  cross-domain ROC-AUC — one per dataset and one pooled across all
  datasets.  ANOVA output is also written as LaTeX matching the
  existing ``paper/tables/`` style.

CLI
---
::

    python scripts/aggregate.py [--results-dir DIR] [--out-dir DIR]

If ``--out-dir`` is omitted, LaTeX files are written next to the CSVs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstad_bench.analysis.anova import effect_size_ranking, run_anova
from nstad_bench.data._paths import resolve_output_root

log = logging.getLogger("aggregate")

EXPECTED_COLS = [
    "dataset", "representation", "model", "da_method", "seed",
    "roc_auc", "pr_auc", "f1",
    "in_domain_auc", "cross_domain_auc", "delta_auc", "gain_over_source",
    "training_time_sec", "timestamp",
]

GROUP_KEYS = ["dataset", "representation", "model", "da_method"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> pd.DataFrame:
    """Concatenate every ``*.csv`` under *results_dir* into one DataFrame.

    Files with unexpected schemas are skipped with a warning so a
    misplaced CSV cannot derail aggregation.  ``seed`` is coerced to int
    and the numeric columns to float; everything else is left as string.
    """
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {results_dir}")

    frames: list[pd.DataFrame] = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            log.warning("Skipping %s: %s", p, exc)
            continue
        missing = set(EXPECTED_COLS) - set(df.columns)
        if missing:
            log.warning("Skipping %s — missing columns %s", p, missing)
            continue
        frames.append(df)
        log.info("Loaded %d rows from %s", len(df), p.name)

    if not frames:
        raise ValueError(f"No usable CSV files in {results_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["seed"] = df["seed"].astype(int)
    for c in ("roc_auc", "pr_auc", "f1", "in_domain_auc",
              "cross_domain_auc", "delta_auc", "gain_over_source",
              "training_time_sec"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.info("Total %d rows from %d files", len(df), len(frames))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds per (dataset, repr, model, da_method)."""
    agg = df.groupby(GROUP_KEYS).agg(
        n_seeds          =("seed",             "nunique"),
        roc_auc_mean     =("roc_auc",          "mean"),
        roc_auc_std      =("roc_auc",          "std"),
        pr_auc_mean      =("pr_auc",           "mean"),
        f1_mean          =("f1",               "mean"),
        in_domain_mean   =("in_domain_auc",    "mean"),
        cross_domain_mean=("cross_domain_auc", "mean"),
        cross_domain_std =("cross_domain_auc", "std"),
        delta_auc_mean   =("delta_auc",        "mean"),
        delta_auc_std    =("delta_auc",        "std"),
        gain_mean        =("gain_over_source", "mean"),
        gain_std         =("gain_over_source", "std"),
        time_mean_sec    =("training_time_sec","mean"),
    ).reset_index()
    return agg


def best_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, return the configs that maximise gain and minimise delta_auc."""
    rows: list[dict] = []
    for ds, sub in summary.groupby("dataset"):
        # Highest gain over source — non-source-only methods only
        non_src = sub[sub["da_method"] != "source-only"]
        if not non_src.empty and non_src["gain_mean"].notna().any():
            best_gain = non_src.loc[non_src["gain_mean"].idxmax()]
            rows.append({
                "dataset":    ds,
                "criterion":  "max gain_over_source",
                "model":      best_gain["model"],
                "repr":       best_gain["representation"],
                "da_method":  best_gain["da_method"],
                "value":      f"{best_gain['gain_mean']:+.4f} ± {best_gain['gain_std']:.4f}",
                "cross_auc":  f"{best_gain['cross_domain_mean']:.4f}",
            })
        # Lowest delta_auc — robustness measure
        if sub["delta_auc_mean"].notna().any():
            best_delta = sub.loc[sub["delta_auc_mean"].idxmin()]
            rows.append({
                "dataset":    ds,
                "criterion":  "min delta_auc",
                "model":      best_delta["model"],
                "repr":       best_delta["representation"],
                "da_method":  best_delta["da_method"],
                "value":      f"{best_delta['delta_auc_mean']:+.4f} ± {best_delta['delta_auc_std']:.4f}",
                "cross_auc":  f"{best_delta['cross_domain_mean']:.4f}",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX rendering (matches paper/tables style)
# ─────────────────────────────────────────────────────────────────────────────

def summary_to_latex(summary: pd.DataFrame, dataset: str) -> str:
    """Render the per-config summary for one dataset as a booktabs LaTeX table."""
    sub = summary[summary["dataset"] == dataset].copy()
    sub = sub.sort_values(["representation", "model", "da_method"])

    def _fmt(mean: float, std: float, prec: int = 4) -> str:
        if pd.isna(mean):
            return "—"
        if pd.isna(std):
            return f"{mean:.{prec}f}"
        return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Per-config results on {dataset} (mean $\pm$ std across seeds).}}",
        rf"\label{{tab:summary-{dataset}}}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Repr. & Model & DA & ROC-AUC & $\Delta$AUC & gain & $n$ \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        lines.append(
            " & ".join([
                str(r["representation"]),
                str(r["model"]),
                str(r["da_method"]),
                _fmt(r["cross_domain_mean"], r["cross_domain_std"]),
                _fmt(r["delta_auc_mean"],    r["delta_auc_std"]),
                _fmt(r["gain_mean"],         r["gain_std"]),
                f"{int(r['n_seeds'])}",
            ]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# ANOVA
# ─────────────────────────────────────────────────────────────────────────────

def _to_anova_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the CSV to the long-format that ``nstad_bench.analysis.anova`` expects."""
    long = df[[*GROUP_KEYS, "seed", "cross_domain_auc"]].copy()
    long = long.rename(columns={
        "representation": "phi",
        "model":          "theta",
        "da_method":      "psi",
        "cross_domain_auc": "metric_value",
    })
    long["metric_name"] = "cross_domain_auc"
    return long


def run_anova_per_dataset(df: pd.DataFrame) -> dict[str, "AnovaResult"]:
    out: dict[str, "AnovaResult"] = {}
    for ds, sub in df.groupby("dataset"):
        try:
            res = run_anova(_to_anova_long(sub), metric="cross_domain_auc")
            out[ds] = res
            log.info(
                "ANOVA(%s) — R²=%.3f n=%d, sig=%s",
                ds, res.r_squared, res.n_obs, res.significant_factors,
            )
        except Exception as exc:
            log.warning("ANOVA failed for dataset %s: %s", ds, exc)
    return out


def run_anova_pooled(df: pd.DataFrame) -> "AnovaResult | None":
    try:
        return run_anova(_to_anova_long(df), metric="cross_domain_auc")
    except Exception as exc:
        log.warning("Pooled ANOVA failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory containing CSV files (default: $OUTPUT_ROOT or results/)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write LaTeX tables (default: alongside CSVs)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = args.results_dir or resolve_output_root()
    out_dir = args.out_dir or results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)

    summary = summary_table(df)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote summary CSV → %s", summary_path)

    best = best_configs(summary)
    if not best.empty:
        best_path = out_dir / "best_configs.csv"
        best.to_csv(best_path, index=False)
        log.info("Wrote best-configs CSV → %s", best_path)
        print("\n=== Best configs per dataset ===")
        print(best.to_string(index=False))

    # Per-dataset LaTeX summary tables
    for ds in sorted(df["dataset"].unique()):
        tex = summary_to_latex(summary, ds)
        tex_path = out_dir / f"summary_{ds}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        log.info("Wrote LaTeX summary → %s", tex_path)

    # Per-dataset ANOVA
    print("\n=== Three-factor ANOVA on cross_domain_auc ===")
    for ds, res in run_anova_per_dataset(df).items():
        print(f"\n[{ds}] R²={res.r_squared:.3f}  n={res.n_obs}")
        print(res.table.to_string(float_format=lambda x: f"{x:.4f}"))
        tex_path = out_dir / f"anova_{ds}.tex"
        res.to_latex(
            tex_path,
            caption=f"Three-factor ANOVA on cross-domain ROC-AUC --- {ds}.",
            label=f"tab:anova-{ds}",
        )
        log.info("Wrote ANOVA LaTeX → %s", tex_path)
        ranking = effect_size_ranking(res)
        print("\nEffect-size ranking (partial η²):")
        print(ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Pooled ANOVA across all datasets
    pooled = run_anova_pooled(df)
    if pooled is not None:
        print(f"\n[POOLED] R²={pooled.r_squared:.3f}  n={pooled.n_obs}")
        print(pooled.table.to_string(float_format=lambda x: f"{x:.4f}"))
        tex_path = out_dir / "anova_pooled.tex"
        pooled.to_latex(
            tex_path,
            caption="Three-factor ANOVA on cross-domain ROC-AUC (pooled across datasets).",
            label="tab:anova-pooled",
        )
        log.info("Wrote pooled ANOVA LaTeX → %s", tex_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

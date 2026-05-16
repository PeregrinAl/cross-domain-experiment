"""Aggregation of Parquet results into summary tables and LaTeX export.

Typical workflow
----------------
::

    from nstad_bench.analysis.summary_tables import (
        load_results, pivot_delta_auc, gain_by_dataset,
        method_summary, screening_summary, to_latex,
    )

    df  = load_results("results/my_experiment.parquet")
    tbl = pivot_delta_auc(df)           # (φ×θ) × dataset heat-map data
    print(to_latex(tbl, caption="ΔAUC by representation and model",
                   label="tab:delta_auc"))
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Column constants (mirrors runner.RESULT_COLS)
# ─────────────────────────────────────────────────────────────────────────────

RESULT_COLS = [
    "config_hash", "dataset", "phi", "theta", "psi", "seed",
    "metric_name", "metric_value", "metric_ci_lower", "metric_ci_upper",
]

_FACTOR_COLS  = ["phi", "theta", "psi", "dataset", "seed"]
_DISPLAY_NAMES = {
    "phi":        "φ",
    "theta":      "θ",
    "psi":        "ψ",
    "delta_auc":  "ΔAUC",
    "roc_auc":    "ROC-AUC",
    "pr_auc":     "PR-AUC",
    "gain":       "Gain",
    "source_roc_auc": "Source ROC-AUC",
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: str | Path) -> pd.DataFrame:
    """Load Parquet results into a long-format DataFrame.

    Parameters
    ----------
    path :
        Path to the ``.parquet`` file written by ``run_experiment()``.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns matching ``RESULT_COLS``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [c for c in RESULT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Results file is missing required columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )
    return df


def _metric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Filter long-format *df* to rows for a single *metric_name*."""
    sub = df[df["metric_name"] == name].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for metric {name!r}. "
            f"Available: {sorted(df['metric_name'].unique())}"
        )
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation tables
# ─────────────────────────────────────────────────────────────────────────────

def pivot_delta_auc(
    df: pd.DataFrame,
    *,
    aggfunc: str | Callable = "mean",
    source_only: bool = True,
) -> pd.DataFrame:
    """Pivot mean ΔAUC with rows=(φ, θ) and columns=dataset.

    Parameters
    ----------
    df :
        Long-format results from ``load_results()``.
    aggfunc :
        Aggregation function applied across seeds (default ``"mean"``).
    source_only :
        If ``True`` (default), restrict to Stage-1 SourceOnly rows, which
        gives a pure measure of the domain gap independent of adaptation.

    Returns
    -------
    pd.DataFrame
        Index = (phi, theta) MultiIndex; columns = dataset names;
        values = mean ΔAUC (higher → larger domain gap).
    """
    sub = _metric(df, "delta_auc")
    if source_only:
        sub = sub[sub["psi"] == "SourceOnly"]

    tbl = sub.pivot_table(
        index=["phi", "theta"],
        columns="dataset",
        values="metric_value",
        aggfunc=aggfunc,
    )
    tbl.index.names  = ["φ", "θ"]
    tbl.columns.name = "dataset"
    return tbl.sort_index()


def gain_by_dataset(
    df: pd.DataFrame,
    *,
    aggfunc: str | Callable = "mean",
    exclude_source_only: bool = True,
) -> pd.DataFrame:
    """Pivot mean Gain with rows=ψ and columns=dataset.

    Parameters
    ----------
    df :
        Long-format results.
    aggfunc :
        Aggregation across seeds and HP trials (default ``"mean"``).
    exclude_source_only :
        Drop SourceOnly rows (gain = 0 by definition) from the table.

    Returns
    -------
    pd.DataFrame
        Index = psi (method); columns = dataset names; values = mean Gain.
    """
    sub = _metric(df, "gain")
    if exclude_source_only:
        sub = sub[sub["psi"] != "SourceOnly"]

    tbl = sub.pivot_table(
        index="psi",
        columns="dataset",
        values="metric_value",
        aggfunc=aggfunc,
    )
    tbl.index.name   = "ψ"
    tbl.columns.name = "dataset"
    return tbl.sort_index()


def method_summary(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("roc_auc", "pr_auc", "delta_auc", "gain"),
    aggfunc: str | Callable = "mean",
    ci_columns: bool = True,
) -> pd.DataFrame:
    """Per-method summary table: mean (and bootstrap CI half-width) per metric.

    Parameters
    ----------
    df :
        Long-format results.
    metrics :
        Which metric names to include.
    aggfunc :
        Central-tendency aggregation (default ``"mean"``).
    ci_columns :
        If ``True``, add ``<metric>_ci_hw`` columns with mean half-width of
        the 95 % bootstrap CI (upper − lower) / 2.  NaN where CI is absent.

    Returns
    -------
    pd.DataFrame
        Index = psi; columns = metric names (and optionally CI half-widths).
    """
    rows: list[dict] = []
    for psi, grp in df.groupby("psi"):
        row: dict = {"ψ": psi}
        for m in metrics:
            sub = grp[grp["metric_name"] == m]
            if sub.empty:
                row[_DISPLAY_NAMES.get(m, m)] = float("nan")
                if ci_columns:
                    row[f"{_DISPLAY_NAMES.get(m, m)} ±"] = float("nan")
                continue
            vals = sub["metric_value"]
            row[_DISPLAY_NAMES.get(m, m)] = float(getattr(vals, aggfunc)())
            if ci_columns:
                hw = (sub["metric_ci_upper"] - sub["metric_ci_lower"]) / 2.0
                hw_clean = hw.replace([float("inf"), float("-inf")], float("nan"))
                row[f"{_DISPLAY_NAMES.get(m, m)} ±"] = float(hw_clean.mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("ψ").sort_index()


def screening_summary(
    df: pd.DataFrame,
    *,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rank (φ, θ) pairs by mean ΔAUC across datasets and seeds.

    Parameters
    ----------
    df :
        Long-format results; SourceOnly rows are used for ranking.
    top_k :
        If given, return only the top-k rows.

    Returns
    -------
    pd.DataFrame
        Columns: φ, θ, mean_delta_auc, std_delta_auc, n_runs.
        Sorted descending by mean_delta_auc.
    """
    sub = _metric(df, "delta_auc")
    sub = sub[sub["psi"] == "SourceOnly"]

    agg = (
        sub.groupby(["phi", "theta"])["metric_value"]
        .agg(mean_delta_auc="mean", std_delta_auc="std", n_runs="count")
        .reset_index()
        .rename(columns={"phi": "φ", "theta": "θ"})
        .sort_values("mean_delta_auc", ascending=False)
        .reset_index(drop=True)
    )
    agg.index = agg.index + 1  # 1-based rank
    agg.index.name = "rank"
    if top_k is not None:
        agg = agg.head(top_k)
    return agg


def scores_matrix(
    df: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    aggfunc: str | Callable = "mean",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a (n_datasets × n_methods) scores matrix for statistical tests.

    Parameters
    ----------
    df :
        Long-format results.
    metric :
        Which metric to use (default ``"roc_auc"``).
    aggfunc :
        Aggregation across seeds and HP trials.

    Returns
    -------
    scores : np.ndarray, shape (n_datasets, n_methods)
    datasets : list[str]
    methods : list[str]
    """
    sub = _metric(df, metric)
    tbl = sub.pivot_table(
        index="dataset",
        columns="psi",
        values="metric_value",
        aggfunc=aggfunc,
    )
    return tbl.values, list(tbl.index), list(tbl.columns)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX export
# ─────────────────────────────────────────────────────────────────────────────

def to_latex(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    caption: str = "",
    label: str = "",
    fmt: str = ".3f",
    na_rep: str = "—",
    booktabs: bool = True,
    position: str = "htbp",
    column_format: str | None = None,
    bold_max: bool = False,
    bold_min: bool = False,
    escape: bool = False,
    header_rename: dict[str, str] | None = None,
) -> str:
    r"""Export a DataFrame to a self-contained LaTeX ``table`` environment.

    Parameters
    ----------
    df :
        DataFrame to export (index and columns become table rows/headers).
    path :
        Optional file path; if given, the LaTeX string is also written there.
    caption :
        Table caption (placed below the table with ``\caption{...}``).
    label :
        LaTeX label for ``\ref{}``.
    fmt :
        Python format string for float values (default ``".3f"``).
    na_rep :
        Replacement string for NaN cells (default ``"—"``).
    booktabs :
        Use ``\toprule`` / ``\midrule`` / ``\bottomrule`` (requires
        ``\usepackage{booktabs}``).
    position :
        Float position specifier (default ``"htbp"``).
    column_format :
        Explicit column format string (e.g. ``"lrrr"``). Auto-inferred if
        ``None``.
    bold_max :
        Bold the maximum value in each column.
    bold_min :
        Bold the minimum value in each column (mutually exclusive with
        *bold_max*).
    escape :
        Escape special LaTeX characters in cell values (default ``False``
        because metric/method names already use safe ASCII).
    header_rename :
        Optional dict to rename column labels before export.

    Returns
    -------
    str
        Complete LaTeX source for the table.
    """
    df = df.copy()
    if header_rename:
        df = df.rename(columns=header_rename)

    # ── Format floats and bold extrema ───────────────────────────────────────
    float_cols = df.select_dtypes(include="number").columns.tolist()

    if (bold_max or bold_min) and float_cols:
        for c in float_cols:
            extreme = df[c].max() if bold_max else df[c].min()
            df[c] = df[c].apply(
                lambda v, e=extreme, f=fmt: (
                    rf"\textbf{{{v:{f}}}}" if (not np.isnan(v) and np.isclose(v, e))
                    else (na_rep if np.isnan(v) else f"{v:{fmt}}")
                )
            )
    else:
        for c in float_cols:
            df[c] = df[c].apply(
                lambda v, f=fmt: na_rep if (isinstance(v, float) and np.isnan(v))
                else f"{v:{f}}"
            )

    # ── Infer column format ───────────────────────────────────────────────────
    if column_format is None:
        n_idx   = df.index.nlevels
        n_cols  = len(df.columns)
        column_format = "l" * n_idx + "r" * n_cols

    # ── Build tabular body via pandas ─────────────────────────────────────────
    tabular = df.to_latex(
        column_format=column_format,
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        na_rep=na_rep,
        escape=escape,
        bold_rows=False,
    )

    # ── Replace rules if booktabs ──────────────────────────────────────────────
    if booktabs:
        tabular = (
            tabular
            .replace(r"\hline", "")
            .replace(r"\toprule", r"\toprule")
        )
        # pandas may not emit \toprule; force it
        lines = tabular.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip().startswith(r"\begin{tabular}"):
                # insert rule after the header line that follows
                continue
        # Easier: just strip the pandas-generated rules and add booktabs ones
        tabular = _apply_booktabs(tabular)

    # ── Wrap in float environment ──────────────────────────────────────────────
    cap_line   = rf"\caption{{{caption}}}" if caption else ""
    label_line = rf"\label{{{label}}}"     if label   else ""

    pieces = [
        rf"\begin{{table}}[{position}]",
        r"\centering",
        tabular.strip(),
    ]
    if cap_line:
        pieces.append(cap_line)
    if label_line:
        pieces.append(label_line)
    pieces.append(r"\end{table}")

    latex = "\n".join(pieces) + "\n"

    if path is not None:
        Path(path).write_text(latex, encoding="utf-8")

    return latex


def _apply_booktabs(tabular: str) -> str:
    """Replace pandas hline-based rules with booktabs rules."""
    lines = tabular.splitlines(keepends=True)
    result = []
    header_done = False
    in_tabular  = False

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(r"\begin{tabular}"):
            in_tabular = True
            result.append(ln)
            continue
        if stripped == r"\end{tabular}":
            # Replace last hline (before \end) with \bottomrule
            if result and result[-1].strip() == r"\hline":
                result[-1] = r"\bottomrule" + "\n"
            result.append(ln)
            continue
        if in_tabular and stripped == r"\hline":
            if not header_done:
                result.append(r"\toprule" + "\n")
                header_done = True
            else:
                result.append(r"\midrule" + "\n")
            continue
        result.append(ln)

    return "".join(result)


def export_suite(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    experiment_name: str = "experiment",
    fmt: str = ".3f",
) -> dict[str, Path]:
    """Export the full standard table suite to *output_dir*.

    Tables produced
    ---------------
    ``delta_auc_pivot.tex``
        Heatmap-style ΔAUC pivot (φ×θ) × dataset.
    ``gain_by_dataset.tex``
        Gain(ψ) × dataset.
    ``method_summary.tex``
        Per-method mean metric table.
    ``screening_summary.tex``
        Screening ranking of (φ,θ) pairs.

    Returns
    -------
    dict[str, Path]
        Map of table name → written file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    tables = {
        "delta_auc_pivot": (
            pivot_delta_auc(df),
            "Mean ΔAUC by representation and model",
            f"tab:{experiment_name}_delta_auc",
        ),
        "gain_by_dataset": (
            gain_by_dataset(df),
            "Mean Gain by adaptation method and dataset",
            f"tab:{experiment_name}_gain",
        ),
        "method_summary": (
            method_summary(df),
            "Per-method performance summary",
            f"tab:{experiment_name}_methods",
        ),
        "screening_summary": (
            screening_summary(df),
            "Screening ranking of (φ, θ) pairs by mean ΔAUC",
            f"tab:{experiment_name}_screening",
        ),
    }

    for name, (tbl, caption, label) in tables.items():
        p = out / f"{name}.tex"
        to_latex(tbl, path=p, caption=caption, label=label, fmt=fmt)
        written[name] = p

    return written


__all__ = [
    "load_results",
    "pivot_delta_auc",
    "gain_by_dataset",
    "method_summary",
    "screening_summary",
    "scores_matrix",
    "to_latex",
    "export_suite",
    "RESULT_COLS",
]

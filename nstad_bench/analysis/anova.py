"""Three-factor ANOVA on ΔAUC ~ φ + θ + ψ + interactions.

Design rationale
----------------
Each ΔAUC observation is the mean over seeds and HP trials for one
(φ, θ, ψ, dataset) combination.  The three fixed factors are:

* **φ** (phi)   — representation / feature extraction
* **θ** (theta) — model architecture
* **ψ** (psi)   — adaptation method

The analysis answers:
1. Which factor (or interaction) explains the most variance in domain gap?
2. Are factor effects statistically significant after controlling for the
   others?

Implementation
--------------
* OLS via ``statsmodels.formula.api.ols``.
* Type II sums of squares (``statsmodels.stats.anova.anova_lm(type=2)``) —
  robust to unbalanced designs without interaction confounding.
* Partial η² = SS_effect / (SS_effect + SS_residual) as effect-size measure.
* LaTeX export via ``to_latex_anova()``.

Usage
-----
::

    from nstad_bench.analysis.anova import run_anova

    df    = load_results("results/experiment.parquet")
    result = run_anova(df)
    print(result.table.to_string())
    result.to_latex("tables/anova.tex",
                    caption="Three-factor ANOVA on ΔAUC",
                    label="tab:anova")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnovaResult:
    """Container for a completed ANOVA run.

    Attributes
    ----------
    table :
        ANOVA table with columns:
        ``df``, ``sum_sq``, ``mean_sq``, ``F``, ``PR(>F)``, ``partial_eta2``.
    formula :
        The Patsy formula string passed to OLS.
    metric :
        Which metric was analysed (typically ``"delta_auc"``).
    r_squared :
        Model R² (proportion of total variance explained).
    r_squared_adj :
        Adjusted R².
    n_obs :
        Number of observations used in the fit.
    model_summary :
        Full ``statsmodels`` result summary as a plain-text string.
    warnings : list[str]
        Any non-fatal warnings raised during fitting.
    """

    table:         pd.DataFrame
    formula:       str
    metric:        str
    r_squared:     float
    r_squared_adj: float
    n_obs:         int
    model_summary: str
    warnings:      list[str] = field(default_factory=list)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def significant_factors(self, alpha: float = 0.05) -> list[str]:
        """Return row names where ``PR(>F) < alpha``."""
        col = "PR(>F)"
        if col not in self.table.columns:
            return []
        sig = self.table[self.table[col] < alpha]
        return [idx for idx in sig.index if idx != "Residual"]

    def factor_pvalue(self, factor: str) -> float | None:
        """Return the p-value for *factor* (or ``None`` if not in table)."""
        col = "PR(>F)"
        if factor not in self.table.index or col not in self.table.columns:
            return None
        return float(self.table.loc[factor, col])

    def factor_partial_eta2(self, factor: str) -> float | None:
        """Return partial η² for *factor* (or ``None`` if not in table)."""
        col = "partial_eta2"
        if factor not in self.table.index or col not in self.table.columns:
            return None
        return float(self.table.loc[factor, col])

    # ── LaTeX export ──────────────────────────────────────────────────────────

    def to_latex(
        self,
        path: str | Path | None = None,
        *,
        caption: str = "",
        label: str = "",
        alpha: float = 0.05,
        fmt_ss: str = ".4f",
        fmt_f:  str = ".3f",
        fmt_p:  str = ".4f",
        fmt_eta: str = ".3f",
        position: str = "htbp",
        note: str = "",
    ) -> str:
        r"""Export the ANOVA table as a standalone LaTeX ``table`` environment.

        Significant rows (p < *alpha*) are highlighted with a ``*`` in the
        p-value column; rows with p < 0.001 get ``***``.

        Parameters
        ----------
        path :
            Optional file path; written if given.
        caption :
            Table caption text.
        label :
            LaTeX ``\label`` key.
        alpha :
            Significance threshold for asterisk marking.
        fmt_ss, fmt_f, fmt_p, fmt_eta :
            Format strings for SS, F-statistic, p-value, and partial η².
        position :
            Float position specifier.
        note :
            Optional note appended below the table as a ``\footnotesize`` line.

        Returns
        -------
        str
            Complete LaTeX source.
        """
        tbl = self.table.copy()

        def _stars(p: float) -> str:
            if np.isnan(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < alpha:
                return "*"
            return ""

        # Format each column
        def _fmt(v: float | str, fmt: str) -> str:
            if isinstance(v, str):
                return v
            if np.isnan(v):
                return "—"
            return f"{v:{fmt}}"

        rows: list[str] = []
        for idx, row in tbl.iterrows():
            p   = row.get("PR(>F)", float("nan"))
            stars = _stars(float(p))
            p_str = (_fmt(float(p), fmt_p) + rf"\textsuperscript{{{stars}}}"
                     if stars else _fmt(float(p), fmt_p))
            cols = [
                str(idx),
                _fmt(row.get("df", float("nan")), ".0f"),
                _fmt(row.get("sum_sq", float("nan")), fmt_ss),
                _fmt(row.get("mean_sq", float("nan")), fmt_ss),
                _fmt(row.get("F", float("nan")), fmt_f),
                p_str,
                _fmt(row.get("partial_eta2", float("nan")), fmt_eta),
            ]
            rows.append(" & ".join(cols) + r" \\")

        header = (
            r"Source & df & SS & MS & $F$ & $p$ & $\hat{\eta}^2_p$ \\"
        )
        cap_line   = rf"\caption{{{caption}}}" if caption else ""
        label_line = rf"\label{{{label}}}"     if label   else ""
        note_line  = (
            rf"\\ \footnotesize{{Note: {note}}}" if note else ""
        )

        body = "\n".join(rows)
        latex = textwrap_table(
            header, body, cap_line, label_line, note_line, position
        )

        if path is not None:
            Path(path).write_text(latex, encoding="utf-8")
        return latex

    def __str__(self) -> str:
        parts = [
            f"Three-factor ANOVA  |  metric={self.metric}",
            f"Formula : {self.formula}",
            f"N       : {self.n_obs}",
            f"R²      : {self.r_squared:.4f}   R²_adj: {self.r_squared_adj:.4f}",
            "",
            self.table.to_string(float_format=lambda x: f"{x:.4f}"),
        ]
        if self.warnings:
            parts += ["", "Warnings:", *[f"  ! {w}" for w in self.warnings]]
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers (private)
# ─────────────────────────────────────────────────────────────────────────────

def textwrap_table(
    header: str,
    body:   str,
    cap_line:   str,
    label_line: str,
    note_line:  str,
    position:   str,
) -> str:
    """Assemble a complete booktabs-style LaTeX table string."""
    col_fmt = "lrrrrrr"
    pieces = [
        rf"\begin{{table}}[{position}]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        header,
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note_line:
        pieces.append(note_line)
    if cap_line:
        pieces.append(cap_line)
    if label_line:
        pieces.append(label_line)
    pieces.append(r"\end{table}")
    return "\n".join(pieces) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

# Default formula — main effects + all two-way interactions.
# Three-way interaction (φ:θ:ψ) is deliberately omitted: it is rarely
# interpretable and often unidentifiable in unbalanced designs.
_DEFAULT_FORMULA = (
    "delta_auc ~ C(phi) + C(theta) + C(psi)"
    " + C(phi):C(theta)"
    " + C(phi):C(psi)"
    " + C(theta):C(psi)"
)


def run_anova(
    df: pd.DataFrame,
    *,
    formula: str | None = None,
    metric: str = "delta_auc",
    aggregate_seeds: bool = True,
    anova_type: int = 2,
    min_cells: int = 2,
) -> AnovaResult:
    """Fit a three-factor ANOVA on *metric* and return an ``AnovaResult``.

    Parameters
    ----------
    df :
        Long-format results DataFrame from ``load_results()``.
    formula :
        Patsy formula string.  Defaults to main effects + two-way
        interactions of φ, θ, ψ.  The dependent variable must be named
        ``delta_auc`` (matching the column created during aggregation).
    metric :
        Which metric to analyse (default ``"delta_auc"``).
    aggregate_seeds :
        If ``True`` (default), average across seeds and HP trials before
        fitting, so each (φ, θ, ψ, dataset) cell contributes one
        observation.  If ``False``, each seed × trial is a separate obs
        (inflates degrees of freedom).
    anova_type :
        Type of sums of squares (2 = Type II, 3 = Type III).  Type II is
        the default and is recommended for unbalanced designs without
        interactions at the highest order.
    min_cells :
        Minimum number of observations required for the ANOVA.  Raises
        ``ValueError`` if fewer are present after aggregation.

    Returns
    -------
    AnovaResult
        Contains the ANOVA table (with partial η²), model R², and the full
        statsmodels summary as a text string.

    Raises
    ------
    ValueError
        If the metric is absent, the dataset is too small, or no variation
        exists in the dependent variable.
    """
    warns: list[str] = []

    # ── Filter to requested metric ────────────────────────────────────────────
    sub = df[df["metric_name"] == metric].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for metric {metric!r}. "
            f"Available: {sorted(df['metric_name'].unique())}"
        )

    # ── Aggregate across seeds / HP trials ───────────────────────────────────
    group_cols = ["phi", "theta", "psi", "dataset"]
    if aggregate_seeds:
        sub = (
            sub.groupby(group_cols)["metric_value"]
            .mean()
            .reset_index()
            .rename(columns={"metric_value": "delta_auc"})
        )
    else:
        sub = sub.rename(columns={"metric_value": "delta_auc"})

    if len(sub) < min_cells:
        raise ValueError(
            f"Too few observations for ANOVA: {len(sub)} < {min_cells}. "
            "Check that results contain multiple (φ, θ, ψ, dataset) combos."
        )

    if sub["delta_auc"].std() < 1e-12:
        raise ValueError(
            "Dependent variable has zero variance — ANOVA is undefined."
        )

    # Warn if any factor has only one level
    for factor in ("phi", "theta", "psi"):
        n_levels = sub[factor].nunique()
        if n_levels < 2:
            warns.append(
                f"Factor '{factor}' has only {n_levels} level(s); "
                "its main effect and interactions are not estimable."
            )

    formula = formula or _DEFAULT_FORMULA

    # ── Fit OLS ──────────────────────────────────────────────────────────────
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model  = smf.ols(formula, data=sub).fit()
        except Exception as exc:
            raise ValueError(
                f"OLS fitting failed with formula {formula!r}: {exc}"
            ) from exc
        for w in caught:
            warns.append(str(w.message))

    # ── ANOVA table ───────────────────────────────────────────────────────────
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anova_tbl = anova_lm(model, typ=anova_type)
        for w in caught:
            warns.append(str(w.message))

    # ── Partial η² = SS_effect / (SS_effect + SS_residual) ──────────────────
    ss_res = anova_tbl.loc["Residual", "sum_sq"] if "Residual" in anova_tbl.index else np.nan
    anova_tbl["mean_sq"] = anova_tbl["sum_sq"] / anova_tbl["df"]

    partial_eta2: list[float] = []
    for idx in anova_tbl.index:
        ss_e = anova_tbl.loc[idx, "sum_sq"]
        if idx == "Residual" or np.isnan(ss_res) or (ss_e + ss_res) < 1e-30:
            partial_eta2.append(float("nan"))
        else:
            partial_eta2.append(float(ss_e / (ss_e + ss_res)))
    anova_tbl["partial_eta2"] = partial_eta2

    # ── Rename index for readability ──────────────────────────────────────────
    rename_map = {
        "C(phi)":              "φ (representation)",
        "C(theta)":            "θ (model)",
        "C(psi)":              "ψ (adaptation)",
        "C(phi):C(theta)":     "φ × θ",
        "C(phi):C(psi)":       "φ × ψ",
        "C(theta):C(psi)":     "θ × ψ",
        "C(phi):C(theta):C(psi)": "φ × θ × ψ",
    }
    anova_tbl = anova_tbl.rename(index=rename_map)

    # Reorder columns for clarity
    col_order = ["df", "sum_sq", "mean_sq", "F", "PR(>F)", "partial_eta2"]
    anova_tbl = anova_tbl[[c for c in col_order if c in anova_tbl.columns]]

    return AnovaResult(
        table         = anova_tbl,
        formula       = formula,
        metric        = metric,
        r_squared     = float(model.rsquared),
        r_squared_adj = float(model.rsquared_adj),
        n_obs         = int(model.nobs),
        model_summary = str(model.summary()),
        warnings      = warns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: effect-size ranking
# ─────────────────────────────────────────────────────────────────────────────

def effect_size_ranking(result: AnovaResult) -> pd.DataFrame:
    """Return factors and interactions ranked by partial η² (descending).

    Parameters
    ----------
    result :
        Completed ``AnovaResult``.

    Returns
    -------
    pd.DataFrame
        Columns: ``source``, ``partial_eta2``, ``F``, ``PR(>F)``.
        Sorted by partial η² descending.  Residual row excluded.
    """
    tbl = result.table.copy()
    tbl = tbl[tbl.index != "Residual"]
    tbl = tbl.reset_index().rename(columns={"index": "source"})

    keep = [c for c in ("source", "partial_eta2", "F", "PR(>F)") if c in tbl.columns]
    return (
        tbl[keep]
        .sort_values("partial_eta2", ascending=False)
        .reset_index(drop=True)
    )


__all__ = [
    "AnovaResult",
    "run_anova",
    "effect_size_ranking",
]

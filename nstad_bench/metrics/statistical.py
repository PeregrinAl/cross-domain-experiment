"""Statistical tests for comparing domain-adaptation methods across datasets.

Tests
-----
wilcoxon_test
    Paired Wilcoxon signed-rank test for two methods.
    Inputs: two 1-D arrays of per-dataset scores (e.g. AUC on each dataset).

friedman_nemenyi
    Friedman omnibus test + post-hoc Nemenyi for ≥ 3 methods.
    Input: (n_datasets × n_methods) score matrix.

Both tests are *non-parametric* and appropriate when:
  * Scores are not normally distributed (common for AUC/F1 across heterogeneous
    datasets).
  * The number of datasets is small (< 30), ruling out CLT-based approaches.

References
----------
Demšar, J. (2006). Statistical comparisons of classifiers over multiple
datasets. *Journal of Machine Learning Research*, 7, 1–30.

Wilcoxon, F. (1945). Individual comparisons by ranking methods.
*Biometrics Bulletin*, 1(6), 80–83.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare, wilcoxon


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank test.

    Attributes
    ----------
    statistic : float
        Wilcoxon W statistic (sum of positive ranks).
    p_value : float
        Two-tailed (or one-tailed) p-value.
    n_pairs : int
        Number of paired observations (after removing zero differences).
    alternative : str
        Hypothesis tested: ``'two-sided'``, ``'greater'``, or ``'less'``.
    """

    statistic: float
    p_value: float
    n_pairs: int
    alternative: str

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Return True if p_value < *alpha*."""
        return self.p_value < alpha

    def __repr__(self) -> str:
        return (
            f"WilcoxonResult(W={self.statistic:.4f}, p={self.p_value:.4f}, "
            f"n={self.n_pairs}, alt='{self.alternative}')"
        )


@dataclass
class FriedmanNemenyiResult:
    """Result of a Friedman test with post-hoc Nemenyi comparisons.

    Attributes
    ----------
    statistic : float
        Friedman χ² statistic.
    p_value : float
        Friedman omnibus p-value.
    n_datasets : int
        Number of datasets (blocks) used in the test.
    n_methods : int
        Number of methods compared.
    method_names : list[str]
        Names corresponding to the columns of the score matrix.
    nemenyi_pvalues : pd.DataFrame
        Symmetric (n_methods × n_methods) DataFrame of Nemenyi pairwise
        p-values.  Diagonal entries are 1.0.
    """

    statistic: float
    p_value: float
    n_datasets: int
    n_methods: int
    method_names: list[str]
    nemenyi_pvalues: pd.DataFrame

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Return True if the Friedman omnibus p-value < *alpha*."""
        return self.p_value < alpha

    def significant_pairs(self, alpha: float = 0.05) -> list[tuple[str, str]]:
        """Return list of (method_a, method_b) pairs with p < *alpha*."""
        pv  = self.nemenyi_pvalues
        out = []
        names = self.method_names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if pv.loc[names[i], names[j]] < alpha:
                    out.append((names[i], names[j]))
        return out

    def __repr__(self) -> str:
        return (
            f"FriedmanNemenyiResult(χ²={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, "
            f"n_datasets={self.n_datasets}, "
            f"methods={self.method_names})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test comparing two methods.

    Parameters
    ----------
    scores_a, scores_b :
        1-D float arrays of per-dataset metric values (e.g. AUC on each
        of K datasets).  Must have the same length.
    alternative :
        ``'two-sided'`` (default), ``'greater'`` (a > b), or ``'less'`` (a < b).

    Returns
    -------
    WilcoxonResult

    Notes
    -----
    Pairs where ``scores_a[i] == scores_b[i]`` are excluded (scipy default
    ``zero_method='wilcox'``).  If *all* differences are zero, scipy raises
    ``ValueError`` — this propagates unchanged as it indicates the two score
    vectors are identical.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(
            "scores_a and scores_b must be 1-D arrays of the same length."
        )
    result = wilcoxon(a, b, alternative=alternative)
    # scipy drops zero-difference pairs; report effective n
    n_nonzero = int(np.sum(a != b))
    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_pairs=n_nonzero,
        alternative=alternative,
    )


def friedman_nemenyi(
    scores: np.ndarray,
    method_names: list[str] | None = None,
) -> FriedmanNemenyiResult:
    """Friedman omnibus test + post-hoc Nemenyi pairwise comparisons.

    Parameters
    ----------
    scores :
        2-D array of shape ``(n_datasets, n_methods)``.  Each row is one
        dataset; each column is one method.
    method_names :
        Optional list of length ``n_methods`` used as row/column labels in
        the Nemenyi p-value table.  Defaults to ``['method_0', ...]``.

    Returns
    -------
    FriedmanNemenyiResult

    Notes
    -----
    The Nemenyi post-hoc is always computed regardless of the Friedman
    p-value, so pairwise differences can be inspected even when the omnibus
    test is not significant.  A Bonferroni-corrected α (e.g. 0.05 / k) should
    be used when interpreting many pairwise comparisons simultaneously.

    The Friedman test requires at least 3 methods and at least 2 datasets;
    both are enforced below.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be 2-D (n_datasets × n_methods), got shape {scores.shape}."
        )
    n_datasets, n_methods = scores.shape
    if n_methods < 3:
        raise ValueError(
            f"Friedman test requires ≥ 3 methods, got {n_methods}. "
            "Use wilcoxon_test() for pairwise comparison."
        )
    if n_datasets < 2:
        raise ValueError(
            f"Friedman test requires ≥ 2 datasets, got {n_datasets}."
        )

    if method_names is None:
        method_names = [f"method_{i}" for i in range(n_methods)]
    if len(method_names) != n_methods:
        raise ValueError(
            f"len(method_names)={len(method_names)} must equal "
            f"n_methods={n_methods}."
        )

    # Friedman: pass each method's scores as a separate positional arg
    stat, p = friedmanchisquare(*scores.T)

    # Nemenyi post-hoc (scores rows = blocks, cols = treatments)
    nemenyi_df: pd.DataFrame = posthoc_nemenyi_friedman(scores)
    nemenyi_df.columns = method_names
    nemenyi_df.index   = method_names

    return FriedmanNemenyiResult(
        statistic=float(stat),
        p_value=float(p),
        n_datasets=n_datasets,
        n_methods=n_methods,
        method_names=list(method_names),
        nemenyi_pvalues=nemenyi_df,
    )

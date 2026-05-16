"""Visualisation utilities for nstad_bench analysis results.

Three plot types
----------------
``plot_delta_auc_heatmap``
    Seaborn heatmap — rows=(φ, θ), columns=dataset, colour=ΔAUC.

``plot_gain_barplot``
    Grouped bar chart — Gain(ψ) for each dataset, with error bars derived
    from bootstrap CI half-widths stored in the results DataFrame.

``plot_cd_diagram``
    Critical-Difference (CD) diagram following Demšar (2006):
    methods ranked by mean rank across datasets, cliques of non-significant
    pairs connected by horizontal bars (Nemenyi post-hoc).

All functions accept an optional ``ax`` (or ``axes``) argument so callers
can embed plots in larger figures. When *ax* is ``None``, a new figure is
created and returned.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import studentized_range

# Use a non-interactive backend if no display is available.
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = "tab10"

def _ensure_ax(ax: plt.Axes | None, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _metric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    sub = df[df["metric_name"] == name]
    if sub.empty:
        raise ValueError(
            f"No rows for metric {name!r}. "
            f"Available: {sorted(df['metric_name'].unique())}"
        )
    return sub.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. ΔAUC heatmap  (φ × θ)  ×  dataset
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_auc_heatmap(
    df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 4),
    fmt: str = ".3f",
    cmap: str = "RdYlGn_r",
    source_only: bool = True,
    annot_fontsize: int = 9,
    title: str = "ΔAUC by (φ, θ) and dataset",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Heatmap of mean ΔAUC with (φ, θ) as rows and dataset as columns.

    ΔAUC = source_AUC − target_AUC; higher → larger domain gap (red).
    Lower values (green) indicate the representation–model pair suffers
    less from distribution shift.

    Parameters
    ----------
    df :
        Long-format results DataFrame from ``load_results()``.
    ax :
        Axes to draw into; a new figure is created if ``None``.
    figsize :
        Figure size when *ax* is ``None``.
    fmt :
        Cell annotation format string.
    cmap :
        Seaborn / matplotlib colormap.  ``"RdYlGn_r"`` maps low ΔAUC
        (small gap) to green and high ΔAUC (large gap) to red.
    source_only :
        Restrict to SourceOnly rows (stage-1 screening values).
    annot_fontsize :
        Font size for in-cell annotations.
    title :
        Axes title.
    vmin, vmax :
        Colour scale limits.  Auto-detected if ``None``.

    Returns
    -------
    plt.Figure
    """
    sub = _metric(df, "delta_auc")
    if source_only:
        sub = sub[sub["psi"] == "SourceOnly"]

    tbl = sub.pivot_table(
        index=["phi", "theta"],
        columns="dataset",
        values="metric_value",
        aggfunc="mean",
    )
    tbl.index = tbl.index.map(lambda t: f"{t[0]} × {t[1]}")
    tbl.columns.name = None
    tbl.index.name   = "(φ, θ)"

    fig, ax = _ensure_ax(ax, figsize)

    sns.heatmap(
        tbl,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": annot_fontsize},
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "ΔAUC", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("(φ, θ)", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gain(ψ) bar-plot by dataset
# ─────────────────────────────────────────────────────────────────────────────

def plot_gain_barplot(
    df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (11, 5),
    method_order: list[str] | None = None,
    exclude_source_only: bool = True,
    palette: str | list | None = None,
    title: str = "Gain(ψ) per dataset",
    show_ci: bool = True,
    alpha_bar: float = 0.85,
    zero_line: bool = True,
) -> plt.Figure:
    """Grouped bar chart: mean Gain per adaptation method, grouped by dataset.

    Error bars show the mean bootstrap-CI half-width (upper−lower)/2
    aggregated across seeds and HP trials.

    Parameters
    ----------
    df :
        Long-format results DataFrame.
    ax :
        Target axes; a new figure is created if ``None``.
    figsize :
        Figure size when *ax* is ``None``.
    method_order :
        Ordered list of ψ names; default sorts alphabetically.
    exclude_source_only :
        Remove SourceOnly rows (Gain = 0 by definition).
    palette :
        Colour palette; defaults to seaborn ``"tab10"``.
    title :
        Axes title.
    show_ci :
        Draw error bars from the stored bootstrap CI.
    alpha_bar :
        Bar transparency.
    zero_line :
        Draw a dashed horizontal line at y=0.

    Returns
    -------
    plt.Figure
    """
    sub = _metric(df, "gain")
    if exclude_source_only:
        sub = sub[sub["psi"] != "SourceOnly"]

    methods  = sorted(sub["psi"].unique()) if method_order is None else method_order
    datasets = sorted(sub["dataset"].unique())

    n_ds  = len(datasets)
    n_m   = len(methods)
    width = 0.8 / max(n_ds, 1)      # bar width
    offsets = np.linspace(-(n_ds - 1) / 2, (n_ds - 1) / 2, n_ds) * width

    palette = palette or sns.color_palette(_PALETTE, n_colors=n_ds)
    if isinstance(palette, str):
        palette = sns.color_palette(palette, n_colors=n_ds)

    fig, ax = _ensure_ax(ax, figsize)

    for di, (ds, off, color) in enumerate(zip(datasets, offsets, palette)):
        means, ci_hws, xs = [], [], []
        for mi, psi in enumerate(methods):
            rows = sub[(sub["dataset"] == ds) & (sub["psi"] == psi)]
            if rows.empty:
                continue
            xs.append(mi)
            means.append(float(rows["metric_value"].mean()))
            if show_ci:
                hw = (rows["metric_ci_upper"] - rows["metric_ci_lower"]) / 2.0
                ci_hws.append(float(hw.mean()))
            else:
                ci_hws.append(0.0)

        if not xs:
            continue
        errs = ci_hws if show_ci else None
        ax.bar(
            [x + off for x in xs],
            means,
            width=width * 0.9,
            color=color,
            alpha=alpha_bar,
            label=ds,
            yerr=errs,
            capsize=3,
            error_kw={"linewidth": 1.2, "ecolor": "black"},
        )

    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_xlabel("Adaptation method (ψ)", fontsize=10)
    ax.set_ylabel("Gain = AUC_adapted − AUC_SourceOnly", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. CD diagram  (Demšar 2006)
# ─────────────────────────────────────────────────────────────────────────────

def _nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference at significance level *alpha*.

    CD = q_α · sqrt(k(k+1) / (6N))

    where q_α is the critical value from the Studentized range distribution
    evaluated at (k, ∞) d.f., divided by sqrt(2).  This matches Table 5 in
    Demšar (2006) to four decimal places.
    """
    q = studentized_range.ppf(1.0 - alpha, n_methods, df=np.inf) / np.sqrt(2)
    return q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))


def _find_cliques(
    mean_ranks: dict[str, float],
    pvalue_matrix: pd.DataFrame,
    alpha: float,
) -> list[list[str]]:
    """Find maximal cliques of methods not significantly different from each other.

    A clique = a set of methods where every pair has p-value >= alpha.
    We use a greedy sweep over the sorted rank list (as in the standard
    CD-diagram presentation).
    """
    methods_sorted = sorted(mean_ranks, key=mean_ranks.__getitem__)
    cliques: list[list[str]] = []

    i = 0
    while i < len(methods_sorted):
        clique = [methods_sorted[i]]
        for j in range(i + 1, len(methods_sorted)):
            m_j = methods_sorted[j]
            # All current members of the clique must be non-significant vs m_j
            if all(
                float(pvalue_matrix.loc[m, m_j]) >= alpha
                for m in clique
            ):
                clique.append(m_j)
        if len(clique) > 1:
            cliques.append(clique)
        i += 1

    # Remove subsumed cliques (keep maximal ones)
    maximal: list[list[str]] = []
    for cl in cliques:
        cl_set = set(cl)
        if not any(cl_set < set(other) for other in cliques):
            if cl not in maximal:
                maximal.append(cl)
    return maximal


def plot_cd_diagram(
    scores: np.ndarray | pd.DataFrame,
    method_names: list[str] | None = None,
    *,
    alpha: float = 0.05,
    nemenyi_pvalues: pd.DataFrame | None = None,
    higher_is_better: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 4),
    title: str | None = None,
    method_fontsize: int = 9,
    show_cd_bracket: bool = True,
) -> plt.Figure:
    """Critical-Difference diagram following Demšar (2006).

    Methods are ranked by mean rank across datasets.  Connected horizontal
    bars indicate groups of methods not significantly different under the
    Nemenyi post-hoc test (or the CD criterion when *nemenyi_pvalues* is
    not provided).

    Parameters
    ----------
    scores :
        Array of shape ``(n_datasets, n_methods)`` where each row is a
        dataset and each column is a method.  May also be a DataFrame whose
        columns are method names and whose index are dataset names.
    method_names :
        List of method names; inferred from DataFrame columns if *scores*
        is a DataFrame.
    alpha :
        Significance level for the Nemenyi test (default 0.05).
    nemenyi_pvalues :
        Square DataFrame of Nemenyi p-values (rows=cols=method names) as
        returned by ``FriedmanNemenyiResult.nemenyi_pvalues``.  When
        provided, cliques are determined from these p-values instead of
        the CD criterion.
    higher_is_better :
        If ``True`` (default), rank 1 = highest score (= best).
        Set to ``False`` for loss metrics (e.g. error rate).
    ax :
        Target axes; a new figure is created if ``None``.
    figsize :
        Figure size when *ax* is ``None``.
    title :
        Axes title.  Defaults to ``"Critical Difference Diagram (α=<alpha>)"``.
    method_fontsize :
        Font size for method labels.
    show_cd_bracket :
        Draw the CD bracket on the top axis.

    Returns
    -------
    plt.Figure

    Notes
    -----
    The diagram style is based on:
        Demšar, J. (2006). Statistical comparisons of classifiers over
        multiple data sets. *JMLR*, 7, 1–30.
    """
    # ── Input normalisation ───────────────────────────────────────────────────
    if isinstance(scores, pd.DataFrame):
        if method_names is None:
            method_names = list(scores.columns)
        scores = scores.values
    scores = np.asarray(scores, dtype=float)

    n_ds, n_m = scores.shape
    if method_names is None:
        method_names = [f"M{i+1}" for i in range(n_m)]
    if len(method_names) != n_m:
        raise ValueError(
            f"len(method_names)={len(method_names)} ≠ n_methods={n_m}"
        )

    # ── Compute mean ranks ────────────────────────────────────────────────────
    from scipy.stats import rankdata  # local import for clarity
    ranked = np.apply_along_axis(
        lambda row: rankdata(-row if higher_is_better else row, method="average"),
        axis=1,
        arr=scores,
    )
    mean_ranks = {m: float(ranked[:, i].mean()) for i, m in enumerate(method_names)}

    # ── CD value ──────────────────────────────────────────────────────────────
    cd = _nemenyi_cd(n_m, n_ds, alpha)

    # ── Cliques ───────────────────────────────────────────────────────────────
    if nemenyi_pvalues is not None:
        cliques = _find_cliques(mean_ranks, nemenyi_pvalues, alpha)
    else:
        # Fall back to CD criterion: connect methods within CD of each other
        sorted_methods = sorted(mean_ranks, key=mean_ranks.__getitem__)
        cliques = []
        for i, m_i in enumerate(sorted_methods):
            group = [m_j for m_j in sorted_methods
                     if abs(mean_ranks[m_i] - mean_ranks[m_j]) < cd]
            if len(group) > 1 and group not in cliques:
                cliques.append(group)

    # ── Layout ────────────────────────────────────────────────────────────────
    sorted_methods = sorted(mean_ranks, key=mean_ranks.__getitem__)
    rank_min, rank_max = 1.0, float(n_m)

    # Methods split: left half on right side of axis, right half on left side
    mid = n_m // 2
    right_methods = sorted_methods[:mid]          # best (low rank) → right column
    left_methods  = sorted_methods[mid:][::-1]    # worst (high rank) → left column

    label_pad  = 0.35     # horizontal padding for labels
    axis_y     = 0.60     # y-position of the rank axis (in axes coordinates)
    label_step = 0.08     # vertical spacing per label row

    fig, ax = _ensure_ax(ax, figsize)
    ax.set_xlim(rank_max + 0.5 + label_pad, rank_min - 0.5 - label_pad)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title is None:
        title = f"Critical Difference Diagram  (α = {alpha})"
    ax.set_title(title, fontsize=11, pad=10)

    # ── Draw rank axis ────────────────────────────────────────────────────────
    ax.annotate(
        "", xy=(rank_min - 0.3, axis_y), xytext=(rank_max + 0.3, axis_y),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
    )
    for r in range(1, n_m + 1):
        ax.plot([r, r], [axis_y - 0.02, axis_y + 0.02], color="black", lw=1.2)
        ax.text(r, axis_y + 0.04, str(r), ha="center", va="bottom", fontsize=8)

    ax.text(
        (rank_min + rank_max) / 2,
        axis_y + 0.13,
        "← better" if higher_is_better else "better →",
        ha="center", va="bottom", fontsize=7, color="gray", style="italic",
    )

    # ── Draw method labels and connector lines ─────────────────────────────────
    def _draw_method(method: str, row: int, side: str) -> None:
        """Draw label + vertical line + tick for one method."""
        rank = mean_ranks[method]
        label_y  = axis_y + 0.20 + row * label_step if side == "top" else \
                   axis_y - 0.20 - row * label_step
        ha       = "right" if side == "top" else "left"
        x_label  = rank_max + label_pad if side == "top" else rank_min - label_pad

        # Horizontal connector from axis tick to the label column
        ax.plot([rank, x_label], [axis_y, label_y], color="black", lw=0.8)
        ax.text(x_label, label_y, f"{method} ({rank:.2f})",
                ha=ha, va="center", fontsize=method_fontsize)

    for row, m in enumerate(right_methods):
        _draw_method(m, row, "top")
    for row, m in enumerate(left_methods):
        _draw_method(m, row, "bottom")

    # ── Draw clique bars ─────────────────────────────────────────────────────
    # Bars drawn BELOW the axis at decreasing y positions
    bar_y_start = axis_y - 0.07
    bar_step    = 0.045
    drawn: list[tuple[float, float]] = []

    for ci_idx, clique in enumerate(cliques):
        r_left  = min(mean_ranks[m] for m in clique)
        r_right = max(mean_ranks[m] for m in clique)

        # Find a y level that doesn't overlap previous bars
        bar_y = bar_y_start
        for prev_l, prev_r in drawn:
            if not (r_right < prev_l - 0.05 or r_left > prev_r + 0.05):
                bar_y -= bar_step
        drawn.append((r_left, r_right))

        ax.plot([r_left, r_right], [bar_y, bar_y],
                color="#333333", lw=3.5, solid_capstyle="round", zorder=5)

    # ── CD bracket ────────────────────────────────────────────────────────────
    if show_cd_bracket:
        cd_y  = axis_y + 0.85
        cd_x0 = rank_min
        cd_x1 = rank_min + cd
        ax.annotate(
            "", xy=(cd_x1, cd_y), xytext=(cd_x0, cd_y),
            arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.5),
        )
        ax.text(
            (cd_x0 + cd_x1) / 2, cd_y + 0.04,
            f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=9, color="steelblue",
        )
        # Tick lines
        for x_cd in (cd_x0, cd_x1):
            ax.plot([x_cd, x_cd], [cd_y - 0.03, cd_y + 0.03],
                    color="steelblue", lw=1.5)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    formats: Sequence[str] = ("pdf", "png"),
) -> dict[str, Path]:
    """Save *fig* to one or more formats derived from *path*.

    Parameters
    ----------
    fig :
        Matplotlib figure to save.
    path :
        Base path (with or without extension).  The stem is reused for
        each format in *formats*.
    dpi :
        Resolution for raster formats (PNG).
    bbox_inches :
        Passed to ``savefig`` (default ``"tight"``).
    formats :
        Iterable of file format strings (default ``("pdf", "png")``).

    Returns
    -------
    dict[str, Path]
        Map of format string → written file path.
    """
    base  = Path(path).with_suffix("")
    saved: dict[str, Path] = {}
    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
        saved[fmt] = p
    return saved


def plot_suite(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    experiment_name: str = "experiment",
    nemenyi_pvalues: pd.DataFrame | None = None,
    scores_matrix: np.ndarray | None = None,
    method_names: list[str] | None = None,
    alpha: float = 0.05,
    dpi: int = 150,
) -> dict[str, dict[str, Path]]:
    """Generate and save all standard plots to *output_dir*.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested map: ``{plot_name: {format: Path}}``.
    """
    out = Path(output_dir)
    saved: dict[str, dict[str, Path]] = {}

    fig1 = plot_delta_auc_heatmap(df)
    saved["delta_auc_heatmap"] = save_figure(
        fig1, out / f"{experiment_name}_delta_auc_heatmap", dpi=dpi
    )
    plt.close(fig1)

    try:
        fig2 = plot_gain_barplot(df)
        saved["gain_barplot"] = save_figure(
            fig2, out / f"{experiment_name}_gain_barplot", dpi=dpi
        )
        plt.close(fig2)
    except ValueError as exc:
        warnings.warn(f"Skipping gain barplot: {exc}", stacklevel=2)

    if scores_matrix is not None and method_names is not None:
        fig3 = plot_cd_diagram(
            scores_matrix, method_names,
            alpha=alpha,
            nemenyi_pvalues=nemenyi_pvalues,
            title=f"CD Diagram — {experiment_name}",
        )
        saved["cd_diagram"] = save_figure(
            fig3, out / f"{experiment_name}_cd_diagram", dpi=dpi
        )
        plt.close(fig3)

    return saved


__all__ = [
    "plot_delta_auc_heatmap",
    "plot_gain_barplot",
    "plot_cd_diagram",
    "save_figure",
    "plot_suite",
]

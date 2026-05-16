"""Single-command analysis pipeline for nstad_bench experiments.

Given a Parquet results file (written by ``run_experiment()``), this module
infers the experiment name, creates a structured output tree, and produces
all standard tables and figures automatically.

Output layout
-------------
::

    results/
    └── <experiment_name>/
        ├── tables/
        │   ├── delta_auc_pivot.tex
        │   ├── gain_by_dataset.tex
        │   ├── method_summary.tex
        │   ├── screening_summary.tex
        │   └── anova.tex
        └── figures/
            ├── delta_auc_heatmap.pdf
            ├── delta_auc_heatmap.png
            ├── gain_barplot.pdf
            ├── gain_barplot.png
            ├── cd_diagram.pdf
            └── cd_diagram.png

Usage — Python
--------------
::

    from nstad_bench.analysis.pipeline import analyze_experiment

    # Minimal: just the Parquet path
    report = analyze_experiment("results/my_experiment.parquet")
    print(report)

    # Override output root or alpha level
    report = analyze_experiment(
        "results/my_experiment.parquet",
        output_root="paper/outputs",
        alpha=0.01,
    )

Usage — CLI
-----------
::

    nstad-analyze results/my_experiment.parquet
    nstad-analyze results/my_experiment.parquet --output-root paper/outputs --alpha 0.01
    nstad-analyze results/my_experiment.parquet --no-plots --no-anova
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: collision-avoidance and metadata
# ─────────────────────────────────────────────────────────────────────────────

def _timestamped_dir(base: Path) -> Path:
    """Return *base* with a UTC timestamp suffix if it already exists.

    e.g. ``results/mitbih_full`` → ``results/mitbih_full_20260517_2030``
    """
    if not base.exists():
        return base
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    candidate = base.parent / f"{base.name}_{ts}"
    # Guard against sub-second re-runs in tests
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = base.parent / f"{base.name}_{ts}_{counter}"
    return candidate


def _git_commit_hash() -> str:
    """Return the current HEAD commit hash, or ``'unknown'`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _library_versions() -> dict[str, str]:
    """Return installed versions of key analysis libraries."""
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "matplotlib", "statsmodels",
                "scikit_posthocs", "scipy", "pyarrow"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _write_metadata(
    directory: Path,
    *,
    source_parquet: Path,
    timestamp: str,
    git_commit_hash: str,
    library_versions: dict[str, str],
    extra: dict | None = None,
) -> Path:
    """Write ``metadata.json`` to *directory* and return the path."""
    meta: dict = {
        "source_parquet":   str(source_parquet.resolve()),
        "timestamp":        timestamp,
        "git_commit_hash":  git_commit_hash,
        "library_versions": library_versions,
    }
    if extra:
        meta.update(extra)
    path = directory / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisReport:
    """Summary of a completed ``analyze_experiment()`` run.

    Attributes
    ----------
    experiment_name :
        Derived from the Parquet file stem (e.g. ``"mitbih_full"``).
    output_dir :
        Root directory that contains ``tables/`` and ``figures/`` sub-dirs.
    tables : dict[str, Path]
        Map of table name → written ``.tex`` path.
    figures : dict[str, dict[str, Path]]
        Map of figure name → {format: Path}.
    anova_summary : str
        One-paragraph ANOVA interpretation (plain text).
    errors : list[str]
        Non-fatal errors encountered (step skipped but run continued).
    """

    experiment_name: str
    output_dir:      Path
    tables:          dict[str, Path]             = field(default_factory=dict)
    figures:         dict[str, dict[str, Path]]  = field(default_factory=dict)
    anova_summary:   str                         = ""
    errors:          list[str]                   = field(default_factory=list)
    metadata:        dict                        = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"AnalysisReport  [{self.experiment_name}]",
            f"Output root : {self.output_dir}",
            "",
            f"Tables  ({len(self.tables)}):",
        ]
        for name, p in self.tables.items():
            lines.append(f"  {name:<25s}  {p}")
        lines += ["", f"Figures ({len(self.figures)}):"]
        for name, fmts in self.figures.items():
            for fmt, p in fmts.items():
                lines.append(f"  {name:<25s}  [{fmt}]  {p}")
        if self.metadata:
            lines += ["", "Metadata:"]
            lines.append(f"  timestamp   : {self.metadata.get('timestamp', '?')}")
            lines.append(f"  git commit  : {self.metadata.get('git_commit_hash', '?')}")
            libs = self.metadata.get("library_versions", {})
            if libs:
                lines.append(f"  numpy       : {libs.get('numpy', '?')}")
                lines.append(f"  pandas      : {libs.get('pandas', '?')}")
                lines.append(f"  matplotlib  : {libs.get('matplotlib', '?')}")
        if self.anova_summary:
            lines += ["", "ANOVA interpretation:", self.anova_summary]
        if self.errors:
            lines += ["", f"Errors ({len(self.errors)} non-fatal):"]
            for e in self.errors:
                lines.append(f"  ! {e}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ANOVA interpretation helper
# ─────────────────────────────────────────────────────────────────────────────

def _interpret_anova(result) -> str:
    """Return a plain-text paragraph interpreting the ANOVA result."""
    from nstad_bench.analysis.anova import effect_size_ranking

    ranking = effect_size_ranking(result)
    lines: list[str] = [
        f"ANOVA on ΔAUC (N={result.n_obs}, R²={result.r_squared:.3f}, "
        f"R²_adj={result.r_squared_adj:.3f}).",
        "",
        "Effect-size ranking (partial η²):",
    ]

    alpha = 0.05
    for _, row in ranking.iterrows():
        src  = row["source"]
        eta2 = row["partial_eta2"]
        p    = row.get("PR(>F)", float("nan"))
        if np.isnan(p):
            sig_label = ""
        elif p < 0.001:
            sig_label = "p<0.001 ***"
        elif p < 0.01:
            sig_label = f"p={p:.3f} **"
        elif p < alpha:
            sig_label = f"p={p:.3f} *"
        else:
            sig_label = f"p={p:.3f} n.s."

        effect_mag = (
            "large"  if eta2 >= 0.14 else
            "medium" if eta2 >= 0.06 else
            "small"
        )
        lines.append(
            f"  {src:<30s}  η²={eta2:.3f} ({effect_mag})  {sig_label}"
        )

    # Main-effect prose
    p_phi   = result.factor_pvalue("φ (representation)")
    p_theta = result.factor_pvalue("θ (model)")
    p_psi   = result.factor_pvalue("ψ (adaptation)")
    p_phipsi = result.factor_pvalue("φ × ψ")

    prose_parts: list[str] = []
    if p_phi is not None:
        verdict = "significantly" if p_phi < alpha else "not significantly"
        prose_parts.append(
            f"Representation φ {verdict} affects the domain gap "
            f"(p={p_phi:.3g}, η²={result.factor_partial_eta2('φ (representation)'):.3f})."
        )
    if p_theta is not None:
        verdict = "significantly" if p_theta < alpha else "not significantly"
        prose_parts.append(
            f"Model architecture θ {verdict} modulates ΔAUC "
            f"(p={p_theta:.3g}, η²={result.factor_partial_eta2('θ (model)'):.3f})."
        )
    if p_psi is not None:
        verdict = "significantly" if p_psi < alpha else "not significantly"
        prose_parts.append(
            f"Adaptation method ψ {verdict} affects raw ΔAUC "
            f"(p={p_psi:.3g}) — ψ is expected to influence Gain rather than ΔAUC."
        )
    if p_phipsi is not None and p_phipsi < alpha:
        prose_parts.append(
            f"The φ×ψ interaction is significant (p={p_phipsi:.3g}), "
            "suggesting that adaptation benefits depend on the chosen representation."
        )

    if prose_parts:
        lines += ["", "Interpretation:"] + [f"  {s}" for s in prose_parts]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_experiment(
    parquet_path: str | Path,
    *,
    output_root: str | Path | None = None,
    alpha: float = 0.05,
    n_bootstrap_ci: bool = True,
    figures: bool = True,
    tables: bool = True,
    anova: bool = True,
    dpi: int = 150,
    figure_formats: tuple[str, ...] = ("pdf", "png"),
    latex_fmt: str = ".3f",
    anova_metric: str = "delta_auc",
) -> AnalysisReport:
    """Run the full analysis pipeline for one experiment.

    Reads ``<parquet_path>``, derives the experiment name from the file stem,
    and writes all outputs under::

        <output_root>/<experiment_name>/tables/
        <output_root>/<experiment_name>/figures/

    Parameters
    ----------
    parquet_path :
        Path to the Parquet file produced by ``run_experiment()``.
        The file stem is used as the experiment name
        (e.g. ``results/mitbih_full.parquet`` → name ``"mitbih_full"``).
    output_root :
        Parent directory for all outputs.  Defaults to the **same directory
        as the Parquet file** so outputs land next to the data that
        produced them.
    alpha :
        Significance level for Friedman/Nemenyi and ANOVA (default 0.05).
    n_bootstrap_ci :
        Whether to include CI columns in method_summary table.
    figures :
        Generate and save plots (default ``True``).
    tables :
        Generate and save LaTeX tables (default ``True``).
    anova :
        Run ANOVA and append ``anova.tex`` (default ``True``).
    dpi :
        Raster figure resolution.
    figure_formats :
        File formats for each figure (default ``("pdf", "png")``).
    latex_fmt :
        Float format string for LaTeX tables.
    anova_metric :
        Dependent variable for ANOVA (default ``"delta_auc"``).

    Returns
    -------
    AnalysisReport
        Contains paths to every written file and an ANOVA interpretation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parquet_path = Path(parquet_path)
    exp_name     = parquet_path.stem

    if output_root is None:
        output_root = parquet_path.parent
    output_root = Path(output_root)

    # ── Collision-avoidance: timestamp suffix if target already exists ─────────
    base_dir = output_root / exp_name
    exp_dir  = _timestamped_dir(base_dir)
    if exp_dir != base_dir:
        log.warning(
            "Output directory '%s' already exists — writing to '%s' instead.",
            base_dir, exp_dir,
        )

    tables_dir  = exp_dir / "tables"
    figures_dir = exp_dir / "figures"

    # ── Collect metadata once (shared between tables/ and figures/) ────────────
    timestamp    = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    git_hash     = _git_commit_hash()
    lib_versions = _library_versions()

    meta_payload: dict = {
        "source_parquet":   str(parquet_path.resolve()),
        "timestamp":        timestamp,
        "git_commit_hash":  git_hash,
        "library_versions": lib_versions,
    }

    log.info("Experiment   : %s", exp_name)
    log.info("Results file : %s", parquet_path)
    log.info("Output dir   : %s", exp_dir)
    log.info("Git commit   : %s", git_hash)
    log.info("Timestamp    : %s", timestamp)

    report = AnalysisReport(
        experiment_name=exp_name,
        output_dir=exp_dir,
        metadata=meta_payload,
    )

    # ── Load results ──────────────────────────────────────────────────────────
    from nstad_bench.analysis.summary_tables import load_results
    df = load_results(parquet_path)
    log.info("Loaded %d rows, metrics: %s",
             len(df), sorted(df["metric_name"].unique()))

    # ── Tables ────────────────────────────────────────────────────────────────
    if tables:
        tables_dir.mkdir(parents=True, exist_ok=True)
        _write_metadata(tables_dir, source_parquet=parquet_path,
                        timestamp=timestamp, git_commit_hash=git_hash,
                        library_versions=lib_versions)
        from nstad_bench.analysis.summary_tables import (
            pivot_delta_auc, gain_by_dataset, method_summary,
            screening_summary, to_latex,
        )

        table_specs = [
            (
                "delta_auc_pivot",
                lambda: pivot_delta_auc(df, source_only=False),
                rf"Mean $\Delta$AUC by $\varphi$ and $\theta$ — {exp_name}",
                f"tab:{exp_name}_delta_auc",
                True,   # bold_max
            ),
            (
                "gain_by_dataset",
                lambda: gain_by_dataset(df),
                rf"Mean Gain$(\psi)$ by dataset — {exp_name}",
                f"tab:{exp_name}_gain",
                True,
            ),
            (
                "method_summary",
                lambda: method_summary(df, ci_columns=n_bootstrap_ci),
                rf"Per-method performance summary — {exp_name}",
                f"tab:{exp_name}_methods",
                False,
            ),
            (
                "screening_summary",
                lambda: screening_summary(df),
                rf"Screening ranking of $(\varphi, \theta)$ pairs — {exp_name}",
                f"tab:{exp_name}_screening",
                False,
            ),
        ]

        for tname, builder, caption, label, bold_max in table_specs:
            try:
                tbl  = builder()
                path = tables_dir / f"{tname}.tex"
                to_latex(tbl, path=path, caption=caption, label=label,
                         fmt=latex_fmt, bold_max=bold_max)
                report.tables[tname] = path
                log.info("  table %-25s → %s", tname, path)
            except Exception as exc:
                msg = f"table '{tname}' skipped: {exc}"
                report.errors.append(msg)
                log.warning("  ! %s", msg)

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    if anova and tables:
        try:
            from nstad_bench.analysis.anova import run_anova
            anova_result = run_anova(df, metric=anova_metric, aggregate_seeds=True)
            path = tables_dir / "anova.tex"
            anova_result.to_latex(
                path=path,
                caption=(
                    rf"Three-factor ANOVA on $\Delta$AUC $\sim$ "
                    rf"$\varphi + \theta + \psi$ + interactions — {exp_name}"
                ),
                label=f"tab:{exp_name}_anova",
                note=r"$^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
                     r"Partial $\hat{\eta}^2_p$ as effect-size measure.",
            )
            report.tables["anova"] = path
            report.anova_summary   = _interpret_anova(anova_result)
            log.info("  table %-25s → %s", "anova", path)
            for warn in anova_result.warnings:
                log.debug("  ANOVA warning: %s", warn[:120])
        except Exception as exc:
            msg = f"ANOVA skipped: {exc}"
            report.errors.append(msg)
            log.warning("  ! %s", msg)

    # ── Figures ───────────────────────────────────────────────────────────────
    if figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        _write_metadata(figures_dir, source_parquet=parquet_path,
                        timestamp=timestamp, git_commit_hash=git_hash,
                        library_versions=lib_versions)
        from nstad_bench.analysis.plots import (
            plot_delta_auc_heatmap, plot_gain_barplot,
            plot_cd_diagram, save_figure,
        )
        from nstad_bench.analysis.summary_tables import scores_matrix

        # 1. ΔAUC heatmap
        try:
            fig = plot_delta_auc_heatmap(
                df, source_only=False,
                title=rf"$\Delta$AUC heatmap — {exp_name}",
            )
            paths = save_figure(fig, figures_dir / "delta_auc_heatmap",
                                dpi=dpi, formats=figure_formats)
            report.figures["delta_auc_heatmap"] = paths
            plt.close(fig)
            log.info("  figure delta_auc_heatmap → %s",
                     ", ".join(str(p) for p in paths.values()))
        except Exception as exc:
            msg = f"figure 'delta_auc_heatmap' skipped: {exc}"
            report.errors.append(msg)
            log.warning("  ! %s", msg)

        # 2. Gain bar-plot
        try:
            fig = plot_gain_barplot(
                df, title=rf"Gain$(\psi)$ per dataset — {exp_name}",
            )
            paths = save_figure(fig, figures_dir / "gain_barplot",
                                dpi=dpi, formats=figure_formats)
            report.figures["gain_barplot"] = paths
            plt.close(fig)
            log.info("  figure gain_barplot → %s",
                     ", ".join(str(p) for p in paths.values()))
        except Exception as exc:
            msg = f"figure 'gain_barplot' skipped: {exc}"
            report.errors.append(msg)
            log.warning("  ! %s", msg)

        # 3. CD diagram (requires Friedman+Nemenyi)
        try:
            from nstad_bench.metrics.statistical import friedman_nemenyi

            mat, dss, meths = scores_matrix(df, metric="roc_auc")
            if len(meths) >= 3 and len(dss) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr = friedman_nemenyi(mat, method_names=meths)
                fig = plot_cd_diagram(
                    mat, meths,
                    nemenyi_pvalues=fr.nemenyi_pvalues,
                    alpha=alpha,
                    title=(
                        rf"CD Diagram (ROC-AUC) — {exp_name}  "
                        rf"($N$={len(dss)} datasets, $k$={len(meths)} methods)"
                    ),
                )
                paths = save_figure(fig, figures_dir / "cd_diagram",
                                    dpi=dpi, formats=figure_formats)
                report.figures["cd_diagram"] = paths
                plt.close(fig)
                log.info("  figure cd_diagram → %s",
                         ", ".join(str(p) for p in paths.values()))
            else:
                log.info("  CD diagram skipped: need ≥3 methods and ≥2 datasets "
                         "(got %d methods, %d datasets)", len(meths), len(dss))
        except Exception as exc:
            msg = f"figure 'cd_diagram' skipped: {exc}"
            report.errors.append(msg)
            log.warning("  ! %s", msg)

    log.info(
        "Done — %d tables, %d figures, %d errors",
        len(report.tables), len(report.figures), len(report.errors),
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Command-line interface: ``nstad-analyze <parquet_file> [options]``."""
    parser = argparse.ArgumentParser(
        prog="nstad-analyze",
        description=(
            "Run the full analysis pipeline for a nstad_bench experiment.\n"
            "Reads a Parquet results file and writes tables + figures under\n"
            "  <output-root>/<experiment-name>/tables/\n"
            "  <output-root>/<experiment-name>/figures/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "parquet",
        metavar="PARQUET",
        help="Path to the .parquet results file (e.g. results/mitbih.parquet).",
    )
    parser.add_argument(
        "--output-root", "-o",
        default=None,
        metavar="DIR",
        help=(
            "Parent directory for outputs. "
            "Default: same directory as the Parquet file."
        ),
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for Nemenyi / ANOVA (default 0.05).",
    )
    parser.add_argument(
        "--no-plots",  dest="figures", action="store_false",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--no-tables", dest="tables",  action="store_false",
        help="Skip LaTeX table generation.",
    )
    parser.add_argument(
        "--no-anova",  dest="anova",   action="store_false",
        help="Skip ANOVA.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Raster figure DPI (default 150).",
    )
    parser.add_argument(
        "--formats", default="pdf,png",
        help="Comma-separated figure formats (default 'pdf,png').",
    )
    parser.add_argument(
        "--anova-metric", default="delta_auc",
        help="Metric for ANOVA dependent variable (default 'delta_auc').",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.set_defaults(figures=True, tables=True, anova=True)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    report = analyze_experiment(
        args.parquet,
        output_root=args.output_root,
        alpha=args.alpha,
        figures=args.figures,
        tables=args.tables,
        anova=args.anova,
        dpi=args.dpi,
        figure_formats=tuple(args.formats.split(",")),
        anova_metric=args.anova_metric,
    )

    print(report)

    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["analyze_experiment", "AnalysisReport"]

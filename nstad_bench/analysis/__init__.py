"""Result analysis, visualisation, and reporting."""

from nstad_bench.analysis.anova import AnovaResult, effect_size_ranking, run_anova
from nstad_bench.analysis.pipeline import AnalysisReport, analyze_experiment
from nstad_bench.analysis.base import BaseAnalyzer, BaseReporter
from nstad_bench.analysis.plots import (
    plot_cd_diagram,
    plot_delta_auc_heatmap,
    plot_gain_barplot,
    plot_suite,
    save_figure,
)
from nstad_bench.analysis.summary_tables import (
    export_suite,
    gain_by_dataset,
    load_results,
    method_summary,
    pivot_delta_auc,
    scores_matrix,
    screening_summary,
    to_latex,
)

__all__ = [
    # pipeline
    "analyze_experiment",
    "AnalysisReport",
    # base
    "BaseAnalyzer",
    "BaseReporter",
    # summary_tables
    "load_results",
    "pivot_delta_auc",
    "gain_by_dataset",
    "method_summary",
    "screening_summary",
    "scores_matrix",
    "to_latex",
    "export_suite",
    # plots
    "plot_delta_auc_heatmap",
    "plot_gain_barplot",
    "plot_cd_diagram",
    "save_figure",
    "plot_suite",
    # anova
    "AnovaResult",
    "run_anova",
    "effect_size_ranking",
]

"""Evaluation metrics and statistical tests for domain-adaptation benchmarks."""

from nstad_bench.metrics.base import BaseDistanceMeasure, BaseMetric
from nstad_bench.metrics.scores import (
    best_threshold,
    delta_auc,
    f1_at_best_threshold,
    gain,
    mcc,
    pr_auc,
    roc_auc,
)
from nstad_bench.metrics.bootstrap import BootstrapCI, bootstrap_ci
from nstad_bench.metrics.statistical import (
    FriedmanNemenyiResult,
    WilcoxonResult,
    friedman_nemenyi,
    wilcoxon_test,
)

__all__ = [
    # base
    "BaseMetric",
    "BaseDistanceMeasure",
    # point metrics
    "roc_auc",
    "pr_auc",
    "best_threshold",
    "f1_at_best_threshold",
    "mcc",
    "delta_auc",
    "gain",
    # bootstrap
    "BootstrapCI",
    "bootstrap_ci",
    # statistical tests
    "WilcoxonResult",
    "FriedmanNemenyiResult",
    "wilcoxon_test",
    "friedman_nemenyi",
]

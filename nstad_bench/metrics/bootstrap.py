"""Bootstrap confidence intervals for evaluation metrics.

Usage
-----
::

    from nstad_bench.metrics.bootstrap import bootstrap_ci
    from nstad_bench.metrics.scores import roc_auc, mcc, best_threshold

    # Simple ranking metric
    ci = bootstrap_ci(roc_auc, y_true, y_score)
    print(ci)  # BootstrapCI(estimate=0.8523, lower=0.8101, upper=0.8912, ...)

    # Threshold-dependent metric: bake the threshold in with a lambda
    t = best_threshold(y_true_val, y_score_val)
    ci = bootstrap_ci(lambda yt, ys: mcc(yt, ys, threshold=t), y_true, y_score)

    # f1_at_best_threshold: val set is fixed; only the test set is resampled
    from functools import partial
    from nstad_bench.metrics.scores import f1_at_best_threshold
    fn = partial(f1_at_best_threshold,
                 y_true_val=y_true_val, y_score_val=y_score_val)
    ci = bootstrap_ci(fn, y_true_test, y_score_test)

Algorithm
---------
Percentile bootstrap (Efron & Tibshirani 1993, Ch. 13):

1. Resample ``(y_true, y_score)`` jointly *with replacement* to produce
   ``n_bootstrap`` bootstrap datasets.
2. Evaluate ``metric_fn`` on each resample.
3. Report the (α/2, 1−α/2) empirical percentiles as the CI bounds.

Degenerate resamples (only one class present) are silently skipped; the
reported ``n_bootstrap`` counts only valid resamples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a bootstrap confidence-interval computation.

    Attributes
    ----------
    estimate : float
        Point estimate on the full (un-resampled) data.
    lower : float
        Lower bound of the percentile CI.
    upper : float
        Upper bound of the percentile CI.
    n_bootstrap : int
        Number of *valid* (non-degenerate) bootstrap resamples used.
    confidence : float
        Nominal coverage level (e.g. 0.95 for a 95 % CI).
    """

    estimate: float
    lower: float
    upper: float
    n_bootstrap: int
    confidence: float

    def __repr__(self) -> str:
        return (
            f"BootstrapCI(estimate={self.estimate:.4f}, "
            f"[{self.lower:.4f}, {self.upper:.4f}], "
            f"n={self.n_bootstrap}, conf={self.confidence:.0%})"
        )

    def contains(self, value: float) -> bool:
        """Return True if *value* lies within ``[lower, upper]``."""
        return self.lower <= value <= self.upper

    @property
    def width(self) -> float:
        """Width of the CI: ``upper − lower``."""
        return self.upper - self.lower


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for any metric that takes ``(y_true, y_score)``.

    Parameters
    ----------
    metric_fn :
        ``Callable(y_true, y_score) -> float``.  For metrics with extra
        arguments (e.g. a pre-computed threshold), wrap with ``functools.partial``
        or a ``lambda``.
    y_true : (N,) binary ground-truth labels (0 or 1).
    y_score : (N,) continuous anomaly scores (higher = more anomalous).
    n_bootstrap :
        Number of bootstrap resamples (default 1000).
    confidence :
        Nominal coverage, e.g. 0.95 for a 95 % CI (default 0.95).
    seed :
        RNG seed for reproducibility (default 42).

    Returns
    -------
    BootstrapCI
        Point estimate + lower/upper percentile bounds.

    Raises
    ------
    ValueError
        If no valid bootstrap resamples could be computed (all degenerate).
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, "
            f"got {y_true.shape} and {y_score.shape}."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    rng = np.random.default_rng(seed)
    n   = len(y_true)
    alpha = 1.0 - confidence

    boot_estimates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        # Skip degenerate resamples (only one class present)
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot_estimates.append(float(metric_fn(yt, ys)))
        except Exception:
            continue

    if not boot_estimates:
        raise ValueError(
            "All bootstrap resamples were degenerate (single class present). "
            "Increase N or check that y_true contains both classes."
        )

    samples = np.array(boot_estimates)
    lower   = float(np.percentile(samples, 100.0 * alpha / 2.0))
    upper   = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapCI(
        estimate=float(metric_fn(y_true, y_score)),
        lower=lower,
        upper=upper,
        n_bootstrap=len(boot_estimates),
        confidence=confidence,
    )

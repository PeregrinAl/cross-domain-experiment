"""Gradient Boosting Machine — boosted decision-tree classifier.

Backed by :class:`sklearn.ensemble.HistGradientBoostingClassifier`,
the histogram-based GBM that ships with sklearn and matches LightGBM
on accuracy/speed without adding an extra dependency.

The YAML benchmark exposes ``n_estimators`` (number of boosting
iterations) for parity with :class:`RandomForest`; internally this is
forwarded to ``HistGradientBoostingClassifier.max_iter``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from nstad_bench.models.statistical.base import StatModel


class GBM(StatModel):
    """Gradient-boosted trees baseline.

    Parameters
    ----------
    n_estimators :
        Number of boosting iterations (forwarded to
        ``HistGradientBoostingClassifier.max_iter``).
    learning_rate, max_depth, random_state, **kwargs :
        Forwarded to
        :class:`sklearn.ensemble.HistGradientBoostingClassifier`.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int | None = 3,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
            **kwargs,
        }
        self._estimator = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self._adapt_transform = None

    def _extract(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)

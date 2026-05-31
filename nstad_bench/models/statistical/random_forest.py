"""RandomForest — ensemble of bagged decision trees.

Wraps :class:`sklearn.ensemble.RandomForestClassifier` behind the
:class:`StatModel` protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nstad_bench.models.statistical.base import StatModel


class RandomForest(StatModel):
    """Random forest baseline.

    Parameters
    ----------
    n_estimators, max_depth, min_samples_leaf, random_state, n_jobs, **kwargs :
        Forwarded to :class:`sklearn.ensemble.RandomForestClassifier`.

    Notes
    -----
    On flattened wide representations (e.g. 12 000-dim LogSTFT-STEAD)
    RF works but is slow due to per-tree feature-subsampling overhead.
    For best speed, pair with hand-crafted features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        n_jobs: int | None = -1,
        **kwargs: Any,
    ) -> None:
        self._config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs,
        }
        self._estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self._adapt_transform = None

    def _extract(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        n = arr.shape[0]
        d = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        return arr.reshape(n, d)

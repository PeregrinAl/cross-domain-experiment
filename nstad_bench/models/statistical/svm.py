"""Support Vector Machine — kernel-based classifier with probability output.

Wraps :class:`sklearn.svm.SVC` with ``probability=True`` (required for
the benchmark's ``predict_proba`` contract).

Performance caveat
------------------
``probability=True`` fits Platt scaling via 5-fold CV after the main
SVM training, which adds an O(N²) overhead.  For N > 5–10 k samples
consider a hand-crafted feature extractor in ``_extract`` to reduce
dimensionality before the kernel evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.svm import SVC

from nstad_bench.models.statistical.base import StatModel


class SVM(StatModel):
    """Kernel SVM baseline (probability-calibrated).

    Parameters
    ----------
    C, kernel, gamma, random_state, **kwargs :
        Forwarded to :class:`sklearn.svm.SVC` (with ``probability=True``).
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
            "random_state": random_state,
            **kwargs,
        }
        self._estimator = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=random_state,
            **kwargs,
        )
        self._adapt_transform = None

    def _extract(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)

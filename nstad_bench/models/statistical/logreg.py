"""LogisticRegression — discriminative linear baseline.

Wraps :class:`sklearn.linear_model.LogisticRegression` behind the
:class:`StatModel` protocol.  Operates on a flattened feature vector
by default; override ``_extract`` to plug in handcrafted features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from nstad_bench.models.statistical.base import StatModel


class LogReg(StatModel):
    """Logistic regression baseline.

    Parameters
    ----------
    C, max_iter, solver, random_state, **kwargs :
        Forwarded to :class:`sklearn.linear_model.LogisticRegression`.

    Notes
    -----
    The default ``_extract`` flattens any input to ``(N, prod(rest))``.
    For very wide flattened representations (e.g. LogSTFT on STEAD →
    ~12 000-dim) lbfgs still converges but is slow; pre-process or
    subclass to use handcrafted features if needed.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = {
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "random_state": random_state,
            **kwargs,
        }
        self._estimator = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            **kwargs,
        )
        self._adapt_transform = None

    def _extract(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)

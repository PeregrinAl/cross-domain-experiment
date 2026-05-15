from __future__ import annotations

import numpy as np

from nstad_bench.representations.base import BaseRepresentation


class RawSignal(BaseRepresentation):
    """Z-score normalisation fitted on source statistics.

    Computes per-feature (column) mean and std during ``fit``, then applies
    z-score standardisation in ``transform``.  Handles both univariate
    ``(N, T)`` and multivariate ``(N, T, C)`` inputs uniformly.
    """

    is_1d: bool = True
    is_2d: bool = False

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.output_shape: tuple[int, ...] | None = None

    def fit(self, X: np.ndarray, **kwargs) -> "RawSignal":
        """Estimate global mean and std from source data *X*.

        Parameters
        ----------
        X:
            Array of shape ``(N, T)`` or ``(N, T, C)``.
        """
        # Compute statistics over the sample axis so that shape broadcasts
        # correctly across both univariate and multivariate inputs.
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.output_shape = X.shape[1:]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.mean_) / (self.std_ + self.eps)

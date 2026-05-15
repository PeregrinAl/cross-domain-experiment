from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAdaptation(ABC):
    """Abstract base for domain adaptation / shift correction methods."""

    @abstractmethod
    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "BaseAdaptation":
        """Estimate adaptation mapping from source to target domain."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned adaptation to samples *X*."""
        ...

    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.fit(X_source, X_target, y_source, **kwargs).transform(X_source)

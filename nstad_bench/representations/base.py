from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseRepresentation(ABC):
    """Abstract base for feature/embedding representations."""

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs: Any) -> "BaseRepresentation":
        """Fit representation parameters on source data *X*."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map raw inputs *X* to a representation space."""
        ...

    def fit_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.fit(X, **kwargs).transform(X)


class BaseKernel(ABC):
    """Abstract base for kernel / similarity functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X, Y)."""
        ...

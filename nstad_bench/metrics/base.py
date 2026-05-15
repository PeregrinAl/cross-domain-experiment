from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    """Abstract base for evaluation metrics."""

    name: str = ""

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return a scalar score for predictions *y_pred* against *y_true*."""
        ...

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.compute(y_true, y_pred)


class BaseDistanceMeasure(ABC):
    """Abstract base for distribution-distance / divergence measures."""

    name: str = ""

    @abstractmethod
    def compute(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Return a scalar distance between distributions *P* and *Q*."""
        ...

    def __call__(self, P: np.ndarray, Q: np.ndarray) -> float:
        return self.compute(P, Q)

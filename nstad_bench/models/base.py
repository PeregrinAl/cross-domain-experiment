from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base for all predictive models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseModel":
        """Train the model on (X, y)."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialise model to *path*."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Deserialise model from *path*."""
        ...

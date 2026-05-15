from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseExperiment(ABC):
    """Abstract base for benchmark experiments."""

    name: str = ""

    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Initialise experiment from *config* dict."""
        ...

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute the experiment and return a results DataFrame."""
        ...

    @abstractmethod
    def save_results(self, results: pd.DataFrame, path: Path) -> None:
        """Persist *results* to *path*."""
        ...

    def run_and_save(self, path: Path) -> pd.DataFrame:
        results = self.run()
        self.save_results(results, path)
        return results

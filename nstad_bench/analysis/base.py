from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseAnalyzer(ABC):
    """Abstract base for result analysis / post-processing."""

    @abstractmethod
    def analyze(self, results: pd.DataFrame) -> pd.DataFrame:
        """Process raw *results* and return an analysis summary."""
        ...

    @abstractmethod
    def plot(self, results: pd.DataFrame, output_dir: Path) -> None:
        """Generate and save plots from *results* to *output_dir*."""
        ...


class BaseReporter(ABC):
    """Abstract base for generating human-readable reports."""

    @abstractmethod
    def generate(self, summary: pd.DataFrame, output_path: Path) -> None:
        """Write a report from *summary* to *output_path*."""
        ...

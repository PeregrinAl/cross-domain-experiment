from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class BaseDataset(ABC):
    """Abstract base for all datasets in the benchmark."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Any]: ...


class BaseDataLoader(ABC):
    """Abstract base for dataset loaders."""

    @abstractmethod
    def load(self, path: Path) -> BaseDataset:
        """Load a dataset from *path* and return a BaseDataset instance."""
        ...

    @abstractmethod
    def save(self, dataset: BaseDataset, path: Path) -> None:
        """Persist *dataset* to *path*."""
        ...


class BaseSplitter(ABC):
    """Abstract base for train/test domain splitters."""

    @abstractmethod
    def split(
        self, dataset: BaseDataset
    ) -> tuple[BaseDataset, BaseDataset]:
        """Return (source, target) splits."""
        ...

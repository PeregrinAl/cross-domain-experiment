"""Shared building blocks for all nstad_bench models.

Every model in the benchmark exposes the same interface:
  - ``forward(x)``         — full forward pass (backbone → projector → head), returns logits
  - ``get_features(X)``    — backbone + projector output, always ``(N, 128)`` numpy array
  - ``predict_proba(X)``   — softmax class probabilities (N, 2)
  - ``predict(X)``         — argmax class predictions (N,)
  - ``fit(X, y, ...)``     — mini-batch Adam training loop
  - ``save(path)`` / ``load(path)``

Pipeline
--------
All models share a three-stage pipeline::

    backbone(x)              → (B, d_backbone)   model-specific dimension
    projector(backbone(x))   → (B, 128)           universal projected features
    head(projector(...))     → (B, 2)             logits

The **projector** is a single ``Linear(d_backbone, 128)`` with no activation,
ensuring that ``get_features`` always returns a 128-dimensional vector
regardless of the backbone architecture.

The shared classification head is::

    Linear(128, 128) → ReLU → Dropout(0.1) → Linear(128, 2)

Subclasses must implement ``_preprocess(X)`` (numpy → tensor, model-specific
reshaping) and set ``self.backbone``, ``self.projector``, ``self.head``,
and ``self._config``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.models.base import BaseModel


class _ClassHead(nn.Module):
    """Shared classification head used by all benchmark models.

    Architecture::

        Linear(d_feat, 128) → ReLU → Dropout(0.1) → Linear(128, 2)

    Parameters
    ----------
    d_feat:
        Dimensionality of the backbone feature vector.
    """

    D_HIDDEN: int = 128
    D_OUT: int = 2

    def __init__(self, d_feat: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_feat, self.D_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.D_HIDDEN, self.D_OUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BenchModel(nn.Module, BaseModel):
    """Abstract base combining ``nn.Module`` with the benchmark I/O protocol.

    Subclasses must:
      1. Call ``super().__init__()`` (calls ``nn.Module.__init__``).
      2. Build and assign ``self.backbone: nn.Module``.
      3. Build and assign ``self.projector: nn.Linear`` — maps backbone output
         to the shared 128-dimensional feature space.
      4. Build and assign ``self.head: _ClassHead``.
      5. Implement ``_preprocess(X: np.ndarray) -> torch.Tensor``.
      6. Set ``self._config: dict[str, Any]`` with constructor kwargs (for
         ``save`` / ``load`` round-trips).
    """

    backbone: nn.Module
    projector: nn.Linear
    head: _ClassHead
    _config: dict[str, Any]

    def __init__(self) -> None:
        nn.Module.__init__(self)

    # ------------------------------------------------------------------ #
    # Core forward                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: backbone → projector → head → logits ``(B, 2)``."""
        return self.head(self.projector(self.backbone(x)))

    # ------------------------------------------------------------------ #
    # Preprocessing hook (model-specific)                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to the tensor shape expected by this model."""
        ...

    # ------------------------------------------------------------------ #
    # Public numpy API                                                     #
    # ------------------------------------------------------------------ #

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def get_features(self, X: np.ndarray) -> np.ndarray:
        """Return projected features — backbone output passed through the
        linear projector, **before** the classification head.

        The returned array always has shape ``(N, 128)`` regardless of the
        underlying backbone architecture, because every model wires a
        ``Linear(d_backbone, 128)`` projector between backbone and head.

        Parameters
        ----------
        X:
            Raw input array (numpy). Shape conventions are model-specific.

        Returns
        -------
        np.ndarray
            Shape ``(N, 128)`` — one 128-dimensional feature vector per sample.
        """
        self.eval()
        t = self._preprocess(X).to(self._device)
        return self.projector(self.backbone(t)).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(N, 2)`` with values in ``[0, 1]`` summing to 1 per row.
        """
        self.eval()
        t = self._preprocess(X).to(self._device)
        logits = self(t)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices (argmax of ``predict_proba``)."""
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> "BenchModel":
        """Train with mini-batch cross-entropy and Adam.

        Parameters
        ----------
        X:
            Input array (numpy).
        y:
            Integer class labels, shape ``(N,)``.
        epochs:
            Training epochs.
        lr:
            Adam learning rate.
        batch_size:
            Mini-batch size.
        """
        X_t = self._preprocess(X)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                criterion(self(xb), yb).backward()
                optimizer.step()

        return self

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self._config,
                "class": self.__class__.__name__,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BenchModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        return model

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

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.models.base import BaseModel

log = logging.getLogger(__name__)


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

    # Default batch size used during inference.  Training uses its own
    # batch_size argument passed to fit().  Inference must also be batched:
    # a full forward pass on N=17 617 samples through InceptionTime1D
    # creates intermediate activations of shape (N, nb_filters×4, T=800)
    # — up to 3.6 GB per module at nb_filters=16 — causing OOM if the
    # entire array is pushed through at once.
    _INFER_BATCH: int = 256

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _run_batched(self, X: np.ndarray, fn) -> np.ndarray:
        """Run *fn(batch_tensor) → tensor* over X in mini-batches.

        Preprocesses X once (CPU), then streams batches of size
        ``_INFER_BATCH`` to the model device.  Concatenates results on CPU.
        Avoids peak-memory spikes from single large forward passes.
        """
        self.eval()
        t = self._preprocess(X)           # stays on CPU until batch loop
        loader = DataLoader(
            TensorDataset(t),
            batch_size=self._INFER_BATCH,
            shuffle=False,
        )
        parts: list[torch.Tensor] = []
        for (xb,) in loader:
            parts.append(fn(xb.to(self._device)).cpu())
        return torch.cat(parts, dim=0).numpy()

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
        return self._run_batched(
            X, lambda xb: self.projector(self.backbone(xb))
        )

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(N, 2)`` with values in ``[0, 1]`` summing to 1 per row.
        """
        return self._run_batched(
            X, lambda xb: torch.softmax(self(xb), dim=-1)
        )

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
        patience: int = 10,
        val_fraction: float = 0.15,
        min_delta: float = 1e-4,
        **kwargs: Any,
    ) -> "BenchModel":
        """Train with mini-batch cross-entropy, Adam, and early stopping on val-AUC.

        Parameters
        ----------
        X:
            Input array (numpy).
        y:
            Integer class labels, shape ``(N,)``.
        epochs:
            Maximum training epochs.
        lr:
            Adam learning rate.
        batch_size:
            Mini-batch size.
        patience:
            Stop if val-AUC has not improved by more than ``min_delta`` for
            this many consecutive epochs.  Set to ``epochs`` to disable.
        val_fraction:
            Fraction of source data held out for early-stopping validation.
            Stratified split so class balance is preserved.
            Set to 0.0 to disable (train on all data, no early stopping).
        min_delta:
            Minimum AUC improvement to count as progress.
        """
        from sklearn.metrics import roc_auc_score

        # ── Stratified train/val split ────────────────────────────────────────
        use_es = val_fraction > 0.0 and patience < epochs
        if use_es:
            rng_split = np.random.default_rng(seed=42)
            classes, counts = np.unique(y, return_counts=True)
            train_idx, val_idx = [], []
            for cls, cnt in zip(classes, counts):
                idx = np.where(y == cls)[0]
                rng_split.shuffle(idx)
                n_val = max(1, int(cnt * val_fraction))
                val_idx.extend(idx[:n_val].tolist())
                train_idx.extend(idx[n_val:].tolist())
            train_idx = np.array(train_idx)
            val_idx   = np.array(val_idx)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            log.info(
                "    early stopping: train=%d  val=%d  patience=%d",
                len(X_tr), len(X_val), patience,
            )
        else:
            X_tr, y_tr = X, y

        X_t = self._preprocess(X_tr)
        y_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64))
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_auc    = -1.0
        best_state  = None
        no_improve  = 0

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0
            self.train()
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # ── Early stopping check ──────────────────────────────────────────
            if use_es:
                val_proba = self.predict_proba(X_val)[:, 1]
                val_auc   = float(roc_auc_score(y_val, val_proba))

                if val_auc > best_auc + min_delta:
                    best_auc   = val_auc
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    log.info(
                        "    epoch %d/%d  loss=%.4f  val_auc=%.4f  patience=%d/%d",
                        epoch + 1, epochs, avg_loss, val_auc, no_improve, patience,
                    )

                if no_improve >= patience:
                    log.info(
                        "    early stop at epoch %d/%d  best_val_auc=%.4f",
                        epoch + 1, epochs, best_auc,
                    )
                    break
            else:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    log.info(
                        "    epoch %d/%d  loss=%.4f",
                        epoch + 1, epochs, avg_loss,
                    )

        # ── Restore best weights ──────────────────────────────────────────────
        if use_es and best_state is not None:
            self.load_state_dict(
                {k: v.to(self._device) for k, v in best_state.items()}
            )

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

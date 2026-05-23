"""Deep CORAL — Correlation Alignment for Deep Domain Adaptation.

Reference
---------
Sun, B., & Saenko, K. (2016).
Deep CORAL: Correlation Alignment for Deep Domain Adaptation.
*ECCV 2016 Workshops*, 443-450.
https://arxiv.org/abs/1607.01719

Algorithm
---------
Fine-tune backbone + projector + head by jointly minimising::

    L = L_ce(X_s, y_s) + λ · L_CORAL(proj(backbone(X_s)), proj(backbone(X_t)))

with the CORAL loss defined as the squared Frobenius distance between the
feature covariance matrices of the two domains, normalised by ``4 d²``::

    L_CORAL = (1 / 4 d²) · ‖C_s − C_t‖²_F

where ``d`` is the projected-feature dimensionality (128 for every BenchModel
subclass) and ``C_s``, ``C_t`` are the per-batch sample covariances.

Relation to MK_MMD
------------------
Both Deep CORAL and MK-MMD are *moment-matching* deep DA methods operating on
the same projected-feature space (128-dim).  CORAL aligns second-order moments
(covariance) directly via a closed-form Frobenius distance; MK-MMD aligns the
entire feature distribution via the multi-kernel MMD with adaptive bandwidth.
The two methods anchor a deliberate axis of the benchmark — "the same alignment
principle applied at shallow (CORAL) vs deep (Deep CORAL / MK-MMD) levels".

Batch-size symmetry
-------------------
Unlike MK-MMD, CORAL does *not* require equal batch sizes — each domain's
covariance is computed independently.  Both DataLoaders still use
``drop_last=True`` for consistency with the MK-MMD scaffold and to avoid the
degenerate ``n=1`` covariance estimate (which divides by ``n - 1``).
"""

from __future__ import annotations

from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.adaptation.base import BaseAdaptation
from nstad_bench.models.common import BenchModel


def _covariance(X: torch.Tensor) -> torch.Tensor:
    """Sample covariance matrix of *X* with rows = samples.

    Returns ``Σ = (1 / (n − 1)) · (X − μ)ᵀ (X − μ)``, shape ``(d, d)``.
    Requires ``n ≥ 2``; callers must enforce this (DataLoaders with
    ``drop_last=True`` and ``batch_size ≥ 2`` suffice).
    """
    n = X.size(0)
    centered = X - X.mean(dim=0, keepdim=True)
    return centered.T @ centered / (n - 1)


def coral_loss(Xs: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss — squared Frobenius distance between covariances.

    ::

        L_CORAL = (1 / 4 d²) · ‖cov(Xs) − cov(Xt)‖²_F

    where ``d = Xs.size(1)`` is the feature dimensionality.  The ``4 d²``
    normaliser is Sun & Saenko's choice (Eq. 1 of the paper).
    """
    d = Xs.size(1)
    diff = _covariance(Xs) - _covariance(Xt)
    return diff.pow(2).sum() / (4.0 * d * d)


class DeepCORAL(BaseAdaptation):
    """Deep CORAL domain adaptation for time-series classification.

    Parameters
    ----------
    X_source :
        Source training samples (same shape convention as the model).
    y_source :
        Source integer class labels, shape ``(N,)``.
    n_epochs :
        Adaptation epochs (default 20).
    lr :
        Adam learning rate (default 1e-4).
    lambda_coral :
        CORAL loss weight λ (default 1.0).  Sun & Saenko sweep λ ∈ [0, 10]
        in their experiments; 1.0 is a reasonable midpoint that keeps the
        adaptation loss on the same order of magnitude as the CE term once
        the model has trained for a few epochs.
    batch_size :
        Mini-batch size.  Both source and target loaders use
        ``drop_last=True`` to keep ``n ≥ 2`` in every covariance estimate.
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        n_epochs: int = 20,
        lr: float = 1e-4,
        lambda_coral: float = 1.0,
        batch_size: int = 32,
    ) -> None:
        self.X_source     = np.asarray(X_source)
        self.y_source     = np.asarray(y_source, dtype=np.int64)
        self.n_epochs     = n_epochs
        self.lr           = lr
        self.lambda_coral = lambda_coral
        self.batch_size   = batch_size

    def _run(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:
        device = model._device

        Xs_t = model._preprocess(self.X_source).to(device)
        ys_t = torch.from_numpy(self.y_source).to(device)
        Xt_t = model._preprocess(X_target).to(device)

        src_loader = DataLoader(
            TensorDataset(Xs_t, ys_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        tgt_loader = DataLoader(
            TensorDataset(Xt_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.n_epochs):
            tgt_iter = cycle(tgt_loader)
            for xb_s, yb_s in src_loader:
                (xb_t,) = next(tgt_iter)

                feat_s  = model.projector(model.backbone(xb_s))   # (B, 128)
                ce_loss = criterion(model.head(feat_s), yb_s)

                feat_t  = model.projector(model.backbone(xb_t))   # (B, 128)
                coral   = coral_loss(feat_s, feat_t)

                loss = ce_loss + self.lambda_coral * coral
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

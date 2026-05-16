"""CoDATS — Convolutional Deep Adaptation Network for Time Series.

Reference
---------
Wilson, G., Doppa, J. R., & Cook, D. J. (2020).
Multi-source deep domain adaptation with weak supervision for time-series
sensor data.
*Proceedings of KDD 2020*, 1768–1778.
https://arxiv.org/abs/2005.10996

Reference implementation (floft/codats):
https://github.com/floft/codats

Algorithm
---------
Standard Domain-Adversarial Neural Network (DANN; Ganin & Lempitsky 2015)
adapted for time-series inputs.  A Gradient Reversal Layer (GRL) is placed
between the shared feature extractor (backbone + projector) and a domain
discriminator::

    X_s / X_t
      └─ backbone → projector ─┬─ head (task,   source only)
                                └─ GRL → disc (domain, both)

Training loss::

    L = L_ce(X_s, y_s) + λ · [L_disc(X_s, dom=1) + L_disc(X_t, dom=0)]

The GRL negates gradients flowing back into backbone + projector, so the
feature extractor learns domain-invariant representations while the
discriminator is simultaneously trained to classify domains correctly.

GRL alpha is linearly annealed 0 → alpha_max over adaptation epochs,
following the schedule in Ganin et al. 2016 (linearised).

Domain discriminator (applied to 128-dim projector output)::

    Linear(128, 256) → ReLU → Dropout(0.5) → Linear(256, 1)
"""

from __future__ import annotations

from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.adaptation.base import BaseAdaptation
from nstad_bench.models.common import BenchModel


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    During forward: identity.
    During backward: multiply gradient by −alpha.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return -ctx.alpha * grad, None


def _grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


# ---------------------------------------------------------------------------
# Domain discriminator
# ---------------------------------------------------------------------------

class _DomainDiscriminator(nn.Module):
    """Two-layer MLP domain classifier (source=1, target=0).

    Operates on 128-dim projected features (after GRL).
    Returns a scalar logit compatible with ``BCEWithLogitsLoss``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)   # (B,)


# ---------------------------------------------------------------------------
# Adaptation method
# ---------------------------------------------------------------------------

class CoDATS(BaseAdaptation):
    """DANN-based domain adaptation for time-series classifiers (CoDATS).

    Parameters
    ----------
    X_source :
        Source training samples.
    y_source :
        Source integer class labels, shape ``(N,)``.
    n_epochs :
        Adaptation epochs (default 20).
    lr :
        Adam learning rate for backbone + projector + head (default 1e-4).
    lr_disc :
        Adam learning rate for the domain discriminator.  Typically higher
        than the feature extractor so the discriminator stays well-trained
        (default 1e-3).
    lambda_domain :
        Domain loss weight λ (default 1.0).
    batch_size :
        Mini-batch size (default 32).
    alpha_max :
        Maximum GRL reversal coefficient; linearly annealed 0 → alpha_max
        over epochs (default 1.0).
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        n_epochs: int = 20,
        lr: float = 1e-4,
        lr_disc: float = 1e-3,
        lambda_domain: float = 1.0,
        batch_size: int = 32,
        alpha_max: float = 1.0,
    ) -> None:
        self.X_source      = np.asarray(X_source)
        self.y_source      = np.asarray(y_source, dtype=np.int64)
        self.n_epochs      = n_epochs
        self.lr            = lr
        self.lr_disc       = lr_disc
        self.lambda_domain = lambda_domain
        self.batch_size    = batch_size
        self.alpha_max     = alpha_max

    def _run(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:
        device = model._device

        Xs_t = model._preprocess(self.X_source).to(device)
        ys_t = torch.from_numpy(self.y_source).to(device)
        Xt_t = model._preprocess(X_target).to(device)

        src_loader = DataLoader(
            TensorDataset(Xs_t, ys_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        tgt_loader = DataLoader(
            TensorDataset(Xt_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        discriminator = _DomainDiscriminator().to(device)

        # Separate optimisers: feature extractor vs discriminator
        feat_opt = torch.optim.Adam(
            list(model.backbone.parameters())
            + list(model.projector.parameters())
            + list(model.head.parameters()),
            lr=self.lr,
        )
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=self.lr_disc)

        ce_loss  = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        for epoch in range(self.n_epochs):
            # Linear GRL alpha schedule: 0 → alpha_max
            alpha = self.alpha_max * epoch / max(self.n_epochs - 1, 1)

            model.train()
            discriminator.train()

            tgt_iter = cycle(tgt_loader)
            for xb_s, yb_s in src_loader:
                (xb_t,) = next(tgt_iter)
                B_s = xb_s.size(0)
                B_t = xb_t.size(0)

                # Feature extraction (shared backbone + projector)
                feat_s = model.projector(model.backbone(xb_s))   # (B_s, 128)
                feat_t = model.projector(model.backbone(xb_t))   # (B_t, 128)

                # Task loss — source classification only
                task_loss = ce_loss(model.head(feat_s), yb_s)

                # Domain loss — both domains via GRL
                # source=1, target=0
                d_s = torch.ones(B_s, device=device)
                d_t = torch.zeros(B_t, device=device)
                domain_loss = (
                    bce_loss(discriminator(_grad_reverse(feat_s, alpha)), d_s)
                    + bce_loss(discriminator(_grad_reverse(feat_t, alpha)), d_t)
                )

                loss = task_loss + self.lambda_domain * domain_loss
                feat_opt.zero_grad()
                disc_opt.zero_grad()
                loss.backward()
                feat_opt.step()
                disc_opt.step()

        return model

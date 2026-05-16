"""MK-MMD — Multi-Kernel Maximum Mean Discrepancy (Deep Adaptation Network).

Reference
---------
Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2015).
Learning transferable features with deep adaptation networks.
*Proceedings of ICML 2015*, 97–105.
https://arxiv.org/abs/1502.02791

Reference implementation (thuml/Transfer-Learning-Library):
https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dan.py
https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/kernels.py

Algorithm
---------
Fine-tune backbone + projector + head by jointly minimising::

    L = L_ce(X_s, y_s) + λ · MMD_k(proj(backbone(X_s)), proj(backbone(X_t)))

Bandwidth selection — adaptive (matching reference exactly)
-----------------------------------------------------------
The reference ``GaussianKernel`` estimates bandwidth **per batch** as::

    σ²_k = α_k · E_{i,j}[ ‖xᵢ − xⱼ‖² ]

where the expectation is over all pairs in the *concatenated* ``[X_s; X_t]``
batch (detached, no gradient).  Three multipliers ``α ∈ {0.5, 1.0, 2.0}``
give three kernels spanning half/unit/double the median-scale bandwidth.

Using fixed σ values (e.g. σ = 0.5 … 16) is wrong for high-dimensional
projected features: with d = 128, ``E[‖Δ‖²] ≈ 2d = 256`` so σ < 8 yields
``k ≈ exp(−256 / 2σ²) ≈ 0`` — a dead kernel that contributes no signal.

Batch-size symmetry
-------------------
The reference requires ``n_s = n_t`` (explicit docstring note; the
``(2n × 2n)`` index matrix assumes equal sizes).  Both DataLoaders use
``drop_last=True`` so every batch has exactly ``batch_size`` samples from
each domain.  The last partial batch is discarded rather than producing
asymmetric kernel matrices with mismatched denominators.
"""

from __future__ import annotations

from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.adaptation.base import BaseAdaptation
from nstad_bench.models.common import BenchModel

# α multipliers used by the reference GaussianKernel (track_running_stats=True)
_DEFAULT_ALPHAS: tuple[float, ...] = (0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Kernel utilities
# ---------------------------------------------------------------------------

def _pairwise_sq_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix D[i,j] = ‖X_i − Y_j‖²."""
    XX = X.pow(2).sum(1, keepdim=True)   # (n, 1)
    YY = Y.pow(2).sum(1, keepdim=True)   # (m, 1)
    return (XX + YY.T - 2.0 * (X @ Y.T)).clamp(min=0.0)


def _adaptive_sigmas(
    Xs: torch.Tensor,
    Xt: torch.Tensor,
    alphas: tuple[float, ...],
) -> list[float]:
    """Estimate per-kernel bandwidths from the current batch.

    Replicates ``GaussianKernel(track_running_stats=True)`` from the
    reference implementation::

        σ²_k = α_k · mean_{i,j}( ‖xᵢ − xⱼ‖² )

    The expectation is taken over *all pairs* in the concatenated
    ``[Xs; Xt]`` batch (detached — no gradient flows through σ).
    """
    with torch.no_grad():
        features = torch.cat([Xs, Xt], dim=0)            # (2n, d)
        mean_sq  = _pairwise_sq_distances(features, features).mean()
    return [float((alpha * mean_sq).sqrt()) for alpha in alphas]


def mk_mmd_loss(
    Xs: torch.Tensor,
    Xt: torch.Tensor,
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
) -> torch.Tensor:
    """Biased multi-kernel MMD² with adaptive bandwidths.

    Bandwidths are estimated once per call from the concatenated
    ``[Xs; Xt]`` features (no gradient through σ), matching the reference
    ``GaussianKernel(track_running_stats=True)`` behaviour.

    Requires ``Xs.size(0) == Xt.size(0)`` (equal batch sizes).

    Parameters
    ----------
    Xs : (n, d) source feature batch  — **must equal Xt.size(0)**.
    Xt : (n, d) target feature batch.
    alphas : bandwidth multipliers α_k; σ²_k = α_k · E[‖Δ‖²].
    """
    n = Xs.size(0)
    if Xt.size(0) != n:
        raise ValueError(
            f"mk_mmd_loss requires equal batch sizes, got n_s={n}, n_t={Xt.size(0)}. "
            "Use drop_last=True in both DataLoaders."
        )

    sigmas = _adaptive_sigmas(Xs, Xt, alphas)

    Dss = _pairwise_sq_distances(Xs, Xs)   # (n, n)
    Dtt = _pairwise_sq_distances(Xt, Xt)   # (n, n)
    Dst = _pairwise_sq_distances(Xs, Xt)   # (n, n)

    loss = Xs.new_zeros(())
    for sigma in sigmas:
        s2 = 2.0 * sigma ** 2
        Kss = torch.exp(-Dss / s2)
        Ktt = torch.exp(-Dtt / s2)
        Kst = torch.exp(-Dst / s2)
        loss = loss + Kss.mean() + Ktt.mean() - 2.0 * Kst.mean()
    return loss / len(alphas)


# ---------------------------------------------------------------------------
# Adaptation method
# ---------------------------------------------------------------------------

class MK_MMD(BaseAdaptation):
    """DAN / MK-MMD domain adaptation for time-series classification.

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
    lambda_mmd :
        MMD loss weight λ (default 1.0).
    batch_size :
        Mini-batch size.  Both source and target loaders use ``drop_last=True``
        to guarantee equal-size batches as required by the reference (default 32).
    alphas :
        Bandwidth multipliers α_k for adaptive σ estimation.  Matches the
        reference ``GaussianKernel`` default: ``(0.5, 1.0, 2.0)``.
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        n_epochs: int = 20,
        lr: float = 1e-4,
        lambda_mmd: float = 1.0,
        batch_size: int = 32,
        alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
    ) -> None:
        self.X_source   = np.asarray(X_source)
        self.y_source   = np.asarray(y_source, dtype=np.int64)
        self.n_epochs   = n_epochs
        self.lr         = lr
        self.lambda_mmd = lambda_mmd
        self.batch_size = batch_size
        self.alphas     = alphas

    def _run(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:
        device = model._device

        Xs_t = model._preprocess(self.X_source).to(device)
        ys_t = torch.from_numpy(self.y_source).to(device)
        Xt_t = model._preprocess(X_target).to(device)

        # drop_last=True guarantees n_s == n_t == batch_size every step
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
                mmd     = mk_mmd_loss(feat_s, feat_t, self.alphas)

                loss = ce_loss + self.lambda_mmd * mmd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

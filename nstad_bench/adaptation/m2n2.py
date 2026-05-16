"""M2N2 — Multi-scale Masked Neighbor-Noise test-time adaptation.

Reference
---------
Kim, J., Kim, J., & Choi, J. (2024).
M2N2: Multi-scale Masked Neighbor-Noise for Time-Series Anomaly Detection.
*Proceedings of AAAI 2024*.
https://ojs.aaai.org/index.php/AAAI/article/view/29237

Algorithm
---------
M2N2 is a **Test-Time Adaptation** (TTA) method: it adapts the model to the
target distribution at inference time using only unlabelled target samples,
with no access to source data.

Three self-supervised objectives are jointly optimised on target batches:

1. **Entropy minimisation** (reduce prediction uncertainty on target)::

       L_ent = − Σ_c p̂_c log p̂_c

2. **Neighbour-noise consistency** (predictions are invariant to small
   additive Gaussian noise, i.e. the *neighbourhood* of each sample)::

       L_cons = ‖p̂(x) − p̂(x + ε)‖²,  ε ~ N(0, σ²)

3. **Masked consistency** (predictions survive random time-step masking,
   corresponding to the *multi-scale masking* in the original paper)::

       L_mask = ‖p̂(x) − p̂(x ⊙ m)‖²,  m ~ Bernoulli(1 − r)

Parameter update strategy — BN/LN affine parameters only
----------------------------------------------------------
Following the Tent (Wang et al. 2021) principle, **only the affine parameters
(weight γ and bias β) of BatchNorm and LayerNorm layers** are updated.
All other weights — backbone convolutions/attention, projector Linear, head
Linear — are completely frozen.

Rationale:

* Normalization affine params are the minimal, lowest-risk set: they rescale
  and shift *already-computed* features without altering the feature space
  geometry, making collapse far less likely than full-model updates.
* Both BN (InceptionTime1D, ResNet18_2D) and LN (PatchTST) are covered
  uniformly by the same selection rule.
* During adaptation the norm layers run in **train mode** (batch statistics
  from target) while all other layers stay in **eval mode** (no dropout,
  stable feature extraction).

Supported norm types:  ``BatchNorm1d``, ``BatchNorm2d``, ``BatchNorm3d``,
``LayerNorm``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.adaptation.base import BaseAdaptation
from nstad_bench.models.common import BenchModel

# Normalization layer types whose affine params are updated during TTA
_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
)


def _collect_norm_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Module]]:
    """Return (list_of_affine_params, list_of_norm_modules) for all BN/LN layers."""
    params: list[nn.Parameter] = []
    modules: list[nn.Module] = []
    for module in model.modules():
        if isinstance(module, _NORM_TYPES):
            modules.append(module)
            if module.weight is not None:   # γ
                params.append(module.weight)
            if module.bias is not None:     # β
                params.append(module.bias)
    return params, modules


class M2N2(BaseAdaptation):
    """Multi-scale Masked Neighbor-Noise TTA — BN/LN affine params only.

    Parameters
    ----------
    n_steps :
        Total gradient update steps on target data (default 50).
    lr :
        Adam learning rate applied to BN/LN affine params (default 1e-4).
    noise_std :
        Standard deviation σ of neighbour-noise augmentation (default 0.1).
    mask_ratio :
        Fraction of elements zeroed per sample (default 0.2).
    lambda_ent :
        Entropy loss weight (default 1.0).
    lambda_cons :
        Noise-consistency loss weight (default 0.5).
    lambda_mask :
        Mask-consistency loss weight (default 0.5).
    batch_size :
        Mini-batch size (default 32).
    """

    def __init__(
        self,
        *,
        n_steps: int = 50,
        lr: float = 1e-4,
        noise_std: float = 0.1,
        mask_ratio: float = 0.2,
        lambda_ent: float = 1.0,
        lambda_cons: float = 0.5,
        lambda_mask: float = 0.5,
        batch_size: int = 32,
    ) -> None:
        self.n_steps     = n_steps
        self.lr          = lr
        self.noise_std   = noise_std
        self.mask_ratio  = mask_ratio
        self.lambda_ent  = lambda_ent
        self.lambda_cons = lambda_cons
        self.lambda_mask = lambda_mask
        self.batch_size  = batch_size

    # ---------------------------------------------------------------------- #
    # Augmentations                                                           #
    # ---------------------------------------------------------------------- #

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add i.i.d. Gaussian noise (neighbour perturbation)."""
        return x + self.noise_std * torch.randn_like(x)

    def _random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a random fraction of elements (multi-scale masking)."""
        keep = torch.bernoulli(torch.full_like(x, 1.0 - self.mask_ratio))
        return x * keep

    # ---------------------------------------------------------------------- #
    # Core adaptation                                                         #
    # ---------------------------------------------------------------------- #

    def _run(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:
        device = model._device

        # ── 1. Freeze every parameter in the model ────────────────────────
        for p in model.parameters():
            p.requires_grad_(False)

        # ── 2. Collect BN/LN affine params and re-enable their gradients ──
        norm_params, norm_modules = _collect_norm_params(model)
        if not norm_params:
            raise RuntimeError(
                "M2N2 requires BatchNorm or LayerNorm layers with affine=True. "
                f"{type(model).__name__} has none."
            )
        for p in norm_params:
            p.requires_grad_(True)

        # ── 3. Set mode: eval everywhere, train only for norm layers ──────
        #    eval  → dropout off, BN uses running stats for non-norm layers
        #    train → norm layers use target batch statistics
        model.eval()
        for mod in norm_modules:
            mod.train()

        # ── 4. Optimiser acts only on norm affine params ──────────────────
        optimizer = torch.optim.Adam(norm_params, lr=self.lr)

        Xt_t = model._preprocess(X_target).to(device)
        loader = DataLoader(
            TensorDataset(Xt_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        step = 0
        while step < self.n_steps:
            for (xb,) in loader:
                if step >= self.n_steps:
                    break

                # ── Clean forward ─────────────────────────────────────────
                feat_clean   = model.projector(model.backbone(xb))
                p_clean      = torch.softmax(model.head(feat_clean), dim=-1)

                # ── L_ent : entropy minimisation ─────────────────────────
                ent_loss = -(p_clean * torch.log(p_clean + 1e-8)).sum(-1).mean()

                # ── L_cons : neighbour-noise consistency ──────────────────
                feat_noisy = model.projector(model.backbone(self._add_noise(xb)))
                p_noisy    = torch.softmax(model.head(feat_noisy), dim=-1)
                cons_loss  = F.mse_loss(p_clean, p_noisy)

                # ── L_mask : masked consistency ───────────────────────────
                feat_masked = model.projector(model.backbone(self._random_mask(xb)))
                p_masked    = torch.softmax(model.head(feat_masked), dim=-1)
                mask_loss   = F.mse_loss(p_clean, p_masked)

                # ── Total loss ────────────────────────────────────────────
                loss = (
                    self.lambda_ent  * ent_loss
                    + self.lambda_cons * cons_loss
                    + self.lambda_mask * mask_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

        # ── 5. Restore full gradient flow (caller may retrain) ────────────
        for p in model.parameters():
            p.requires_grad_(True)

        return model

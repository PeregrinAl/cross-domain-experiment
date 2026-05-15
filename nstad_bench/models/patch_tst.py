"""PatchTST — Patch Time-Series Transformer for 1-D time series.

Reference
---------
Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
A time series is worth 64 words: Long-term forecasting with transformers.
*ICLR 2023*. https://arxiv.org/abs/2211.14730

Architecture
------------
The classification variant used here::

    Input (B, C, T)
    ├─ Patch extraction: T  → n_patches non-overlapping or overlapping windows
    │     patch_len = patch_len, stride = stride
    │     n_patches = ⌊(T - patch_len) / stride⌋ + 1
    ├─ Patch projection: (B, n_patches, C × patch_len) → (B, n_patches, d_model)
    ├─ Learnable positional embedding (1, n_patches, d_model)
    ├─ Dropout
    ├─ TransformerEncoder (n_layers × TransformerEncoderLayer)
    │     d_model, n_heads, dim_ff = 4 × d_model, dropout
    ├─ Mean over patch dim → (B, d_model)   ← backbone output
    └─ ClassHead(d_feat=d_model)

Input shape
-----------
``(N, T)``     univariate  — reshaped to (N, 1, T) internally.
``(N, C, T)``  multivariate — used as-is.

Parameters
----------
in_channels:
    Number of input channels *C* (1 for univariate).
seq_len:
    Expected time-series length *T*.  Must be consistent with the data.
patch_len:
    Length of each patch window (default 16).
stride:
    Step between consecutive patches (default 8, giving 50 % overlap).
d_model:
    Transformer embedding / hidden dimension (default 128).
n_heads:
    Number of self-attention heads (default 4; must divide d_model).
n_layers:
    Number of Transformer encoder layers (default 3).
dropout:
    Dropout rate applied after patch embedding and inside each Transformer
    layer (default 0.1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nstad_bench.models.common import BenchModel, _ClassHead


class _PatchBackbone(nn.Module):
    """Patch-based Transformer backbone."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        n_patches = (seq_len - patch_len) // stride + 1
        patch_dim = in_channels * patch_len  # flattened patch size

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN variant — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Initialise positional embedding from a small normal
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Extract overlapping patches from ``x: (B, C, T)``."""
        B, C, T = x.shape
        patches = []
        pos = 0
        while pos + self.patch_len <= T:
            patch = x[:, :, pos: pos + self.patch_len]  # (B, C, patch_len)
            patches.append(patch.reshape(B, -1))          # (B, C*patch_len)
            pos += self.stride
        # Stack → (B, n_patches, C*patch_len)
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(x)                          # (B, n, C*patch_len)
        n = patches.size(1)
        h = self.patch_proj(patches)                         # (B, n, d_model)
        h = self.dropout(h + self.pos_emb[:, :n])
        h = self.transformer(h)                              # (B, n, d_model)
        return h.mean(dim=1)                                 # (B, d_model)


class PatchTST(BenchModel):
    """PatchTST classifier for 1-D time series inputs.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for univariate, default).
    seq_len:
        Time-series length expected during ``fit`` and ``predict``.
    patch_len:
        Patch window size (default 16).
    stride:
        Patch step (default 8, 50 % overlap).
    d_model:
        Transformer hidden dimension (default 128).
    n_heads:
        Self-attention heads (default 4).
    n_layers:
        Transformer encoder depth (default 3).
    dropout:
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 512,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._config = dict(
            in_channels=in_channels,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.backbone = _PatchBackbone(
            in_channels, seq_len, patch_len, stride, d_model, n_heads, n_layers, dropout
        )
        self.projector = nn.Linear(d_model, 128)
        self.head = _ClassHead(128)

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Accept ``(N, T)`` or ``(N, C, T)``; return float32 tensor ``(N, C, T)``.

        Zero-padding
        ------------
        If the time dimension T is shorter than the model's ``seq_len``, the
        sequence is right-padded with zeros to exactly ``seq_len``.  This mirrors
        the behaviour of ``LogSTFT``, which pads short signals to ``n_fft``
        before computing the STFT, and keeps inference consistent regardless of
        input length.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:          # (N, T) → (N, 1, T)
            X = X[:, None, :]
        seq_len = self._config["seq_len"]
        T = X.shape[-1]
        if T < seq_len:
            pad_width = [(0, 0)] * (X.ndim - 1) + [(0, seq_len - T)]
            X = np.pad(X, pad_width)          # zero-pad along time axis
        return torch.from_numpy(X)

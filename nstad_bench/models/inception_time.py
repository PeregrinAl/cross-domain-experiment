"""InceptionTime1D — Inception-based model for univariate/multivariate time series.

Reference
---------
Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J.,
Webb, G. I., Idoumghar, L., Muller, P.-A., & Petitjean, F. (2020).
InceptionTime: Finding AlexNet for time series classification.
*Data Mining and Knowledge Discovery*, 34(6), 1936–1962.
https://doi.org/10.1007/s10618-020-00710-y

Architecture
------------
Three stacked InceptionModules with residual shortcuts, followed by global
average pooling and the shared benchmark classification head::

    Input (B, C, T)
    ├─ InceptionModule(C_in → nb_filters×4)  ← 4 parallel paths, each nb_filters wide
    ├─ InceptionModule(nb_filters×4 → nb_filters×4)
    ├─ InceptionModule(nb_filters×4 → nb_filters×4)
    ├─ Residual shortcut: Conv1d(C_in, nb_filters×4, 1) → BN → +
    ├─ GlobalAvgPool1d → (B, nb_filters×4)   ← backbone output
    └─ ClassHead(d_feat=nb_filters×4)

Each InceptionModule::

    Bottleneck Conv1d(C_in, bottleneck, 1)
    ├─ Conv1d(bottleneck, nb_filters, kernel_size=39, padding='same')
    ├─ Conv1d(bottleneck, nb_filters, kernel_size=19, padding='same')
    ├─ Conv1d(bottleneck, nb_filters, kernel_size=9,  padding='same')
    └─ MaxPool1d(3, stride=1, padding=1) → Conv1d(C_in, nb_filters, 1)
    → Concat(4 paths) → BN → ReLU   →  (B, nb_filters×4, T)

Input shape
-----------
``(N, T)``        univariate    — reshaped to (N, 1, T) internally.
``(N, C, T)``     multivariate  — used as-is.
``(N, embed_dim)`` CARLA-SSL embedding — reshaped to (N, 1, embed_dim).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nstad_bench.models.common import BenchModel, _ClassHead


class _InceptionModule(nn.Module):
    """One Inception block with four parallel convolutional paths."""

    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        bottleneck: int,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
    ) -> None:
        super().__init__()
        # Shared bottleneck before the three variable-kernel convolutions
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck, kernel_size=1, bias=False
        )
        self.conv_paths = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck, nb_filters, kernel_size=k, padding="same", bias=False
                )
                for k in kernel_sizes
            ]
        )
        # Max-pool path bypasses the bottleneck and acts on the raw input
        self.maxpool_path = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_out = self.bottleneck(x)
        paths = [conv(bottleneck_out) for conv in self.conv_paths]
        paths.append(self.maxpool_path(x))
        return self.relu(self.bn(torch.cat(paths, dim=1)))


class _InceptionBackbone(nn.Module):
    """Stack of InceptionModules with one residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        bottleneck: int,
        depth: int,
    ) -> None:
        super().__init__()
        n_paths = 4  # 3 conv paths + 1 maxpool path
        d_out = nb_filters * n_paths  # channel width after each module

        self.modules_: nn.ModuleList = nn.ModuleList()
        for i in range(depth):
            c_in = in_channels if i == 0 else d_out
            self.modules_.append(
                _InceptionModule(c_in, nb_filters, bottleneck)
            )

        # Residual shortcut: map input channels → d_out
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, d_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_out),
        )
        self.shortcut_relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, d_out, T) → (B, d_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = x
        for mod in self.modules_:
            h = mod(h)
        h = self.shortcut_relu(h + residual)
        return self.gap(h).squeeze(-1)  # (B, d_out)


class InceptionTime1D(BenchModel):
    """InceptionTime classifier for 1-D time series inputs.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for univariate / CARLA embeddings).
    nb_filters:
        Filters per inception path; total backbone output dim = nb_filters × 4.
    bottleneck:
        Bottleneck width inside each InceptionModule.
    depth:
        Number of stacked InceptionModules (default 3, as in the paper).
    """

    def __init__(
        self,
        in_channels: int = 1,
        nb_filters: int = 32,
        bottleneck: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self._config = dict(
            in_channels=in_channels,
            nb_filters=nb_filters,
            bottleneck=bottleneck,
            depth=depth,
        )
        d_backbone = nb_filters * 4  # 4 parallel paths per InceptionModule
        self.backbone = _InceptionBackbone(in_channels, nb_filters, bottleneck, depth)
        self.projector = nn.Linear(d_backbone, 128)
        self.head = _ClassHead(128)

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Accept ``(N, T)`` or ``(N, C, T)``; return float32 tensor ``(N, C, T)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:          # (N, T) → (N, 1, T)
            X = X[:, None, :]
        return torch.from_numpy(X)

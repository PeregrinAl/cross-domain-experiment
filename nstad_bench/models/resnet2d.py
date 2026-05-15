"""ResNet18_2D — ResNet-18 backbone for 2-D spectral representations.

This is a from-scratch implementation of the ResNet-18 architecture
(He et al., 2016) adapted for single-channel 2-D inputs (STFT spectrograms
or CWT scalograms).  It intentionally avoids the ``torchvision`` dependency
so the package remains self-contained.

Reference
---------
He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep residual learning for image recognition.
*CVPR 2016*. https://arxiv.org/abs/1512.03385

Architecture
------------
Standard ResNet-18 with one modification: the first convolution accepts
``in_channels`` input channels (default 1) instead of 3::

    Conv2d(in_channels, 64, 7, stride=2, padding=3) → BN → ReLU → MaxPool
    Layer 1: BasicBlock(64→64)  × 2
    Layer 2: BasicBlock(64→128) × 2   (stride 2 at first block)
    Layer 3: BasicBlock(128→256) × 2  (stride 2)
    Layer 4: BasicBlock(256→512) × 2  (stride 2)
    AdaptiveAvgPool2d(1,1) → flatten → (B, 512)   ← backbone output
    ClassHead(d_feat=512)

Input shape
-----------
``(N, F, T)``    single-channel 2-D — reshaped to (N, 1, F, T) internally.
``(N, 1, F, T)`` already batched with explicit channel dim.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nstad_bench.models.common import BenchModel, _ClassHead


class _BasicBlock(nn.Module):
    """ResNet-18/34 basic residual block (two 3×3 convolutions)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample: nn.Sequential | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = [_BasicBlock(in_ch, out_ch, stride)]
    for _ in range(1, blocks):
        layers.append(_BasicBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class _ResNet18Backbone(nn.Module):
    """ResNet-18 up to (and including) the global average pool."""

    D_FEAT: int = 512

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = _make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = _make_layer(256, 512, blocks=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.gap(x).flatten(1)   # (B, 512)


class ResNet18_2D(BenchModel):
    """ResNet-18 classifier for 2-D spectral representations (STFT / CWT).

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for single-channel spectrograms).
    """

    D_FEAT: int = 512

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self._config = dict(in_channels=in_channels)
        self.backbone = _ResNet18Backbone(in_channels)
        self.projector = nn.Linear(self.D_FEAT, 128)
        self.head = _ClassHead(128)

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Accept ``(N, F, T)`` or ``(N, 1, F, T)``; return float32 tensor."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:          # (N, F, T) → (N, 1, F, T)
            X = X[:, None, :, :]
        return torch.from_numpy(X)

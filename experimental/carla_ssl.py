"""CARLA: Self-supervised Contrastive Representation Learning for Time Series
Anomaly Detection.

Reference
---------
Darban, Z. Z., Webb, G. I., Pan, S., Aggarwal, C. C., & Rashidinejad, M.
"CARLA: Self-supervised contrastive representation learning for time series
anomaly detection." *Pattern Recognition*, 2025.
https://doi.org/10.1016/j.patcog.2024.110Prevention
GitHub: https://github.com/zamanzadeh/CARLA

Architecture (this implementation)
------------------------------------
*  Encoder: three-block 1-D ResNet (kernel sizes 8-5-3, 64 channels) followed
   by global average-pooling and a linear projection to ``embed_dim``.
*  Training signal: triplet loss with
       - anchor  = source window  x_a
       - positive = adjacent source window  x_p  (nearest neighbour in time)
       - negative = x_a with one of five synthetic anomaly types injected
*  Five synthetic anomaly types (see ``_inject_anomaly``):
       1. point_global      – spike to global extreme
       2. point_contextual  – spike relative to local statistics
       3. subseq_seasonal   – random phase-shift of a sinusoidal patch
       4. subseq_trend      – ramp injection into a sub-window
       5. subseq_shapelet   – replacement with a shuffled sub-window
*  Representation at inference: encoder output (``embed_dim``-dimensional).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nstad_bench.representations.base import BaseRepresentation


# ---------------------------------------------------------------------------
# Synthetic anomaly injection
# ---------------------------------------------------------------------------

def _inject_anomaly(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of *x* (shape ``(T,)``) with one synthetic anomaly."""
    x = x.copy()
    T = len(x)
    anomaly_type = rng.choice(
        ["point_global", "point_contextual", "subseq_seasonal",
         "subseq_trend", "subseq_shapelet"]
    )

    if anomaly_type == "point_global":
        idx = int(rng.integers(0, T))
        global_std = float(x.std()) or 1.0
        sign = rng.choice([-1, 1])
        x[idx] = x.mean() + sign * 4.0 * global_std

    elif anomaly_type == "point_contextual":
        win = min(20, T)
        idx = int(rng.integers(win // 2, T - win // 2))
        local = x[idx - win // 2: idx + win // 2]
        local_std = float(local.std()) or 1.0
        sign = rng.choice([-1, 1])
        x[idx] = local.mean() + sign * 3.0 * local_std

    elif anomaly_type == "subseq_seasonal":
        length = max(8, int(T * rng.uniform(0.1, 0.3)))
        start = int(rng.integers(0, T - length))
        freq = rng.uniform(0.05, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = float(x[start: start + length].std()) or 1.0
        patch = amplitude * np.sin(
            2 * np.pi * freq * np.arange(length) + phase
        )
        x[start: start + length] += patch

    elif anomaly_type == "subseq_trend":
        length = max(8, int(T * rng.uniform(0.1, 0.3)))
        start = int(rng.integers(0, T - length))
        slope = float(x.std()) * rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.0)
        ramp = slope * np.linspace(0, 1, length)
        x[start: start + length] += ramp

    else:  # subseq_shapelet
        length = max(8, int(T * rng.uniform(0.1, 0.3)))
        start = int(rng.integers(0, T - length))
        patch = x[start: start + length].copy()
        rng.shuffle(patch)
        x[start: start + length] = patch

    return x


# ---------------------------------------------------------------------------
# 1-D ResNet encoder
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        # padding='same' preserves temporal length for any kernel size (stride=1 only).
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding="same", bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.proj = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.proj(x))


class _ResNetEncoder(nn.Module):
    """Three-block 1-D ResNet that maps (B, 1, T) → (B, embed_dim)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _ResBlock(1, 64, 7),   # odd kernel → no zero-copy warning with padding='same'
            _ResBlock(64, 64, 5),
            _ResBlock(64, 64, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        h = self.blocks(x)          # (B, 64, T)
        h = self.pool(h).squeeze(-1)  # (B, 64)
        return self.head(h)           # (B, embed_dim)


# ---------------------------------------------------------------------------
# Triplet dataset
# ---------------------------------------------------------------------------

class _TripletDataset(torch.utils.data.Dataset):
    """Yield (anchor, positive, negative) triplets from source windows."""

    def __init__(self, X: np.ndarray, seed: int) -> None:
        self.X = X.astype(np.float32)
        self.N = len(X)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        anchor = self.X[idx]

        # Positive: adjacent window (wraps around)
        pos_idx = (idx + 1) % self.N
        positive = self.X[pos_idx]

        # Negative: anchor with synthetic anomaly
        negative = _inject_anomaly(anchor, self.rng).astype(np.float32)

        # Shape: (1, T) for Conv1d
        return (
            torch.from_numpy(anchor[None]),
            torch.from_numpy(positive[None]),
            torch.from_numpy(negative[None]),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CARLA_SSL(BaseRepresentation):
    """CARLA contrastive SSL representation.

    Parameters
    ----------
    embed_dim:
        Dimensionality of the learned representation vector.
    margin:
        Margin for the triplet loss.
    n_epochs:
        Number of training epochs over the source pool.
    batch_size:
        Mini-batch size during training.
    lr:
        Adam learning rate.
    seed:
        Global seed for reproducibility of training *and* transform.
    device:
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    """

    is_1d: bool = True
    is_2d: bool = False

    def __init__(
        self,
        embed_dim: int = 64,
        margin: float = 1.0,
        n_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.margin = margin
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_: _ResNetEncoder | None = None
        self.output_shape: tuple[int] | None = None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        return F.relu(d_pos - d_neg + self.margin).mean()

    # ------------------------------------------------------------------
    # BaseRepresentation interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, **kwargs: Any) -> "CARLA_SSL":
        """Train the ResNet encoder on source windows *X* ``(N, T)``."""
        self._seed_everything()

        self.encoder_ = _ResNetEncoder(self.embed_dim).to(self.device)
        optimiser = torch.optim.Adam(self.encoder_.parameters(), lr=self.lr)

        dataset = _TripletDataset(X, seed=self.seed)
        _g = torch.Generator()
        _g.manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=_g, num_workers=0,
        )

        self.encoder_.train()
        for _ in range(self.n_epochs):
            for anchor, positive, negative in loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                z_a = self.encoder_(anchor)
                z_p = self.encoder_(positive)
                z_n = self.encoder_(negative)

                loss = self._triplet_loss(z_a, z_p, z_n)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        self.output_shape = (self.embed_dim,)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode *X* ``(N, T)`` → embeddings ``(N, embed_dim)``."""
        if self.encoder_ is None:
            raise RuntimeError("Call fit() before transform().")

        # Deterministic inference: fix seed so dropout (if any) is stable.
        torch.manual_seed(self.seed)

        tensor = torch.from_numpy(X.astype(np.float32))[:, None, :].to(self.device)
        self.encoder_.eval()
        with torch.no_grad():
            embeddings = self.encoder_(tensor)
        return embeddings.cpu().numpy()

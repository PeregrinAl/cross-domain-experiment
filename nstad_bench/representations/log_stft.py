from __future__ import annotations

import numpy as np
from scipy.signal import stft

from nstad_bench.representations.base import BaseRepresentation


class LogSTFT(BaseRepresentation):
    """Log-magnitude linear STFT representation.

    Applies a Short-Time Fourier Transform with a Hann window and returns
    ``log1p(|STFT|)``.  The output is a real-valued 2-D spectrogram
    ``(F, T')`` per sample, where ``F = n_fft // 2 + 1`` frequency bins and
    ``T'`` is the number of frames determined by *hop_length*.

    Input must be univariate: ``(N, T)``.
    """

    is_1d: bool = False
    is_2d: bool = True

    def __init__(self, n_fft: int = 256, hop_length: int = 64) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _stft_sample(self, x: np.ndarray) -> np.ndarray:
        """Return log-magnitude spectrogram for a single signal *x* of shape (T,)."""
        # Zero-pad to at least n_fft so that noverlap < nperseg always holds.
        if len(x) < self.n_fft:
            x = np.pad(x, (0, self.n_fft - len(x)))
        _, _, Zxx = stft(
            x,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window="hann",
            boundary="zeros",
            padded=True,
        )
        return np.log1p(np.abs(Zxx))  # (F, T')

    # ------------------------------------------------------------------
    # BaseRepresentation interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, **kwargs) -> "LogSTFT":
        """Determine output shape from the first sample in *X* ``(N, T)``."""
        spec = self._stft_sample(X[0])
        self.output_shape = spec.shape  # (F, T')
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return log-magnitude spectrograms ``(N, F, T')``."""
        if self.output_shape is None:
            raise RuntimeError("Call fit() before transform().")
        return np.stack([self._stft_sample(x) for x in X], axis=0)

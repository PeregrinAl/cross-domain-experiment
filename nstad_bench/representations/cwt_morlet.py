from __future__ import annotations

import numpy as np
import pywt

from nstad_bench.representations.base import BaseRepresentation

# Scales are spaced geometrically so that low and high frequency ranges
# are sampled equally on a log scale – a common choice for Morlet CWT.
_DEFAULT_SCALES = np.geomspace(1, 128, num=64)


class CWT_Morlet(BaseRepresentation):
    """Continuous Wavelet Transform (CWT) with a Morlet wavelet.

    Produces a real-valued scalogram ``|CWT|`` of shape ``(n_scales, T)``
    per sample.  The 64 scales are distributed geometrically between 1 and
    128 by default, giving good frequency resolution at both high and low
    frequencies.

    Input must be univariate: ``(N, T)``.
    """

    is_1d: bool = False
    is_2d: bool = True

    def __init__(
        self,
        n_scales: int = 64,
        scales: np.ndarray | None = None,
        sampling_period: float = 1.0,
    ) -> None:
        self.n_scales = n_scales
        self.scales: np.ndarray = (
            scales if scales is not None else np.geomspace(1, 128, num=n_scales)
        )
        self.sampling_period = sampling_period
        self.output_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _cwt_sample(self, x: np.ndarray) -> np.ndarray:
        """Return scalogram for a single signal *x* of shape (T,)."""
        coeffs, _ = pywt.cwt(
            x,
            scales=self.scales,
            wavelet="morl",
            sampling_period=self.sampling_period,
        )
        return np.abs(coeffs)  # (n_scales, T)

    # ------------------------------------------------------------------
    # BaseRepresentation interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, **kwargs) -> "CWT_Morlet":
        """Determine output shape from the first sample in *X* ``(N, T)``."""
        scalogram = self._cwt_sample(X[0])
        self.output_shape = scalogram.shape  # (n_scales, T)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return scalograms ``(N, n_scales, T)``."""
        if self.output_shape is None:
            raise RuntimeError("Call fit() before transform().")
        return np.stack([self._cwt_sample(x) for x in X], axis=0)

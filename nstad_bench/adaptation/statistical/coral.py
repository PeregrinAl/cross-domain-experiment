"""CORAL — CORrelation ALignment.

Reference
---------
Sun, B., Feng, J., & Saenko, K. (2016). Return of frustratingly easy
domain adaptation.  *AAAI 2016*.  https://arxiv.org/abs/1511.05547

Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for
deep domain adaptation.  *ECCV Workshop 2016*.
https://arxiv.org/abs/1607.01719

Reference implementation
------------------------
``jindongwang/transferlearning`` ─ traditional/CORAL/CORAL.py

Algorithm (classical, feature-level)
------------------------------------
Given source features ``F_s`` (n_s × D) and target features ``F_t``
(n_t × D), CORAL finds a linear transform that aligns the second-order
statistics of the source to those of the target::

    C_s = cov(F_s) + λI
    C_t = cov(F_t) + λI
    A   = C_s^{-1/2} · C_t^{1/2}

The aligned source ``F_s_aligned = F_s · A`` then has covariance
≈ ``C_t``.  The underlying classifier is re-fitted on this aligned
source; at inference, *raw* target features are fed unchanged because
they already match the training distribution by construction.

The Tikhonov regulariser ``λ`` guards against ill-conditioning when
``D > n``.  The optional ``align_mean`` flag additionally shifts the
source mean to the target mean (AAAI'16 formulation).

Runner-compatibility note
-------------------------
The runner uses the adapted model both for target inference *and* for
threshold-selection on a held-out source-validation split.  CORAL's
classifier is trained on target-distributed (= aligned-source) features,
so target predictions are clean while source-val predictions are
slightly out-of-distribution.  This is the standard CORAL trade-off and
matches the reference implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power

from nstad_bench.adaptation.statistical.base import BaseStatAdaptation
from nstad_bench.models.statistical.base import StatModel


def _real_psd_fractional_power(C: np.ndarray, p: float) -> np.ndarray:
    """``C^p`` for symmetric PSD ``C``.

    ``scipy.linalg.fractional_matrix_power`` returns complex values for
    PSD matrices when eigenvalues are tiny-negative due to floating-point
    noise.  We take the real part — safe because the true matrix is real
    and the imaginary component is round-off.
    """
    return np.asarray(fractional_matrix_power(C, p)).real


class CORAL(BaseStatAdaptation):
    """CORrelation ALignment (classical, feature-level).

    Parameters
    ----------
    X_source, y_source :
        Labelled source data (stored at construction; passed by the
        runner mirroring the neural ``MK_MMD`` / ``CoDATS`` contract).
    lambda_reg :
        Tikhonov regulariser added to the source and target covariance
        before computing matrix square roots.
    align_mean :
        If True, also shift the aligned source by ``μ_t − μ_s`` so that
        ``mean(F_s_aligned) ≈ μ_t`` (AAAI'16 formulation).
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        lambda_reg: float = 1e-3,
        align_mean: bool = True,
    ) -> None:
        self.X_s = X_source
        self.y_s = y_source
        self.lambda_reg = float(lambda_reg)
        self.align_mean = bool(align_mean)

    def _run(self, model: StatModel, X_target: np.ndarray) -> StatModel:
        F_s = np.asarray(model.get_features(self.X_s), dtype=np.float64)
        F_t = np.asarray(model.get_features(X_target),  dtype=np.float64)

        D = F_s.shape[1]
        eye = self.lambda_reg * np.eye(D)

        # Centre before computing covariance so the mean-shift is
        # explicit and the matrix-power step operates on pure cov.
        mu_s = F_s.mean(axis=0)
        mu_t = F_t.mean(axis=0)
        F_s_c = F_s - mu_s
        F_t_c = F_t - mu_t

        # rowvar=False → each row is an observation.  Use ddof=1 to
        # match np.cov's default unbiased estimator.
        cov_s = np.cov(F_s_c, rowvar=False) + eye
        cov_t = np.cov(F_t_c, rowvar=False) + eye

        # Guard against the degenerate D=1 case (np.cov returns a scalar).
        if cov_s.ndim == 0:
            cov_s = cov_s.reshape(1, 1)
            cov_t = cov_t.reshape(1, 1)

        A_coral = _real_psd_fractional_power(cov_s, -0.5) @ \
                  _real_psd_fractional_power(cov_t,  0.5)

        F_s_aligned = F_s_c @ A_coral
        if self.align_mean:
            F_s_aligned = F_s_aligned + mu_t
        else:
            # Restore source mean so the classifier still sees an
            # un-centred input (sklearn handles offsets fine, but staying
            # close to the original is gentler for the inference step
            # below where target features are raw).
            F_s_aligned = F_s_aligned + mu_s

        # Refit the estimator on aligned source; _adapt_transform stays
        # ``None`` so that at inference raw features pass straight to
        # the estimator — matching the transferlearning reference.
        model._estimator.fit(
            F_s_aligned, np.asarray(self.y_s, dtype=np.int64)
        )
        return model

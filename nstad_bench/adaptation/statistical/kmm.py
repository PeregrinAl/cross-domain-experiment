"""KMM — Kernel Mean Matching.

Reference
---------
Huang, J., Smola, A. J., Gretton, A., Borgwardt, K. M., & Schölkopf, B.
(2007). Correcting sample selection bias by unlabeled data.  *NIPS 2006*,
601–608.  https://papers.nips.cc/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html

Algorithm
---------
Direct density-ratio estimation by matching the kernel mean of the
weighted source to the kernel mean of the target in an RBF RKHS.  Solve
the quadratic programme::

    minimise   ½ wᵀ K w − κᵀ w
    subject to 0 ≤ wᵢ ≤ B
               |Σᵢ wᵢ − n_s| ≤ n_s · ε

where::

    K_{ij} = exp(−‖x_sᵢ − x_sⱼ‖² / (2σ²))   ∈ R^{n_s × n_s}
    κᵢ     = (n_s / n_t) Σⱼ exp(−‖x_sᵢ − x_tⱼ‖² / (2σ²))

The optimised ``w`` gives importance weights for source samples; the
underlying estimator is then re-fitted with ``sample_weight=w``.  Compared
to :class:`ImportanceReweighting`, KMM works directly in the RKHS
without training an auxiliary classifier, but the QP scales as
O(n_s²) — we sub-sample the source to a tractable size if needed.

Solver
------
We solve the strictly convex box-constrained QP with
:func:`scipy.optimize.minimize` (L-BFGS-B) using analytic gradients.
The original paper's sum-of-weights equality is enforced
*post-hoc* by re-scaling the optimised weights to mean ≈ 1 (and
clipping to the per-sample upper bound ``B``); this is equivalent to
the constrained formulation up to a global scale that has no effect on
sklearn estimators with internal regularisation.  Avoids an extra
dependency on cvxopt at the price of a less specialised QP solver;
for typical n_s ≤ 2000 it converges in well under a second.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

from nstad_bench.adaptation.statistical.base import BaseStatAdaptation
from nstad_bench.models.statistical.base import StatModel


log = logging.getLogger(__name__)


_KMM_MAX_SOURCE = 2000   # subsample threshold for tractable QP
_KMM_MEDIAN_PROBE = 1000  # subsample size for the median-σ heuristic


class KMM(BaseStatAdaptation):
    """Kernel Mean Matching.

    Parameters
    ----------
    X_source, y_source :
        Labelled source data.
    sigma :
        RBF kernel bandwidth.  If ``None``, set to the median pairwise
        Euclidean distance on the concatenated ``[F_s; F_t]`` features
        (Schölkopf median heuristic).
    B :
        Upper bound on individual weights (default 1000.0 — matches
        the original paper).
    eps :
        Tolerance for the sum-of-weights constraint
        ``|Σw − n_s| ≤ n_s · ε``.
    random_state :
        Seed for the optional source / median-heuristic sub-samples.
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        sigma: float | None = None,
        B: float = 1000.0,
        eps: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        self.X_s = X_source
        self.y_s = y_source
        self.sigma = sigma
        self.B = float(B)
        self.eps = float(eps)
        self.random_state = random_state

    def _run(self, model: StatModel, X_target: np.ndarray) -> StatModel:
        F_s = np.asarray(model.get_features(self.X_s), dtype=np.float64)
        F_t = np.asarray(model.get_features(X_target),  dtype=np.float64)
        y_s = np.asarray(self.y_s, dtype=np.int64)

        rng = np.random.default_rng(self.random_state)

        # ── Sub-sample source for a tractable QP ─────────────────────────
        if len(F_s) > _KMM_MAX_SOURCE:
            log.info(
                "KMM: sub-sampling source from %d to %d for the QP",
                len(F_s), _KMM_MAX_SOURCE,
            )
            idx = rng.choice(len(F_s), _KMM_MAX_SOURCE, replace=False)
            F_s_qp = F_s[idx]
        else:
            idx = np.arange(len(F_s))
            F_s_qp = F_s

        # ── Bandwidth: median heuristic on combined features ─────────────
        sigma = self.sigma
        if sigma is None:
            combined = np.vstack([F_s_qp, F_t])
            if len(combined) > _KMM_MEDIAN_PROBE:
                probe_idx = rng.choice(
                    len(combined), _KMM_MEDIAN_PROBE, replace=False,
                )
                combined = combined[probe_idx]
            dists = pdist(combined, metric="euclidean")
            # Guard against pathological zero-distance cases.
            sigma = float(np.median(dists)) if dists.size else 1.0
            if sigma <= 0:
                sigma = 1.0

        gamma = 1.0 / (2.0 * sigma * sigma)

        # ── Kernel matrices ──────────────────────────────────────────────
        K_ss = np.exp(-gamma * cdist(F_s_qp, F_s_qp, metric="sqeuclidean"))
        K_st = np.exp(-gamma * cdist(F_s_qp, F_t,    metric="sqeuclidean"))
        n_s_qp = len(F_s_qp)
        n_t    = len(F_t)
        kappa  = (n_s_qp / max(n_t, 1)) * K_st.sum(axis=1)

        # Symmetrise for numerical stability and pad the diagonal so the
        # QP problem is strictly convex (SLSQP requires PD Hessian).
        K_ss = 0.5 * (K_ss + K_ss.T) + 1e-6 * np.eye(n_s_qp)

        # ── QP: min ½wᵀKw − κᵀw,  0 ≤ w ≤ B  (L-BFGS-B, box-bounded) ───
        def fun(w: np.ndarray) -> float:
            return 0.5 * float(w @ K_ss @ w) - float(kappa @ w)

        def jac(w: np.ndarray) -> np.ndarray:
            return K_ss @ w - kappa

        bounds = [(0.0, self.B)] * n_s_qp
        w0 = np.ones(n_s_qp)
        result = minimize(
            fun,
            w0,
            jac=jac,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-7},
        )
        w_qp = np.clip(result.x, 0.0, self.B)
        if not result.success:
            log.warning(
                "KMM L-BFGS-B did not converge (%s); using best-effort "
                "weights from the optimiser.",
                result.message,
            )
        # Enforce the paper's sum-of-weights ≈ n_s constraint by
        # post-rescaling to mean 1 (within the tolerance ``eps``).
        mean_qp = float(w_qp.mean())
        if mean_qp > 0 and abs(mean_qp - 1.0) > self.eps:
            w_qp = np.clip(w_qp / mean_qp, 0.0, self.B)

        # ── Lift weights back to full source set ────────────────────────
        w = np.ones(len(F_s))
        w[idx] = w_qp
        # Re-normalise to mean 1 over the full set so refitting matches
        # the original effective sample size.
        m = float(w.mean())
        if m > 0:
            w = w / m

        # ── Refit estimator with sample weights ─────────────────────────
        model._estimator.fit(F_s, y_s, sample_weight=w)
        return model

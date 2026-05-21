"""Subspace Alignment (SA).

Reference
---------
Fernando, B., Habrard, A., Sebban, M., & Tuytelaars, T. (2013).
Unsupervised visual domain adaptation using subspace alignment.
*ICCV 2013*.  https://arxiv.org/abs/1409.5241

Algorithm
---------
Let ``P_s ∈ R^{d × D}`` and ``P_t ∈ R^{d × D}`` be the top-``d`` PCA
components of the source and target feature matrices (with their PCA
means ``μ_s`` and ``μ_t``).  Define the alignment matrix::

    M = P_s P_t^⊤    ∈ R^{d × d}

Source data are projected into the source subspace and then aligned to
the target subspace::

    F_s_aligned = (F_s − μ_s) P_s^⊤ M

Target data are projected into the target subspace at inference::

    F_t_proj    = (F_t − μ_t) P_t^⊤

Both representations live in ``R^d`` and are approximately co-distributed
(``P_s^⊤ M ≈ P_t^⊤`` when subspaces are similar).  The underlying
classifier is re-fitted on ``F_s_aligned``; at inference we apply
``P_t^⊤`` to *any* incoming features via :class:`AffineFeatureTransform`,
which delivers the canonical target-side SA projection.

Runner-compatibility note
-------------------------
Because the runner calls the adapted model on both target (``X_t``) and
a held-out source-validation split (``X_val``), and both have to go
through the same ``predict_proba`` path, we apply ``P_t^⊤`` to every
test input.  Target predictions are exact SA (paper protocol); source-val
predictions get the source data projected into the target subspace,
which differs slightly from the training distribution
(``(F_s − μ_s) P_s^⊤ M``).  Threshold-selection on source-val is
therefore best-effort, mirroring the same trade-off as :class:`CORAL`.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from nstad_bench.adaptation.statistical.base import BaseStatAdaptation
from nstad_bench.models.statistical.base import (
    AffineFeatureTransform,
    StatModel,
)


class SubspaceAlignment(BaseStatAdaptation):
    """Subspace Alignment (SA).

    Parameters
    ----------
    X_source, y_source :
        Labelled source data.
    n_components :
        Target subspace dimensionality ``d``.  Clipped to a feasible
        value inside ``_run`` based on the data shape (PCA requires
        ``d ≤ min(n_samples − 1, D)``).
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        n_components: int = 50,
    ) -> None:
        self.X_s = X_source
        self.y_s = y_source
        self.n_components = int(n_components)

    def _run(self, model: StatModel, X_target: np.ndarray) -> StatModel:
        F_s = np.asarray(model.get_features(self.X_s), dtype=np.float64)
        F_t = np.asarray(model.get_features(X_target),  dtype=np.float64)

        D = F_s.shape[1]
        # PCA requires d ≤ min(n_samples, D).  We further subtract one
        # from n_samples for the centred-data degree of freedom.
        d = min(
            self.n_components,
            D,
            max(1, F_s.shape[0] - 1),
            max(1, F_t.shape[0] - 1),
        )

        pca_s = PCA(n_components=d).fit(F_s)
        pca_t = PCA(n_components=d).fit(F_t)

        P_s = pca_s.components_   # (d, D)
        P_t = pca_t.components_   # (d, D)
        mu_s = pca_s.mean_        # (D,)
        mu_t = pca_t.mean_        # (D,)

        M = P_s @ P_t.T           # (d, d) — source basis ↔ target basis

        # ── Train on source aligned to target subspace ───────────────────
        F_s_aligned = (F_s - mu_s) @ P_s.T @ M   # (n_s, d)
        model._estimator.fit(
            F_s_aligned, np.asarray(self.y_s, dtype=np.int64)
        )

        # ── Install target-side projection for inference ─────────────────
        # AffineFeatureTransform applies ``(X - shift) @ matrix``, so
        # shift=μ_t and matrix=P_t^⊤ implements ``(F - μ_t) P_t^⊤``.
        model._adapt_transform = AffineFeatureTransform(
            matrix=P_t.T.copy(),
            shift=mu_t.copy(),
        )
        return model

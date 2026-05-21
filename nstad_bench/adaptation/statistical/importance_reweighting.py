"""Importance Reweighting — covariate-shift correction by density ratio.

Reference
---------
Shimodaira, H. (2000). Improving predictive inference under covariate
shift by weighting the log-likelihood function.  *Journal of Statistical
Planning and Inference*, 90(2), 227–244.

Sugiyama, M., Suzuki, T., & Kanamori, T. (2012). *Density Ratio
Estimation in Machine Learning.*  Cambridge University Press.

Algorithm
---------
Estimate ``w(x) = p_t(x) / p_s(x)`` and refit the underlying estimator
on the source data with ``sample_weight=w``.

Density-ratio estimation strategy
---------------------------------
We support a single, robust estimator out of the box: the
**probabilistic-classifier** approach, which trains a binary domain
classifier ``g(x) = P(domain = target | x)`` on the combined
``[F_s; F_t]`` set with labels ``(0, 1)``.  By Bayes' rule::

    w(x_s) = (n_s / n_t) · g(x_s) / (1 − g(x_s))

A :class:`sklearn.preprocessing.StandardScaler` is fitted on the
combined features before the classifier to stabilise convergence on
wide raw representations.

KLIEP and uLSIF are exposed as parameter values for forward
compatibility but currently fall back to the classifier estimator with
a logged warning — implementing them would require an additional
hyperparameter (kernel bandwidth) and a separate QP / closed-form
solver.  The classifier method covers the same covariate-shift use
cases and is the default in modern DA libraries.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nstad_bench.adaptation.statistical.base import BaseStatAdaptation
from nstad_bench.models.statistical.base import StatModel


log = logging.getLogger(__name__)


DREstimator = Literal["classifier", "kliep", "ulsif"]


class ImportanceReweighting(BaseStatAdaptation):
    """Importance reweighting for covariate shift.

    Parameters
    ----------
    X_source, y_source :
        Labelled source data.
    estimator :
        Density-ratio estimator.  Only ``classifier`` is implemented;
        ``kliep`` and ``ulsif`` are accepted (for HP-search
        compatibility) but currently delegate to ``classifier``.
    clip :
        Weights are clipped to ``[1/clip, clip]`` to limit the influence
        of a handful of target-rare source samples.
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        *,
        estimator: DREstimator = "classifier",
        clip: float = 10.0,
    ) -> None:
        if estimator not in {"classifier", "kliep", "ulsif"}:
            raise ValueError(
                f"Unknown estimator {estimator!r}; expected one of "
                "{'classifier', 'kliep', 'ulsif'}"
            )
        self.X_s = X_source
        self.y_s = y_source
        self.estimator = estimator
        self.clip = float(clip)

    def _run(self, model: StatModel, X_target: np.ndarray) -> StatModel:
        F_s = np.asarray(model.get_features(self.X_s), dtype=np.float64)
        F_t = np.asarray(model.get_features(X_target),  dtype=np.float64)

        if self.estimator != "classifier":
            log.warning(
                "ImportanceReweighting(estimator=%r) is not implemented; "
                "falling back to the 'classifier' density-ratio estimator.",
                self.estimator,
            )

        w = self._classifier_weights(F_s, F_t)
        # Clip first, then scale to mean 1 so the effective sample size
        # is comparable to the source size (otherwise sklearn estimators
        # with internal L2 regularisation see a different effective
        # regularisation strength).
        w = np.clip(w, 1.0 / self.clip, self.clip)
        mean_w = float(w.mean())
        if mean_w > 0:
            w = w / mean_w

        # Refit estimator on raw source features with sample weights.
        # `_adapt_transform` stays untouched — IR only reweights samples,
        # the feature space is unchanged.
        model._estimator.fit(
            F_s,
            np.asarray(self.y_s, dtype=np.int64),
            sample_weight=w,
        )
        return model

    # ------------------------------------------------------------------ #
    # Probabilistic-classifier density ratio                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classifier_weights(
        F_s: np.ndarray,
        F_t: np.ndarray,
    ) -> np.ndarray:
        """Estimate ``w(x_s) = (n_s/n_t) · g(x_s)/(1−g(x_s))``."""
        n_s, n_t = len(F_s), len(F_t)
        X = np.vstack([F_s, F_t])
        d = np.concatenate([np.zeros(n_s), np.ones(n_t)]).astype(np.int64)

        scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
        X_scaled = scaler.transform(X)

        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_scaled, d)

        # P(target | x_s)
        p_target = clf.predict_proba(scaler.transform(F_s))[:, 1]
        eps = 1e-8
        p_target = np.clip(p_target, eps, 1.0 - eps)

        return (n_s / max(n_t, 1)) * p_target / (1.0 - p_target)

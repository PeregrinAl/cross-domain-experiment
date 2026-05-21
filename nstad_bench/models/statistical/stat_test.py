"""Statistical-test based classifier.

Maps each input window to a low-dimensional summary statistic and
classifies that summary with a logistic regression.  Three modes:

  - ``moments`` : per-sample k-th moment (1: mean, 2: variance, 3: skew,
                  4: kurtosis).  The bearing-fault analysis in
                  ``experimental/README.md`` motivates kurtosis as a
                  near-perfect impulsivity detector.
  - ``chi2``    : per-sample normalised histogram across fixed bin
                  edges (n_bins features per sample).  The classifier
                  then learns weights on histogram bins — a poor-man's
                  χ²-based density model.
  - ``ks``      : per-sample Kolmogorov–Smirnov statistic against
                  per-class reference samples pooled from the training
                  set (one feature per class).

All three modes flatten the input to a 1-D series per sample first,
so they are agnostic to representation shape ``(N, T)`` or ``(N, F, T)``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.stats import kurtosis, ks_2samp, skew
from sklearn.linear_model import LogisticRegression

from nstad_bench.models.statistical.base import StatModel


TestKind = Literal["moments", "chi2", "ks"]


_KS_REF_CAP = 5000   # max pooled reference points per class for KS speed


class StatTestClassifier(StatModel):
    """Per-sample statistical-test classifier.

    Parameters
    ----------
    test :
        Which feature extractor to use (``moments`` | ``chi2`` | ``ks``).
    moment_order :
        Order of the moment when ``test='moments'``.
        1=mean, 2=variance, 3=skewness, 4=kurtosis.
    n_bins :
        Histogram bin count when ``test='chi2'``.
    random_state :
        RNG seed for the KS-reference subsample.

    Notes
    -----
    State set during ``fit`` (in addition to the estimator):

      - ``_bin_edges`` for ``chi2`` — fixed bin edges derived from the
        source training set.  Reused at inference for both source-val
        and target so histograms are comparable across domains.
      - ``_ref_samples`` for ``ks`` — pooled reference samples per
        class (subsampled to ``_KS_REF_CAP`` for tractable computation).
    """

    def __init__(
        self,
        test: TestKind = "moments",
        moment_order: int = 4,
        n_bins: int = 32,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        if test not in {"moments", "chi2", "ks"}:
            raise ValueError(
                f"Unknown test {test!r}; expected one of "
                "{'moments', 'chi2', 'ks'}"
            )
        if test == "moments" and moment_order not in {1, 2, 3, 4}:
            raise ValueError(
                f"moment_order must be 1..4, got {moment_order}"
            )

        self._config = {
            "test": test,
            "moment_order": moment_order,
            "n_bins": n_bins,
            "random_state": random_state,
            **kwargs,
        }
        self._test: TestKind = test
        self._moment_order: int = moment_order
        self._n_bins: int = n_bins
        self._random_state: int | None = random_state
        self._estimator = LogisticRegression(max_iter=1000)
        self._adapt_transform = None
        self._bin_edges: np.ndarray | None = None
        self._ref_samples: list[np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)

    def _extract(self, X: np.ndarray) -> np.ndarray:
        Xf = self._flatten(X)
        if self._test == "moments":
            return self._extract_moments(Xf)
        if self._test == "chi2":
            return self._extract_chi2(Xf)
        # ks
        return self._extract_ks(Xf)

    def _extract_moments(self, Xf: np.ndarray) -> np.ndarray:
        if self._moment_order == 1:
            vals = Xf.mean(axis=1)
        elif self._moment_order == 2:
            vals = Xf.var(axis=1)
        elif self._moment_order == 3:
            vals = skew(Xf, axis=1, bias=False)
        else:  # 4
            vals = kurtosis(Xf, axis=1, fisher=True, bias=False)
        return np.nan_to_num(vals, nan=0.0).reshape(-1, 1)

    def _extract_chi2(self, Xf: np.ndarray) -> np.ndarray:
        if self._bin_edges is None:
            # Defensive fallback (used only if predict_proba is called
            # before fit, e.g. in tests that probe _extract directly).
            edges = np.linspace(Xf.min(), Xf.max(), self._n_bins + 1)
        else:
            edges = self._bin_edges
        # Vectorised histogram per row (loop is unavoidable for
        # variable-width samples but the inner call is BLAS-fast).
        hists = np.empty((len(Xf), self._n_bins), dtype=np.float64)
        for i, row in enumerate(Xf):
            hists[i] = np.histogram(row, bins=edges)[0]
        sums = hists.sum(axis=1, keepdims=True)
        return hists / np.where(sums > 0, sums, 1.0)

    def _extract_ks(self, Xf: np.ndarray) -> np.ndarray:
        if self._ref_samples is None:
            raise RuntimeError(
                "StatTestClassifier(test='ks') needs reference distributions; "
                "call fit() before predict_proba()."
            )
        feats = np.empty((len(Xf), len(self._ref_samples)), dtype=np.float64)
        for i, sample in enumerate(Xf):
            for j, ref in enumerate(self._ref_samples):
                feats[i, j] = ks_2samp(sample, ref).statistic
        return feats

    # ------------------------------------------------------------------ #
    # fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "StatTestClassifier":
        Xf = self._flatten(X)
        y_arr = np.asarray(y, dtype=np.int64)

        # Set up test-specific state from the (X_source, y_source) pair.
        if self._test == "chi2":
            self._bin_edges = np.linspace(
                Xf.min(), Xf.max(), self._n_bins + 1
            )
        elif self._test == "ks":
            rng = np.random.default_rng(self._random_state)
            self._ref_samples = []
            for cls in np.unique(y_arr):
                pooled = Xf[y_arr == cls].ravel()
                if len(pooled) > _KS_REF_CAP:
                    pooled = rng.choice(pooled, _KS_REF_CAP, replace=False)
                self._ref_samples.append(pooled)

        # Now extract and fit the underlying logistic regression.
        # Use the base-class fit so neural training keys are dropped and
        # ``sample_weight`` survives intact for adaptation methods.
        return super().fit(X, y_arr, **kwargs)

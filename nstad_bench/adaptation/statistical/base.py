"""Abstract base class for *statistical* domain-adaptation methods.

Mirror of :class:`nstad_bench.adaptation.base.BaseAdaptation` but typed
against :class:`nstad_bench.models.statistical.base.StatModel` instead of
the neural :class:`BenchModel`.

Two complementary flavours of statistical adaptation are supported by
this base class:

A. **Feature-space alignment** (CORAL, Subspace Alignment, KMM)
   The method computes a *transform* ``T : R^D → R^D`` on the feature
   space such that ``T(get_features(X_t))`` matches the statistics of
   ``get_features(X_s)``.  The estimator inside the model is then re-fit
   (or applied unchanged) on the aligned features.

B. **Sample reweighting** (importance reweighting / KMM weights)
   The method computes per-source-sample weights
   ``w(x_s) ≈ p_t(x_s) / p_s(x_s)``; the source estimator is re-fit on
   the same (X_s, y_s) but with ``sample_weight=w``.

Both flavours fit the same single-method contract::

    adapted = method.adapt(pretrained_model, X_target)

The original model is never modified — ``adapt`` always works on a deep
copy internally.  Methods are free to refit the wrapped estimator, swap
the feature pipeline, or both.

Subclasses needing access to the labelled source data should accept
``(X_source, y_source)`` in ``__init__`` (the runner already does this
for neural ``MK_MMD`` / ``CoDATS`` via the ``_build_adapt`` switch).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from nstad_bench.models.statistical.base import StatModel


class BaseStatAdaptation(ABC):
    """Common interface for all statistical domain-adaptation methods.

    Subclasses implement ``_run(model, X_target)`` which receives an
    already deep-copied :class:`StatModel` and returns the adapted model.
    The public ``adapt`` method handles the copy so callers never need to
    worry about in-place mutation.
    """

    def adapt(self, model: StatModel, X_target: np.ndarray) -> StatModel:
        """Return an adapted deep-copy of *model* using unlabelled *X_target*.

        The original *model* is **never modified in-place**.

        Parameters
        ----------
        model :
            Pre-trained source :class:`StatModel`.
        X_target :
            Unlabelled target samples.  Shape follows the model's
            ``_extract`` contract (raw or representation-space input —
            same as what was passed to ``fit``).
        """
        adapted = copy.deepcopy(model)
        return self._run(adapted, X_target)

    @abstractmethod
    def _run(self, model: StatModel, X_target: np.ndarray) -> StatModel:
        """Adaptation logic.  *model* is a fresh deepcopy — modify freely."""
        ...

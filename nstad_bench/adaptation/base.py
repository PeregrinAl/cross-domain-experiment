"""Abstract base class for model-level domain adaptation methods.

Every adaptation method in the benchmark shares one contract:

    adapted = method.adapt(pretrained_model, X_target)

The original model is **never modified** — ``adapt`` always works on a
deep copy internally.  ``X_target`` is unlabelled target data; no target
labels are required by any method.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from nstad_bench.models.common import BenchModel


class BaseAdaptation(ABC):
    """Common interface for all domain-adaptation / TTA methods.

    Subclasses implement ``_run(model, X_target)`` which receives an
    already deep-copied model and returns the adapted model.  The public
    ``adapt`` method handles the copy so callers never need to worry about
    in-place mutation.
    """

    def adapt(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:
        """Return an adapted deep-copy of *model* using unlabelled *X_target*.

        The original *model* is **never modified in-place**.

        Parameters
        ----------
        model :
            Pre-trained source model (any ``BenchModel`` subclass).
        X_target :
            Unlabelled target samples.  Shape follows the model's
            ``_preprocess`` contract (e.g. ``(N, T)`` for 1-D models).

        Returns
        -------
        BenchModel
            A new model instance adapted to the target distribution.
        """
        adapted = copy.deepcopy(model)
        return self._run(adapted, X_target)

    @abstractmethod
    def _run(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:
        """Adaptation logic.  *model* is a fresh deepcopy — modify freely."""
        ...

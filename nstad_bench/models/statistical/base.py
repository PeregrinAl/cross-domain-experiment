"""Shared interface for statistical (non-neural) benchmark models.

Every statistical model in the benchmark exposes the same public surface as
:class:`nstad_bench.models.common.BenchModel`, but is implemented in pure
NumPy / scikit-learn — no PyTorch, no GPU, no mini-batch training loop.

Public protocol
---------------
  - ``fit(X, y, **kwargs)``        — train on (X, y); returns ``self``
  - ``predict_proba(X) -> (N, 2)`` — class probabilities
  - ``predict(X) -> (N,)``         — argmax of ``predict_proba``
  - ``get_features(X) -> (N, D)``  — feature vector used for adaptation
                                      methods that operate in feature space
                                      (e.g. CORAL, Subspace Alignment, KMM)
  - ``save(path)`` / ``load(path)``

Pipeline
--------
Statistical models share a three-stage pipeline::

    _extract(X)               → (N, D₀)    raw per-sample features
    _adapt_transform(F)       → (N, D₁)    optional feature transform set
                                            by an adaptation method (CORAL,
                                            SubspaceAlignment, …) — identity
                                            on a freshly trained model
    estimator(features)       → (N, 2)    sklearn-style probabilities

Subclasses must implement ``_extract`` and set ``self._estimator`` and
``self._config``.  The ``_adapt_transform`` hook is managed by adaptation
methods and lives in this base class.

Compatibility with the runner
-----------------------------
The two-stage runner only assumes the following from a model object::

    model.fit(X_s, y_s, **train_kwargs)
    model.predict_proba(X)[:, 1]

so any subclass of :class:`StatModel` is plug-compatible with both
:mod:`nstad_bench.experiments.runner` and
:mod:`nstad_bench.experiments.runner_stat`.  Neural training keys
(``epochs`` / ``lr`` / ``batch_size``) passed via ``**kwargs`` are
silently ignored by :meth:`StatModel.fit`.
"""

from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from nstad_bench.models.base import BaseModel

log = logging.getLogger(__name__)


# Neural training keys that the runner forwards from YAML.  Statistical
# models ignore them — fit() drops these before passing **kwargs onwards.
_NEURAL_TRAIN_KEYS = frozenset({
    "epochs", "lr", "batch_size",
    "patience", "val_fraction", "min_delta",
})


@dataclass
class AffineFeatureTransform:
    """Pickle-friendly affine transform on feature vectors.

    Applied as ``(X - shift) @ matrix``.  Used by adaptation methods that
    introduce a feature-space transform (CORAL, SubspaceAlignment): the
    transform is attached to the model via ``StatModel._adapt_transform``
    after deep-copy, so it survives both deep-copy and pickle round-trips.

    A ``shift`` of zero degenerates to a linear projection; a square
    ``matrix`` of identity degenerates to a pure centering operation.
    """

    matrix: np.ndarray   # shape (D_in, D_out)
    shift: np.ndarray    # shape (D_in,) — subtracted before projection

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.shift) @ self.matrix


class StatModel(BaseModel):
    """Abstract base for statistical (non-neural) benchmark models.

    Subclasses must:
      1. Build and assign ``self._estimator`` — any object with
         ``fit(X, y, [sample_weight=...])`` and
         ``predict_proba(X) -> (N, 2)`` (sklearn estimators satisfy this).
      2. Implement ``_extract(X: np.ndarray) -> np.ndarray`` returning a
         feature matrix of shape ``(N, D)``.
      3. Set ``self._config: dict[str, Any]`` with constructor kwargs
         (for ``save`` / ``load`` round-trips).
    """

    _estimator: Any
    _config: dict[str, Any]
    # Optional feature-space hook installed by adaptation methods.
    # When non-None, ``get_features``/``predict_proba`` apply it to the
    # output of ``_extract`` before passing to the estimator.
    _adapt_transform: Callable[[np.ndarray], np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # Hooks subclasses must implement                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _extract(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample feature matrix of shape ``(N, D)``."""
        ...

    # ------------------------------------------------------------------ #
    # Public numpy API                                                     #
    # ------------------------------------------------------------------ #

    def get_features(self, X: np.ndarray) -> np.ndarray:
        """Return features as seen by the estimator.

        Applies the optional adaptation-time feature transform on top of
        the model-specific ``_extract``.  Adaptation methods like CORAL
        and SubspaceAlignment can therefore install a feature transform
        on a deep-copied model without touching subclass code.
        """
        feats = self._extract(X)
        if self._adapt_transform is not None:
            feats = self._adapt_transform(feats)
        return feats

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax-style class probabilities, shape ``(N, 2)``."""
        feats = self.get_features(X)
        proba = np.asarray(self._estimator.predict_proba(feats))
        # Guarantee shape (N, 2) for the binary benchmark protocol.
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices (argmax of ``predict_proba``)."""
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "StatModel":
        """Fit the estimator on extracted features.

        Neural training keys (``epochs`` / ``lr`` / ``batch_size`` / …)
        passed via ``**kwargs`` are silently dropped so the runner can
        forward its YAML training block unchanged.  Remaining kwargs
        (e.g. ``sample_weight``) are forwarded to the sklearn estimator.
        """
        feats = self._extract(X)
        sklearn_kwargs = {
            k: v for k, v in kwargs.items() if k not in _NEURAL_TRAIN_KEYS
        }
        self._estimator.fit(feats, np.asarray(y, dtype=np.int64), **sklearn_kwargs)
        return self

    # ------------------------------------------------------------------ #
    # Device shim (runner compatibility — statistical models are CPU-only) #
    # ------------------------------------------------------------------ #

    def to(self, device: Any) -> "StatModel":  # noqa: ARG002
        """No-op for parity with ``BenchModel.to(device)``."""
        return self

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #
    #
    # We pickle the whole instance so subclass-specific state survives
    # automatically (e.g. StatTestClassifier's ``_bin_edges`` for χ²,
    # ``_ref_samples`` for KS, or any state an adaptation method
    # attached after deep-copy).  ``cls.load`` then validates that the
    # loaded object is actually an instance of *cls* to avoid silent
    # class confusion across model types.

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "StatModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle at {path} contains a {type(obj).__name__}, "
                f"but {cls.__name__}.load() was called."
            )
        return obj

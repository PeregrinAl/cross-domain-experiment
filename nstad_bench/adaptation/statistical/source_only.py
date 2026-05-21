"""SourceOnly (statistical branch) — no-op baseline.

Parallel to :class:`nstad_bench.adaptation.source_only.SourceOnly` but
typed for :class:`StatModel`.  Returns a deep copy of the source model
without any modification.  Every benchmark run should include
``SourceOnly`` as the reference point: any adaptation that scores
*worse* on the target than ``SourceOnly`` indicates negative transfer.
"""

from __future__ import annotations

import numpy as np

from nstad_bench.adaptation.statistical.base import BaseStatAdaptation
from nstad_bench.models.statistical.base import StatModel


class SourceOnly(BaseStatAdaptation):
    """Baseline: return a deep copy of the source model unchanged."""

    def _run(self, model: StatModel, X_target: np.ndarray) -> StatModel:  # noqa: ARG002
        return model  # deepcopy already performed by BaseStatAdaptation.adapt

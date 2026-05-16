"""SourceOnly — no-op baseline.

Returns a deep copy of the source model without any modification.
Every benchmark evaluation should include SourceOnly as the reference point:
any adaptation method that scores *worse* than SourceOnly on the target
domain indicates negative transfer.
"""

from __future__ import annotations

import numpy as np

from nstad_bench.adaptation.base import BaseAdaptation
from nstad_bench.models.common import BenchModel


class SourceOnly(BaseAdaptation):
    """Baseline: return a deep copy of the source model unchanged.

    No target data is used.  The method exists solely to give a fair
    lower-bound reference in the benchmark table.
    """

    def _run(self, model: BenchModel, X_target: np.ndarray) -> BenchModel:  # noqa: ARG002
        return model  # deepcopy already performed by BaseAdaptation.adapt

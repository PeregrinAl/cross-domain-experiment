"""Statistical (non-neural) domain-adaptation methods.

Sibling subpackage to the top-level ``nstad_bench.adaptation`` neural
methods.  Every method shares the
:class:`nstad_bench.adaptation.statistical.base.BaseStatAdaptation`
interface — same single-method ``adapt(model, X_target)`` contract as
the neural ``BaseAdaptation`` but typed against
:class:`nstad_bench.models.statistical.base.StatModel`.

Available stubs (to be implemented)
-----------------------------------
  - :class:`SourceOnly`            — baseline (no adaptation)
  - :class:`CORAL`                 — correlation alignment
  - :class:`SubspaceAlignment`     — PCA-subspace alignment (SA)
  - :class:`ImportanceReweighting` — covariate-shift density-ratio reweighting
  - :class:`KMM`                   — Kernel Mean Matching (RKHS reweighting)
"""

from nstad_bench.adaptation.statistical.base import BaseStatAdaptation
from nstad_bench.adaptation.statistical.coral import CORAL
from nstad_bench.adaptation.statistical.importance_reweighting import (
    ImportanceReweighting,
)
from nstad_bench.adaptation.statistical.kmm import KMM
from nstad_bench.adaptation.statistical.source_only import SourceOnly
from nstad_bench.adaptation.statistical.subspace_alignment import SubspaceAlignment

__all__ = [
    "BaseStatAdaptation",
    "SourceOnly",
    "CORAL",
    "SubspaceAlignment",
    "ImportanceReweighting",
    "KMM",
]

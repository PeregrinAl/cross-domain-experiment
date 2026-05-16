"""Domain adaptation and test-time adaptation methods.

All methods share the interface::

    adapted_model = method.adapt(pretrained_model, X_target)

The original model is never modified in-place.
"""

from nstad_bench.adaptation.base import BaseAdaptation
from nstad_bench.adaptation.source_only import SourceOnly
from nstad_bench.adaptation.mk_mmd import MK_MMD
from nstad_bench.adaptation.codats import CoDATS
from nstad_bench.adaptation.m2n2 import M2N2

__all__ = ["BaseAdaptation", "SourceOnly", "MK_MMD", "CoDATS", "M2N2"]

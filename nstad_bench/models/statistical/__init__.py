"""Statistical (non-neural) benchmark models.

Sibling subpackage to the top-level ``nstad_bench.models`` neural models.
Every class shares the :class:`StatModel` protocol — same public surface
as the neural :class:`BenchModel` (fit, predict, predict_proba,
get_features, save, load) but implemented in pure NumPy / scikit-learn.

Available stubs (to be implemented)
-----------------------------------
  - :class:`LogReg`              — logistic regression
  - :class:`RandomForest`        — bagged decision trees
  - :class:`GBM`                 — gradient-boosted trees
  - :class:`SVM`                 — kernel SVM with probability output
  - :class:`StatTestClassifier`  — KS / χ² / moment-based decision rule
"""

from nstad_bench.models.statistical.base import StatModel
from nstad_bench.models.statistical.gbm import GBM
from nstad_bench.models.statistical.logreg import LogReg
from nstad_bench.models.statistical.random_forest import RandomForest
from nstad_bench.models.statistical.stat_test import StatTestClassifier
from nstad_bench.models.statistical.svm import SVM

__all__ = [
    "StatModel",
    "LogReg",
    "RandomForest",
    "GBM",
    "SVM",
    "StatTestClassifier",
]

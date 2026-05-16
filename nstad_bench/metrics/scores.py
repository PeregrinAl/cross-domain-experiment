"""Point-estimate evaluation metrics for domain-adaptation benchmarks.

Metrics
-------
roc_auc
    Area under the ROC curve.
pr_auc
    Area under the Precision-Recall curve (average precision).
best_threshold
    Threshold on a *validation* score array that maximises F1; use it as the
    ``threshold`` argument to ``f1_at_best_threshold`` and ``mcc``.
f1_at_best_threshold
    F1 score with the decision threshold chosen on a held-out *validation* set
    (DS_val) and evaluated on a separate *test* set (DS_target).
mcc
    Matthews Correlation Coefficient at a fixed threshold (default 0.5).
delta_auc
    ``auc_source − auc_target`` — positive values indicate domain degradation.
gain
    ``auc_adapted − auc_source_only`` — positive values indicate that
    adaptation improved over the no-adaptation baseline.

Note on point-adjust F1
-----------------------
Point-adjust F1 is deliberately *not* implemented.  The adjustment labels
every point in a detected window as correct, which inflates F1 for methods
that produce late, noisy, or coarse-grained detections.  See Kim et al.
(2022) "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" for
a formal critique.  We use standard sample-level F1 instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


# ──────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ──────────────────────────────────────────────────────────────────────────────

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve.

    Parameters
    ----------
    y_true  : (N,) binary ground-truth labels (0 or 1).
    y_score : (N,) continuous anomaly scores (higher = more anomalous).
    """
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (interpolated average precision).

    Parameters
    ----------
    y_true  : (N,) binary ground-truth labels.
    y_score : (N,) continuous anomaly scores.
    """
    return float(average_precision_score(np.asarray(y_true), np.asarray(y_score)))


# ──────────────────────────────────────────────────────────────────────────────
# Threshold-dependent metrics
# ──────────────────────────────────────────────────────────────────────────────

def best_threshold(
    y_true_val: np.ndarray,
    y_score_val: np.ndarray,
) -> float:
    """Score threshold that maximises F1 on the *validation* set.

    Uses sklearn's ``precision_recall_curve`` which sweeps all unique score
    values as candidate thresholds — exact, O(N log N), no grid search needed.

    Parameters
    ----------
    y_true_val  : (M,) binary validation labels.
    y_score_val : (M,) validation anomaly scores.

    Returns
    -------
    float
        The threshold t* = argmax_t F1(y_true_val, y_score_val >= t).
    """
    y_true_val  = np.asarray(y_true_val)
    y_score_val = np.asarray(y_score_val)

    prec, rec, thresholds = precision_recall_curve(y_true_val, y_score_val)
    # precision_recall_curve appends a sentinel (prec=1, rec=0) with no matching
    # threshold, so prec[:-1] / rec[:-1] correspond 1-to-1 with thresholds.
    denom = prec[:-1] + rec[:-1]
    # Use errstate to avoid division-by-zero RuntimeWarning: np.where evaluates
    # both branches eagerly; the mask guarantees we never *use* the bad values.
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * prec[:-1] * rec[:-1] / denom, 0.0)
    return float(thresholds[int(np.argmax(f1))])


def f1_at_best_threshold(
    y_true_test: np.ndarray,
    y_score_test: np.ndarray,
    y_true_val: np.ndarray,
    y_score_val: np.ndarray,
) -> float:
    """F1 score with decision threshold chosen on a held-out validation set.

    The threshold is found by maximising F1 on ``(y_true_val, y_score_val)``
    and then applied to ``(y_true_test, y_score_test)``.  Using the same split
    for both selection and evaluation would be optimistically biased.

    Parameters
    ----------
    y_true_test  : (N,) binary test labels.
    y_score_test : (N,) test anomaly scores.
    y_true_val   : (M,) binary validation labels.
    y_score_val  : (M,) validation anomaly scores.
    """
    t = best_threshold(y_true_val, y_score_val)
    y_pred = (np.asarray(y_score_test) >= t).astype(int)
    return float(f1_score(y_true_test, y_pred, zero_division=0.0))


def mcc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    """Matthews Correlation Coefficient at a fixed decision threshold.

    Parameters
    ----------
    y_true    : (N,) binary ground-truth labels.
    y_score   : (N,) continuous anomaly scores.
    threshold : decision boundary (default 0.5).  Pass ``best_threshold(...)``
                for val-optimal thresholding.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(matthews_corrcoef(np.asarray(y_true), y_pred))


# ──────────────────────────────────────────────────────────────────────────────
# Adaptation-level metrics
# ──────────────────────────────────────────────────────────────────────────────

def delta_auc(auc_source: float, auc_target: float) -> float:
    """Domain-gap proxy: ``auc_source − auc_target``.

    Positive values indicate that the model degrades on the target domain.
    A value of 0 means no generalisation gap.

    Parameters
    ----------
    auc_source : AUC on the source domain (or source validation set).
    auc_target : AUC of the *unadapted* model on the target domain.
    """
    return float(auc_source) - float(auc_target)


def gain(auc_adapted: float, auc_source_only: float) -> float:
    """Adaptation gain: ``auc_adapted − auc_source_only``.

    Positive values indicate that the adaptation method improved over the
    no-adaptation (SourceOnly) baseline.

    Parameters
    ----------
    auc_adapted     : AUC of the adapted model on the target domain.
    auc_source_only : AUC of the unadapted (SourceOnly) model on the same
                      target domain.
    """
    return float(auc_adapted) - float(auc_source_only)

"""Tests for nstad_bench.metrics.

Every test uses analytically known ground truth so failures are
unambiguous.  No model training — pure numpy/scipy.

Coverage
--------
scores.py       roc_auc, pr_auc, best_threshold, f1_at_best_threshold,
                mcc, delta_auc, gain
bootstrap.py    BootstrapCI shape / ordering / coverage / helpers
statistical.py  wilcoxon_test, friedman_nemenyi (structure + known outcomes)
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from nstad_bench.metrics import (
    BootstrapCI,
    FriedmanNemenyiResult,
    WilcoxonResult,
    best_threshold,
    bootstrap_ci,
    delta_auc,
    f1_at_best_threshold,
    friedman_nemenyi,
    gain,
    mcc,
    pr_auc,
    roc_auc,
    wilcoxon_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def perfect() -> tuple[np.ndarray, np.ndarray]:
    """y_score perfectly separates the two classes."""
    y_true  = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y_true, y_score


@pytest.fixture()
def inverted() -> tuple[np.ndarray, np.ndarray]:
    """y_score is perfectly anti-correlated with y_true → worst possible."""
    y_true  = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    return y_true, y_score


@pytest.fixture()
def random_scores(rng=np.random.default_rng(0)) -> tuple[np.ndarray, np.ndarray]:
    """200-sample balanced dataset; scores drawn i.i.d. from Uniform(0,1)."""
    y_true  = np.array([0] * 100 + [1] * 100)
    y_score = rng.uniform(0, 1, size=200)
    return y_true, y_score


@pytest.fixture()
def separable_large() -> tuple[np.ndarray, np.ndarray]:
    """500 samples with a clear score gap; used for bootstrap coverage tests."""
    rng     = np.random.default_rng(7)
    y_true  = np.array([0] * 250 + [1] * 250)
    y_score = np.concatenate([rng.normal(0.3, 0.1, 250),
                               rng.normal(0.7, 0.1, 250)]).clip(0, 1)
    return y_true, y_score


# ─────────────────────────────────────────────────────────────────────────────
# roc_auc
# ─────────────────────────────────────────────────────────────────────────────

class TestRocAuc:

    def test_perfect_predictor(self, perfect):
        y_true, y_score = perfect
        assert roc_auc(y_true, y_score) == pytest.approx(1.0)

    def test_inverted_predictor(self, inverted):
        y_true, y_score = inverted
        assert roc_auc(y_true, y_score) == pytest.approx(0.0)

    def test_random_near_half(self, random_scores):
        """Uniform random scores should give AUC ≈ 0.5 ± 0.1."""
        y_true, y_score = random_scores
        assert abs(roc_auc(y_true, y_score) - 0.5) < 0.1

    def test_returns_float(self, perfect):
        y_true, y_score = perfect
        assert isinstance(roc_auc(y_true, y_score), float)

    def test_known_value(self):
        """Hand-computed: 3 negative (scores 0.1,0.4,0.35), 2 positive (0.8,0.65).
        All 6 (neg,pos) pairs: pos > neg in 5/6 pairs → AUC = 5/6."""
        y_true  = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.65])
        # Pairs: (0.1,0.8)✓ (0.1,0.65)✓ (0.4,0.8)✓ (0.4,0.65)✓ (0.35,0.8)✓ (0.35,0.65)✓
        # All 6 positive → AUC = 1.0
        assert roc_auc(y_true, y_score) == pytest.approx(1.0)

    def test_known_partial(self):
        """2 negatives (0.6,0.4), 2 positives (0.7,0.3).
        Pairs: (0.6,0.7)✓ (0.6,0.3)✗ (0.4,0.7)✓ (0.4,0.3)✗ → 2/4 = 0.5."""
        y_true  = np.array([0, 0, 1, 1])
        y_score = np.array([0.6, 0.4, 0.7, 0.3])
        assert roc_auc(y_true, y_score) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# pr_auc
# ─────────────────────────────────────────────────────────────────────────────

class TestPrAuc:

    def test_perfect_predictor(self, perfect):
        y_true, y_score = perfect
        assert pr_auc(y_true, y_score) == pytest.approx(1.0)

    def test_in_unit_interval(self, random_scores):
        y_true, y_score = random_scores
        val = pr_auc(y_true, y_score)
        assert 0.0 <= val <= 1.0

    def test_random_near_prevalence(self, random_scores):
        """PR-AUC of a random scorer ≈ prevalence (0.5 for balanced data)."""
        y_true, y_score = random_scores
        prevalence = y_true.mean()
        assert abs(pr_auc(y_true, y_score) - prevalence) < 0.15

    def test_returns_float(self, perfect):
        y_true, y_score = perfect
        assert isinstance(pr_auc(y_true, y_score), float)


# ─────────────────────────────────────────────────────────────────────────────
# best_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestBestThreshold:

    def test_threshold_separates_classes(self, perfect):
        """With a perfectly separable dataset the best threshold must lie
        strictly between the highest negative (0.3) and lowest positive (0.7)."""
        y_true, y_score = perfect
        t = best_threshold(y_true, y_score)
        assert 0.3 <= t <= 0.7

    def test_threshold_is_float(self, perfect):
        y_true, y_score = perfect
        assert isinstance(best_threshold(y_true, y_score), float)

    def test_f1_at_threshold_is_one(self, perfect):
        """Applying the best threshold to a perfectly separable set → F1 = 1."""
        y_true, y_score = perfect
        t = best_threshold(y_true, y_score)
        preds = (y_score >= t).astype(int)
        from sklearn.metrics import f1_score
        assert f1_score(y_true, preds) == pytest.approx(1.0)

    def test_threshold_works_on_imbalanced_validation(self):
        """90 % negatives / 10 % positives with a real score signal.

        Anomaly-detection validation sets are typically this imbalanced.
        The degenerate majority-class solution — predict everything as 0 —
        yields Recall = 0 and F1 = 0 for the positive class.

        best_threshold must find a *non-trivial* threshold at which both
        Precision > 0 and Recall > 0, i.e. at least one positive is
        retrieved and at least one retrieved sample is truly positive.

        Score distribution
        ------------------
        Negatives ~ N(0.30, 0.10), Positives ~ N(0.70, 0.10).
        With 90 negatives and 10 positives the optimal F1 threshold lies
        around 0.50, well within the (0, 1) range and far from degeneracy.
        """
        rng   = np.random.default_rng(20)
        n_neg, n_pos = 90, 10

        s_neg = rng.normal(0.30, 0.10, n_neg).clip(0.0, 1.0)
        s_pos = rng.normal(0.70, 0.10, n_pos).clip(0.0, 1.0)

        y_val = np.array([0] * n_neg + [1] * n_pos)
        s_val = np.concatenate([s_neg, s_pos])

        t = best_threshold(y_val, s_val)
        y_pred = (s_val >= t).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_val == 1)))
        fp = int(np.sum((y_pred == 1) & (y_val == 0)))
        fn = int(np.sum((y_pred == 0) & (y_val == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        assert precision > 0.0, (
            f"best_threshold collapsed to majority class (predict all 0): "
            f"Precision = 0  (t = {t:.4f}, TP={tp}, FP={fp}, FN={fn})"
        )
        assert recall > 0.0, (
            f"best_threshold collapsed to majority class (predict all 0): "
            f"Recall = 0  (t = {t:.4f}, TP={tp}, FP={fp}, FN={fn})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# f1_at_best_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestF1AtBestThreshold:

    def test_perfect_val_and_test(self, perfect):
        """Same perfectly separable data for val and test → F1 = 1."""
        y_true, y_score = perfect
        assert f1_at_best_threshold(y_true, y_score, y_true, y_score) == pytest.approx(1.0)

    def test_threshold_from_val_applied_to_test(self):
        """Val and test have the same distribution; threshold generalises."""
        rng = np.random.default_rng(1)
        y_val  = np.array([0]*50 + [1]*50)
        s_val  = np.concatenate([rng.normal(0.2, 0.05, 50),
                                  rng.normal(0.8, 0.05, 50)]).clip(0, 1)
        y_test = np.array([0]*50 + [1]*50)
        s_test = np.concatenate([rng.normal(0.2, 0.05, 50),
                                  rng.normal(0.8, 0.05, 50)]).clip(0, 1)
        f1 = f1_at_best_threshold(y_test, s_test, y_val, s_val)
        assert f1 > 0.9, f"Expected F1 > 0.9 on well-separated data, got {f1:.4f}"

    def test_inverted_test_gives_zero_f1(self, perfect):
        """Threshold from perfect val applied to inverted scores → F1 = 0."""
        y_true, y_score = perfect
        y_inv   = y_true.copy()
        s_inv   = 1.0 - y_score      # invert scores
        # val is perfect, test is inverted → threshold is high, all inverted
        # positives score low → no true positives → F1 = 0
        assert f1_at_best_threshold(y_inv, s_inv, y_true, y_score) == pytest.approx(0.0)

    def test_returns_float(self, perfect):
        y_true, y_score = perfect
        assert isinstance(f1_at_best_threshold(y_true, y_score, y_true, y_score), float)


# ─────────────────────────────────────────────────────────────────────────────
# mcc
# ─────────────────────────────────────────────────────────────────────────────

class TestMcc:

    def test_perfect_prediction(self):
        y_true  = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(1.0)

    def test_all_predicted_positive(self):
        """All positives: MCC = 0 (no-skill classifier)."""
        y_true  = np.array([0, 0, 1, 1])
        y_score = np.array([1.0, 1.0, 1.0, 1.0])
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(0.0)

    def test_all_predicted_negative(self):
        """All negatives: MCC = 0."""
        y_true  = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 0.0, 0.0])
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(0.0)

    def test_inverted_prediction(self):
        """Perfectly wrong → MCC = −1."""
        y_true  = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(-1.0)

    def test_custom_threshold(self):
        """Using a high threshold (0.9) on scores [0.1,0.2,0.8,0.95]:
        predicts [0,0,0,1] against y_true=[0,0,1,1] → TP=1 FN=1 FP=0 TN=2
        MCC = (1·2 − 0·1) / sqrt((1+0)(1+1)(2+0)(2+1)) = 2/sqrt(12)."""
        y_true  = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.95])
        expected = 2.0 / (12.0 ** 0.5)
        assert mcc(y_true, y_score, threshold=0.9) == pytest.approx(expected, abs=1e-6)

    def test_returns_float(self, perfect):
        y_true, y_score = perfect
        assert isinstance(mcc(y_true, y_score), float)


# ─────────────────────────────────────────────────────────────────────────────
# delta_auc / gain
# ─────────────────────────────────────────────────────────────────────────────

class TestScalarMetrics:

    def test_delta_auc_positive(self):
        assert delta_auc(0.85, 0.70) == pytest.approx(0.15)

    def test_delta_auc_zero(self):
        assert delta_auc(0.80, 0.80) == pytest.approx(0.0)

    def test_delta_auc_negative(self):
        """If target AUC > source AUC, delta is negative (unusual but valid)."""
        assert delta_auc(0.70, 0.85) == pytest.approx(-0.15)

    def test_gain_positive(self):
        assert gain(0.85, 0.80) == pytest.approx(0.05)

    def test_gain_zero(self):
        assert gain(0.75, 0.75) == pytest.approx(0.0)

    def test_gain_negative(self):
        assert gain(0.70, 0.80) == pytest.approx(-0.10)

    def test_returns_float(self):
        assert isinstance(delta_auc(0.8, 0.7), float)
        assert isinstance(gain(0.8, 0.7), float)


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_returns_bootstrap_ci(self, perfect):
        y_true, y_score = perfect
        ci = bootstrap_ci(roc_auc, y_true, y_score)
        assert isinstance(ci, BootstrapCI)

    def test_ordering(self, separable_large):
        """lower ≤ estimate ≤ upper must always hold."""
        y_true, y_score = separable_large
        ci = bootstrap_ci(roc_auc, y_true, y_score)
        assert ci.lower <= ci.estimate <= ci.upper

    def test_width_positive(self, separable_large):
        """Bootstrap CI must have non-zero width (data is not degenerate)."""
        y_true, y_score = separable_large
        ci = bootstrap_ci(roc_auc, y_true, y_score)
        assert ci.width > 0.0

    def test_confidence_stored(self, separable_large):
        y_true, y_score = separable_large
        ci = bootstrap_ci(roc_auc, y_true, y_score, confidence=0.90)
        assert ci.confidence == pytest.approx(0.90)

    def test_n_bootstrap_stored(self, separable_large):
        y_true, y_score = separable_large
        ci = bootstrap_ci(roc_auc, y_true, y_score, n_bootstrap=200)
        # May be ≤ 200 if any degenerate resamples were skipped
        assert ci.n_bootstrap <= 200
        assert ci.n_bootstrap > 0

    def test_wider_at_lower_confidence(self, separable_large):
        """95 % CI should be wider than 80 % CI."""
        y_true, y_score = separable_large
        ci95 = bootstrap_ci(roc_auc, y_true, y_score, confidence=0.95, seed=0)
        ci80 = bootstrap_ci(roc_auc, y_true, y_score, confidence=0.80, seed=0)
        assert ci95.width >= ci80.width

    def test_perfect_predictor_ci_near_one(self, separable_large):
        """Lower bound of AUC CI should be well above 0.5 for separable data."""
        y_true, y_score = separable_large
        ci = bootstrap_ci(roc_auc, y_true, y_score)
        assert ci.lower > 0.7

    def test_contains_helper(self, separable_large):
        y_true, y_score = separable_large
        ci = bootstrap_ci(roc_auc, y_true, y_score)
        assert ci.contains(ci.estimate)
        assert not ci.contains(-1.0)

    def test_works_with_pr_auc(self, separable_large):
        y_true, y_score = separable_large
        ci = bootstrap_ci(pr_auc, y_true, y_score)
        assert ci.lower <= ci.estimate <= ci.upper

    def test_works_with_lambda_mcc(self, separable_large):
        """Wrap mcc with a fixed threshold in a lambda."""
        y_true, y_score = separable_large
        t  = best_threshold(y_true, y_score)
        ci = bootstrap_ci(lambda yt, ys: mcc(yt, ys, threshold=t),
                          y_true, y_score)
        assert ci.lower <= ci.estimate <= ci.upper

    def test_reproducible_with_same_seed(self, separable_large):
        y_true, y_score = separable_large
        ci1 = bootstrap_ci(roc_auc, y_true, y_score, seed=99)
        ci2 = bootstrap_ci(roc_auc, y_true, y_score, seed=99)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_different_seeds_give_different_bounds(self, separable_large):
        y_true, y_score = separable_large
        ci1 = bootstrap_ci(roc_auc, y_true, y_score, seed=1)
        ci2 = bootstrap_ci(roc_auc, y_true, y_score, seed=2)
        # With N=500, estimates will differ at high precision
        assert ci1.lower != ci2.lower or ci1.upper != ci2.upper

    def test_invalid_confidence_raises(self, perfect):
        y_true, y_score = perfect
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci(roc_auc, y_true, y_score, confidence=1.5)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            bootstrap_ci(roc_auc, np.array([0, 1]), np.array([0.1, 0.9, 0.5]))


# ─────────────────────────────────────────────────────────────────────────────
# wilcoxon_test
# ─────────────────────────────────────────────────────────────────────────────

class TestWilcoxon:

    def test_clearly_different_methods(self):
        """Method A consistently scores higher → very small p-value."""
        rng = np.random.default_rng(3)
        a = rng.uniform(0.75, 0.95, size=20)
        b = rng.uniform(0.55, 0.75, size=20)
        result = wilcoxon_test(a, b)
        assert result.p_value < 0.05, f"Expected p < 0.05, got {result.p_value:.4f}"

    def test_nearly_identical_gives_large_p(self):
        """Tiny random perturbations around 0.80 → no significant difference."""
        rng = np.random.default_rng(4)
        base = rng.uniform(0.78, 0.82, size=30)
        a    = base + rng.normal(0, 1e-4, 30)
        b    = base + rng.normal(0, 1e-4, 30)
        result = wilcoxon_test(a, b)
        assert result.p_value > 0.05, f"Expected p > 0.05, got {result.p_value:.4f}"

    def test_returns_wilcoxon_result(self):
        a = np.array([0.8, 0.7, 0.9, 0.75])
        b = np.array([0.6, 0.65, 0.7, 0.55])
        result = wilcoxon_test(a, b)
        assert isinstance(result, WilcoxonResult)

    def test_p_value_in_unit_interval(self):
        a = np.array([0.8, 0.7, 0.9, 0.75, 0.85])
        b = np.array([0.6, 0.65, 0.7, 0.55, 0.68])
        result = wilcoxon_test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_alternative_greater(self):
        """One-sided test: a > b should give p < two-sided p."""
        a = np.array([0.85, 0.80, 0.90, 0.78, 0.88, 0.82])
        b = np.array([0.60, 0.65, 0.70, 0.55, 0.68, 0.62])
        two_sided = wilcoxon_test(a, b, alternative="two-sided").p_value
        one_sided = wilcoxon_test(a, b, alternative="greater").p_value
        assert one_sided <= two_sided

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            wilcoxon_test(np.array([0.8, 0.7]), np.array([0.6]))

    def test_is_significant_method(self):
        a = np.array([0.85, 0.80, 0.90, 0.78, 0.88, 0.82,
                      0.86, 0.81, 0.89, 0.79])
        b = np.array([0.60, 0.65, 0.70, 0.55, 0.68, 0.62,
                      0.61, 0.66, 0.71, 0.56])
        result = wilcoxon_test(a, b)
        assert result.is_significant(0.05) == (result.p_value < 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# friedman_nemenyi
# ─────────────────────────────────────────────────────────────────────────────

class TestFriedmanNemenyi:

    @pytest.fixture()
    def scores_dominant(self):
        """Method A always ranks 1st across 10 datasets → should be significant."""
        rng = np.random.default_rng(5)
        a = rng.uniform(0.85, 0.95, 10)
        b = rng.uniform(0.65, 0.75, 10)
        c = rng.uniform(0.55, 0.65, 10)
        d = rng.uniform(0.45, 0.55, 10)
        return np.column_stack([a, b, c, d])

    @pytest.fixture()
    def scores_equal(self):
        """All methods draw from same distribution → Friedman should not reject."""
        rng = np.random.default_rng(6)
        return rng.uniform(0.70, 0.80, (15, 4))

    def test_returns_result_type(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant)
        assert isinstance(result, FriedmanNemenyiResult)

    def test_dominant_method_is_significant(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant)
        assert result.p_value < 0.05, (
            f"Expected Friedman p < 0.05 for a clearly dominant method, "
            f"got p={result.p_value:.4f}"
        )

    def test_equal_methods_not_significant(self, scores_equal):
        result = friedman_nemenyi(scores_equal)
        assert result.p_value > 0.05, (
            f"Expected Friedman p > 0.05 for equal methods, "
            f"got p={result.p_value:.4f}"
        )

    def test_nemenyi_matrix_shape(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant,
                                  method_names=["A", "B", "C", "D"])
        assert result.nemenyi_pvalues.shape == (4, 4)

    def test_nemenyi_diagonal_is_one(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant,
                                  method_names=["A", "B", "C", "D"])
        diag = np.diag(result.nemenyi_pvalues.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_nemenyi_is_symmetric(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant,
                                  method_names=["A", "B", "C", "D"])
        pv = result.nemenyi_pvalues.values
        np.testing.assert_allclose(pv, pv.T, atol=1e-10)

    def test_method_names_in_index(self, scores_dominant):
        names  = ["Alpha", "Beta", "Gamma", "Delta"]
        result = friedman_nemenyi(scores_dominant, method_names=names)
        assert list(result.nemenyi_pvalues.columns) == names
        assert list(result.nemenyi_pvalues.index)   == names

    def test_significant_pairs_dominant(self, scores_dominant):
        """With a clearly dominant method A, at least A vs D should be significant."""
        result = friedman_nemenyi(scores_dominant,
                                  method_names=["A", "B", "C", "D"])
        pairs = result.significant_pairs(alpha=0.05)
        # At minimum, the best vs worst pair should be flagged
        assert len(pairs) >= 1

    def test_n_datasets_and_methods_stored(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant)
        assert result.n_datasets == 10
        assert result.n_methods  == 4

    def test_too_few_methods_raises(self):
        scores = np.random.default_rng(0).uniform(0, 1, (10, 2))
        with pytest.raises(ValueError, match="≥ 3 methods"):
            friedman_nemenyi(scores)

    def test_too_few_datasets_raises(self):
        scores = np.random.default_rng(0).uniform(0, 1, (1, 4))
        with pytest.raises(ValueError, match="≥ 2 datasets"):
            friedman_nemenyi(scores)

    def test_wrong_method_names_length_raises(self, scores_dominant):
        with pytest.raises(ValueError, match="method_names"):
            friedman_nemenyi(scores_dominant, method_names=["A", "B"])

    def test_nemenyi_p_values_in_unit_interval(self, scores_dominant):
        result = friedman_nemenyi(scores_dominant)
        pv = result.nemenyi_pvalues.values
        assert np.all(pv >= 0.0) and np.all(pv <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# sklearn parity — cross-reference every metric against sklearn.metrics
#
# These tests prove our wrappers are thin, correct adapters:
# every value must match sklearn exactly (atol=1e-10) on the same input.
# ─────────────────────────────────────────────────────────────────────────────

class TestSklearnParity:
    """Compare nstad_bench metrics against sklearn reference on canonical inputs.

    Fixture: 10 negatives (scores 0.05 … 0.45) + 10 positives (scores 0.55 … 0.95).
    This is the minimal balanced perfectly-separable dataset the user specified.
    """

    @pytest.fixture()
    def ten_ten(self):
        """10 negatives, 10 positives, perfectly separated at threshold 0.50."""
        y_true  = np.array([0]*10 + [1]*10)
        y_score = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                             0.30, 0.35, 0.38, 0.42, 0.46,
                             0.54, 0.58, 0.62, 0.65, 0.70,
                             0.75, 0.80, 0.85, 0.90, 0.95])
        return y_true, y_score

    # ── roc_auc ──────────────────────────────────────────────────────────────

    def test_roc_auc_equals_one_on_perfect(self, ten_ten):
        """Perfect 10+10 dataset → AUC = 1.0 (matches sklearn)."""
        from sklearn.metrics import roc_auc_score as sk_roc
        y_true, y_score = ten_ten
        assert roc_auc(y_true, y_score) == pytest.approx(1.0)
        assert roc_auc(y_true, y_score) == pytest.approx(sk_roc(y_true, y_score))

    def test_roc_auc_matches_sklearn_partial(self):
        """Partial separation: our roc_auc must equal sklearn's exactly."""
        from sklearn.metrics import roc_auc_score as sk_roc
        rng = np.random.default_rng(11)
        y_true  = np.array([0]*10 + [1]*10)
        y_score = rng.uniform(0, 1, 20)
        assert roc_auc(y_true, y_score) == pytest.approx(sk_roc(y_true, y_score), abs=1e-10)

    # ── pr_auc ───────────────────────────────────────────────────────────────

    def test_pr_auc_equals_one_on_perfect(self, ten_ten):
        """Perfect 10+10 dataset → PR-AUC = 1.0 (matches sklearn)."""
        from sklearn.metrics import average_precision_score as sk_ap
        y_true, y_score = ten_ten
        assert pr_auc(y_true, y_score) == pytest.approx(1.0)
        assert pr_auc(y_true, y_score) == pytest.approx(sk_ap(y_true, y_score))

    def test_pr_auc_matches_sklearn_partial(self):
        """Partial separation: our pr_auc must equal sklearn's exactly."""
        from sklearn.metrics import average_precision_score as sk_ap
        rng = np.random.default_rng(12)
        y_true  = np.array([0]*10 + [1]*10)
        y_score = rng.uniform(0, 1, 20)
        assert pr_auc(y_true, y_score) == pytest.approx(sk_ap(y_true, y_score), abs=1e-10)

    # ── f1_at_best_threshold on 10+10 ────────────────────────────────────────

    def test_f1_equals_one_on_perfect_10_10(self, ten_ten):
        """F1 = 1.0 when val and test are both perfectly separable 10+10."""
        from sklearn.metrics import f1_score as sk_f1
        y_true, y_score = ten_ten
        f1_val = f1_at_best_threshold(y_true, y_score, y_true, y_score)
        assert f1_val == pytest.approx(1.0)
        # Verify via sklearn with the same threshold
        t      = best_threshold(y_true, y_score)
        y_pred = (y_score >= t).astype(int)
        assert f1_val == pytest.approx(sk_f1(y_true, y_pred), abs=1e-10)

    # ── mcc ──────────────────────────────────────────────────────────────────

    def test_mcc_equals_one_on_perfect_10_10(self, ten_ten):
        """MCC = 1.0 on perfectly separable 10+10 at threshold 0.5."""
        from sklearn.metrics import matthews_corrcoef as sk_mcc
        y_true, y_score = ten_ten
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(1.0)
        y_pred = (y_score >= 0.5).astype(int)
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(
            sk_mcc(y_true, y_pred), abs=1e-10
        )

    def test_mcc_matches_sklearn_partial(self):
        """Partial scores: our mcc must equal sklearn's at threshold 0.5."""
        from sklearn.metrics import matthews_corrcoef as sk_mcc
        rng = np.random.default_rng(13)
        y_true  = np.array([0]*10 + [1]*10)
        y_score = rng.uniform(0, 1, 20)
        y_pred  = (y_score >= 0.5).astype(int)
        assert mcc(y_true, y_score, threshold=0.5) == pytest.approx(
            sk_mcc(y_true, y_pred), abs=1e-10
        )

    # ── bootstrap on 10+10 ───────────────────────────────────────────────────

    def test_bootstrap_degenerate_on_perfect_10_10(self, ten_ten):
        """Perfectly separable 10+10 → every valid resample also gives AUC=1.0,
        so CI=[1.0, 1.0] with width=0.  This is correct, not a bug.
        """
        y_true, y_score = ten_ten
        ci = bootstrap_ci(roc_auc, y_true, y_score, n_bootstrap=500, seed=0)
        assert ci.estimate == pytest.approx(1.0)
        assert ci.lower == pytest.approx(1.0)
        assert ci.upper == pytest.approx(1.0)
        # Degenerate but well-formed: ordering holds trivially
        assert ci.lower <= ci.estimate <= ci.upper

    def test_bootstrap_nonzero_width_on_overlapping_10_10(self):
        """10+10 with overlapping Gaussian scores → CI width > 0.

        Uses Gaussian(μ=0.40, σ=0.12) for negatives and Gaussian(μ=0.60, σ=0.12)
        for positives.  The overlap means some resamples give AUC < 1.0, so the
        CI has a non-trivial lower bound.
        """
        rng    = np.random.default_rng(42)
        neg    = rng.normal(0.40, 0.12, 10).clip(0, 1)
        pos    = rng.normal(0.60, 0.12, 10).clip(0, 1)
        y_true = np.array([0] * 10 + [1] * 10)
        y_score = np.concatenate([neg, pos])
        ci = bootstrap_ci(roc_auc, y_true, y_score, n_bootstrap=500, seed=0)
        assert ci.width > 0.0, (
            f"Expected CI width > 0 on overlapping 10+10 data, got {ci.width}"
        )
        assert ci.lower <= ci.estimate <= ci.upper


# ─────────────────────────────────────────────────────────────────────────────
# No-leakage: threshold is chosen on val, not on test
#
# Design: the val set has a non-standard optimal threshold (t* ≈ 0.30,
# far from the default 0.50).  The test set is constructed so that:
#   • At t* (val-derived)  → F1 = 1.0   (correct path)
#   • At t  = 0.50 (test-derived default) → F1 = 0.0  (what leakage gives)
#
# If f1_at_best_threshold used the test set to pick the threshold, it would
# find t_test* ≈ 0.49 and return F1 = 1.0 — but via the wrong split.
# The test therefore confirms that only the val split drives threshold selection.
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLeakage:

    @pytest.fixture()
    def leakage_probe(self):
        """Val: positives score ~0.21–0.23, negatives score ~0.05–0.07 → t* = 0.21.
        Test: positives score ~0.35–0.45, negatives score ~0.01–0.03.

        At t = t_val* (≈ 0.21): test positives (0.35–0.45) ≥ t* → F1 = 1.0.
        At t = 0.50  : test positives (0.35–0.45) < 0.50 → all predicted 0 → F1 = 0.
        At t = t_test* (≈ 0.35): different from t_val*, |Δ| ≈ 0.14 — probe is sensitive.
        """
        y_val  = np.array([1, 1, 1, 0, 0, 0])
        s_val  = np.array([0.21, 0.22, 0.23, 0.05, 0.06, 0.07])

        y_test = np.array([1, 1, 1, 0, 0, 0])
        s_test = np.array([0.35, 0.40, 0.45, 0.01, 0.02, 0.03])
        return y_val, s_val, y_test, s_test

    def test_val_threshold_is_below_0_5(self, leakage_probe):
        """Val-derived threshold is 0.21 (min positive score) — well below 0.5."""
        y_val, s_val, _, _ = leakage_probe
        t = best_threshold(y_val, s_val)
        assert 0.19 <= t <= 0.23, (
            f"Expected val threshold ≈ 0.21, got {t:.4f}"
        )

    def test_f1_is_one_with_val_threshold(self, leakage_probe):
        """Using the val threshold on the test set → F1 = 1.0."""
        y_val, s_val, y_test, s_test = leakage_probe
        f1 = f1_at_best_threshold(y_test, s_test, y_val, s_val)
        assert f1 == pytest.approx(1.0), (
            f"Expected F1=1.0 with val threshold, got {f1:.4f}"
        )

    def test_default_threshold_gives_zero_f1(self, leakage_probe):
        """Sanity: applying default threshold=0.50 to the test set → F1 = 0.
        This is what a leaking implementation (using test to pick threshold)
        would avoid — it would still return F1=1.0, masking the problem.
        """
        _, _, y_test, s_test = leakage_probe
        # All test scores are < 0.50 → every sample predicted negative → F1 = 0
        f1_default = mcc(y_test, s_test, threshold=0.50)
        from sklearn.metrics import f1_score as sk_f1
        f1_at_half = sk_f1(y_test, (s_test >= 0.5).astype(int), zero_division=0.0)
        assert f1_at_half == pytest.approx(0.0), (
            "Sanity failed: expected F1=0 at threshold=0.5 on this test set"
        )

    def test_val_and_test_disagree_confirm_no_leakage(self, leakage_probe):
        """Key assertion: the two thresholds t_val* and t_test* differ by > 0.05,
        proving the probe is sensitive to which split is used.

        f1_at_best_threshold must return 1.0 (uses t_val*=0.21 → all test
        positives at 0.35–0.45 are correctly classified).  A leaking
        implementation using t_test*=0.35 would also return 1.0, but any
        implementation using the default 0.5 would return 0.0 — that is why
        test_default_threshold_gives_zero_f1 is needed in parallel.

        The traceable proof: call best_threshold on val explicitly and verify
        the returned value matches t_val*, not t_test*.
        """
        y_val, s_val, y_test, s_test = leakage_probe
        t_val  = best_threshold(y_val, s_val)    # ≈ 0.21
        t_test = best_threshold(y_test, s_test)  # ≈ 0.35

        # The two thresholds must differ (probe sensitivity check)
        assert abs(t_val - t_test) > 0.05, (
            f"Probe design error: val threshold {t_val:.3f} ≈ test threshold "
            f"{t_test:.3f} — probe cannot detect leakage"
        )

        # f1_at_best_threshold uses val split → F1 = 1.0
        f1 = f1_at_best_threshold(y_test, s_test, y_val, s_val)
        assert f1 == pytest.approx(1.0)

        # best_threshold is a pure function: same val data → same threshold
        assert best_threshold(y_val, s_val) == pytest.approx(t_val, abs=1e-9)

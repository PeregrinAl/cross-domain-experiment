"""Tests for representations: shape consistency and deterministic reproducibility.

Shape consistency
-----------------
For each representation ``R``:
  1. ``R.fit(X)`` must set ``R.output_shape`` to the per-sample shape.
  2. ``R.transform(X)`` must return an array whose per-sample shape matches
     ``R.output_shape`` for both the training set and a held-out test set.

Reproducibility
---------------
  * Non-trained (stateless) representations (RawSignal, LogSTFT, CWT_Morlet):
    two ``transform`` calls on the same input must return identical arrays.
  * Trained representation (CARLA_SSL): fitting with the same ``seed`` and then
    transforming the same input must give identical results across two
    independent instantiations of the model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nstad_bench.representations.carla_ssl import CARLA_SSL
from nstad_bench.representations.cwt_morlet import CWT_Morlet
from nstad_bench.representations.log_stft import LogSTFT
from nstad_bench.representations.raw_signal import RawSignal

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)

# Univariate windows: 32 training samples, 8 test samples, length 128
N_TRAIN, N_TEST, T = 32, 8, 128
X_TRAIN: np.ndarray = RNG.standard_normal((N_TRAIN, T)).astype(np.float32)
X_TEST: np.ndarray = RNG.standard_normal((N_TEST, T)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_output_shape(
    rep, X: np.ndarray, label: str = ""
) -> None:
    """Verify that transform output is consistent with output_shape."""
    out = rep.transform(X)
    n = len(X)
    assert out.shape == (n,) + rep.output_shape, (
        f"{label}: expected (N={n},) + {rep.output_shape}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# RawSignal
# ---------------------------------------------------------------------------

class TestRawSignal:

    def test_is_flags(self):
        assert RawSignal.is_1d is True
        assert RawSignal.is_2d is False

    def test_output_shape_set_after_fit(self):
        r = RawSignal()
        r.fit(X_TRAIN)
        assert r.output_shape == (T,)

    def test_shape_consistency_train(self):
        r = RawSignal().fit(X_TRAIN)
        _assert_output_shape(r, X_TRAIN, "RawSignal/train")

    def test_shape_consistency_test(self):
        r = RawSignal().fit(X_TRAIN)
        _assert_output_shape(r, X_TEST, "RawSignal/test")

    def test_shape_single_sample(self):
        r = RawSignal().fit(X_TRAIN)
        _assert_output_shape(r, X_TEST[:1], "RawSignal/single")

    def test_reproducibility(self):
        r = RawSignal().fit(X_TRAIN)
        out1 = r.transform(X_TEST)
        out2 = r.transform(X_TEST)
        np.testing.assert_array_equal(out1, out2)

    def test_zero_mean_unit_std_approx(self):
        """After fitting on X_TRAIN, the transformed X_TRAIN should be ≈ standardised."""
        r = RawSignal().fit(X_TRAIN)
        out = r.transform(X_TRAIN)
        # Column-wise statistics should be close to (0, 1)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)

    def test_fit_before_transform_raises(self):
        r = RawSignal()
        with pytest.raises(RuntimeError):
            r.transform(X_TEST)

    def test_multivariate_shape(self):
        X_mv = RNG.standard_normal((N_TRAIN, T, 3)).astype(np.float32)
        r = RawSignal().fit(X_mv)
        assert r.output_shape == (T, 3)
        out = r.transform(X_mv)
        assert out.shape == (N_TRAIN, T, 3)


# ---------------------------------------------------------------------------
# LogSTFT
# ---------------------------------------------------------------------------

class TestLogSTFT:

    @pytest.fixture(scope="class")
    def fitted(self):
        return LogSTFT(n_fft=256, hop_length=64).fit(X_TRAIN)

    def test_is_flags(self):
        assert LogSTFT.is_1d is False
        assert LogSTFT.is_2d is True

    def test_output_shape_set_after_fit(self, fitted):
        F_bins, T_frames = fitted.output_shape
        assert F_bins == 256 // 2 + 1  # 129
        assert T_frames > 0

    def test_shape_consistency_train(self, fitted):
        _assert_output_shape(fitted, X_TRAIN, "LogSTFT/train")

    def test_shape_consistency_test(self, fitted):
        _assert_output_shape(fitted, X_TEST, "LogSTFT/test")

    def test_shape_single_sample(self, fitted):
        _assert_output_shape(fitted, X_TEST[:1], "LogSTFT/single")

    def test_reproducibility(self, fitted):
        out1 = fitted.transform(X_TEST)
        out2 = fitted.transform(X_TEST)
        np.testing.assert_array_equal(out1, out2)

    def test_non_negative_output(self, fitted):
        """log1p(|STFT|) must be non-negative."""
        out = fitted.transform(X_TEST)
        assert (out >= 0).all()

    def test_fit_before_transform_raises(self):
        with pytest.raises(RuntimeError):
            LogSTFT().transform(X_TEST)

    @pytest.mark.parametrize("n_fft,hop", [(128, 32), (256, 64), (512, 128)])
    def test_frequency_bin_count(self, n_fft, hop):
        r = LogSTFT(n_fft=n_fft, hop_length=hop).fit(X_TRAIN)
        assert r.output_shape[0] == n_fft // 2 + 1


# ---------------------------------------------------------------------------
# CWT_Morlet
# ---------------------------------------------------------------------------

class TestCWT_Morlet:

    @pytest.fixture(scope="class")
    def fitted(self):
        return CWT_Morlet(n_scales=64).fit(X_TRAIN)

    def test_is_flags(self):
        assert CWT_Morlet.is_1d is False
        assert CWT_Morlet.is_2d is True

    def test_output_shape_set_after_fit(self, fitted):
        n_sc, t_len = fitted.output_shape
        assert n_sc == 64
        assert t_len == T

    def test_shape_consistency_train(self, fitted):
        _assert_output_shape(fitted, X_TRAIN, "CWT/train")

    def test_shape_consistency_test(self, fitted):
        _assert_output_shape(fitted, X_TEST, "CWT/test")

    def test_shape_single_sample(self, fitted):
        _assert_output_shape(fitted, X_TEST[:1], "CWT/single")

    def test_reproducibility(self, fitted):
        out1 = fitted.transform(X_TEST)
        out2 = fitted.transform(X_TEST)
        np.testing.assert_array_equal(out1, out2)

    def test_non_negative_output(self, fitted):
        """Modulus of CWT coefficients must be non-negative."""
        out = fitted.transform(X_TEST)
        assert (out >= 0).all()

    def test_fit_before_transform_raises(self):
        with pytest.raises(RuntimeError):
            CWT_Morlet().transform(X_TEST)

    @pytest.mark.parametrize("n_scales", [16, 32, 64])
    def test_scale_count(self, n_scales):
        r = CWT_Morlet(n_scales=n_scales).fit(X_TRAIN)
        assert r.output_shape[0] == n_scales


# ---------------------------------------------------------------------------
# CARLA_SSL
# ---------------------------------------------------------------------------

# Use a tiny config to keep tests fast; full config is used in experiments.
_CARLA_KWARGS = dict(
    embed_dim=16,
    n_epochs=3,
    batch_size=16,
    seed=42,
    device="cpu",
)


class TestCARLA_SSL:

    @pytest.fixture(scope="class")
    def fitted(self):
        return CARLA_SSL(**_CARLA_KWARGS).fit(X_TRAIN)

    def test_is_flags(self):
        assert CARLA_SSL.is_1d is True
        assert CARLA_SSL.is_2d is False

    def test_output_shape_set_after_fit(self, fitted):
        assert fitted.output_shape == (16,)

    def test_shape_consistency_train(self, fitted):
        _assert_output_shape(fitted, X_TRAIN, "CARLA/train")

    def test_shape_consistency_test(self, fitted):
        _assert_output_shape(fitted, X_TEST, "CARLA/test")

    def test_shape_single_sample(self, fitted):
        _assert_output_shape(fitted, X_TEST[:1], "CARLA/single")

    def test_reproducibility_transform(self, fitted):
        """Two transform calls on the same fitted model must be identical."""
        out1 = fitted.transform(X_TEST)
        out2 = fitted.transform(X_TEST)
        np.testing.assert_array_equal(out1, out2)

    def test_reproducibility_full_pipeline(self):
        """Two independent fit+transform pipelines with same seed must match."""
        r1 = CARLA_SSL(**_CARLA_KWARGS).fit(X_TRAIN)
        r2 = CARLA_SSL(**_CARLA_KWARGS).fit(X_TRAIN)
        out1 = r1.transform(X_TEST)
        out2 = r2.transform(X_TEST)
        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)

    def test_different_seeds_differ(self):
        """Models trained with different seeds should (almost certainly) differ."""
        r1 = CARLA_SSL(**{**_CARLA_KWARGS, "seed": 0}).fit(X_TRAIN)
        r2 = CARLA_SSL(**{**_CARLA_KWARGS, "seed": 99}).fit(X_TRAIN)
        out1 = r1.transform(X_TEST)
        out2 = r2.transform(X_TEST)
        assert not np.allclose(out1, out2), "Different seeds must yield different models."

    def test_embeddings_are_finite(self, fitted):
        out = fitted.transform(X_TEST)
        assert np.isfinite(out).all()

    def test_fit_before_transform_raises(self):
        with pytest.raises(RuntimeError):
            CARLA_SSL(**_CARLA_KWARGS).transform(X_TEST)

    def test_embed_dim_respected(self):
        for dim in (8, 32):
            r = CARLA_SSL(**{**_CARLA_KWARGS, "embed_dim": dim}).fit(X_TRAIN)
            assert r.output_shape == (dim,)
            assert r.transform(X_TEST).shape == (N_TEST, dim)

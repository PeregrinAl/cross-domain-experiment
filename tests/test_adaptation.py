"""Tests for domain-adaptation methods.

Checks per method
-----------------
1. ``adapt()`` returns a *different* object — original model never mutated.
2. Adapted model runs forward on **target** without error; output is valid
   probability distribution (shape (N, 2), values ∈ [0,1], rows sum to 1).
3. Adapted model runs forward on **source** without error — same validity.
4. No full collapse: source predictions are not all identical after adaptation
   (both class labels appear), indicating the model retains discriminative
   capacity on the source domain.
5. Method-specific structural checks (e.g. M2N2 leaves backbone unchanged).

Design notes
------------
* All fixtures are ``scope="module"`` — the source model is trained once and
  shared across all test classes, keeping the suite fast (< 20 s on CPU).
* We use a tiny InceptionTime1D (nb_filters=8, depth=1) with N=40 samples
  and T=64 time steps.  The two classes are visually distinct (flat vs sine)
  so 5 training epochs suffice for the model to be non-trivially discriminative.
* Adaptation hyperparameters are intentionally aggressive (high LR, few
  epochs/steps) to keep the test suite fast while still exercising the code
  path; they are not tuned for best domain-adaptation performance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torch.nn as nn

from nstad_bench.adaptation import CoDATS, MK_MMD, M2N2, SourceOnly
from nstad_bench.models import InceptionTime1D

# Normalization layer types whose affine params M2N2 is allowed to update
_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)


def _norm_param_names(model: nn.Module) -> set[str]:
    """Return fully-qualified names of all BN/LN affine parameters."""
    names: set[str] = set()
    for mname, mod in model.named_modules():
        if isinstance(mod, _NORM_TYPES):
            for pname in ("weight", "bias"):
                if getattr(mod, pname, None) is not None:
                    names.add(f"{mname}.{pname}" if mname else pname)
    return names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T          = 64    # time-series length
N_PER_CLS  = 20   # samples per class
NB_FILTERS = 8    # tiny model — fast training


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _make_data(
    n_per_class: int,
    *,
    noise: float = 0.15,
    shift: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary dataset: class 0 ≈ constant 0, class 1 ≈ sin(·).

    ``shift`` displaces the mean to simulate a target-domain distribution.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 2 * np.pi, T, dtype=np.float32)

    X0 = rng.normal(shift, noise, (n_per_class, T)).astype(np.float32)
    X1 = (np.sin(t) + rng.normal(shift, noise, (n_per_class, T))).astype(np.float32)

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Module-scoped fixtures — built once, shared by all test classes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def source_data() -> tuple[np.ndarray, np.ndarray]:
    return _make_data(N_PER_CLS, seed=1)


@pytest.fixture(scope="module")
def target_data(source_data) -> np.ndarray:
    """Unlabelled target — same structure, mean-shifted by 0.4."""
    X, _ = _make_data(N_PER_CLS, shift=0.4, seed=2)
    return X


@pytest.fixture(scope="module")
def trained_model(source_data) -> InceptionTime1D:
    """Source model trained for 5 epochs — non-trivially discriminative."""
    X, y = source_data
    m = InceptionTime1D(
        in_channels=1, nb_filters=NB_FILTERS, bottleneck=NB_FILTERS, depth=1
    )
    m.fit(X, y, epochs=5, lr=5e-3, batch_size=16)
    return m


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_valid_proba(proba: np.ndarray, tag: str) -> None:
    assert proba.ndim == 2 and proba.shape[1] == 2, \
        f"{tag}: expected (N, 2), got {proba.shape}"
    assert np.all(proba >= -1e-6) and np.all(proba <= 1 + 1e-6), \
        f"{tag}: probabilities outside [0, 1]"
    np.testing.assert_allclose(
        proba.sum(axis=1), 1.0, atol=1e-5,
        err_msg=f"{tag}: rows do not sum to 1",
    )


def _assert_no_collapse(proba: np.ndarray, tag: str) -> None:
    """Both class labels must appear in argmax predictions."""
    preds = proba.argmax(axis=1)
    assert len(set(preds.tolist())) > 1, (
        f"{tag}: model collapsed — all {len(preds)} predictions are "
        f"class {preds[0]}; adaptation may have destroyed source knowledge"
    )


def _assert_original_unchanged(
    original: InceptionTime1D,
    adapted: InceptionTime1D,
) -> None:
    """Every parameter of *original* must be bitwise-equal before/after adapt."""
    for (n_o, p_o), (n_a, p_a) in zip(
        original.named_parameters(), adapted.named_parameters()
    ):
        assert n_o == n_a
        assert torch.equal(p_o, p_a), \
            f"Original model parameter '{n_o}' was mutated by adapt()"


# ---------------------------------------------------------------------------
# SourceOnly
# ---------------------------------------------------------------------------

class TestSourceOnly:

    @pytest.fixture(scope="class")
    def adapted(self, trained_model, target_data):
        return SourceOnly().adapt(trained_model, target_data)

    def test_returns_new_object(self, trained_model, adapted):
        assert adapted is not trained_model

    def test_weights_identical(self, trained_model, adapted):
        """SourceOnly returns an exact copy — weights must match."""
        for p_orig, p_copy in zip(trained_model.parameters(), adapted.parameters()):
            assert torch.equal(p_orig, p_copy)

    def test_valid_proba_on_target(self, adapted, target_data):
        _assert_valid_proba(adapted.predict_proba(target_data), "SourceOnly/target")

    def test_valid_proba_on_source(self, adapted, source_data):
        X, _ = source_data
        _assert_valid_proba(adapted.predict_proba(X), "SourceOnly/source")

    def test_original_unchanged(self, trained_model, target_data):
        params_snap = {n: p.clone() for n, p in trained_model.named_parameters()}
        SourceOnly().adapt(trained_model, target_data)
        for n, p in trained_model.named_parameters():
            assert torch.equal(p, params_snap[n]), f"Original mutated at '{n}'"


# ---------------------------------------------------------------------------
# MK_MMD
# ---------------------------------------------------------------------------

class TestMK_MMD:

    @pytest.fixture(scope="class")
    def adapted(self, trained_model, source_data, target_data):
        X_s, y_s = source_data
        return MK_MMD(
            X_s, y_s,
            n_epochs=3, lr=1e-3, lambda_mmd=1.0, batch_size=16,
        ).adapt(trained_model, target_data)

    def test_returns_new_object(self, trained_model, adapted):
        assert adapted is not trained_model

    def test_parameters_changed(self, trained_model, adapted):
        """At least one parameter must differ after adaptation."""
        any_changed = any(
            not torch.equal(p_o, p_a)
            for p_o, p_a in zip(trained_model.parameters(), adapted.parameters())
        )
        assert any_changed, "MK_MMD did not update any model parameters"

    def test_valid_proba_on_target(self, adapted, target_data):
        _assert_valid_proba(adapted.predict_proba(target_data), "MK_MMD/target")

    def test_valid_proba_on_source(self, adapted, source_data):
        X_s, _ = source_data
        _assert_valid_proba(adapted.predict_proba(X_s), "MK_MMD/source")

    def test_no_collapse_on_source(self, adapted, source_data):
        X_s, _ = source_data
        _assert_no_collapse(adapted.predict_proba(X_s), "MK_MMD")

    def test_original_unchanged(self, trained_model, source_data, target_data):
        X_s, y_s = source_data
        params_snap = {n: p.clone() for n, p in trained_model.named_parameters()}
        MK_MMD(X_s, y_s, n_epochs=1, batch_size=16).adapt(trained_model, target_data)
        for n, p in trained_model.named_parameters():
            assert torch.equal(p, params_snap[n]), f"Original mutated at '{n}'"


# ---------------------------------------------------------------------------
# CoDATS
# ---------------------------------------------------------------------------

class TestCoDATS:

    @pytest.fixture(scope="class")
    def adapted(self, trained_model, source_data, target_data):
        X_s, y_s = source_data
        return CoDATS(
            X_s, y_s,
            n_epochs=3, lr=5e-3, lr_disc=5e-2, lambda_domain=1.0, batch_size=16,
        ).adapt(trained_model, target_data)

    def test_returns_new_object(self, trained_model, adapted):
        assert adapted is not trained_model

    def test_parameters_changed(self, trained_model, adapted):
        any_changed = any(
            not torch.equal(p_o, p_a)
            for p_o, p_a in zip(trained_model.parameters(), adapted.parameters())
        )
        assert any_changed, "CoDATS did not update any model parameters"

    def test_valid_proba_on_target(self, adapted, target_data):
        _assert_valid_proba(adapted.predict_proba(target_data), "CoDATS/target")

    def test_valid_proba_on_source(self, adapted, source_data):
        X_s, _ = source_data
        _assert_valid_proba(adapted.predict_proba(X_s), "CoDATS/source")

    def test_no_collapse_on_source(self, adapted, source_data):
        X_s, _ = source_data
        _assert_no_collapse(adapted.predict_proba(X_s), "CoDATS")

    def test_original_unchanged(self, trained_model, source_data, target_data):
        X_s, y_s = source_data
        params_snap = {n: p.clone() for n, p in trained_model.named_parameters()}
        CoDATS(X_s, y_s, n_epochs=1, batch_size=16).adapt(trained_model, target_data)
        for n, p in trained_model.named_parameters():
            assert torch.equal(p, params_snap[n]), f"Original mutated at '{n}'"


# ---------------------------------------------------------------------------
# M2N2
# ---------------------------------------------------------------------------

class TestM2N2:

    @pytest.fixture(scope="class")
    def adapted(self, trained_model, target_data):
        return M2N2(n_steps=15, lr=1e-3, batch_size=16).adapt(trained_model, target_data)

    def test_returns_new_object(self, trained_model, adapted):
        assert adapted is not trained_model

    def test_only_norm_params_changed(self, trained_model, adapted):
        """Every non-BN/LN parameter must be bitwise-identical after adaptation."""
        allowed = _norm_param_names(trained_model)
        orig = dict(trained_model.named_parameters())
        adpt = dict(adapted.named_parameters())
        for name, p_orig in orig.items():
            if name in allowed:
                continue   # BN/LN affine: allowed to change
            assert torch.equal(p_orig, adpt[name]), (
                f"M2N2 modified non-norm parameter '{name}' — only BN/LN "
                "affine params (γ, β) should be updated"
            )

    def test_some_norm_params_changed(self, trained_model, adapted):
        """At least one BN/LN affine parameter must differ after adaptation."""
        allowed = _norm_param_names(trained_model)
        orig = dict(trained_model.named_parameters())
        adpt = dict(adapted.named_parameters())
        any_changed = any(
            not torch.equal(orig[n], adpt[n]) for n in allowed if n in orig
        )
        assert any_changed, "M2N2 did not update any BN/LN affine parameters"

    def test_valid_proba_on_target(self, adapted, target_data):
        _assert_valid_proba(adapted.predict_proba(target_data), "M2N2/target")

    def test_valid_proba_on_source(self, adapted, source_data):
        X_s, _ = source_data
        _assert_valid_proba(adapted.predict_proba(X_s), "M2N2/source")

    def test_no_collapse_on_source(self, adapted, source_data):
        X_s, _ = source_data
        _assert_no_collapse(adapted.predict_proba(X_s), "M2N2")

    def test_original_unchanged(self, trained_model, target_data):
        params_snap = {n: p.clone() for n, p in trained_model.named_parameters()}
        M2N2(n_steps=5, lr=1e-3, batch_size=16).adapt(trained_model, target_data)
        for n, p in trained_model.named_parameters():
            assert torch.equal(p, params_snap[n]), f"Original mutated at '{n}'"

    def test_all_grads_restored(self, trained_model, target_data):
        """After adapt(), every parameter must have requires_grad=True again."""
        adapted = M2N2(n_steps=5, lr=1e-3, batch_size=16).adapt(trained_model, target_data)
        for name, p in adapted.named_parameters():
            assert p.requires_grad, \
                f"Parameter '{name}' still frozen after adapt() returned"

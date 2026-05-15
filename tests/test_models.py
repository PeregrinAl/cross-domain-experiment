"""Tests for benchmark model implementations.

Coverage
--------
1. Forward-pass output shape  — ``(B, 2)`` logits for every model.
2. ``get_features`` shape     — always ``(B, 128)`` after the projector.
3. ``get_features → head → predict_proba`` equivalence:
       model.head(torch.from_numpy(feats)).softmax(-1) ≈ model.predict_proba(X)
4. ``predict`` shape          — argmax of ``predict_proba``.
5. Cross-model feature-dim invariant — all three models return 128-dim features.
6. ``configs/compatibility.yaml`` is a valid YAML with the expected keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nstad_bench.models import InceptionTime1D, PatchTST, ResNet18_2D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parents[1] / "configs"
KNOWN_REPRS = {"RawSignal", "LogSTFT", "CWT_Morlet", "CARLA_SSL"}
KNOWN_MODELS = {"InceptionTime1D", "ResNet18_2D", "PatchTST"}

BATCH = 4  # small batch — keeps tests fast on CPU


def _rng_array(*shape: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# InceptionTime1D — 1-D univariate input  (N, T)
# ---------------------------------------------------------------------------

class TestInceptionTime1D:
    T = 256  # time-series length

    @pytest.fixture(scope="class")
    def model(self):
        m = InceptionTime1D(in_channels=1, nb_filters=16, bottleneck=16, depth=2)
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def X_1d(self):
        return _rng_array(BATCH, self.T)

    @pytest.fixture(scope="class")
    def X_mc(self):
        return _rng_array(BATCH, 2, self.T)  # 2-channel multivariate

    # -- forward shape -------------------------------------------------------

    def test_forward_shape_univariate(self, model, X_1d):
        t = torch.from_numpy(X_1d[:, None, :])   # (B, 1, T)
        with torch.no_grad():
            out = model(t)
        assert out.shape == (BATCH, 2), f"expected (B, 2), got {out.shape}"

    def test_forward_shape_multivariate(self, X_mc):
        """Model with in_channels=2 must accept 2-channel input."""
        mc_model = InceptionTime1D(in_channels=2, nb_filters=16, bottleneck=16, depth=2)
        mc_model.eval()
        t = torch.from_numpy(X_mc)               # (B, 2, T)
        with torch.no_grad():
            out = mc_model(t)
        assert out.shape == (BATCH, 2)

    # -- get_features shape --------------------------------------------------

    def test_get_features_shape_1d(self, model, X_1d):
        feats = model.get_features(X_1d)
        assert feats.shape == (BATCH, 128), f"expected (B, 128), got {feats.shape}"

    # -- get_features → head → predict_proba equivalence --------------------

    def test_features_head_equivalence(self, model, X_1d):
        feats = model.get_features(X_1d)          # (B, d_feat) numpy
        feats_t = torch.from_numpy(feats)
        with torch.no_grad():
            proba_from_head = torch.softmax(model.head(feats_t), dim=-1).numpy()
        proba_direct = model.predict_proba(X_1d)
        np.testing.assert_allclose(
            proba_from_head, proba_direct, atol=1e-5,
            err_msg="get_features → head ≠ predict_proba"
        )

    # -- predict shape -------------------------------------------------------

    def test_predict_shape(self, model, X_1d):
        preds = model.predict(X_1d)
        assert preds.shape == (BATCH,)
        assert set(preds).issubset({0, 1})

    # -- CARLA-style embedding input (N, embed_dim) --------------------------

    def test_forward_embedding_input(self, model):
        """InceptionTime1D must accept (N, embed_dim) CARLA embeddings."""
        X_emb = _rng_array(BATCH, 64)   # (N, embed_dim) → treated as (N,T)
        proba = model.predict_proba(X_emb)
        assert proba.shape == (BATCH, 2)


# ---------------------------------------------------------------------------
# ResNet18_2D — 2-D spectrogram input  (N, F, T)
# ---------------------------------------------------------------------------

class TestResNet18_2D:
    F, T = 64, 128  # spectrogram height × width

    @pytest.fixture(scope="class")
    def model(self):
        m = ResNet18_2D(in_channels=1)
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def X_2d(self):
        return _rng_array(BATCH, self.F, self.T)  # (N, F, T)

    @pytest.fixture(scope="class")
    def X_4d(self):
        return _rng_array(BATCH, 1, self.F, self.T)  # already (N,1,F,T)

    # -- forward shape -------------------------------------------------------

    def test_forward_shape_3d(self, model, X_2d):
        t = torch.from_numpy(X_2d[:, None, :, :])  # (B, 1, F, T)
        with torch.no_grad():
            out = model(t)
        assert out.shape == (BATCH, 2)

    def test_forward_shape_4d(self, model, X_4d):
        t = torch.from_numpy(X_4d)
        with torch.no_grad():
            out = model(t)
        assert out.shape == (BATCH, 2)

    # -- get_features shape --------------------------------------------------

    def test_get_features_shape(self, model, X_2d):
        feats = model.get_features(X_2d)
        assert feats.shape == (BATCH, 128), f"expected (B, 128), got {feats.shape}"

    # -- equivalence ---------------------------------------------------------

    def test_features_head_equivalence(self, model, X_2d):
        feats = model.get_features(X_2d)
        feats_t = torch.from_numpy(feats)
        with torch.no_grad():
            proba_from_head = torch.softmax(model.head(feats_t), dim=-1).numpy()
        proba_direct = model.predict_proba(X_2d)
        np.testing.assert_allclose(
            proba_from_head, proba_direct, atol=1e-5,
            err_msg="get_features → head ≠ predict_proba"
        )

    # -- predict shape -------------------------------------------------------

    def test_predict_shape(self, model, X_2d):
        preds = model.predict(X_2d)
        assert preds.shape == (BATCH,)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# PatchTST — 1-D time-series input  (N, T)
# ---------------------------------------------------------------------------

class TestPatchTST:
    T = 128  # must satisfy (T - patch_len) % stride == 0 is not required, but must be >= patch_len

    @pytest.fixture(scope="class")
    def model(self):
        # Small config so the test stays fast
        m = PatchTST(
            in_channels=1,
            seq_len=self.T,
            patch_len=16,
            stride=8,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def X_1d(self):
        return _rng_array(BATCH, self.T)

    @pytest.fixture(scope="class")
    def X_mc(self):
        return _rng_array(BATCH, 3, self.T)  # 3-channel multivariate

    # -- forward shape -------------------------------------------------------

    def test_forward_shape_univariate(self, model, X_1d):
        t = torch.from_numpy(X_1d[:, None, :])   # (B, 1, T)
        with torch.no_grad():
            out = model(t)
        assert out.shape == (BATCH, 2)

    def test_forward_shape_multivariate(self, X_mc):
        """Model with in_channels=3 must accept 3-channel input."""
        mc_model = PatchTST(
            in_channels=3,
            seq_len=self.T,
            patch_len=16,
            stride=8,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        mc_model.eval()
        t = torch.from_numpy(X_mc)
        with torch.no_grad():
            out = mc_model(t)
        assert out.shape == (BATCH, 2)

    # -- get_features shape --------------------------------------------------

    def test_get_features_shape(self, model, X_1d):
        feats = model.get_features(X_1d)
        assert feats.shape == (BATCH, 128), f"expected (B, 128), got {feats.shape}"

    # -- equivalence ---------------------------------------------------------

    def test_features_head_equivalence(self, model, X_1d):
        feats = model.get_features(X_1d)
        feats_t = torch.from_numpy(feats)
        with torch.no_grad():
            proba_from_head = torch.softmax(model.head(feats_t), dim=-1).numpy()
        proba_direct = model.predict_proba(X_1d)
        np.testing.assert_allclose(
            proba_from_head, proba_direct, atol=1e-5,
            err_msg="get_features → head ≠ predict_proba"
        )

    # -- predict shape -------------------------------------------------------

    def test_predict_shape(self, model, X_1d):
        preds = model.predict(X_1d)
        assert preds.shape == (BATCH,)
        assert set(preds).issubset({0, 1})

    # -- zero-padding for inputs shorter than seq_len ------------------------

    def test_short_input_is_zero_padded(self):
        """Input shorter than seq_len must be right-padded with zeros.

        Consistent with LogSTFT, which pads to n_fft before computing STFT.
        The model must produce the same output shape as with a full-length input.
        """
        SEQ_LEN = 128
        T_short = 48          # strictly less than seq_len
        m = PatchTST(
            in_channels=1, seq_len=SEQ_LEN,
            patch_len=16, stride=8, d_model=32, n_heads=4, n_layers=1,
            dropout=0.0,
        )
        m.eval()
        X_short = _rng_array(BATCH, T_short)
        # Must not crash and must return the standard output shape
        proba = m.predict_proba(X_short)
        assert proba.shape == (BATCH, 2)
        feats = m.get_features(X_short)
        assert feats.shape == (BATCH, 128)

    def test_padded_output_equals_manual_pad(self):
        """_preprocess zero-pad must produce the same result as padding manually."""
        SEQ_LEN = 128
        T_short = 60
        m = PatchTST(
            in_channels=1, seq_len=SEQ_LEN,
            patch_len=16, stride=8, d_model=32, n_heads=4, n_layers=1,
            dropout=0.0,
        )
        m.eval()
        X_short = _rng_array(BATCH, T_short)
        # Manually right-pad to seq_len, then run through the model
        X_padded = np.pad(X_short, [(0, 0), (0, SEQ_LEN - T_short)])
        proba_auto  = m.predict_proba(X_short)   # padding done inside _preprocess
        proba_manual = m.predict_proba(X_padded) # already full-length
        np.testing.assert_allclose(
            proba_auto, proba_manual, atol=1e-5,
            err_msg="_preprocess padding ≠ manual zero-padding"
        )


# ---------------------------------------------------------------------------
# Cross-model feature-dimension invariant
# ---------------------------------------------------------------------------

class TestFeatureDimInvariant:
    """All models must expose the same 128-dimensional feature space."""

    def test_inception_features_dim(self):
        m = InceptionTime1D(in_channels=1, nb_filters=32, bottleneck=32, depth=3)
        m.eval()
        feats = m.get_features(_rng_array(2, 256))
        assert feats.shape == (2, 128)

    def test_resnet_features_dim(self):
        m = ResNet18_2D(in_channels=1)
        m.eval()
        feats = m.get_features(_rng_array(2, 64, 128))
        assert feats.shape == (2, 128)

    def test_patchtst_features_dim(self):
        m = PatchTST(in_channels=1, seq_len=128, patch_len=16, stride=8, d_model=256)
        m.eval()
        feats = m.get_features(_rng_array(2, 128))
        assert feats.shape == (2, 128)

    def test_all_dims_equal(self):
        """Sanity-check: feature vectors from different backbones are
        interchangeable in shape."""
        X_1d = _rng_array(3, 128)
        X_2d = _rng_array(3, 32, 64)

        dims = [
            InceptionTime1D().eval().get_features(X_1d).shape[1],
            ResNet18_2D().eval().get_features(X_2d).shape[1],
            PatchTST(seq_len=128).eval().get_features(X_1d).shape[1],
        ]
        assert len(set(dims)) == 1 and dims[0] == 128, (
            f"Feature dims differ across models: {dims}"
        )


# ---------------------------------------------------------------------------
# Compatibility YAML
# ---------------------------------------------------------------------------

class TestCompatibilityYaml:
    @pytest.fixture(scope="class")
    def compat(self):
        path = CONFIGS_DIR / "compatibility.yaml"
        assert path.exists(), f"Missing {path}"
        with path.open() as f:
            data = yaml.safe_load(f)
        return data["compatibility"]

    def test_all_representations_present(self, compat):
        assert set(compat.keys()) == KNOWN_REPRS, (
            f"Missing/extra representations: {set(compat.keys()) ^ KNOWN_REPRS}"
        )

    def test_all_models_listed_per_repr(self, compat):
        for repr_name, models in compat.items():
            assert set(models.keys()) == KNOWN_MODELS, (
                f"repr '{repr_name}': missing/extra models: "
                f"{set(models.keys()) ^ KNOWN_MODELS}"
            )

    def test_values_are_booleans(self, compat):
        for repr_name, models in compat.items():
            for model_name, val in models.items():
                assert isinstance(val, bool), (
                    f"compat[{repr_name!r}][{model_name!r}] = {val!r}, expected bool"
                )

    def test_2d_models_not_used_with_1d_reprs(self, compat):
        """ResNet18_2D must be False for 1-D representations."""
        for repr_name in ("RawSignal", "CARLA_SSL"):
            assert not compat[repr_name]["ResNet18_2D"], (
                f"ResNet18_2D should not be compatible with {repr_name}"
            )

    def test_2d_reprs_not_used_with_1d_models(self, compat):
        """1-D models must be False for 2-D representations."""
        for repr_name in ("LogSTFT", "CWT_Morlet"):
            for model_name in ("InceptionTime1D", "PatchTST"):
                assert not compat[repr_name][model_name], (
                    f"{model_name} should not be compatible with {repr_name}"
                )

    def test_at_least_one_valid_pair_per_repr(self, compat):
        for repr_name, models in compat.items():
            assert any(models.values()), (
                f"Representation '{repr_name}' has no compatible model"
            )

    def test_carla_patchtst_is_valid(self, compat):
        """CARLA embeddings are 1-D sequences; PatchTST is a valid consumer."""
        assert compat["CARLA_SSL"]["PatchTST"], (
            "CARLA_SSL × PatchTST should be marked compatible"
        )

    def test_carla_patchtst_forward_smoke(self):
        """PatchTST must accept a CARLA-style (N, embed_dim) array end-to-end."""
        embed_dim = 64
        # seq_len must match embed_dim because _preprocess treats embed_dim as T
        m = PatchTST(
            in_channels=1, seq_len=embed_dim,
            patch_len=8, stride=4, d_model=32, n_heads=4, n_layers=1,
        )
        m.eval()
        X_emb = _rng_array(BATCH, embed_dim)   # CARLA output shape
        proba = m.predict_proba(X_emb)
        assert proba.shape == (BATCH, 2)
        feats = m.get_features(X_emb)
        assert feats.shape == (BATCH, 128)

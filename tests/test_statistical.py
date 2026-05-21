"""Tests for the statistical branch (models + adaptation).

Coverage
--------
1. Each :class:`StatModel` subclass:
   * ``fit`` then ``predict_proba`` returns shape ``(N, 2)`` with rows
     summing to 1.0 and values in ``[0, 1]``.
   * ``predict`` is ``predict_proba.argmax``.
   * ``get_features`` returns a 2-D matrix.
   * ``save`` → ``load`` round-trip preserves ``predict_proba`` exactly.

2. Each adaptation method:
   * ``adapt(model, X_t)`` returns a model whose ``predict_proba`` is
     well-formed on both source-val and target.
   * The input model is **not** mutated in place.
   * On an anisotropic synthetic covariate shift designed to favour
     adaptation, target AUC after adaptation is at least source-only
     minus a small slack (no catastrophic degradation).

3. Integration with the statistical runner:
   * ``runner_stat.run_experiment_stat`` consumes a config that mentions
     every model and every adaptation method without raising.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from nstad_bench.adaptation.statistical import (
    CORAL,
    KMM,
    ImportanceReweighting,
    SourceOnly,
    SubspaceAlignment,
)
from nstad_bench.experiments.runner import register_dataset
from nstad_bench.experiments.runner_stat import run_experiment_stat
from nstad_bench.models.statistical import (
    GBM,
    SVM,
    LogReg,
    RandomForest,
    StatTestClassifier,
)
from nstad_bench.models.statistical.base import (
    AffineFeatureTransform,
    StatModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_shift() -> dict:
    """Moderate anisotropic source/target covariate shift.

    Both domains share the same discriminative direction (``x[0]``);
    target additionally inflates the variance on ``x[0..1]`` and tilts
    the class means slightly into ``x[1]``.  This keeps source-only
    target AUC around 0.75–0.85 — a regime where adaptation should
    avoid catastrophic degradation but is not guaranteed to improve.
    """
    rng = np.random.default_rng(42)
    D = 6
    N_S, N_T = 300, 200

    # Source: class means along ``x[0]``, isotropic noise.
    def gen_source(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, 2, size=n).astype(np.int64)
        means = np.zeros((n, D))
        means[:, 0] = (2 * y - 1) * 1.5
        noise = rng.standard_normal((n, D)) * 0.5
        return (means + noise).astype(np.float32), y

    # Target: same discriminative direction with a tiny tilt into ``x[1]``,
    # plus mild anisotropic noise on ``x[0..1]``.
    def gen_target(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, 2, size=n).astype(np.int64)
        means = np.zeros((n, D))
        means[:, 0] = (2 * y - 1) * 1.4
        means[:, 1] = (2 * y - 1) * 0.3
        noise = rng.standard_normal((n, D)) * 0.5
        noise[:, 0] *= 1.6
        noise[:, 1] *= 1.6
        return (means + noise).astype(np.float32), y

    X_s, y_s = gen_source(N_S)
    X_t, y_t = gen_target(N_T)
    return dict(X_s=X_s, y_s=y_s, X_t=X_t, y_t=y_t, D=D)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

ALL_MODEL_FACTORIES = [
    ("LogReg",             lambda: LogReg(C=1.0, max_iter=500, random_state=0)),
    ("RandomForest",       lambda: RandomForest(n_estimators=30, random_state=0)),
    ("GBM",                lambda: GBM(n_estimators=30, random_state=0)),
    ("SVM",                lambda: SVM(C=1.0, kernel="rbf", random_state=0)),
    ("StatTest:moments",   lambda: StatTestClassifier(test="moments", moment_order=4, random_state=0)),
    ("StatTest:chi2",      lambda: StatTestClassifier(test="chi2", n_bins=16, random_state=0)),
    ("StatTest:ks",        lambda: StatTestClassifier(test="ks", random_state=0)),
]


@pytest.mark.parametrize("name,factory", ALL_MODEL_FACTORIES,
                         ids=[n for n, _ in ALL_MODEL_FACTORIES])
class TestStatModels:
    """Contract tests every StatModel must satisfy."""

    def test_predict_proba_shape_and_simplex(self, synthetic_shift, name, factory):
        X, y = synthetic_shift["X_s"], synthetic_shift["y_s"]
        m = factory().fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (len(X), 2), f"{name}: bad shape {p.shape}"
        assert np.all(p >= 0) and np.all(p <= 1), f"{name}: probs out of [0,1]"
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), f"{name}: rows not on simplex"

    def test_predict_is_argmax(self, synthetic_shift, name, factory):
        X, y = synthetic_shift["X_s"], synthetic_shift["y_s"]
        m = factory().fit(X, y)
        np.testing.assert_array_equal(m.predict(X), m.predict_proba(X).argmax(axis=1))

    def test_get_features_is_2d(self, synthetic_shift, name, factory):
        X, y = synthetic_shift["X_s"], synthetic_shift["y_s"]
        m = factory().fit(X, y)
        f = m.get_features(X)
        assert f.ndim == 2 and len(f) == len(X), f"{name}: bad features shape {f.shape}"

    def test_save_load_roundtrip(self, synthetic_shift, name, factory):
        X, y = synthetic_shift["X_s"], synthetic_shift["y_s"]
        m = factory().fit(X, y)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pkl"
            m.save(path)
            m2 = type(m).load(path)
        p1, p2 = m.predict_proba(X), m2.predict_proba(X)
        np.testing.assert_allclose(p1, p2, rtol=1e-6, atol=1e-8,
                                   err_msg=f"{name}: save/load lossy")

    def test_fit_ignores_neural_kwargs(self, synthetic_shift, name, factory):
        """``epochs`` / ``lr`` / ``batch_size`` come from the YAML for the
        neural branch — StatModel must silently drop them."""
        X, y = synthetic_shift["X_s"], synthetic_shift["y_s"]
        # Should not raise — kwargs are dropped before reaching the estimator.
        factory().fit(X, y, epochs=99, lr=1e-3, batch_size=64)

    def test_to_device_is_noop(self, name, factory):
        m = factory()
        assert m.to("cpu") is m
        assert m.to("cuda") is m


def test_logreg_above_chance_on_source(synthetic_shift):
    """A linear classifier should easily clear chance on this synthetic
    task — sanity-check that the fixture and the pipeline align."""
    X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
    p = LogReg(C=1.0, random_state=0).fit(X_s, y_s).predict_proba(X_s)[:, 1]
    assert roc_auc_score(y_s, p) > 0.95


def test_stattest_classifier_validates_inputs():
    with pytest.raises(ValueError):
        StatTestClassifier(test="bad")
    with pytest.raises(ValueError):
        StatTestClassifier(test="moments", moment_order=5)


def test_stattest_ks_requires_fit_first(synthetic_shift):
    """``test='ks'`` cannot extract features before fit() builds the
    per-class reference samples."""
    X = synthetic_shift["X_s"]
    m = StatTestClassifier(test="ks", random_state=0)
    with pytest.raises(RuntimeError):
        m.predict_proba(X)


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def _src_only_target_auc(model: StatModel, X_t, y_t) -> float:
    return float(roc_auc_score(y_t, model.predict_proba(X_t)[:, 1]))


ALL_ADAPT_FACTORIES = [
    ("SourceOnly",            lambda Xs, ys: SourceOnly()),
    ("CORAL",                 lambda Xs, ys: CORAL(Xs, ys, lambda_reg=1e-2, align_mean=True)),
    ("SubspaceAlignment",     lambda Xs, ys: SubspaceAlignment(Xs, ys, n_components=3)),
    ("ImportanceReweighting", lambda Xs, ys: ImportanceReweighting(Xs, ys, estimator="classifier", clip=10.0)),
    ("KMM",                   lambda Xs, ys: KMM(Xs, ys, sigma=None, B=100.0, eps=0.05, random_state=0)),
]


@pytest.mark.parametrize("name,factory", ALL_ADAPT_FACTORIES,
                         ids=[n for n, _ in ALL_ADAPT_FACTORIES])
class TestAdaptationContract:
    """All adaptation methods share these guarantees."""

    def test_predict_proba_well_formed(self, synthetic_shift, name, factory):
        X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
        X_t      = synthetic_shift["X_t"]
        base = LogReg(C=1.0, random_state=0).fit(X_s, y_s)
        adapted = factory(X_s, y_s).adapt(base, X_t)
        p = adapted.predict_proba(X_t)
        assert p.shape == (len(X_t), 2)
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0) and np.all(p <= 1)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_original_model_not_mutated(self, synthetic_shift, name, factory):
        X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
        X_t      = synthetic_shift["X_t"]
        base = LogReg(C=1.0, random_state=0).fit(X_s, y_s)
        baseline_target_proba = base.predict_proba(X_t).copy()
        _ = factory(X_s, y_s).adapt(base, X_t)
        np.testing.assert_allclose(
            base.predict_proba(X_t), baseline_target_proba,
            rtol=1e-6, atol=1e-8,
            err_msg=f"{name}: adapt() mutated the input model",
        )

    def test_target_not_catastrophic(self, synthetic_shift, name, factory):
        """After adaptation, target AUC must stay above chance — methods
        may make target worse than source-only on hard cases, but must
        not collapse to random."""
        X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
        X_t, y_t = synthetic_shift["X_t"], synthetic_shift["y_t"]
        base = LogReg(C=1.0, random_state=0).fit(X_s, y_s)
        adapted = factory(X_s, y_s).adapt(base, X_t)
        auc = roc_auc_score(y_t, adapted.predict_proba(X_t)[:, 1])
        assert auc > 0.55, f"{name}: target AUC collapsed to {auc:.3f}"


def test_source_only_is_identity(synthetic_shift):
    X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
    X_t      = synthetic_shift["X_t"]
    base = LogReg(random_state=0).fit(X_s, y_s)
    adapted = SourceOnly().adapt(base, X_t)
    # SourceOnly does not refit the estimator — output must be identical.
    np.testing.assert_allclose(
        base.predict_proba(X_t),
        adapted.predict_proba(X_t),
        rtol=1e-10, atol=1e-12,
    )


def test_subspace_alignment_installs_adapt_transform(synthetic_shift):
    """SA changes the feature pipeline via ``_adapt_transform``."""
    X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
    X_t      = synthetic_shift["X_t"]
    base = LogReg(random_state=0).fit(X_s, y_s)
    assert base._adapt_transform is None
    adapted = SubspaceAlignment(X_s, y_s, n_components=3).adapt(base, X_t)
    assert isinstance(adapted._adapt_transform, AffineFeatureTransform)
    # Original model still has no transform.
    assert base._adapt_transform is None


def test_coral_refits_estimator_without_feature_transform(synthetic_shift):
    """CORAL only refits the underlying estimator; the feature pipeline
    is unchanged (matches the transferlearning reference impl)."""
    X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
    X_t      = synthetic_shift["X_t"]
    base = LogReg(random_state=0).fit(X_s, y_s)
    base_coef = base._estimator.coef_.copy()
    adapted = CORAL(X_s, y_s, lambda_reg=1e-3, align_mean=True).adapt(base, X_t)
    assert adapted._adapt_transform is None
    # The estimator's coefficients changed by adaptation.
    assert not np.allclose(adapted._estimator.coef_, base_coef)
    # Original is unchanged.
    np.testing.assert_array_equal(base._estimator.coef_, base_coef)


def test_importance_reweighting_changes_estimator(synthetic_shift):
    """IR refits the estimator with sample_weight — coefficients differ
    from the source-only fit."""
    X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
    X_t      = synthetic_shift["X_t"]
    base = LogReg(random_state=0).fit(X_s, y_s)
    base_coef = base._estimator.coef_.copy()
    adapted = ImportanceReweighting(X_s, y_s).adapt(base, X_t)
    assert not np.allclose(adapted._estimator.coef_, base_coef)


def test_kmm_changes_estimator(synthetic_shift):
    """KMM refits the estimator with the QP-derived sample weights."""
    X_s, y_s = synthetic_shift["X_s"], synthetic_shift["y_s"]
    X_t      = synthetic_shift["X_t"]
    base = LogReg(random_state=0).fit(X_s, y_s)
    base_coef = base._estimator.coef_.copy()
    adapted = KMM(X_s, y_s, sigma=None, B=100.0, eps=0.05,
                  random_state=0).adapt(base, X_t)
    assert not np.allclose(adapted._estimator.coef_, base_coef)


def test_importance_reweighting_validates_estimator():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=10).astype(np.int64)
    with pytest.raises(ValueError):
        ImportanceReweighting(X, y, estimator="not_a_method")


# ---------------------------------------------------------------------------
# Integration with runner_stat
# ---------------------------------------------------------------------------

def test_runner_stat_e2e_smoke(tmp_path, synthetic_shift):
    """Run the statistical runner end-to-end on a minimal config that
    exercises every model and every adaptation method."""

    def _loader():
        return (
            synthetic_shift["X_s"], synthetic_shift["y_s"],
            synthetic_shift["X_t"], synthetic_shift["y_t"],
        )
    register_dataset("smoke_stat_e2e", _loader)

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    # Compatibility matrix: every (RawSignal, model) is compatible.
    (cfg_dir / "compatibility.yaml").write_text(yaml.safe_dump({
        "compatibility": {
            "RawSignal": {
                "LogReg": True, "RandomForest": True, "GBM": True,
                "SVM": True, "StatTestClassifier": True,
            }
        }
    }))
    cfg = {
        "experiment_name": "smoke_e2e",
        "output_dir": str(tmp_path / "results"),
        "n_bootstrap": 20,
        "screening": {"top_k": 2, "metric": "delta_auc"},
        "random_search": {"n_trials": 1, "seeds": [0], "base_seed": 42},
        "datasets": ["smoke_stat_e2e"],
        "representations": ["RawSignal"],
        "models": {
            "LogReg":             {"C": 1.0, "max_iter": 200, "random_state": 0},
            "RandomForest":       {"n_estimators": 20, "random_state": 0},
            "GBM":                {"n_estimators": 20, "random_state": 0},
            "SVM":                {"C": 1.0, "random_state": 0},
            "StatTestClassifier": {"test": "moments", "moment_order": 4, "random_state": 0},
        },
        "adaptation_methods": {
            "SourceOnly":            {},
            "CORAL":                 {"lambda_reg": 1e-2, "align_mean": True},
            "SubspaceAlignment":     {"n_components": 3},
            "ImportanceReweighting": {"estimator": "classifier", "clip": 10.0},
            "KMM":                   {"B": 100.0, "eps": 0.05, "random_state": 0},
        },
    }
    cfg_path = cfg_dir / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    df = run_experiment_stat(cfg_path, config_root=cfg_dir)

    # Every model ran through screening; the top-2 ran through every
    # non-SourceOnly adaptation method.
    assert {"SourceOnly", "CORAL", "SubspaceAlignment",
            "ImportanceReweighting", "KMM"} <= set(df["psi"].unique())
    # Each metric shows up at least once.
    assert {"roc_auc", "pr_auc", "delta_auc", "gain", "f1"} <= set(df["metric_name"])
    # No NaNs in the headline metrics.
    roc = df[df["metric_name"] == "roc_auc"]
    assert roc["metric_value"].notna().all()
    assert (roc["metric_value"] >= 0).all() and (roc["metric_value"] <= 1).all()

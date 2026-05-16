"""Tests for nstad_bench.experiments.runner.

Design principles
-----------------
* No real datasets, no file I/O, no network calls.
* A tiny synthetic dataset (N=80, T=64, binary) is used throughout.
* InceptionTime1D with the smallest possible config (nb_filters=4, depth=1)
  and very few training epochs keeps each _run_single call < 3 s on CPU.
* All YAML configs are provided as in-memory strings (io.StringIO / tmp file).

Coverage
--------
_config_hash            determinism, collision-resistance
_sample_hp              all four spec types, fixed values, edge cases
RunConfig               hash stability, hp_dict round-trip
_screening_configs      compat filter, correct psi="SourceOnly"
_adaptation_configs     HP-trial expansion, SourceOnly = 1 trial
_select_top_k           ranking, k clipping, fallback on empty metric
_add_gain_rows          gain = adapted - source_only, source_only gain = 0
_run_single             smoke test (fast model, real pipeline)
run_experiment          dry_run mode + end-to-end with tmp dir
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ── Imports under test ────────────────────────────────────────────────────────
from nstad_bench.experiments.runner import (
    RESULT_COLS,
    RunConfig,
    _adaptation_configs,
    _add_gain_rows,
    _config_hash,
    _run_single,
    _sample_hp,
    _screening_configs,
    _select_top_k,
    register_dataset,
    run_experiment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_data(seed: int = 0) -> tuple[np.ndarray, ...]:
    """Return (X_s, y_s, X_t, y_t) — tiny binary, two separable classes."""
    rng  = np.random.default_rng(seed)
    T    = 64
    n    = 40   # 20 per class
    t    = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    X0_s = rng.normal(0.0, 0.15, (n // 2, T)).astype(np.float32)
    X1_s = (np.sin(t) + rng.normal(0.0, 0.15, (n // 2, T))).astype(np.float32)
    X_s  = np.vstack([X0_s, X1_s])
    y_s  = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    # Target: mean-shifted version
    X0_t = rng.normal(0.3, 0.15, (n // 2, T)).astype(np.float32)
    X1_t = (np.sin(t) + 0.3 + rng.normal(0.0, 0.15, (n // 2, T))).astype(np.float32)
    X_t  = np.vstack([X0_t, X1_t])
    y_t  = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    return X_s, y_s, X_t, y_t


@pytest.fixture(autouse=True)
def _register_test_dataset():
    """Register 'test_ds' in the data registry for every test."""
    register_dataset("test_ds", _make_data)
    yield


@pytest.fixture()
def compat() -> dict:
    """Minimal compatibility mask: only RawSignal × InceptionTime1D."""
    return {
        "RawSignal": {"InceptionTime1D": True, "ResNet18_2D": False, "PatchTST": False},
        "LogSTFT":   {"InceptionTime1D": False, "ResNet18_2D": True,  "PatchTST": False},
    }


def _minimal_exp_cfg(
    *,
    top_k: int = 1,
    n_trials: int = 2,
    seeds: list = None,
    datasets: list = None,
    phis: list = None,
    thetas: list = None,
    psi_spaces: dict = None,
) -> dict:
    """Build a minimal experiment config dict for testing."""
    return {
        "experiment_name":   "test_exp",
        "output_dir":        "results/",
        "n_bootstrap":       50,
        "screening": {"top_k": top_k, "metric": "delta_auc"},
        "random_search":     {"n_trials": n_trials, "seeds": seeds or [0], "base_seed": 0},
        "datasets":          datasets or ["test_ds"],
        "representations":   phis or ["RawSignal"],
        "models":            {t: {"epochs": 1, "lr": 1e-3, "batch_size": 16, "nb_filters": 4, "depth": 1}
                              for t in (thetas or ["InceptionTime1D"])},
        "adaptation_methods": psi_spaces or {"SourceOnly": {}},
    }


# ─────────────────────────────────────────────────────────────────────────────
# _config_hash
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigHash:

    def test_deterministic(self):
        d = {"a": 1, "b": [1, 2]}
        assert _config_hash(d) == _config_hash(d)

    def test_key_order_independent(self):
        assert _config_hash({"x": 1, "y": 2}) == _config_hash({"y": 2, "x": 1})

    def test_different_dicts_differ(self):
        assert _config_hash({"a": 1}) != _config_hash({"a": 2})

    def test_length_16(self):
        h = _config_hash({"foo": "bar"})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# _sample_hp
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleHp:

    def _rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_log_float_in_range(self):
        spec = {"lr": {"type": "log_float", "low": 1e-5, "high": 1e-3}}
        for _ in range(20):
            hp = _sample_hp(spec, self._rng())
            assert 1e-5 <= hp["lr"] <= 1e-3

    def test_float_in_range(self):
        spec = {"lambda": {"type": "float", "low": 0.1, "high": 5.0}}
        for _ in range(20):
            hp = _sample_hp(spec, self._rng())
            assert 0.1 <= hp["lambda"] <= 5.0

    def test_int_in_range(self):
        spec = {"n_epochs": {"type": "int", "low": 5, "high": 20}}
        values = {_sample_hp(spec, np.random.default_rng(i))["n_epochs"]
                  for i in range(50)}
        assert all(5 <= v <= 20 for v in values)
        assert len(values) > 1, "int sampling should produce different values"

    def test_choice(self):
        spec = {"bs": {"type": "choice", "choices": [16, 32, 64]}}
        values = {_sample_hp(spec, np.random.default_rng(i))["bs"] for i in range(30)}
        assert values <= {16, 32, 64}
        assert len(values) > 1

    def test_fixed_value_passed_through(self):
        spec = {"alphas": [0.5, 1.0, 2.0], "flag": True}
        hp = _sample_hp(spec, self._rng())
        assert hp["alphas"] == [0.5, 1.0, 2.0]
        assert hp["flag"] is True

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown HP spec type"):
            _sample_hp({"x": {"type": "uniform_log"}}, self._rng())

    def test_deterministic_same_rng(self):
        spec = {"lr": {"type": "log_float", "low": 1e-5, "high": 1e-2}}
        hp1 = _sample_hp(spec, np.random.default_rng(99))
        hp2 = _sample_hp(spec, np.random.default_rng(99))
        assert hp1["lr"] == pytest.approx(hp2["lr"])


# ─────────────────────────────────────────────────────────────────────────────
# RunConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestRunConfig:

    def _make(self, **kwargs) -> RunConfig:
        defaults = dict(
            dataset="ds1", phi="RawSignal", theta="InceptionTime1D",
            psi="SourceOnly", seed=0, hp_trial=0, hp_params=(), stage="screening"
        )
        defaults.update(kwargs)
        return RunConfig(**defaults)

    def test_hp_dict_round_trip(self):
        hp = (("batch_size", 32), ("lr", 1e-4))
        cfg = self._make(hp_params=hp)
        assert cfg.hp_dict == {"batch_size": 32, "lr": 1e-4}

    def test_config_hash_stable(self):
        cfg = self._make()
        assert cfg.config_hash == cfg.config_hash

    def test_different_seeds_different_hash(self):
        cfg0 = self._make(seed=0)
        cfg1 = self._make(seed=1)
        assert cfg0.config_hash != cfg1.config_hash

    def test_different_hp_different_hash(self):
        cfg_a = self._make(hp_params=(("lr", 1e-4),))
        cfg_b = self._make(hp_params=(("lr", 5e-4),))
        assert cfg_a.config_hash != cfg_b.config_hash

    def test_str_contains_stage(self):
        cfg = self._make(stage="adaptation")
        assert "adaptation" in str(cfg)

    def test_frozen(self):
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.seed = 999  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# _screening_configs
# ─────────────────────────────────────────────────────────────────────────────

class TestScreeningConfigs:

    def test_psi_is_always_source_only(self, compat):
        cfgs = _screening_configs(_minimal_exp_cfg(), compat)
        assert all(c.psi == "SourceOnly" for c in cfgs)

    def test_stage_is_screening(self, compat):
        cfgs = _screening_configs(_minimal_exp_cfg(), compat)
        assert all(c.stage == "screening" for c in cfgs)

    def test_incompatible_pairs_excluded(self, compat):
        exp = _minimal_exp_cfg(phis=["LogSTFT"], thetas=["InceptionTime1D"])
        cfgs = _screening_configs(exp, compat)
        assert len(cfgs) == 0, "LogSTFT × InceptionTime1D is incompatible"

    def test_compatible_pair_included(self, compat):
        exp = _minimal_exp_cfg(phis=["RawSignal"], thetas=["InceptionTime1D"])
        cfgs = _screening_configs(exp, compat)
        assert len(cfgs) > 0

    def test_count_seeds_x_datasets(self, compat):
        exp = _minimal_exp_cfg(seeds=[0, 1, 2], datasets=["test_ds"])
        cfgs = _screening_configs(exp, compat)
        # 1 phi × 1 theta × 1 dataset × 3 seeds = 3
        assert len(cfgs) == 3

    def test_config_hash_unique_per_seed(self, compat):
        exp  = _minimal_exp_cfg(seeds=[0, 1, 2])
        cfgs = _screening_configs(exp, compat)
        hashes = [c.config_hash for c in cfgs]
        assert len(set(hashes)) == len(hashes), "each seed must give a unique hash"


# ─────────────────────────────────────────────────────────────────────────────
# _adaptation_configs
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptationConfigs:

    def test_source_only_single_trial(self, compat):
        exp  = _minimal_exp_cfg(psi_spaces={"SourceOnly": {}}, n_trials=10)
        cfgs = _adaptation_configs(exp, compat, [("RawSignal", "InceptionTime1D")])
        so   = [c for c in cfgs if c.psi == "SourceOnly"]
        assert len(so) == 1   # 1 dataset × 1 seed × 1 trial

    def test_method_with_hp_gets_n_trials(self, compat):
        psi_spaces = {
            "SourceOnly": {},
            "M2N2": {"n_steps": {"type": "int", "low": 10, "high": 50}},
        }
        exp  = _minimal_exp_cfg(psi_spaces=psi_spaces, n_trials=5)
        cfgs = _adaptation_configs(exp, compat, [("RawSignal", "InceptionTime1D")])
        m2n2 = [c for c in cfgs if c.psi == "M2N2"]
        # 1 dataset × 1 seed × 5 trials = 5
        assert len(m2n2) == 5

    def test_hp_trials_deterministic_across_calls(self, compat):
        psi_spaces = {"MK_MMD": {"lr": {"type": "log_float", "low": 1e-5, "high": 1e-3}}}
        exp  = _minimal_exp_cfg(psi_spaces=psi_spaces, n_trials=3)
        pair = [("RawSignal", "InceptionTime1D")]
        cfgs1 = _adaptation_configs(exp, compat, pair, base_seed=0)
        cfgs2 = _adaptation_configs(exp, compat, pair, base_seed=0)
        lrs1 = [dict(c.hp_params).get("lr") for c in cfgs1 if c.psi == "MK_MMD"]
        lrs2 = [dict(c.hp_params).get("lr") for c in cfgs2 if c.psi == "MK_MMD"]
        assert lrs1 == lrs2, "HP sampling must be deterministic given base_seed"

    def test_different_base_seeds_give_different_hp(self, compat):
        psi_spaces = {"MK_MMD": {"lr": {"type": "log_float", "low": 1e-5, "high": 1e-3}}}
        exp  = _minimal_exp_cfg(psi_spaces=psi_spaces, n_trials=3)
        pair = [("RawSignal", "InceptionTime1D")]
        cfgs_a = _adaptation_configs(exp, compat, pair, base_seed=0)
        cfgs_b = _adaptation_configs(exp, compat, pair, base_seed=7)
        lrs_a = sorted(dict(c.hp_params).get("lr", 0) for c in cfgs_a if c.psi == "MK_MMD")
        lrs_b = sorted(dict(c.hp_params).get("lr", 0) for c in cfgs_b if c.psi == "MK_MMD")
        assert lrs_a != lrs_b

    def test_stage_is_adaptation(self, compat):
        exp  = _minimal_exp_cfg()
        cfgs = _adaptation_configs(exp, compat, [("RawSignal", "InceptionTime1D")])
        assert all(c.stage == "adaptation" for c in cfgs)


# ─────────────────────────────────────────────────────────────────────────────
# _select_top_k
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectTopK:

    def _rows(self, pairs_and_scores: list[tuple[str, str, float]]) -> list[dict]:
        rows = []
        for phi, theta, score in pairs_and_scores:
            rows.append({
                "phi": phi, "theta": theta, "psi": "SourceOnly",
                "dataset": "ds", "seed": 0,
                "metric_name": "delta_auc", "metric_value": score,
                "config_hash": f"{phi}_{theta}",
                "metric_ci_lower": float("nan"),
                "metric_ci_upper": float("nan"),
            })
        return rows

    def test_returns_top_k_highest(self):
        rows = self._rows([
            ("RawSignal", "InceptionTime1D", 0.20),
            ("RawSignal", "PatchTST",        0.05),
            ("LogSTFT",   "ResNet18_2D",     0.15),
        ])
        top = _select_top_k(rows, k=2)
        assert ("RawSignal", "InceptionTime1D") in top
        assert ("LogSTFT",   "ResNet18_2D")     in top
        assert len(top) == 2

    def test_k_clipped_to_available(self):
        rows = self._rows([("A", "B", 0.1)])
        top  = _select_top_k(rows, k=10)
        assert len(top) == 1

    def test_empty_metric_returns_all_pairs(self):
        rows = self._rows([("A", "B", 0.1)])
        # Ask for a metric that doesn't exist
        top = _select_top_k(rows, k=5, metric="nonexistent_metric")
        assert ("A", "B") in top

    def test_averages_across_seeds(self):
        rows = []
        for seed, score in [(0, 0.30), (1, 0.10)]:
            rows.append({
                "phi": "A", "theta": "B", "psi": "SourceOnly",
                "dataset": "ds", "seed": seed,
                "metric_name": "delta_auc", "metric_value": score,
                "config_hash": f"A_B_{seed}",
                "metric_ci_lower": float("nan"),
                "metric_ci_upper": float("nan"),
            })
        rows.append({
            "phi": "C", "theta": "D", "psi": "SourceOnly",
            "dataset": "ds", "seed": 0,
            "metric_name": "delta_auc", "metric_value": 0.19,
            "config_hash": "C_D",
            "metric_ci_lower": float("nan"),
            "metric_ci_upper": float("nan"),
        })
        # (A,B) mean = 0.20; (C,D) mean = 0.19 → (A,B) is top-1
        top = _select_top_k(rows, k=1)
        assert top == [("A", "B")]


# ─────────────────────────────────────────────────────────────────────────────
# _add_gain_rows
# ─────────────────────────────────────────────────────────────────────────────

class TestAddGainRows:

    def _base_df(self) -> pd.DataFrame:
        """Three rows: SourceOnly=0.70, MK_MMD=0.80, CoDATS=0.65 (same run params)."""
        common = dict(dataset="ds", phi="R", theta="M", seed=0,
                      metric_ci_lower=float("nan"), metric_ci_upper=float("nan"))
        rows = [
            {**common, "psi": "SourceOnly", "config_hash": "h0",
             "metric_name": "roc_auc", "metric_value": 0.70},
            {**common, "psi": "MK_MMD",     "config_hash": "h1",
             "metric_name": "roc_auc", "metric_value": 0.80},
            {**common, "psi": "CoDATS",     "config_hash": "h2",
             "metric_name": "roc_auc", "metric_value": 0.65},
        ]
        return pd.DataFrame(rows)[RESULT_COLS]

    def test_gain_rows_added(self):
        df  = _add_gain_rows(self._base_df())
        assert "gain" in df["metric_name"].values

    def test_source_only_gain_is_zero(self):
        df   = _add_gain_rows(self._base_df())
        so_g = df[(df["psi"] == "SourceOnly") & (df["metric_name"] == "gain")]["metric_value"]
        assert so_g.values == pytest.approx(0.0)

    def test_mk_mmd_gain_is_correct(self):
        df = _add_gain_rows(self._base_df())
        g  = df[(df["psi"] == "MK_MMD") & (df["metric_name"] == "gain")]["metric_value"]
        assert g.iloc[0] == pytest.approx(0.10)

    def test_codats_negative_gain(self):
        df = _add_gain_rows(self._base_df())
        g  = df[(df["psi"] == "CoDATS") & (df["metric_name"] == "gain")]["metric_value"]
        assert g.iloc[0] == pytest.approx(-0.05)

    def test_no_gain_without_source_only(self):
        """If no SourceOnly rows present, gain should not be added."""
        df = self._base_df()
        df = df[df["psi"] != "SourceOnly"].copy()
        df_out = _add_gain_rows(df)
        assert "gain" not in df_out["metric_name"].values


# ─────────────────────────────────────────────────────────────────────────────
# _run_single — smoke test (real pipeline, tiny model)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSingle:
    """Lightweight smoke tests: exercises the full _run_single path."""

    def _cfg(self, psi: str = "SourceOnly", hp_params: tuple = ()) -> RunConfig:
        return RunConfig(
            dataset="test_ds", phi="RawSignal", theta="InceptionTime1D",
            psi=psi, seed=0, hp_trial=0, hp_params=hp_params, stage="screening",
        )

    def _train_cfg(self) -> dict:
        return {"InceptionTime1D": {"epochs": 1, "lr": 1e-3, "batch_size": 16,
                                    "nb_filters": 4, "depth": 1}}

    def test_returns_list_of_dicts(self):
        rows = _run_single(self._cfg(), self._train_cfg(), n_bootstrap=20)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_expected_metric_names(self):
        rows = _run_single(self._cfg(), self._train_cfg(), n_bootstrap=20)
        names = {r["metric_name"] for r in rows}
        assert {"roc_auc", "pr_auc", "source_roc_auc", "delta_auc"} <= names

    def test_roc_auc_in_unit_interval(self):
        rows = _run_single(self._cfg(), self._train_cfg(), n_bootstrap=20)
        roc  = next(r["metric_value"] for r in rows if r["metric_name"] == "roc_auc")
        assert 0.0 <= roc <= 1.0

    def test_ci_ordering(self):
        """CI must satisfy lower ≤ estimate ≤ upper."""
        rows = _run_single(self._cfg(), self._train_cfg(), n_bootstrap=20)
        roc  = next(r for r in rows if r["metric_name"] == "roc_auc")
        assert roc["metric_ci_lower"] <= roc["metric_value"] <= roc["metric_ci_upper"]

    def test_schema_keys_present(self):
        rows = _run_single(self._cfg(), self._train_cfg(), n_bootstrap=20)
        for r in rows:
            for col in RESULT_COLS:
                assert col in r, f"Missing column {col!r}"

    def test_unregistered_dataset_raises(self):
        cfg = RunConfig(
            dataset="does_not_exist", phi="RawSignal", theta="InceptionTime1D",
            psi="SourceOnly", seed=0, hp_trial=0, hp_params=(), stage="screening",
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            _run_single(cfg, self._train_cfg(), n_bootstrap=10)

    def test_m2n2_pipeline(self):
        hp  = (("batch_size", 16), ("lr", 1e-4), ("n_steps", 5))
        cfg = self._cfg(psi="M2N2", hp_params=hp)
        rows = _run_single(cfg, self._train_cfg(), n_bootstrap=20)
        assert any(r["metric_name"] == "roc_auc" for r in rows)


# ─────────────────────────────────────────────────────────────────────────────
# run_experiment — dry_run + integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunExperiment:

    def _write_yaml(self, tmp_path: Path, cfg: dict) -> Path:
        p = tmp_path / "experiment.yaml"
        p.write_text(yaml.dump(cfg))
        return p

    def test_dry_run_returns_empty_df(self, tmp_path):
        exp  = _minimal_exp_cfg()
        exp["output_dir"] = str(tmp_path / "results")
        p    = self._write_yaml(tmp_path, exp)
        compat_root = Path(__file__).parents[1] / "configs"
        df   = run_experiment(p, config_root=compat_root, dry_run=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_dry_run_creates_no_files(self, tmp_path):
        exp = _minimal_exp_cfg()
        exp["output_dir"] = str(tmp_path / "results")
        p   = self._write_yaml(tmp_path, exp)
        compat_root = Path(__file__).parents[1] / "configs"
        run_experiment(p, config_root=compat_root, dry_run=True)
        assert not (tmp_path / "results").exists()

    def test_end_to_end_creates_parquet(self, tmp_path):
        """Full two-stage run: 1 dataset, RawSignal × InceptionTime1D, SourceOnly only."""
        exp = _minimal_exp_cfg(
            top_k=1,
            n_trials=1,
            seeds=[0],
            psi_spaces={"SourceOnly": {}},
        )
        exp["output_dir"] = str(tmp_path / "results")
        p   = self._write_yaml(tmp_path, exp)
        compat_root = Path(__file__).parents[1] / "configs"
        df  = run_experiment(p, config_root=compat_root)
        parquet = tmp_path / "results" / "test_exp.parquet"
        assert parquet.exists(), "Parquet file must be created"
        loaded = pd.read_parquet(parquet)
        assert len(loaded) > 0

    def test_end_to_end_result_schema(self, tmp_path):
        exp = _minimal_exp_cfg(top_k=1, n_trials=1, seeds=[0])
        exp["output_dir"] = str(tmp_path / "results")
        p   = self._write_yaml(tmp_path, exp)
        compat_root = Path(__file__).parents[1] / "configs"
        df  = run_experiment(p, config_root=compat_root)
        for col in RESULT_COLS:
            assert col in df.columns, f"Missing column {col!r} in result DataFrame"

    def test_end_to_end_gain_present(self, tmp_path):
        """Stage 2 with one adaptation method → gain rows must appear."""
        psi_spaces = {
            "SourceOnly": {},
            "M2N2": {"n_steps": {"type": "int", "low": 3, "high": 5}},
        }
        exp = _minimal_exp_cfg(
            top_k=1, n_trials=1, seeds=[0], psi_spaces=psi_spaces,
        )
        exp["output_dir"] = str(tmp_path / "results")
        p   = self._write_yaml(tmp_path, exp)
        compat_root = Path(__file__).parents[1] / "configs"
        df  = run_experiment(p, config_root=compat_root)
        assert "gain" in df["metric_name"].values, (
            "Expected 'gain' rows in output DataFrame"
        )

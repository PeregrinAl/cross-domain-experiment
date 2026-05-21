"""Tests for nstad_bench.data._paths — DATA_ROOT precedence + OUTPUT_ROOT."""

from __future__ import annotations

from pathlib import Path

import pytest

from nstad_bench.data._paths import resolve_data_root, resolve_output_root


def test_data_root_takes_precedence_over_nstad(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "kaggle"))
    monkeypatch.setenv("NSTAD_DATA_ROOT", str(tmp_path / "local"))
    assert resolve_data_root("mitbih") == tmp_path / "kaggle" / "mitbih"


def test_falls_back_to_nstad_when_data_root_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setenv("NSTAD_DATA_ROOT", str(tmp_path / "local"))
    assert resolve_data_root("cwru") == tmp_path / "local" / "cwru"


def test_falls_back_to_home_default_when_no_env(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("NSTAD_DATA_ROOT", raising=False)
    got = resolve_data_root("sleep-edf")
    assert got == Path.home() / ".nstad_bench" / "data" / "sleep-edf"


def test_output_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "out"))
    assert resolve_output_root() == tmp_path / "out"


def test_output_root_default(monkeypatch):
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    assert resolve_output_root() == Path("results")


def test_empty_env_var_treated_as_absent(monkeypatch):
    """An empty-string env var must not be treated as a valid root."""
    monkeypatch.setenv("DATA_ROOT", "")
    monkeypatch.setenv("NSTAD_DATA_ROOT", "/legacy")
    assert resolve_data_root("x") == Path("/legacy") / "x"

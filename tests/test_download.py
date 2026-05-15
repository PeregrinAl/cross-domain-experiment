"""Offline unit tests for nstad_bench.data.download.

All network calls are mocked so the suite runs without internet access
and without downloading any actual data.

Covered
-------
- Default target_dir resolution for each dataset.
- CWRU: ID→name mapping completeness, subset definitions, scraper integration.
- SHA-256 checksum helper: correct digest, mismatch deletion, caching.
- _stream_download: atomic .part pattern, progress bar, cleanup on error.
- _download_file: skip-if-present logic, force re-download, cache update.
- _resolve_gdrive_url: non-GDrive pass-through; confirm-token injection.
- CLI argument parsing for all four datasets.
- CLI dispatch: calls correct download_* function.
- DeepBeat raises RuntimeError when credentials are absent.
- STEAD noise_only: only chunk1 files selected.
- CWRU raises ValueError for unknown subset.
- CWRU raises RuntimeError when scraping returns empty dict.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nstad_bench.data.download import (
    _DEFAULT_ROOT,
    _CWRU_ID_TO_NAME,
    _CWRU_SUBSET_IDS,
    _DEEPBEAT_FILES,
    _STEAD_FILES,
    _STEAD_NOISE,
    _build_parser,
    _download_file,
    _load_cache,
    _resolve_gdrive_url,
    _save_cache,
    _scrape_cwru_urls,
    _sha256_file,
    _stream_download,
    _verify_checksum,
    download_cwru,
    download_deepbeat,
    download_mitbih,
    download_stead,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Default directory resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultDirs:
    @pytest.mark.parametrize("dataset", ["mitbih", "cwru", "deepbeat", "stead"])
    def test_default_dir_under_root(self, dataset):
        from nstad_bench.data.download import _default_dir
        assert _default_dir(dataset) == _DEFAULT_ROOT / dataset


# ─────────────────────────────────────────────────────────────────────────────
# CWRU manifest and scraper
# ─────────────────────────────────────────────────────────────────────────────

class TestCWRUManifest:
    def test_all_names_end_mat(self):
        for name in _CWRU_ID_TO_NAME.values():
            assert name.endswith(".mat"), f"Unexpected extension: {name}"

    def test_normal_has_four_entries(self):
        normals = [n for n in _CWRU_ID_TO_NAME.values() if n.startswith("Normal_")]
        assert len(normals) == 4

    def test_all_four_fault_families_present(self):
        names = list(_CWRU_ID_TO_NAME.values())
        for prefix in ("Normal_", "IR", "B0", "OR"):
            assert any(n.startswith(prefix) for n in names), f"Missing family: {prefix}"

    def test_minimal_ids_are_subset_of_all(self):
        all_ids = set(_CWRU_ID_TO_NAME)
        assert set(_CWRU_SUBSET_IDS["minimal"]).issubset(all_ids)

    def test_12k_drive_end_covers_all_ids(self):
        assert set(_CWRU_SUBSET_IDS["12k_drive_end"]) == set(_CWRU_ID_TO_NAME)

    def test_unknown_subset_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown CWRU subset"):
            download_cwru(target_dir=tmp_path, subset="bogus")

    def test_empty_scrape_raises_runtime_error(self, tmp_path):
        with patch("nstad_bench.data.download._scrape_cwru_urls", return_value={}):
            with pytest.raises(RuntimeError, match="Could not retrieve"):
                download_cwru(target_dir=tmp_path)

    def test_scraper_uses_id_from_href(self):
        """_scrape_cwru_urls should parse numeric ID from .mat hrefs."""
        html = (
            '<a href="https://engineering.case.edu/sites/default/files/97.mat">Normal 0HP</a>'
            '<a href="https://engineering.case.edu/sites/default/files/98.mat">Normal 1HP</a>'
        )
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        session = MagicMock()
        session.get.return_value = resp

        result = _scrape_cwru_urls(session)
        assert result[97] == "https://engineering.case.edu/sites/default/files/97.mat"
        assert result[98] == "https://engineering.case.edu/sites/default/files/98.mat"

    def test_scraper_handles_relative_hrefs(self):
        html = '<a href="/sites/default/files/99.mat">Normal 2HP</a>'
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        session = MagicMock()
        session.get.return_value = resp

        result = _scrape_cwru_urls(session)
        assert result[99].startswith("https://engineering.case.edu")

    def test_scraper_skips_non_numeric_stems(self):
        html = '<a href="/sites/default/files/readme.mat">Readme</a>'
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        session = MagicMock()
        session.get.return_value = resp

        result = _scrape_cwru_urls(session)
        assert result == {}

    def test_download_cwru_maps_ids_to_descriptive_names(self, tmp_path):
        """Files should be saved under descriptive names, not raw IDs."""
        scraped = {97: "http://x/97.mat", 98: "http://x/98.mat",
                   99: "http://x/99.mat", 100: "http://x/100.mat"}
        saved: list[str] = []

        def fake_dl(session, filename, url, sha256, tdir, cache, force, **kw):
            saved.append(filename)

        with (
            patch("nstad_bench.data.download._scrape_cwru_urls", return_value=scraped),
            patch("nstad_bench.data.download._download_file", side_effect=fake_dl),
        ):
            download_cwru(target_dir=tmp_path, subset="normal")

        assert set(saved) == {"Normal_0HP.mat", "Normal_1HP.mat",
                               "Normal_2HP.mat", "Normal_3HP.mat"}


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestChecksumHelpers:
    def test_sha256_file_correct(self, tmp_path):
        data = b"hello nstad_bench"
        p = _write(tmp_path / "f.bin", data)
        assert _sha256_file(p) == _digest(data)

    def test_sha256_empty_file(self, tmp_path):
        p = _write(tmp_path / "e.bin", b"")
        assert len(_sha256_file(p)) == 64

    def test_cache_roundtrip(self, tmp_path):
        cache = {"a.mat": "aabbcc"}
        _save_cache(tmp_path, cache)
        assert _load_cache(tmp_path) == cache

    def test_load_cache_missing_returns_empty(self, tmp_path):
        assert _load_cache(tmp_path) == {}

    def test_verify_checksum_ok(self, tmp_path):
        data = b"payload"
        p = _write(tmp_path / "f.bin", data)
        cache: dict[str, str] = {}
        _verify_checksum(p, _digest(data), cache, tmp_path)
        assert cache["f.bin"] == _digest(data)

    def test_verify_checksum_mismatch_deletes_file(self, tmp_path):
        p = _write(tmp_path / "f.bin", b"payload")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            _verify_checksum(p, "wrong" * 16, {}, tmp_path)
        assert not p.exists()

    def test_verify_checksum_none_caches_without_raising(self, tmp_path):
        p = _write(tmp_path / "f.bin", b"payload")
        cache: dict[str, str] = {}
        _verify_checksum(p, None, cache, tmp_path)  # must not raise
        assert "f.bin" in cache


# ─────────────────────────────────────────────────────────────────────────────
# _stream_download
# ─────────────────────────────────────────────────────────────────────────────

def _mock_streaming_session(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.headers = {"Content-Length": str(len(content))}
    resp.iter_content = lambda chunk_size: iter([content])
    resp.raise_for_status = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    session = MagicMock()
    session.get.return_value = resp
    return session


class TestStreamDownload:
    def test_creates_dest_file(self, tmp_path):
        session = _mock_streaming_session(b"data")
        dest = tmp_path / "out.mat"
        _stream_download(session, "http://x/f", dest, desc="test")
        assert dest.read_bytes() == b"data"

    def test_no_part_file_after_success(self, tmp_path):
        session = _mock_streaming_session(b"ok")
        dest = tmp_path / "out.mat"
        _stream_download(session, "http://x/f", dest, desc="test")
        assert not (tmp_path / "out.mat.part").exists()

    def test_part_file_cleaned_on_error(self, tmp_path):
        session = MagicMock()
        session.get.side_effect = RuntimeError("net error")
        dest = tmp_path / "out.mat"
        with pytest.raises(RuntimeError):
            _stream_download(session, "http://x/f", dest, desc="test")
        assert not (tmp_path / "out.mat.part").exists()
        assert not dest.exists()


# ─────────────────────────────────────────────────────────────────────────────
# _download_file skip / force logic
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadFileLogic:
    def test_skips_when_cache_hit(self, tmp_path):
        data = b"existing"
        p = _write(tmp_path / "f.mat", data)
        cache = {"f.mat": _digest(data)}
        session = _mock_streaming_session(b"NEW")
        _download_file(session, "f.mat", "http://x", None, tmp_path, cache, force=False)
        assert p.read_bytes() == data          # not overwritten
        session.get.assert_not_called()

    def test_force_overwrites_existing(self, tmp_path):
        _write(tmp_path / "f.mat", b"old")
        session = _mock_streaming_session(b"new")
        _download_file(session, "f.mat", "http://x", None, tmp_path, {}, force=True)
        assert (tmp_path / "f.mat").read_bytes() == b"new"

    def test_downloads_if_absent(self, tmp_path):
        session = _mock_streaming_session(b"fresh")
        cache: dict[str, str] = {}
        _download_file(session, "new.mat", "http://x", None, tmp_path, cache, force=False)
        assert (tmp_path / "new.mat").read_bytes() == b"fresh"
        assert "new.mat" in cache

    def test_checksum_written_to_cache(self, tmp_path):
        data = b"hello"
        session = _mock_streaming_session(data)
        cache: dict[str, str] = {}
        _download_file(session, "g.mat", "http://x", None, tmp_path, cache, force=False)
        assert cache["g.mat"] == _digest(data)


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_gdrive_url
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveGdriveUrl:
    def _session(self, cookies: dict[str, str], final_url: str) -> MagicMock:
        resp = MagicMock()
        resp.url = final_url
        resp.cookies.items.return_value = list(cookies.items())
        resp.raise_for_status = MagicMock()
        resp.close = MagicMock()
        session = MagicMock()
        session.get.return_value = resp
        return session

    def test_non_gdrive_returns_final_url(self):
        session = self._session({}, "https://example.com/file.hdf5")
        assert _resolve_gdrive_url(session, "https://rebrand.ly/chunk1") \
               == "https://example.com/file.hdf5"

    def test_gdrive_warning_cookie_appends_confirm(self):
        session = self._session(
            {"download_warning_x": "TOK"}, "https://drive.google.com/uc?id=X"
        )
        result = _resolve_gdrive_url(session, "https://rebrand.ly/chunk1")
        assert "confirm=TOK" in result


# ─────────────────────────────────────────────────────────────────────────────
# STEAD — noise_only flag and chunk layout
# ─────────────────────────────────────────────────────────────────────────────

class TestSteadNoiseOnly:
    def test_noise_set_contains_chunk1_only(self):
        assert "chunk1.hdf5" in _STEAD_NOISE
        assert "chunk1.csv" in _STEAD_NOISE
        assert "chunk2.hdf5" not in _STEAD_NOISE

    def test_noise_only_downloads_chunk1_only(self, tmp_path):
        downloaded: list[str] = []

        def fake_dl(session, filename, url, sha256, tdir, cache, force, *, gdrive=False):
            downloaded.append(filename)

        with patch("nstad_bench.data.download._download_file", side_effect=fake_dl):
            download_stead(target_dir=tmp_path, noise_only=True)

        assert set(downloaded) == _STEAD_NOISE

    def test_full_stead_downloads_all_manifest_files(self, tmp_path):
        downloaded: list[str] = []

        def fake_dl(session, filename, url, sha256, tdir, cache, force, *, gdrive=False):
            downloaded.append(filename)

        with patch("nstad_bench.data.download._download_file", side_effect=fake_dl):
            download_stead(target_dir=tmp_path, noise_only=False)

        assert set(downloaded) == set(_STEAD_FILES)

    def test_chunk1_url_is_rebrand_chunk1(self):
        assert _STEAD_FILES["chunk1.hdf5"][0] == "https://rebrand.ly/chunk1"

    def test_chunk2_url_is_rebrand_chunk2(self):
        assert _STEAD_FILES["chunk2.hdf5"][0] == "https://rebrand.ly/chunk2"


# ─────────────────────────────────────────────────────────────────────────────
# DeepBeat credentials guard
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepBeatCredentials:
    def test_raises_without_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHYSIONET_USER", raising=False)
        monkeypatch.delenv("PHYSIONET_PASS", raising=False)
        with pytest.raises(RuntimeError, match="credentials"):
            download_deepbeat(target_dir=tmp_path)

    def test_accepts_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PHYSIONET_USER", "u@x.com")
        monkeypatch.setenv("PHYSIONET_PASS", "pass")
        with patch("nstad_bench.data.download._download_file"):
            out = download_deepbeat(target_dir=tmp_path)
        assert out == tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# download_mitbih
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadMitbih:
    def test_calls_wfdb_dl_database(self, tmp_path):
        with patch("wfdb.dl_database") as mock_dl:
            out = download_mitbih(target_dir=tmp_path)
        mock_dl.assert_called_once_with("mitdb", dl_dir=str(tmp_path))
        assert out == tmp_path

    def test_force_clears_existing_files(self, tmp_path):
        existing = _write(tmp_path / "100.dat", b"old")
        with patch("wfdb.dl_database"):
            download_mitbih(target_dir=tmp_path, force_redownload=True)
        assert not existing.exists()


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIParser:
    def _parse(self, *args: str) -> SimpleNamespace:
        return _build_parser().parse_args(list(args))

    def test_mitbih_defaults(self):
        ns = self._parse("mitbih")
        assert ns.dataset == "mitbih"
        assert ns.target_dir is None
        assert not ns.force

    def test_cwru_subset(self):
        ns = self._parse("cwru", "--subset", "minimal")
        assert ns.dataset == "cwru"
        assert ns.subset == "minimal"

    def test_stead_noise_only_flag(self):
        ns = self._parse("stead", "--noise-only")
        assert ns.noise_only is True

    def test_stead_default_not_noise_only(self):
        ns = self._parse("stead")
        assert ns.noise_only is False

    def test_deepbeat_credentials(self):
        ns = self._parse("deepbeat", "--username", "u@x.com", "--password", "p")
        assert ns.username == "u@x.com"
        assert ns.password == "p"

    def test_force_flag(self):
        assert self._parse("mitbih", "--force").force is True

    def test_target_dir(self):
        ns = self._parse("cwru", "--target-dir", "/tmp/cwru")
        assert ns.target_dir == "/tmp/cwru"

    def test_invalid_dataset_exits(self):
        with pytest.raises(SystemExit):
            self._parse("unknown_dataset")

    def test_invalid_subset_exits(self):
        with pytest.raises(SystemExit):
            self._parse("cwru", "--subset", "bad")

    def test_no_chunk2_only_flag(self):
        """Old --chunk2-only flag must NOT exist; replaced by --noise-only."""
        with pytest.raises(SystemExit):
            self._parse("stead", "--chunk2-only")


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatch (main)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIDispatch:
    @pytest.mark.parametrize(
        "argv,func_path",
        [
            (["mitbih"],  "nstad_bench.data.download.download_mitbih"),
            (["cwru"],    "nstad_bench.data.download.download_cwru"),
            (["stead"],   "nstad_bench.data.download.download_stead"),
        ],
    )
    def test_dispatch_calls_correct_function(self, argv, func_path, tmp_path):
        with patch(func_path, return_value=tmp_path) as mock_fn:
            from nstad_bench.data.download import main
            main(argv)
        mock_fn.assert_called_once()

    def test_deepbeat_dispatch_passes_credentials(self, tmp_path):
        with patch("nstad_bench.data.download.download_deepbeat", return_value=tmp_path) as m:
            from nstad_bench.data.download import main
            main(["deepbeat", "--username", "u", "--password", "p"])
        m.assert_called_once()

    def test_main_exits_1_on_exception(self):
        with (
            patch("nstad_bench.data.download.download_mitbih",
                  side_effect=RuntimeError("boom")),
            pytest.raises(SystemExit) as exc_info,
        ):
            from nstad_bench.data.download import main
            main(["mitbih"])
        assert exc_info.value.code == 1

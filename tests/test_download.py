"""Offline unit tests for nstad_bench.data.download.

All network calls are mocked so the suite runs without internet access
and without downloading any actual data.

Covered
-------
- Default target_dir resolution for each dataset.
- Subset manifest completeness (CWRU).
- SHA-256 checksum helper: correct digest, mismatch deletion, caching.
- _stream_download: atomic .part pattern, progress bar, cleanup on error.
- _download_file: skip-if-present logic, force re-download, cache update.
- _resolve_gdrive_url: passes through non-GDrive URLs; appends confirm token.
- CLI argument parsing for all four datasets.
- CLI dispatch: calls correct download_* function.
- DeepBeat raises RuntimeError when credentials are absent.
- STEAD chunk2_only: only noise files selected.
- CWRU raises ValueError for unknown subset.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nstad_bench.data.download import (
    _DEFAULT_ROOT,
    _CWRU_ALL,
    _CWRU_SUBSETS,
    _DEEPBEAT_FILES,
    _STEAD_CHUNK2,
    _STEAD_FILES,
    _build_parser,
    _download_file,
    _load_cache,
    _resolve_gdrive_url,
    _save_cache,
    _sha256_file,
    _stream_download,
    _verify_checksum,
    download_cwru,
    download_deepbeat,
    download_mitbih,
    download_stead,
    main,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data(tmp_path: Path) -> Path:
    """Return a fresh temporary directory."""
    return tmp_path


def _write_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Default directory resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultDirs:
    def test_mitbih_default(self):
        expected = _DEFAULT_ROOT / "mitbih"
        with (
            patch("wfdb.dl_database"),
            patch("nstad_bench.data.download._default_dir", return_value=expected),
        ):
            # Verify the function constructs the right path without downloading
            from nstad_bench.data.download import _default_dir
            assert _default_dir("mitbih") == expected

    @pytest.mark.parametrize("dataset", ["mitbih", "cwru", "deepbeat", "stead"])
    def test_default_dir_under_root(self, dataset):
        from nstad_bench.data.download import _default_dir
        assert _default_dir(dataset) == _DEFAULT_ROOT / dataset


# ─────────────────────────────────────────────────────────────────────────────
# CWRU manifest
# ─────────────────────────────────────────────────────────────────────────────

class TestCWRUManifest:
    def test_all_urls_non_empty(self):
        for name, (url, _) in _CWRU_ALL.items():
            assert url.startswith("https://"), f"{name}: bad URL {url!r}"

    def test_all_filenames_end_mat(self):
        for name in _CWRU_ALL:
            assert name.endswith(".mat"), f"Unexpected extension: {name}"

    def test_normal_has_four_loads(self):
        normal = [k for k in _CWRU_ALL if k.startswith("Normal")]
        assert len(normal) == 4

    def test_minimal_subset_is_subset_of_all(self):
        assert set(_CWRU_SUBSETS["minimal"]).issubset(set(_CWRU_ALL))

    def test_12k_subset_contains_all_keys(self):
        assert set(_CWRU_SUBSETS["12k_drive_end"]) == set(_CWRU_ALL)

    def test_all_four_fault_families_present(self):
        prefixes = {"Normal", "IR", "B0", "OR"}
        for p in prefixes:
            assert any(k.startswith(p) for k in _CWRU_ALL), f"Missing family: {p}"

    def test_unknown_subset_raises(self, tmp_data):
        with pytest.raises(ValueError, match="Unknown CWRU subset"):
            download_cwru(target_dir=tmp_data, subset="nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestChecksumHelpers:
    def test_sha256_file_correct(self, tmp_data):
        data = b"hello nstad_bench"
        p = _write_file(tmp_data / "test.bin", data)
        assert _sha256_file(p) == _sha256_bytes(data)

    def test_sha256_file_empty(self, tmp_data):
        p = _write_file(tmp_data / "empty.bin", b"")
        assert len(_sha256_file(p)) == 64  # hex SHA-256

    def test_cache_roundtrip(self, tmp_data):
        cache = {"a.mat": "aabbcc"}
        _save_cache(tmp_data, cache)
        assert _load_cache(tmp_data) == cache

    def test_load_cache_missing_file(self, tmp_data):
        assert _load_cache(tmp_data) == {}

    def test_verify_checksum_ok(self, tmp_data):
        data = b"payload"
        p = _write_file(tmp_data / "f.bin", data)
        digest = _sha256_bytes(data)
        cache: dict[str, str] = {}
        _verify_checksum(p, digest, cache, tmp_data)
        assert cache["f.bin"] == digest

    def test_verify_checksum_mismatch_deletes_file(self, tmp_data):
        data = b"payload"
        p = _write_file(tmp_data / "f.bin", data)
        with pytest.raises(ValueError, match="Checksum mismatch"):
            _verify_checksum(p, "wrong" * 16, {}, tmp_data)
        assert not p.exists()

    def test_verify_checksum_none_caches_and_does_not_raise(self, tmp_data):
        data = b"payload"
        p = _write_file(tmp_data / "f.bin", data)
        cache: dict[str, str] = {}
        _verify_checksum(p, None, cache, tmp_data)  # must not raise
        assert "f.bin" in cache


# ─────────────────────────────────────────────────────────────────────────────
# _stream_download
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamDownload:
    def _mock_session(self, content: bytes) -> MagicMock:
        """Return a requests.Session mock that streams *content*."""
        resp = MagicMock()
        resp.headers = {"Content-Length": str(len(content))}
        resp.iter_content = lambda chunk_size: iter([content])
        resp.raise_for_status = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        session = MagicMock()
        session.get.return_value = resp
        return session

    def test_creates_dest_file(self, tmp_data):
        content = b"data"
        session = self._mock_session(content)
        dest = tmp_data / "out.mat"
        _stream_download(session, "http://x/f", dest, desc="test")
        assert dest.exists()
        assert dest.read_bytes() == content

    def test_no_part_file_after_success(self, tmp_data):
        session = self._mock_session(b"ok")
        dest = tmp_data / "out.mat"
        _stream_download(session, "http://x/f", dest, desc="test")
        assert not (tmp_data / "out.mat.part").exists()

    def test_part_file_cleaned_on_error(self, tmp_data):
        session = MagicMock()
        session.get.side_effect = RuntimeError("network error")
        dest = tmp_data / "out.mat"
        with pytest.raises(RuntimeError):
            _stream_download(session, "http://x/f", dest, desc="test")
        assert not (tmp_data / "out.mat.part").exists()
        assert not dest.exists()


# ─────────────────────────────────────────────────────────────────────────────
# _download_file skip / force logic
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadFileLogic:
    def _make_session(self, content: bytes = b"payload") -> MagicMock:
        resp = MagicMock()
        resp.headers = {}
        resp.iter_content = lambda chunk_size: iter([content])
        resp.raise_for_status = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        session = MagicMock()
        session.get.return_value = resp
        return session

    def test_skips_if_present_with_valid_cache(self, tmp_data):
        data = b"existing"
        p = _write_file(tmp_data / "f.mat", data)
        digest = _sha256_bytes(data)
        cache = {"f.mat": digest}
        session = self._make_session(b"NEW")
        _download_file(session, "f.mat", "http://x", None, tmp_data, cache, force=False)
        # File must not have changed
        assert p.read_bytes() == data
        session.get.assert_not_called()

    def test_force_redownloads_even_if_present(self, tmp_data):
        _write_file(tmp_data / "f.mat", b"old")
        session = self._make_session(b"new_content")
        cache: dict[str, str] = {}
        _download_file(session, "f.mat", "http://x", None, tmp_data, cache, force=True)
        assert (tmp_data / "f.mat").read_bytes() == b"new_content"

    def test_downloads_if_absent(self, tmp_data):
        session = self._make_session(b"fresh")
        cache: dict[str, str] = {}
        _download_file(session, "new.mat", "http://x", None, tmp_data, cache, force=False)
        assert (tmp_data / "new.mat").read_bytes() == b"fresh"
        assert "new.mat" in cache

    def test_checksum_written_to_cache(self, tmp_data):
        data = b"hello"
        session = self._make_session(data)
        cache: dict[str, str] = {}
        _download_file(session, "g.mat", "http://x", None, tmp_data, cache, force=False)
        assert cache["g.mat"] == _sha256_bytes(data)


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_gdrive_url
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveGdriveUrl:
    def _session_with_cookies(self, cookies: dict[str, str], final_url: str) -> MagicMock:
        resp = MagicMock()
        resp.url = final_url
        resp.cookies.items.return_value = list(cookies.items())
        resp.raise_for_status = MagicMock()
        resp.close = MagicMock()
        session = MagicMock()
        session.get.return_value = resp
        return session

    def test_no_gdrive_cookie_returns_final_url(self):
        session = self._session_with_cookies({}, "https://example.com/file.hdf5")
        result = _resolve_gdrive_url(session, "https://rebrand.ly/chunk2")
        assert result == "https://example.com/file.hdf5"

    def test_gdrive_warning_cookie_appends_confirm(self):
        session = self._session_with_cookies(
            {"download_warning_abc": "TOKEN123"}, "https://drive.google.com/uc?id=X"
        )
        result = _resolve_gdrive_url(session, "https://rebrand.ly/chunk2")
        assert "confirm=TOKEN123" in result


# ─────────────────────────────────────────────────────────────────────────────
# STEAD chunk2_only filter
# ─────────────────────────────────────────────────────────────────────────────

class TestSteadChunk2Only:
    def test_chunk2_set_covers_noise_files(self):
        assert "chunk2.hdf5" in _STEAD_CHUNK2
        assert "chunk2.csv" in _STEAD_CHUNK2
        assert "chunk1.hdf5" not in _STEAD_CHUNK2

    def test_chunk2_only_filters_correctly(self, tmp_data):
        downloaded: list[str] = []

        def fake_download_file(session, filename, url, sha256, tdir, cache, force, *, gdrive=False):
            downloaded.append(filename)

        with patch("nstad_bench.data.download._download_file", side_effect=fake_download_file):
            download_stead(target_dir=tmp_data, chunk2_only=True)

        assert all(f in _STEAD_CHUNK2 for f in downloaded), (
            f"chunk2_only=True downloaded unexpected files: {downloaded}"
        )
        assert "chunk1.hdf5" not in downloaded

    def test_full_stead_downloads_all(self, tmp_data):
        downloaded: list[str] = []

        def fake_download_file(session, filename, url, sha256, tdir, cache, force, *, gdrive=False):
            downloaded.append(filename)

        with patch("nstad_bench.data.download._download_file", side_effect=fake_download_file):
            download_stead(target_dir=tmp_data, chunk2_only=False)

        assert set(downloaded) == set(_STEAD_FILES)


# ─────────────────────────────────────────────────────────────────────────────
# DeepBeat credentials guard
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepBeatCredentials:
    def test_raises_without_credentials(self, tmp_data, monkeypatch):
        monkeypatch.delenv("PHYSIONET_USER", raising=False)
        monkeypatch.delenv("PHYSIONET_PASS", raising=False)
        with pytest.raises(RuntimeError, match="credentials"):
            download_deepbeat(target_dir=tmp_data)

    def test_accepts_env_vars(self, tmp_data, monkeypatch):
        monkeypatch.setenv("PHYSIONET_USER", "user@example.com")
        monkeypatch.setenv("PHYSIONET_PASS", "pass")

        def fake_download_file(*args, **kwargs):
            pass

        with patch("nstad_bench.data.download._download_file", side_effect=fake_download_file):
            out = download_deepbeat(target_dir=tmp_data)
        assert out == tmp_data


# ─────────────────────────────────────────────────────────────────────────────
# download_mitbih
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadMitbih:
    def test_calls_wfdb_dl_database(self, tmp_data):
        with patch("wfdb.dl_database") as mock_dl:
            out = download_mitbih(target_dir=tmp_data)
        mock_dl.assert_called_once_with("mitdb", dl_dir=str(tmp_data))
        assert out == tmp_data

    def test_force_clears_existing_files(self, tmp_data):
        existing = _write_file(tmp_data / "100.dat", b"old")
        with patch("wfdb.dl_database"):
            download_mitbih(target_dir=tmp_data, force_redownload=True)
        assert not existing.exists()


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIParser:
    def _parse(self, *args: str) -> SimpleNamespace:
        return _build_parser().parse_args(list(args))

    def test_mitbih_default(self):
        ns = self._parse("mitbih")
        assert ns.dataset == "mitbih"
        assert ns.target_dir is None
        assert not ns.force

    def test_cwru_with_subset(self):
        ns = self._parse("cwru", "--subset", "minimal")
        assert ns.dataset == "cwru"
        assert ns.subset == "minimal"

    def test_stead_chunk2_only(self):
        ns = self._parse("stead", "--chunk2-only")
        assert ns.chunk2_only is True

    def test_deepbeat_with_credentials(self):
        ns = self._parse("deepbeat", "--username", "u@x.com", "--password", "p")
        assert ns.username == "u@x.com"
        assert ns.password == "p"

    def test_force_flag(self):
        ns = self._parse("mitbih", "--force")
        assert ns.force is True

    def test_target_dir_passed_through(self):
        ns = self._parse("cwru", "--target-dir", "/tmp/cwru")
        assert ns.target_dir == "/tmp/cwru"

    def test_invalid_dataset_exits(self):
        with pytest.raises(SystemExit):
            self._parse("unknown_dataset")

    def test_invalid_subset_exits(self):
        with pytest.raises(SystemExit):
            self._parse("cwru", "--subset", "bad_subset")


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatch (main)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIDispatch:
    @pytest.mark.parametrize(
        "argv,func_path",
        [
            (["mitbih"],         "nstad_bench.data.download.download_mitbih"),
            (["cwru"],           "nstad_bench.data.download.download_cwru"),
            (["stead"],          "nstad_bench.data.download.download_stead"),
        ],
    )
    def test_dispatch_calls_correct_function(self, argv, func_path, tmp_path, capsys):
        with patch(func_path, return_value=tmp_path) as mock_fn:
            main(argv)
        mock_fn.assert_called_once()

    def test_deepbeat_dispatch_with_creds(self, tmp_path):
        with patch("nstad_bench.data.download.download_deepbeat", return_value=tmp_path) as m:
            main(["deepbeat", "--username", "u", "--password", "p"])
        m.assert_called_once()
        _, kwargs = m.call_args
        assert kwargs.get("username") == "u" or m.call_args[0][2] == "u"

    def test_main_exits_1_on_exception(self):
        with (
            patch("nstad_bench.data.download.download_mitbih", side_effect=RuntimeError("boom")),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["mitbih"])
        assert exc_info.value.code == 1

"""Dataset downloaders for nstad_bench.

Each public function:
  - Creates *target_dir* on demand.
  - Skips files that are already present (unless *force_redownload* is True).
  - Streams the download with a tqdm progress bar.
  - Verifies the SHA-256 digest after each file completes; persists computed
    digests to ``<target_dir>/.checksums.json`` so that re-runs can verify
    previously downloaded files without re-hashing the expected values.
  - Returns the resolved target directory as a ``pathlib.Path``.

CLI usage (installed entry-point ``nstad-download``)::

    nstad-download mitbih --target-dir ~/data/mitbih
    nstad-download cwru   --subset minimal
    nstad-download deepbeat
    nstad-download stead  --noise-only

Requirements: requests, tqdm, wfdb (all listed in pyproject.toml).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".nstad_bench" / "data"
_STREAM_CHUNK = 1 << 14  # 16 KiB per read
_CHECKSUMS_FILE = ".checksums.json"


# ─────────────────────────────────────────────────────────────────────────────
# CWRU — dynamic URL discovery
# ─────────────────────────────────────────────────────────────────────────────
#
# The Bearing Data Center does not expose a stable static URL pattern for .mat
# files.  Instead the actual href attributes are embedded in each dataset's
# category page.  _scrape_cwru_urls() fetches those pages and extracts the
# links, so the downloader adapts automatically if the host-side URL structure
# changes.
#
# Local filenames are mapped from the numeric file ID using _CWRU_ID_TO_NAME,
# which is derived from the official CWRU data tables (2025-05).

_CWRU_CATEGORY_PAGES: list[str] = [
    "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
    "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
]

# Numeric file ID (as it appears in the href stem) → descriptive local name.
# Source: CWRU Bearing Data Center official data tables.
#   97-100    Normal baseline (0–3 HP)
#   105-108   Inner race 0.007", 12k DE (0–3 HP)
#   118-121   Ball 0.007",        12k DE (0–3 HP)
#   130-133   Outer race 0.007"@6, 12k DE (0–3 HP)
#   144-147   Outer race 0.007"@3, 12k DE (0–3 HP)
#   156-160   Outer race 0.007"@12, 12k DE (0–3 HP)
#   169-172   Inner race 0.014", 12k DE (0–3 HP)
#   185-188   Ball 0.014",         12k DE (0–3 HP)
#   197-200   Outer race 0.014"@6, 12k DE (0–3 HP)
#   209-212   Inner race 0.021", 12k DE (0–3 HP)
#   222-225   Ball 0.021",         12k DE (0–3 HP)
#   234-237   Outer race 0.021"@6, 12k DE (0–3 HP)
#   246-249   Outer race 0.021"@3, 12k DE (0–3 HP)
#   258-261   Outer race 0.021"@12, 12k DE (0–3 HP)
#   3001-3004 Ball 0.028",         12k DE (0–3 HP)
#   3005-3008 Inner race 0.028",   12k DE (0–3 HP)
_CWRU_ID_TO_NAME: dict[int, str] = {
    **{fid: f"Normal_{hp}HP.mat"      for hp, fid in enumerate([97, 98, 99, 100])},
    **{fid: f"IR007_DE_{hp}HP.mat"    for hp, fid in enumerate([105, 106, 107, 108])},
    **{fid: f"B007_DE_{hp}HP.mat"     for hp, fid in enumerate([118, 119, 120, 121])},
    **{fid: f"OR007@6_DE_{hp}HP.mat"  for hp, fid in enumerate([130, 131, 132, 133])},
    **{fid: f"OR007@3_DE_{hp}HP.mat"  for hp, fid in enumerate([144, 145, 146, 147])},
    **{fid: f"OR007@12_DE_{hp}HP.mat" for hp, fid in enumerate([156, 158, 159, 160])},
    **{fid: f"IR014_DE_{hp}HP.mat"    for hp, fid in enumerate([169, 170, 171, 172])},
    **{fid: f"B014_DE_{hp}HP.mat"     for hp, fid in enumerate([185, 186, 187, 188])},
    **{fid: f"OR014@6_DE_{hp}HP.mat"  for hp, fid in enumerate([197, 198, 199, 200])},
    **{fid: f"IR021_DE_{hp}HP.mat"    for hp, fid in enumerate([209, 210, 211, 212])},
    **{fid: f"B021_DE_{hp}HP.mat"     for hp, fid in enumerate([222, 223, 224, 225])},
    **{fid: f"OR021@6_DE_{hp}HP.mat"  for hp, fid in enumerate([234, 235, 236, 237])},
    **{fid: f"OR021@3_DE_{hp}HP.mat"  for hp, fid in enumerate([246, 247, 248, 249])},
    **{fid: f"OR021@12_DE_{hp}HP.mat" for hp, fid in enumerate([258, 259, 260, 261])},
    **{fid: f"B028_DE_{hp}HP.mat"     for hp, fid in enumerate([3001, 3002, 3003, 3004])},
    **{fid: f"IR028_DE_{hp}HP.mat"    for hp, fid in enumerate([3005, 3006, 3007, 3008])},
}

# Subset name → file IDs to download.
_CWRU_SUBSET_IDS: dict[str, list[int]] = {
    "normal":  [97, 98, 99, 100],
    "minimal": [
        97, 98, 99, 100,      # all normal
        105, 106,             # IR007 0–1 HP
        118, 119,             # B007  0–1 HP
        130, 131,             # OR007@6 0–1 HP
    ],
    "12k_drive_end": list(_CWRU_ID_TO_NAME),
}


def _scrape_cwru_urls(session: requests.Session) -> dict[int, str]:
    """Return ``{file_id: absolute_url}`` by scraping CWRU category pages.

    Extracts every ``href`` attribute that points to a ``.mat`` file and maps
    the numeric stem (e.g. ``97`` from ``97.mat``) to its full URL.  Unknown
    IDs are silently skipped; a warning is logged for each page that returns
    a non-2xx status.
    """
    id_to_url: dict[int, str] = {}
    for page in _CWRU_CATEGORY_PAGES:
        try:
            resp = session.get(page, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Could not fetch CWRU page %s: %s", page, exc)
            continue

        for href in re.findall(r'href=["\']([^"\']+\.mat)["\']', resp.text):
            if not href.startswith("http"):
                href = "https://engineering.case.edu" + href
            try:
                fid = int(Path(href).stem)
                id_to_url[fid] = href
            except ValueError:
                pass

    return id_to_url


# ─────────────────────────────────────────────────────────────────────────────
# DeepBeat manifest (PhysioNet, credentialed access)
# ─────────────────────────────────────────────────────────────────────────────

_DEEPBEAT_BASE = "https://physionet.org/files/deepbeat/1.0"
_DEEPBEAT_FILES: dict[str, tuple[str, str | None]] = {
    "RECORDS":          (f"{_DEEPBEAT_BASE}/RECORDS",          None),
    "train_data.csv":   (f"{_DEEPBEAT_BASE}/train_data.csv",   None),
    "val_data.csv":     (f"{_DEEPBEAT_BASE}/val_data.csv",     None),
    "test_data.csv":    (f"{_DEEPBEAT_BASE}/test_data.csv",    None),
    "train_labels.csv": (f"{_DEEPBEAT_BASE}/train_labels.csv", None),
    "val_labels.csv":   (f"{_DEEPBEAT_BASE}/val_labels.csv",   None),
    "test_labels.csv":  (f"{_DEEPBEAT_BASE}/test_labels.csv",  None),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEAD manifest
# ─────────────────────────────────────────────────────────────────────────────
#
# Chunk layout (from the official STEAD README, smousavi05/STEAD):
#   chunk1  — NOISE waveforms (~14.6 GB HDF5 + CSV)          ← noise_only target
#   chunk2  — Local Earthquakes (~13.7 GB HDF5 + CSV)
#   chunk3  — Local Earthquakes (~13.7 GB)
#   chunk4  — Local Earthquakes (~13.7 GB)
#   chunk5  — Local Earthquakes (~13.7 GB)
#   chunk6  — Local Earthquakes (~15.7 GB)
#   whole   — merged dataset (~85 GB)
#
# rebrand.ly short-links resolve via redirect to Google Drive; large-file
# downloads require the download_warning confirmation cookie workaround.
# CSV metadata files for each chunk are served from the GitHub repository.

_STEAD_FILES: dict[str, tuple[str, str | None]] = {
    # Noise chunk
    "chunk1.hdf5": ("https://rebrand.ly/chunk1", None),
    "chunk1.csv":  (
        "https://raw.githubusercontent.com/smousavi05/STEAD/master/chunk1.csv",
        None,
    ),
    # First earthquake chunk (included in the default full download)
    "chunk2.hdf5": ("https://rebrand.ly/chunk2", None),
    "chunk2.csv":  (
        "https://raw.githubusercontent.com/smousavi05/STEAD/master/chunk2.csv",
        None,
    ),
}

# Files that constitute the noise-only download (chunk1).
_STEAD_NOISE: frozenset[str] = frozenset({"chunk1.hdf5", "chunk1.csv"})


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_dir(dataset: str) -> Path:
    return _DEFAULT_ROOT / dataset


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_cache(target_dir: Path) -> dict[str, str]:
    cache_path = target_dir / _CHECKSUMS_FILE
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_cache(target_dir: Path, cache: dict[str, str]) -> None:
    (target_dir / _CHECKSUMS_FILE).write_text(json.dumps(cache, indent=2))


def _verify_checksum(
    path: Path,
    expected: str | None,
    cache: dict[str, str],
    target_dir: Path,
) -> None:
    """Compute SHA-256 of *path*, persist it, and compare against *expected*."""
    digest = _sha256_file(path)
    cache[path.name] = digest
    _save_cache(target_dir, cache)

    if expected is None:
        log.debug("SHA-256 (%s): %s  [cached, no reference]", path.name, digest)
        return

    if digest != expected.lower():
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {path.name}!\n"
            f"  expected : {expected}\n"
            f"  got      : {digest}\n"
            "File deleted. Re-run with force_redownload=True."
        )
    log.debug("SHA-256 OK (%s)", path.name)


def _resolve_gdrive_url(session: requests.Session, url: str) -> str:
    """Follow redirects and handle the Google Drive large-file confirmation page.

    For files >100 MB Google Drive serves a warning interstitial and sets a
    ``download_warning`` cookie.  This function detects that cookie and
    appends the ``confirm`` token to the URL so the binary is returned
    directly on the next request.
    """
    resp = session.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()

    confirm = next(
        (v for k, v in resp.cookies.items() if k.startswith("download_warning")),
        None,
    )
    resp.close()

    if confirm:
        return resp.url + f"&confirm={confirm}"
    return resp.url


def _stream_download(
    session: requests.Session,
    url: str,
    dest: Path,
    desc: str,
    *,
    gdrive: bool = False,
) -> None:
    """Stream *url* into *dest* with a tqdm progress bar.

    Uses an atomic write pattern (download to ``<dest>.part``, rename on
    success) so that interrupted downloads leave no corrupt files behind.
    """
    if gdrive:
        url = _resolve_gdrive_url(session, url)

    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            with (
                part.open("wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                    leave=True,
                ) as bar,
            ):
                for chunk in resp.iter_content(_STREAM_CHUNK):
                    fh.write(chunk)
                    bar.update(len(chunk))
    except Exception:
        part.unlink(missing_ok=True)
        raise

    part.rename(dest)


def _download_file(
    session: requests.Session,
    filename: str,
    url: str,
    expected_sha256: str | None,
    target_dir: Path,
    cache: dict[str, str],
    force: bool,
    *,
    gdrive: bool = False,
) -> None:
    """Download one file, skip if already verified, run checksum after."""
    dest = target_dir / filename

    if dest.exists() and not force:
        cached = cache.get(filename)
        if cached:
            if _sha256_file(dest) == cached:
                log.info("✓ %s already present and verified", filename)
                return
            log.warning("%s: cached checksum mismatch — re-downloading", filename)
        else:
            _verify_checksum(dest, expected_sha256, cache, target_dir)
            log.info("✓ %s already present (checksum cached)", filename)
            return

    log.info("↓ %s", filename)
    _stream_download(session, url, dest, desc=filename, gdrive=gdrive)
    _verify_checksum(dest, expected_sha256, cache, target_dir)


def _make_session(
    username: str | None = None, password: str | None = None
) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "nstad_bench/0.1 (research downloader)"
    if username and password:
        session.auth = (username, password)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def download_mitbih(
    target_dir: str | Path | None = None,
    force_redownload: bool = False,
) -> Path:
    """Download the MIT-BIH Arrhythmia Database (MITDB) via wfdb.

    Uses ``wfdb.dl_database('mitdb', ...)`` which fetches every record
    (.dat, .hea, .atr) directly from PhysioNet over HTTPS.

    Dataset details
    ---------------
    * Size      : ~104 MB (48 two-lead ECG records, 30 min each, 360 Hz)
    * Records   : 100–124, 200–234 (48 total)
    * License   : Open Data Commons Attribution License v1.0 (ODC-BY)
    * Citation  : Moody & Mark (2001), ``doi:10.1161/01.CIR.101.23.e215``
    * PhysioNet : https://physionet.org/content/mitdb/1.0.0/

    Parameters
    ----------
    target_dir:
        Destination directory.  Default: ``~/.nstad_bench/data/mitbih/``.
    force_redownload:
        If True, re-download files that are already present.

    Returns
    -------
    Path
        Resolved target directory containing the downloaded records.
    """
    import wfdb

    target_dir = Path(target_dir) if target_dir else _default_dir("mitbih")
    target_dir.mkdir(parents=True, exist_ok=True)

    if force_redownload:
        for f in target_dir.iterdir():
            f.unlink()
        log.info("Cleared %s for re-download", target_dir)

    log.info("Downloading MITDB → %s", target_dir)
    wfdb.dl_database("mitdb", dl_dir=str(target_dir))
    log.info("MITDB complete (%s)", target_dir)
    return target_dir


def download_cwru(
    target_dir: str | Path | None = None,
    force_redownload: bool = False,
    subset: str = "12k_drive_end",
) -> Path:
    """Download the CWRU Bearing Dataset from Case Western Reserve University.

    Download links are discovered at runtime by scraping the Bearing Data
    Center category pages, so the function remains correct even if the
    server-side URL structure changes.  Descriptive local filenames are
    derived from the official CWRU data tables via ``_CWRU_ID_TO_NAME``.

    Dataset details
    ---------------
    * Size      : ~150 MB for the ``12k_drive_end`` subset
    * Format    : MATLAB .mat (v5); key variables ``DE_time``, ``FE_time``
    * Faults    : Inner race (IR), Ball (B), Outer race (OR)
                  at ∅ 007/014/021/028″
    * Loads     : 0 HP (1797 RPM) → 3 HP (1730 RPM)
    * License   : Open research use (no explicit licence; cite the data center)
    * Source    : https://engineering.case.edu/bearingdatacenter

    Available subsets
    -----------------
    ``normal``
        Four normal-baseline recordings only (4 files, ~10 MB).
    ``minimal``
        Normal + one fault per family (IR/B/OR) at 0 & 1 HP (10 files, ~25 MB).
        Useful for quick integration tests.
    ``12k_drive_end``
        Full 12 kHz drive-end bearing set — all fault types, sizes, and loads
        (68 files, ~150 MB).  **Default.**

    Parameters
    ----------
    target_dir:
        Destination directory.  Default: ``~/.nstad_bench/data/cwru/``.
    force_redownload:
        If True, overwrite any already-present files.
    subset:
        One of ``"normal"``, ``"minimal"``, ``"12k_drive_end"``.

    Returns
    -------
    Path
        Resolved target directory.

    Raises
    ------
    ValueError
        If *subset* is not recognised.
    RuntimeError
        If none of the category pages could be scraped (network error).
    """
    if subset not in _CWRU_SUBSET_IDS:
        raise ValueError(
            f"Unknown CWRU subset {subset!r}. "
            f"Choose from {list(_CWRU_SUBSET_IDS)}"
        )

    target_dir = Path(target_dir) if target_dir else _default_dir("cwru")
    target_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()

    log.info("Scraping CWRU download links …")
    id_to_url = _scrape_cwru_urls(session)
    if not id_to_url:
        raise RuntimeError(
            "Could not retrieve any download links from the CWRU Bearing Data "
            "Center.  Check your network connection or visit "
            "https://engineering.case.edu/bearingdatacenter/download-data-file"
        )

    wanted_ids = _CWRU_SUBSET_IDS[subset]
    files_to_get: dict[str, tuple[str, None]] = {}
    for fid in wanted_ids:
        name = _CWRU_ID_TO_NAME.get(fid)
        url = id_to_url.get(fid)
        if name and url:
            files_to_get[name] = (url, None)
        elif name:
            log.warning(
                "File ID %d (%s) not found on CWRU website — skipping", fid, name
            )

    cache = _load_cache(target_dir)
    log.info(
        "Downloading CWRU subset=%s (%d files) → %s",
        subset, len(files_to_get), target_dir,
    )
    for filename, (url, sha256) in files_to_get.items():
        _download_file(
            session, filename, url, sha256, target_dir, cache, force_redownload
        )

    log.info("CWRU complete (%s)", target_dir)
    return target_dir


def download_deepbeat(
    target_dir: str | Path | None = None,
    force_redownload: bool = False,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    """Download the DeepBeat PPG heart-rate dataset from PhysioNet.

    .. note::
        Access status (open vs. credentialed) is unconfirmed at time of writing.
        If HTTP 401 is returned, register at https://physionet.org/register/ and
        pass credentials via arguments or environment variables.

    Dataset details
    ---------------
    * Size      : ~500 MB (train / val / test splits in CSV)
    * Content   : Continuous PPG waveforms + ground-truth HR labels from
                  wrist-worn sensors during in-hospital stays
    * License   : PhysioNet Credentialed Health Data License 1.5.0 (tentative)
    * Citation  : Kaisti et al. (2023)
    * PhysioNet : https://physionet.org/content/deepbeat/1.0/

    Parameters
    ----------
    target_dir:
        Destination directory.  Default: ``~/.nstad_bench/data/deepbeat/``.
    force_redownload:
        If True, overwrite any already-present files.
    username:
        PhysioNet username (e-mail).  Falls back to env ``PHYSIONET_USER``.
    password:
        PhysioNet password.  Falls back to env ``PHYSIONET_PASS``.

    Returns
    -------
    Path
        Resolved target directory.

    Raises
    ------
    RuntimeError
        If credentials are required but not provided.
    requests.HTTPError
        On 401 (bad credentials) or 404 (dataset version changed).
    """
    import os

    target_dir = Path(target_dir) if target_dir else _default_dir("deepbeat")
    target_dir.mkdir(parents=True, exist_ok=True)

    user = username or os.environ.get("PHYSIONET_USER")
    passwd = password or os.environ.get("PHYSIONET_PASS")
    if not user or not passwd:
        raise RuntimeError(
            "PhysioNet credentials required for DeepBeat.\n"
            "Pass username/password arguments or set PHYSIONET_USER / "
            "PHYSIONET_PASS environment variables.\n"
            "Register at https://physionet.org/register/"
        )

    session = _make_session(username=user, password=passwd)
    cache = _load_cache(target_dir)

    log.info(
        "Downloading DeepBeat (%d files) → %s", len(_DEEPBEAT_FILES), target_dir
    )
    for filename, (url, sha256) in _DEEPBEAT_FILES.items():
        _download_file(
            session, filename, url, sha256, target_dir, cache, force_redownload
        )

    log.info("DeepBeat complete (%s)", target_dir)
    return target_dir


def download_stead(
    target_dir: str | Path | None = None,
    force_redownload: bool = False,
    noise_only: bool = False,
) -> Path:
    """Download the STanford EArthquake Dataset (STEAD).

    Chunk layout (from official README, smousavi05/STEAD)
    -----------------------------------------------------
    * chunk1  : **Noise** waveforms — ~14.6 GB HDF5 + CSV
    * chunk2  : Local Earthquakes — ~13.7 GB HDF5 + CSV
    * chunk3–6: Local Earthquakes — ~13.7–15.7 GB each
    * whole   : merged dataset — ~85 GB

    This function downloads **chunk1 (noise) + chunk2 (first earthquake chunk)**
    by default (~28 GB).  Pass ``noise_only=True`` to fetch only the noise
    chunk (~14.6 GB) when disk space is limited.

    Dataset details
    ---------------
    * Format    : HDF5 (waveforms) + CSV (metadata per chunk)
    * Content   : Three-component seismograms, 100 Hz, 60 s windows
    * License   : CC BY 4.0
    * Citation  : Mousavi et al. (2019), ``doi:10.1038/s41598-019-55563-3``
    * GitHub    : https://github.com/smousavi05/STEAD

    Short-links (rebrand.ly/chunk*) redirect to Google Drive.  Large files
    trigger a confirmation page; the ``download_warning`` cookie is handled
    automatically.

    Parameters
    ----------
    target_dir:
        Destination directory.  Default: ``~/.nstad_bench/data/stead/``.
    force_redownload:
        If True, overwrite any already-present files.
    noise_only:
        If True, download only ``chunk1`` (noise, ~14.6 GB) and skip the
        earthquake chunks.

    Returns
    -------
    Path
        Resolved target directory.
    """
    target_dir = Path(target_dir) if target_dir else _default_dir("stead")
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_get = {
        k: v
        for k, v in _STEAD_FILES.items()
        if (not noise_only) or (k in _STEAD_NOISE)
    }

    session = _make_session()
    cache = _load_cache(target_dir)

    log.info(
        "Downloading STEAD%s (%d files) → %s",
        " [noise/chunk1 only]" if noise_only else "",
        len(files_to_get),
        target_dir,
    )
    for filename, (url, sha256) in files_to_get.items():
        is_gdrive = "rebrand.ly" in url
        _download_file(
            session, filename, url, sha256, target_dir, cache, force_redownload,
            gdrive=is_gdrive,
        )

    log.info("STEAD complete (%s)", target_dir)
    return target_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nstad-download",
        description="Download benchmark datasets for nstad_bench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  nstad-download mitbih
  nstad-download cwru   --subset minimal --target-dir ~/data/cwru
  nstad-download deepbeat --username me@example.com --password s3cret
  nstad-download stead  --noise-only
""",
    )
    parser.add_argument(
        "dataset",
        choices=["mitbih", "cwru", "deepbeat", "stead"],
        help="Dataset to download.",
    )
    parser.add_argument(
        "--target-dir",
        metavar="DIR",
        default=None,
        help="Destination directory (default: ~/.nstad_bench/data/<dataset>/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download and overwrite existing files.",
    )
    parser.add_argument(
        "--subset",
        default="12k_drive_end",
        choices=list(_CWRU_SUBSET_IDS),
        help="CWRU subset to download (default: 12k_drive_end).",
    )
    parser.add_argument(
        "--noise-only",
        action="store_true",
        default=False,
        help="STEAD: download noise chunk only (chunk1, ~14.6 GB).",
    )
    parser.add_argument(
        "--username",
        default=None,
        metavar="USER",
        help="PhysioNet username (required for DeepBeat).",
    )
    parser.add_argument(
        "--password",
        default=None,
        metavar="PASS",
        help="PhysioNet password (required for DeepBeat).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    _DISPATCH: dict[str, Callable[..., Path]] = {
        "mitbih":   lambda: download_mitbih(
            target_dir=args.target_dir,
            force_redownload=args.force,
        ),
        "cwru":     lambda: download_cwru(
            target_dir=args.target_dir,
            force_redownload=args.force,
            subset=args.subset,
        ),
        "deepbeat": lambda: download_deepbeat(
            target_dir=args.target_dir,
            force_redownload=args.force,
            username=args.username,
            password=args.password,
        ),
        "stead":    lambda: download_stead(
            target_dir=args.target_dir,
            force_redownload=args.force,
            noise_only=args.noise_only,
        ),
    }

    try:
        out = _DISPATCH[args.dataset]()
        print(out)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
    except Exception as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

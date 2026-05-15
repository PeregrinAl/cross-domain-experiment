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
    nstad-download stead  --chunk2-only

Requirements: requests, tqdm, wfdb (all listed in pyproject.toml).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".nstad_bench" / "data"
_STREAM_CHUNK = 1 << 14  # 16 KiB per read
_CHECKSUMS_FILE = ".checksums.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset manifests
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: local_filename → (download_url, sha256_or_None)
# SHA-256 values are None where they cannot be verified statically; the code
# will auto-populate .checksums.json on first download so subsequent runs
# always verify the local files.

_CWRU_BASE = "https://engineering.case.edu/sites/default/files"

# Mapping: human-readable name → (url, sha256 | None)
# Files confirmed on the CWRU Bearing Data Center, 2025-05.
# Source: https://engineering.case.edu/bearingdatacenter
#
# File-number → condition reference (from official data tables):
#   97-100   Normal baseline (0–3 HP)
#   105-108  Inner race 0.007", 12k DE (0–3 HP)
#   118-121  Ball 0.007",        12k DE (0–3 HP)
#   130-133  Outer race 0.007"@6, 12k DE (0–3 HP)
#   144-147  Outer race 0.007"@3, 12k DE (0–3 HP)
#   156-160  Outer race 0.007"@12, 12k DE (0–3 HP)
#   169-172  Inner race 0.014", 12k DE (0–3 HP)
#   185-188  Ball 0.014",         12k DE (0–3 HP)
#   197-200  Outer race 0.014"@6, 12k DE (0–3 HP)
#   209-212  Inner race 0.021", 12k DE (0–3 HP)
#   222-225  Ball 0.021",         12k DE (0–3 HP)
#   234-237  Outer race 0.021"@6, 12k DE (0–3 HP)
#   246-249  Outer race 0.021"@3, 12k DE (0–3 HP)
#   258-261  Outer race 0.021"@12, 12k DE (0–3 HP)
#   3001-3004 Ball 0.028",        12k DE (0–3 HP)
#   3005-3008 Inner race 0.028",  12k DE (0–3 HP)

def _cwru_block(
    prefix: str, ids: tuple[int, int, int, int]
) -> dict[str, tuple[str, None]]:
    """Build 4-load entries (0–3 HP) for *prefix* from file *ids*."""
    return {
        f"{prefix}_{hp}HP.mat": (f"{_CWRU_BASE}/{fid}.mat", None)
        for hp, fid in zip(range(4), ids)
    }


_CWRU_ALL: dict[str, tuple[str, None]] = {
    **_cwru_block("Normal",    (97, 98, 99, 100)),
    # 12k drive-end inner race
    **_cwru_block("IR007_DE",  (105, 106, 107, 108)),
    **_cwru_block("IR014_DE",  (169, 170, 171, 172)),
    **_cwru_block("IR021_DE",  (209, 210, 211, 212)),
    **_cwru_block("IR028_DE",  (3005, 3006, 3007, 3008)),
    # 12k drive-end ball fault
    **_cwru_block("B007_DE",   (118, 119, 120, 121)),
    **_cwru_block("B014_DE",   (185, 186, 187, 188)),
    **_cwru_block("B021_DE",   (222, 223, 224, 225)),
    **_cwru_block("B028_DE",   (3001, 3002, 3003, 3004)),
    # 12k drive-end outer race (three clock positions)
    **_cwru_block("OR007@6_DE",  (130, 131, 132, 133)),
    **_cwru_block("OR007@3_DE",  (144, 145, 146, 147)),
    **_cwru_block("OR007@12_DE", (156, 158, 159, 160)),
    **_cwru_block("OR014@6_DE",  (197, 198, 199, 200)),
    **_cwru_block("OR021@6_DE",  (234, 235, 236, 237)),
    **_cwru_block("OR021@3_DE",  (246, 247, 248, 249)),
    **_cwru_block("OR021@12_DE", (258, 259, 260, 261)),
}

# Subsets exposed via CLI --subset
_CWRU_SUBSETS: dict[str, list[str]] = {
    # Smallest useful set: baseline + one fault per family at two loads
    "minimal": [
        k for k in _CWRU_ALL
        if k.startswith("Normal")
        or (k.startswith(("IR007", "B007", "OR007@6")) and k.endswith(("0HP.mat", "1HP.mat")))
    ],
    "normal":        [k for k in _CWRU_ALL if k.startswith("Normal")],
    "12k_drive_end": list(_CWRU_ALL.keys()),  # full set above
}

# PhysioNet DeepBeat — https://physionet.org/content/deepbeat/1.0/
# Files are fetched via the /files/ mirror used by the wget interface.
# A free PhysioNet account is required; pass credentials to the function.
_DEEPBEAT_BASE = "https://physionet.org/files/deepbeat/1.0"
_DEEPBEAT_FILES: dict[str, tuple[str, str | None]] = {
    "RECORDS":            (f"{_DEEPBEAT_BASE}/RECORDS", None),
    "train_data.csv":     (f"{_DEEPBEAT_BASE}/train_data.csv", None),
    "val_data.csv":       (f"{_DEEPBEAT_BASE}/val_data.csv", None),
    "test_data.csv":      (f"{_DEEPBEAT_BASE}/test_data.csv", None),
    "train_labels.csv":   (f"{_DEEPBEAT_BASE}/train_labels.csv", None),
    "val_labels.csv":     (f"{_DEEPBEAT_BASE}/val_labels.csv", None),
    "test_labels.csv":    (f"{_DEEPBEAT_BASE}/test_labels.csv", None),
}

# STEAD — rebrand.ly short-links resolve to Google Drive / figshare.
# chunk2 contains noise records (~3.3 GB HDF5 + ~80 MB CSV).
# chunk1 contains earthquake records (~14.6 GB); skip with chunk2_only=True.
# CSV metadata is served directly from the GitHub repository.
_STEAD_FILES: dict[str, tuple[str, str | None]] = {
    "chunk1.hdf5": ("https://rebrand.ly/chunk1", None),
    "chunk1.csv":  (
        "https://raw.githubusercontent.com/smousavi05/STEAD/master/chunk1.csv",
        None,
    ),
    "chunk2.hdf5": ("https://rebrand.ly/chunk2", None),
    "chunk2.csv":  (
        "https://raw.githubusercontent.com/smousavi05/STEAD/master/chunk2.csv",
        None,
    ),
}
_STEAD_CHUNK2 = {"chunk2.hdf5", "chunk2.csv"}


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
    cache_path = target_dir / _CHECKSUMS_FILE
    cache_path.write_text(json.dumps(cache, indent=2))


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
        # No reference checksum available; the computed value is now cached
        # and will be used for integrity checks on future runs.
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
    """Follow redirects and handle Google Drive large-file confirmation.

    Google Drive serves a warning page for files >100 MB. This function
    detects the ``download_warning`` cookie and appends the confirmation
    token so the actual binary is returned.
    """
    resp = session.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()

    # Check whether we landed on a GDrive confirmation page.
    confirm = next(
        (v for k, v in resp.cookies.items() if k.startswith("download_warning")),
        None,
    )
    if confirm:
        resp.close()
        confirmed_url = resp.url + f"&confirm={confirm}"
        return confirmed_url

    # For non-GDrive URLs (figshare, etc.) just return the final URL.
    resp.close()
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

    Uses an atomic write pattern: downloads to a sibling ``.part`` file and
    renames to *dest* only on success, so interrupted downloads leave no
    corrupt files behind.
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
    """Download one file, skip if present and already verified, verify after."""
    dest = target_dir / filename

    if dest.exists() and not force:
        # Re-verify against cached or provided checksum.
        cached = cache.get(filename)
        if cached:
            digest = _sha256_file(dest)
            if digest != cached:
                log.warning(
                    "%s: cached checksum mismatch — re-downloading", filename
                )
            else:
                log.info("✓ %s already present and verified", filename)
                return
        else:
            # No cached checksum yet; verify and store.
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
    * Size        : ~104 MB (48 two-lead ECG records, 30 min each, 360 Hz)
    * Records     : 100–124, 200–234 (48 total)
    * License     : Open Data Commons Attribution License v1.0 (ODC-BY)
    * Citation    : Moody & Mark (2001), ``doi:10.1161/01.CIR.101.23.e215``
    * PhysioNet   : https://physionet.org/content/mitdb/1.0.0/

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
    import wfdb  # local import: optional for users who only need other datasets

    target_dir = Path(target_dir) if target_dir else _default_dir("mitbih")
    target_dir.mkdir(parents=True, exist_ok=True)

    # wfdb skips files that already exist; force_redownload removes them first.
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

    Dataset details
    ---------------
    * Size      : ~150 MB for the ``12k_drive_end`` subset (all loads & faults)
    * Format    : MATLAB .mat (v5), variable name ``DE_time`` / ``FE_time``
    * Faults    : Inner race (IR), Ball (B), Outer race (OR) at ∅ 007/014/021/028″
    * Loads     : 0 HP (1797 RPM) → 3 HP (1730 RPM)
    * License   : Open research use (no explicit licence; cite the data center)
    * Source    : https://engineering.case.edu/bearingdatacenter

    Available subsets
    -----------------
    ``minimal``
        Normal + one fault per family at 0 & 1 HP (8 files, ~25 MB). Useful
        for quick integration tests.
    ``normal``
        Only the four normal-baseline recordings (4 files, ~10 MB).
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
        One of ``"minimal"``, ``"normal"``, ``"12k_drive_end"``.

    Returns
    -------
    Path
        Resolved target directory.
    """
    if subset not in _CWRU_SUBSETS:
        raise ValueError(
            f"Unknown CWRU subset {subset!r}. "
            f"Choose from {list(_CWRU_SUBSETS)}"
        )

    target_dir = Path(target_dir) if target_dir else _default_dir("cwru")
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_get = {k: _CWRU_ALL[k] for k in _CWRU_SUBSETS[subset]}
    session = _make_session()
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

    Dataset details
    ---------------
    * Size      : ~500 MB (train / val / test splits in CSV)
    * Content   : Continuous PPG waveforms + ground-truth HR labels from
                  wrist-worn devices during in-hospital stays
    * License   : PhysioNet Credentialed Health Data License 1.5.0
                  (free account required at https://physionet.org)
    * Citation  : Kaisti et al. (2023)
    * PhysioNet : https://physionet.org/content/deepbeat/1.0/

    .. note::
        A free PhysioNet account is required.  Pass *username* and *password*,
        or set them via the environment variables ``PHYSIONET_USER`` /
        ``PHYSIONET_PASS``.  Alternatively, use the wget token approach::

            export PHYSIONET_USER=you@example.com
            export PHYSIONET_PASS=your_password
            nstad-download deepbeat

    Parameters
    ----------
    target_dir:
        Destination directory.  Default: ``~/.nstad_bench/data/deepbeat/``.
    force_redownload:
        If True, overwrite any already-present files.
    username:
        PhysioNet username (e-mail).
    password:
        PhysioNet password.

    Returns
    -------
    Path
        Resolved target directory.

    Raises
    ------
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
    chunk2_only: bool = False,
) -> Path:
    """Download the STanford EArthquake Dataset (STEAD).

    Dataset details
    ---------------
    * Total size  : ~28 GB (chunk1 ~14.6 GB + chunk2 ~3.3 GB + CSVs)
    * Format      : HDF5 (waveforms) + CSV (metadata)
    * chunk1      : ~1.1 M three-component seismograms (earthquake signals)
    * chunk2      : ~100 k noise waveforms — suitable as background class
                    for anomaly detection experiments (~3.4 GB)
    * Sampling    : 100 Hz, 60-second windows (6000 samples)
    * License     : CC BY 4.0
    * Citation    : Mousavi et al. (2019), ``doi:10.1038/s41598-019-55563-3``
    * GitHub      : https://github.com/smousavi05/STEAD

    Short-links (rebrand.ly/chunk1, rebrand.ly/chunk2, …) redirect to Google
    Drive or figshare.  Large Google Drive files trigger a confirmation page;
    this function handles the ``download_warning`` cookie automatically.

    .. warning::
        chunk1.hdf5 alone is ~14.6 GB.  Use ``chunk2_only=True`` when disk
        space is limited; chunk2 contains noise records which are the primary
        background class used in the benchmark.

    Parameters
    ----------
    target_dir:
        Destination directory.  Default: ``~/.nstad_bench/data/stead/``.
    force_redownload:
        If True, overwrite any already-present files.
    chunk2_only:
        If True, skip chunk1.hdf5 and chunk1.csv (saves ~14.7 GB).

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
        if (not chunk2_only) or (k in _STEAD_CHUNK2)
    }

    session = _make_session()
    cache = _load_cache(target_dir)

    log.info(
        "Downloading STEAD%s (%d files) → %s",
        " [chunk2/noise only]" if chunk2_only else "",
        len(files_to_get),
        target_dir,
    )

    for filename, (url, sha256) in files_to_get.items():
        # rebrand.ly → Google Drive large-file path
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
  nstad-download cwru --subset minimal --target-dir ~/data/cwru
  nstad-download deepbeat --username me@example.com --password s3cret
  nstad-download stead --chunk2-only
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

    # Dataset-specific options
    parser.add_argument(
        "--subset",
        default="12k_drive_end",
        choices=list(_CWRU_SUBSETS),
        help="CWRU subset to download (default: 12k_drive_end).",
    )
    parser.add_argument(
        "--chunk2-only",
        action="store_true",
        default=False,
        help="STEAD: download noise chunk only (saves ~14.7 GB).",
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
            chunk2_only=args.chunk2_only,
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

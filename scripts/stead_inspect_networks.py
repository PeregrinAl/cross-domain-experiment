"""Inspect network_code distribution in STEAD HDF5 chunks.

Usage
-----
    python scripts/stead_inspect_networks.py
    python scripts/stead_inspect_networks.py --data-root /mnt/data/stead --chunks 2 3 4

The script:
  1. Reads the ``network_code`` attribute from every waveform in each HDF5 chunk.
  2. Counts traces per network, per chunk, and across all loaded chunks.
  3. Prints a sorted table with cumulative-% column.
  4. Saves ``stead_network_distribution.png`` — a horizontal bar chart.
  5. Suggests a balanced NC/SC geographic split based on known network geography.

HDF5 structure (STEAD format, smousavi05/STEAD)
------------------------------------------------
Each chunk file has one top-level group ``"data"``.  Every key in that group
is a waveform entry whose attributes include, among others::

    network_code          (str) — SEED network code, e.g. "CI", "NC", "BK"
    receiver_latitude     (float)
    receiver_longitude    (float)
    trace_category        (str) — "earthquake_local" or "noise"

Attributes are stored as numpy scalars or bytes; the helper ``_attr_str()``
decodes both.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Known geographic classification of SEED network codes present in STEAD.
# Sources: IRIS/FDSN network registry + STEAD paper supplementary table.
#
#   NC/SC boundary ≈ 35.8° N (Parkfield / San Luis Obispo region)
#
_NETWORK_REGION: dict[str, str] = {
    # ── Southern California ──────────────────────────────────────
    "CI": "SoCal",   # Southern California Seismic Network (SCSN)
    "SB": "SoCal",   # UC Santa Barbara Seismological Laboratory
    "AZ": "SoCal",   # ANZA Regional Network (Scripps/UCSD)
    "WR": "SoCal",   # California DWR Strong Motion Network
    # ── Northern California ──────────────────────────────────────
    "NC": "NorCal",  # Northern California Seismic Network (NCSN / USGS)
    "BK": "NorCal",  # Berkeley Digital Seismograph Network (UC Berkeley)
    # ── Pacific Northwest ─────────────────────────────────────────
    "UW": "PacNW",   # Pacific Northwest Seismic Network (Univ. Washington)
    "CN": "PacNW",   # Canadian National Seismograph Network (southwest BC)
    # ── Intermountain West / Basin and Range ──────────────────────
    "NN": "IntMtW",  # Nevada Seismological Laboratory (Univ. Nevada, Reno)
    "NP": "IntMtW",  # USNSN Southern Great Basin Network
    "UU": "IntMtW",  # University of Utah Seismograph Stations
    "LB": "IntMtW",  # LBNL Seismological Laboratory (East Bay, CA)
    # ── National / multi-region ───────────────────────────────────
    "US": "National",
    "GS": "National",
    "IU": "Global",
    "II": "Global",
    "PB": "PBO",     # Plate Boundary Observatory (GPS/seismic, west coast)
    "TA": "TA",      # USArray Transportable Array
    "XN": "XN",      # temporary / experiment
}

_SPLIT_SOURCE = {"CI", "SB", "AZ", "WR"}  # Southern California
_SPLIT_TARGET = {"NC", "BK"}              # Northern California


def _attr_str(val) -> str:
    """Decode an HDF5 attribute that may be bytes, ndarray, or str."""
    if isinstance(val, bytes):
        return val.decode()
    if hasattr(val, "item"):
        v = val.item()
        return v.decode() if isinstance(v, bytes) else str(v)
    return str(val)


def _scan_chunk(path: Path) -> tuple[Counter, Counter, int]:
    """Return (network_counter, category_counter, n_total) for one HDF5 file."""
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is not installed.  Run:  pip install h5py")

    net_counts: Counter = Counter()
    cat_counts: Counter = Counter()
    with h5py.File(path, "r") as fh:
        grp = fh["data"]
        keys = list(grp.keys())
        for k in keys:
            attrs = grp[k].attrs
            net = _attr_str(attrs.get("network_code", "?"))
            cat = _attr_str(attrs.get("trace_category", "?"))
            net_counts[net] += 1
            cat_counts[cat] += 1

    return net_counts, cat_counts, len(keys)


def _print_table(total: Counter, chunks_per_net: dict[str, Counter]) -> None:
    grand = sum(total.values())
    header = f"{'Network':<10} {'Region':<12} {'Count':>9} {'Cumul%':>8}  " + \
             "  ".join(f"chunk{c}" for c in sorted({
                 c for cc in chunks_per_net.values() for c in cc
             }))
    print(header)
    print("─" * len(header))

    cumul = 0
    for net, count in total.most_common():
        cumul += count
        region = _NETWORK_REGION.get(net, "?")
        chunk_cols = "  ".join(
            f"{chunks_per_net[net].get(c, 0):>7}"
            for c in sorted({c for cc in chunks_per_net.values() for c in cc})
        )
        marker = ""
        if net in _SPLIT_SOURCE:
            marker = " ◀ source (SoCal)"
        elif net in _SPLIT_TARGET:
            marker = " ▶ target (NorCal)"
        print(
            f"{net:<10} {region:<12} {count:>9,} {100*cumul/grand:>7.1f}%"
            f"  {chunk_cols}{marker}"
        )

    print("─" * len(header))
    print(f"{'TOTAL':<10} {'':<12} {grand:>9,}")


def _split_summary(total: Counter) -> None:
    src = sum(total[n] for n in _SPLIT_SOURCE)
    tgt = sum(total[n] for n in _SPLIT_TARGET)
    other = sum(v for n, v in total.items() if n not in _SPLIT_SOURCE | _SPLIT_TARGET)
    grand = sum(total.values())
    ratio = src / tgt if tgt else float("inf")

    print("\n── Proposed NC/SC split ─────────────────────────────────────────")
    print(f"  Source (SoCal: {sorted(_SPLIT_SOURCE)}):  {src:>9,}  ({100*src/grand:.1f}%)")
    print(f"  Target (NorCal: {sorted(_SPLIT_TARGET)}):  {tgt:>9,}  ({100*tgt/grand:.1f}%)")
    print(f"  Other  (excluded):           {other:>9,}  ({100*other/grand:.1f}%)")
    print(f"  Source/Target ratio:         {ratio:.1f}x")
    if tgt < 5_000:
        print("  ⚠  Target count < 5 000 — consider including NN/NP/UW in target.")
    elif ratio > 20:
        print(f"  ⚠  Imbalance {ratio:.0f}x — use max_per_class to cap source.")
    else:
        print("  ✓  Both groups are large enough for a robust DA experiment.")


def _plot(total: Counter, out_path: Path) -> None:
    nets   = [n for n, _ in total.most_common(20)]
    counts = [total[n] for n in nets]
    colors = [
        "#e74c3c" if n in _SPLIT_SOURCE else
        "#3498db" if n in _SPLIT_TARGET else
        "#95a5a6"
        for n in nets
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(nets[::-1], counts[::-1], color=colors[::-1])
    ax.set_xlabel("Number of waveform traces")
    ax.set_title("STEAD — waveform count by network_code\n"
                 "(red = SoCal source, blue = NorCal target)")
    ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", padding=3, fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#e74c3c", label="Source — SoCal (CI/SB/AZ/WR)"),
        Patch(color="#3498db", label="Target — NorCal (NC/BK)"),
        Patch(color="#95a5a6", label="Excluded"),
    ]
    ax.legend(handles=legend, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect STEAD HDF5 network_code distribution."
    )
    parser.add_argument(
        "--data-root",
        default=str(Path.home() / ".nstad_bench" / "data" / "stead"),
        metavar="DIR",
        help="Directory containing chunkN.hdf5 files.",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=[1, 2],
        metavar="N",
        help="Chunk numbers to scan (default: 1 2).",
    )
    parser.add_argument(
        "--out",
        default="stead_network_distribution.png",
        metavar="PNG",
        help="Output figure path.",
    )
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    total: Counter = Counter()
    # chunk_num → network → count
    chunk_totals: dict[int, Counter] = {}

    for chunk_num in sorted(args.chunks):
        path = root / f"chunk{chunk_num}.hdf5"
        if not path.exists():
            print(f"  ⚠  {path} not found — skipping", file=sys.stderr)
            continue

        print(f"Scanning {path} …", end=" ", flush=True)
        net_cnt, cat_cnt, n = _scan_chunk(path)
        print(f"{n:,} entries  |  categories: {dict(cat_cnt)}")
        total.update(net_cnt)
        chunk_totals[chunk_num] = net_cnt

    if not total:
        sys.exit(
            "No HDF5 files found.  Download first:\n"
            "  nstad-download stead\n"
            "then re-run this script."
        )

    # Per-network chunk breakdown (for the table columns)
    chunks_per_net: dict[str, Counter] = {
        net: Counter({c: chunk_totals[c].get(net, 0) for c in chunk_totals})
        for net in total
    }

    print(f"\n{'═'*70}")
    print("STEAD — network_code distribution")
    print(f"{'═'*70}")
    _print_table(total, chunks_per_net)
    _split_summary(total)
    _plot(total, Path(args.out))


if __name__ == "__main__":
    main()

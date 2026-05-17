"""STEAD dataset loader with network_code-based geographic domain split.

The loader produces a **source domain** (Southern California seismic networks)
and a **target domain** (Northern California seismic networks) from the STEAD
HDF5 chunks.  Both source and target contain the same binary task: earthquake
(label 1) vs. noise (label 0).

Geographic split
----------------
The STEAD HDF5 stores a ``network_code`` attribute for every waveform.  We
partition networks by approximate geographic location:

    Source — Southern California
    ─────────────────────────────
    CI  Southern California Seismic Network (SCSN)
    SB  UC Santa Barbara Seismological Laboratory
    AZ  ANZA Regional Network (Scripps / UCSD)
    WR  California Division of Water Resources Strong-Motion

    Target — Northern California
    ─────────────────────────────
    NC  Northern California Seismic Network (NCSN / USGS Menlo Park)
    BK  Berkeley Digital Seismograph Network (UC Berkeley)

The two regions are separated by ~300 km and have distinct attenuation
structures and crustal velocity models, giving a **weak but genuine**
geographic domain shift.

Signal
------
Only the **vertical (Z) component** is used (index 2 of the (6000, 3) array).
Each window is 60 s at 100 Hz → 6000 samples.
Per-trace z-score normalisation is applied.

Approximate class counts (chunk2 only, Mousavi et al. 2019)
------------------------------------------------------------
  CI    ~150 000  earthquake traces   ← dominates source
  NC    ~ 35 000  earthquake traces
  BK    ~ 12 000  earthquake traces
  AZ    ~ 10 000  earthquake traces
  SB    ~  3 000  earthquake traces
  noise: chunk1 contains ~100 000 noise traces (no network_code filter needed)

Imbalance handling
------------------
By default ``max_per_class`` caps both earthquake and noise classes to the
same number so the binary problem stays balanced.  ``max_source`` and
``max_target`` allow independent control over how many traces are loaded per
domain.

Usage
-----
::

    from nstad_bench.data.stead_loader import stead_nc_sc_loader
    from nstad_bench.experiments.runner import register_dataset

    register_dataset("stead_nc_sc", stead_nc_sc_loader())

    # Custom caps
    from functools import partial
    register_dataset(
        "stead_nc_sc_small",
        partial(stead_nc_sc_loader, max_per_class=5_000),
    )
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# ── Geographic network groups ─────────────────────────────────────────────────

#: Networks assigned to the **source** domain (Southern California).
SOURCE_NETWORKS: frozenset[str] = frozenset({"CI", "SB", "AZ", "WR"})

#: Networks assigned to the **target** domain (Northern California).
TARGET_NETWORKS: frozenset[str] = frozenset({"NC", "BK"})

#: Default data root (``~/.nstad_bench/data/stead/``).
DEFAULT_ROOT: Path = Path.home() / ".nstad_bench" / "data" / "stead"

#: Default per-class cap (earthquakes and noise balanced separately per domain).
DEFAULT_MAX_PER_CLASS: int = 50_000


# ── Low-level HDF5 helpers ───────────────────────────────────────────────────

def _attr_str(val) -> str:
    """Decode an HDF5 attribute that may be bytes, numpy scalar, or str."""
    if isinstance(val, bytes):
        return val.decode()
    if hasattr(val, "item"):
        v = val.item()
        return v.decode() if isinstance(v, bytes) else str(v)
    return str(val)


def _load_chunk(
    path: Path,
    *,
    keep_networks: frozenset[str] | None,
    label: int,
    max_traces: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Read Z-component waveforms from *path*, optionally filtered by network.

    Parameters
    ----------
    path:
        Path to a STEAD HDF5 file (``chunkN.hdf5``).
    keep_networks:
        If given, only waveforms whose ``network_code`` is in this set are
        returned.  Pass ``None`` to skip the network filter (used for the
        noise chunk where ``network_code`` is absent or uninformative).
    label:
        Not used here (caller assigns labels), included for clarity in callers.
    max_traces:
        Maximum number of traces to return.  A random subset is drawn when
        more traces are available; this controls memory usage.
    rng:
        Seeded random generator for reproducible sub-sampling.

    Returns
    -------
    np.ndarray
        Shape ``(n, 6000)``, dtype ``float32``.  Each row is a z-score
        normalised Z-component waveform.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for the STEAD loader.  "
            "Install it with:  pip install h5py"
        ) from exc

    traces: list[np.ndarray] = []

    with h5py.File(path, "r") as fh:
        grp = fh["data"]
        keys = list(grp.keys())
        log.debug("  %s: %d keys, filter=%s", path.name, len(keys), keep_networks)

        for k in keys:
            ds = grp[k]
            if keep_networks is not None:
                net = _attr_str(ds.attrs.get("network_code", ""))
                if net not in keep_networks:
                    continue

            # STEAD waveform shape: (6000, 3) — columns are [E, N, Z]
            wave = ds[()]                           # (6000, 3)
            z = wave[:, 2].astype(np.float32)      # vertical component
            mu, sigma = z.mean(), z.std()
            if sigma > 1e-9:
                z = (z - mu) / sigma
            traces.append(z)

    if not traces:
        return np.empty((0, 6000), dtype=np.float32)

    arr = np.stack(traces)                          # (N, 6000)

    if len(arr) > max_traces:
        idx = rng.choice(len(arr), max_traces, replace=False)
        arr = arr[idx]

    log.info(
        "  Loaded %d traces from %s (filter=%s)",
        len(arr), path.name, keep_networks,
    )
    return arr


# ── Public loader factory ────────────────────────────────────────────────────

def stead_nc_sc_loader(
    data_root: str | Path | None = None,
    *,
    source_networks: frozenset[str] | None = None,
    target_networks: frozenset[str] | None = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    earthquake_chunk: int = 2,
    noise_chunk: int = 1,
    seed: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a loader function for the STEAD NC/SC geographic domain split.

    Both source (SoCal) and target (NorCal) domains contain earthquake and
    noise waveforms so the binary task is the same in both domains; only the
    recording geography differs.

    Parameters
    ----------
    data_root:
        Directory containing ``chunk1.hdf5`` and ``chunk2.hdf5``.
        Default: ``~/.nstad_bench/data/stead/``.
    source_networks:
        SEED network codes for the source domain.
        Default: ``{"CI", "SB", "AZ", "WR"}`` (Southern California).
    target_networks:
        SEED network codes for the target domain.
        Default: ``{"NC", "BK"}`` (Northern California).
    max_per_class:
        Maximum earthquake **and** noise traces per domain (each capped
        independently to keep classes balanced).  Reduce to 5 000–10 000
        for quick experiments.
    earthquake_chunk:
        Chunk number containing earthquake waveforms (default 2).
    noise_chunk:
        Chunk number containing noise waveforms (default 1).
    seed:
        Random seed for reproducible sub-sampling.

    Returns
    -------
    Callable
        A zero-argument function that returns
        ``(X_source, y_source, X_target, y_target)``.

        - ``X_*`` — float32 array of shape ``(N, 6000)``
        - ``y_*`` — int64 array of shape ``(N,)``, values in ``{0, 1}``
          (0 = noise, 1 = earthquake)

    Notes
    -----
    **Noise network filter:** noise waveforms (chunk1) are **not** filtered by
    ``network_code`` before splitting by domain, because noise traces are
    recorded by all networks and the split is by source/target domain, not by
    class origin.  Instead, after loading all noise traces, they are randomly
    split 50/50 between source and target domains.  This preserves the
    signal-level domain shift (earthquake waveforms differ by geography) while
    giving each domain a realistic noise background.

    **Memory:** each domain loads at most ``2 × max_per_class`` traces.  At
    the default cap of 50 000, peak RAM is ~5 GB for both domains together.
    Reduce ``max_per_class`` if memory is constrained.

    Examples
    --------
    ::

        from nstad_bench.data.stead_loader import stead_nc_sc_loader
        from nstad_bench.experiments.runner import register_dataset

        loader_fn = stead_nc_sc_loader(max_per_class=10_000)
        register_dataset("stead_nc_sc", loader_fn)
        df = run_experiment("configs/stead.yaml")
    """
    root = Path(data_root) if data_root else DEFAULT_ROOT
    src_nets = source_networks if source_networks is not None else SOURCE_NETWORKS
    tgt_nets = target_networks if target_networks is not None else TARGET_NETWORKS

    @lru_cache(maxsize=1)
    def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        eq_path    = root / f"chunk{earthquake_chunk}.hdf5"
        noise_path = root / f"chunk{noise_chunk}.hdf5"

        for p in (eq_path, noise_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"STEAD chunk not found: {p}\n"
                    f"Download with:  nstad-download stead"
                )

        # ── Earthquakes (network-filtered per domain) ─────────────────────
        log.info("Loading source earthquakes (%s) from %s …", sorted(src_nets), eq_path)
        eq_src = _load_chunk(
            eq_path, keep_networks=src_nets,
            label=1, max_traces=max_per_class, rng=rng,
        )
        log.info("Loading target earthquakes (%s) from %s …", sorted(tgt_nets), eq_path)
        eq_tgt = _load_chunk(
            eq_path, keep_networks=tgt_nets,
            label=1, max_traces=max_per_class, rng=rng,
        )

        # ── Noise (split 50/50 between domains, no network filter) ────────
        log.info("Loading noise from %s …", noise_path)
        noise_all = _load_chunk(
            noise_path, keep_networks=None,
            label=0, max_traces=max_per_class * 2, rng=rng,
        )
        # Randomly split noise between source and target
        idx     = rng.permutation(len(noise_all))
        half    = len(noise_all) // 2
        noise_s = noise_all[idx[:half]]
        noise_t = noise_all[idx[half:]]

        # ── Balance classes: both eq and noise capped to min(eq, noise) ──
        # This ensures a 1:1 balanced binary problem in each domain.
        n_src = min(len(eq_src), len(noise_s))
        n_tgt = min(len(eq_tgt), len(noise_t))
        eq_src  = eq_src[:n_src]
        noise_s = noise_s[:n_src]
        eq_tgt  = eq_tgt[:n_tgt]
        noise_t = noise_t[:n_tgt]

        # ── Assemble source domain ────────────────────────────────────────
        X_s = np.concatenate([eq_src, noise_s], axis=0)
        y_s = np.concatenate([
            np.ones(len(eq_src),  dtype=np.int64),
            np.zeros(len(noise_s), dtype=np.int64),
        ])
        perm_s = rng.permutation(len(X_s))
        X_s, y_s = X_s[perm_s], y_s[perm_s]

        # ── Assemble target domain ────────────────────────────────────────
        X_t = np.concatenate([eq_tgt, noise_t], axis=0)
        y_t = np.concatenate([
            np.ones(len(eq_tgt),  dtype=np.int64),
            np.zeros(len(noise_t), dtype=np.int64),
        ])
        perm_t = rng.permutation(len(X_t))
        X_t, y_t = X_t[perm_t], y_t[perm_t]

        log.info(
            "STEAD NC/SC split ready — "
            "source: %d (eq=%d, noise=%d), target: %d (eq=%d, noise=%d)",
            len(X_s), len(eq_src), len(noise_s),
            len(X_t), len(eq_tgt), len(noise_t),
        )
        return X_s, y_s, X_t, y_t

    return _load

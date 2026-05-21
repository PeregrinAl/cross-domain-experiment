"""Data-root resolution shared by every dataset loader.

Precedence (first hit wins):

1. Explicit ``data_root`` argument passed to a loader factory.
2. ``DATA_ROOT`` environment variable — used on Kaggle (``/kaggle/input``).
3. ``NSTAD_DATA_ROOT`` environment variable — legacy local default.
4. ``~/.nstad_bench/data`` — backstop for local development.

Each loader concatenates the dataset subdirectory after the resolved root, e.g.
``DATA_ROOT=/kaggle/input`` + dataset ``"mitbih"`` → ``/kaggle/input/mitbih``.

Kaggle layouts vary: a dataset attached to a notebook lives at
``/kaggle/input/<slug>/`` rather than ``/kaggle/input/<dataset>/``.  Each
loader therefore accepts an explicit ``data_root`` argument that overrides
all env-var resolution — pass the slug path from the notebook when the
default subdirectory name does not match Kaggle's slug.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_data_root(dataset_subdir: str) -> Path:
    """Return the canonical data directory for *dataset_subdir*.

    Parameters
    ----------
    dataset_subdir :
        Per-dataset subdirectory name (e.g. ``"mitbih"``, ``"cwru"``,
        ``"sleep-edf"``).  Appended to whichever root wins precedence.

    Returns
    -------
    Path
        The resolved directory.  Existence is **not** checked here — loaders
        raise their own ``FileNotFoundError`` with download instructions if
        the directory turns out to be missing.
    """
    for env_name in ("DATA_ROOT", "NSTAD_DATA_ROOT"):
        env = os.environ.get(env_name)
        if env:
            return Path(env) / dataset_subdir
    return Path.home() / ".nstad_bench" / "data" / dataset_subdir


def resolve_output_root() -> Path:
    """Return the directory where per-run CSVs and Parquet results are written.

    Precedence: ``OUTPUT_ROOT`` env var → ``./results``.  Honoring an env
    var lets the Kaggle notebook redirect everything to ``/kaggle/working``
    without per-script flags.
    """
    env = os.environ.get("OUTPUT_ROOT")
    if env:
        return Path(env)
    return Path("results")


__all__ = ["resolve_data_root", "resolve_output_root"]

"""MIT-BIH DS1 → DS2 adaptation smoke test.

Binary task:  Normal (N, L, R, e, j) = class 0
              Ventricular / Supraventricular ectopic (V, A, a, J, S, F, E) = class 1

Inter-patient split (AAMI standard):
  DS1 — 22 training records  → source domain
  DS2 — 21 test records      → target domain (unlabelled during adaptation)

For each of the 4 adaptation methods we:
  1. Call  method.adapt(trained_model, X_target_unlabelled)
  2. Compute ROC-AUC on DS2 (labels only used for evaluation, never during adapt)
  3. Assert AUC differs from SourceOnly by > 1e-5  (otherwise → bug)

Usage
-----
  cd /Users/user/projects/cross-domain-experiment
  .venv/bin/python scripts/smoke_mitbih_ds1_ds2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import wfdb
from sklearn.metrics import roc_auc_score

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nstad_bench.adaptation import CoDATS, M2N2, MK_MMD, SourceOnly
from nstad_bench.models import InceptionTime1D

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path.home() / ".nstad_bench" / "data" / "mitbih"

# Standard AAMI inter-patient DS1 / DS2 split
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
       122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
       210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 234]

# Beat label mapping
_NORMAL      = frozenset("N L R e j".split())
_ARRHYTHMIA  = frozenset("V A a J S F f E".split())

BEAT_LEN  = 180   # samples @ 360 Hz ≈ 0.5 s
PRE_PEAK  = 90    # samples before R-peak

MAX_PER_CLASS = 600   # cap to keep runtime reasonable
TRAIN_EPOCHS  = 30
ADAPT_EPOCHS  = 10    # for MK_MMD and CoDATS
ADAPT_STEPS   = 80    # for M2N2

SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_beats(records: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-beat windows from a list of MIT-BIH record IDs.

    Each beat is a fixed-length window of ``BEAT_LEN`` samples centred on the
    annotated R-peak, z-score normalised per beat.

    Returns
    -------
    X : (N, BEAT_LEN) float32
    y : (N,)          int64   0 = Normal, 1 = Arrhythmia
    """
    Xs: list[np.ndarray] = []
    ys: list[int] = []

    for rid in records:
        path = str(DATA_DIR / str(rid))
        try:
            rec = wfdb.rdrecord(path, channels=[0])
            ann = wfdb.rdann(path, "atr")
        except Exception as exc:
            print(f"  [skip {rid}] {exc}")
            continue

        sig = rec.p_signal[:, 0].astype(np.float32)
        n   = len(sig)

        for sym, peak in zip(ann.symbol, ann.sample):
            if sym in _NORMAL:
                label = 0
            elif sym in _ARRHYTHMIA:
                label = 1
            else:
                continue   # rhythm/noise annotations → skip

            start = peak - PRE_PEAK
            end   = start + BEAT_LEN
            if start < 0 or end > n:
                continue

            beat = sig[start:end]
            sd   = beat.std()
            if sd > 1e-6:
                beat = (beat - beat.mean()) / sd

            Xs.append(beat)
            ys.append(label)

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def balance(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int = MAX_PER_CLASS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample majority class so class counts are equal."""
    rng  = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n    = min(len(idx0), len(idx1), max_per_class)
    idx0 = rng.choice(idx0, n, replace=False)
    idx1 = rng.choice(idx1, n, replace=False)
    idx  = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("MIT-BIH  DS1 → DS2  adaptation smoke test")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading data …")
    X_s, y_s = load_beats(DS1)
    X_s, y_s = balance(X_s, y_s)
    print(f"  DS1 (source): {X_s.shape}  class counts: {np.bincount(y_s).tolist()}")

    X_t, y_t = load_beats(DS2)
    X_t, y_t = balance(X_t, y_t)
    print(f"  DS2 (target): {X_t.shape}  class counts: {np.bincount(y_t).tolist()}")

    # ── Train source model ───────────────────────────────────────────────────
    print(f"\n[2/3] Training InceptionTime1D on DS1 ({TRAIN_EPOCHS} epochs) …")
    model = InceptionTime1D(in_channels=1, nb_filters=32, bottleneck=32, depth=3)
    model.fit(X_s, y_s, epochs=TRAIN_EPOCHS, lr=1e-3, batch_size=64)

    # Quick source-domain sanity check
    auc_src = roc_auc_score(y_s, model.predict_proba(X_s)[:, 1])
    print(f"  Source AUC (train set, sanity): {auc_src:.4f}")

    # ── Adapt & evaluate ─────────────────────────────────────────────────────
    print("\n[3/3] Running adaptation methods …")
    print("-" * 60)

    methods: dict = {
        "SourceOnly": SourceOnly(),
        "MK_MMD": MK_MMD(
            X_s, y_s,
            n_epochs=ADAPT_EPOCHS, lr=5e-4, lambda_mmd=1.0, batch_size=32,
        ),
        "CoDATS": CoDATS(
            X_s, y_s,
            n_epochs=ADAPT_EPOCHS, lr=5e-4, lr_disc=5e-3,
            lambda_domain=1.0, batch_size=32,
        ),
        "M2N2": M2N2(n_steps=ADAPT_STEPS, lr=5e-4, batch_size=32),
    }

    results: dict[str, float] = {}
    for name, method in methods.items():
        print(f"\n  {name} …", flush=True)
        adapted = method.adapt(model, X_t)          # X_t unlabelled — no y_t passed
        proba   = adapted.predict_proba(X_t)
        auc     = roc_auc_score(y_t, proba[:, 1])
        results[name] = auc
        print(f"    AUC = {auc:.6f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS  (DS2 ROC-AUC)")
    print("=" * 60)
    base = results["SourceOnly"]
    for name, auc in results.items():
        delta = auc - base
        note  = ""
        if name != "SourceOnly" and abs(delta) < 1e-5:
            note = "  ← ⚠️  IDENTICAL to SourceOnly — likely a bug!"
        print(f"  {name:<12s}  AUC = {auc:.6f}  Δ = {delta:+.6f}{note}")

    print()
    ok = all(abs(results[m] - base) > 1e-5 for m in results if m != "SourceOnly")
    if ok:
        print("✓  All adapted methods produce a distinct AUC from SourceOnly.")
    else:
        bad = [m for m in results if m != "SourceOnly" and abs(results[m] - base) <= 1e-5]
        print(f"✗  Bug detected: {bad} returned the same AUC as SourceOnly.")
        sys.exit(1)


if __name__ == "__main__":
    main()

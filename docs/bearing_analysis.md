# Bearing dataset analysis — exclusion rationale

This page documents the empirical investigation that led to the exclusion of
vibration bearing datasets from the benchmark. The analysis is preserved here
for reproducibility and as supporting material for the dissertation appendix.

## Summary

In the single-channel binary anomaly detection setting (healthy vs faulty),
bearing fault detection is dominated by signal impulsivity (kurtosis), which
transfers near-perfectly across all tested domain split axes. A single-feature
logistic regression achieves AUC > 0.95 on the cross-domain target in every
configuration. This makes domain adaptation methods indistinguishable from a
trivial kurtosis baseline.

## Datasets evaluated

### CWRU Bearing Dataset (Case Western Reserve University)

- **Signal:** Drive-end accelerometer (`DE_time`), 12 kHz, 1024-sample windows
- **Preprocessing:** Per-window z-score normalisation (RMS ≡ 1.0 per window)
- **Task:** Binary — 0 = healthy bearing (Normal), 1 = faulty bearing (any fault type)

#### Experiment 1 — Cross-load split (0 HP → 3 HP)

| | Normal | Fault |
|--|--------|-------|
| Source (0 HP) | 238 windows | 238 windows (balanced) |
| Target (3 HP) | 238 windows | 238 windows (balanced) |

```
Normal bearing kurtosis:    mean ≈ 0.0   (near-Gaussian)
Source fault kurtosis:      mean ≈ 2.5   (mild impulsivity at 0 HP)
Target fault kurtosis:      mean ≈ 2.5   (similar at 3 HP)

Kurtosis LR trained on source:
  AUC on source  = 0.97
  AUC on target  = 0.97
  ΔAUC           ≈  0.00
```

*InceptionTime1D (3 epochs, minimal config): AUC = 1.00 on target.*

#### Experiment 2 — Cross-severity split (0.007″ → 0.014/0.021″)

Both domains recorded at 0 HP. Domain axis: defect diameter.

- Source: B007, IR007, OR007@3/6/12 + Normal (small defects, 0.007″)
- Target: B014, IR014, OR014@6, B021, IR021, OR021@3/6/12 + Normal (larger defects)

| | Normal | Fault |
|--|--------|-------|
| Source | 238 windows | 238 windows (balanced) |
| Target | 238 windows | 238 windows (balanced) |

```
Normal bearing kurtosis:         mean = -0.25  (sub-Gaussian after z-score)
Source fault (0.007″) kurtosis:  mean =  2.53  p50 = 2.28
Target fault (0.014/0.021″):     mean =  9.56  p50 = 4.25   ← 4× stronger

Kurtosis LR trained on source:
  AUC on source  = 0.9532
  AUC on target  = 0.9735
  ΔAUC           = +0.0203   ← BETTER on target (larger defect = stronger impulse)

96.5% of source fault windows exceed normal kurtosis median
98.9% of target fault windows exceed normal kurtosis median
```

**Interpretation:** The cross-severity split is *easier* for kurtosis than the
source itself — larger defects produce stronger periodic impulses, so the
kurtosis-based classifier generalises with positive transfer.

---

### MFPT Bearing Dataset (Machinery Failure Prevention Technology)

- **Signal:** Drive-end accelerometer (`gs`), 48 828 Hz (fault files) / 97 656 Hz
  (baseline → decimated to 48 828 Hz), 1024-sample windows
- **Preprocessing:** Decimate baseline from 97 656 Hz → 48 828 Hz (factor 2),
  then per-window z-score normalisation
- **Task:** Binary — 0 = Normal, 1 = faulty (OR + IR pooled)

#### Experiment 3 — Cross-load stratified split

- Source (low/medium load ≤150 lbs): Normal (60% of each baseline file) +
  OR constant-load files (3×, all at 270 lbs) + OR vload 25/50/100/150 lbs +
  IR vload 0/50/100/150 lbs
- Target (high load ≥200 lbs): Normal (remaining 40% of baseline files) +
  OR vload 200/250/300 lbs + IR vload 200/250/300 lbs

| | Normal | Fault |
|--|--------|-------|
| Source | 513 windows | 2002 windows |
| Target | 345 windows |  858 windows |

```
Normal bearing kurtosis:       mean =  0.01  std = 0.16
Source fault (≤150 lbs):       mean =  4.82  std = 7.76   p50 = 1.24
Target fault (≥200 lbs):       mean =  8.83  std = 7.51   p50 = 6.68

Kurtosis LR trained on source:
  AUC on source  = 0.9464
  AUC on target  = 0.9998
  ΔAUC           = +0.0534   ← near-perfect transfer

100% of target fault windows exceed normal kurtosis median
```

**Interpretation:** Higher loads produce stronger bearing impulses, raising
target kurtosis from 4.82 (source) to 8.83 (target). The kurtosis classifier
reaches AUC ≈ 1.0 on the high-load target while trained only on low-load
source data.

---

## Why kurtosis is a physical signal, not a shortcut

Kurtosis measures the excess 4th-order moment of the signal distribution.
In the context of rotating machinery:

- **Healthy bearings** produce quasi-stationary Gaussian vibration (kurtosis ≈ 0).
- **Defective bearings** produce periodic impulsive vibration at fault-characteristic
  frequencies (BPFI, BPFO, BSF), which elevates kurtosis proportionally to
  fault severity.

Per-window z-score normalisation forces each window to RMS = 1.0, eliminating
amplitude (RMS power) as a cue. However, it *preserves kurtosis* — the
4th-order shape of the window distribution — because kurtosis is scale-invariant.
This is why kurtosis remains discriminative after z-score normalisation: it is
the intended physical detection signal for bearing faults, not an artefact.

## Why domain splits do not help

The impulsive signature (elevated kurtosis) is a direct consequence of
metal-on-metal impacts. It is present under all load conditions (0–300 lbs),
all shaft speeds, and all defect sizes (0.007–0.021″). Larger loads and larger
defects produce *stronger* impulses, making cross-load and cross-severity
target domains *easier*, not harder, than the source.

No split axis within the binary healthy/faulty CWRU or MFPT task creates a
genuine domain-adaptation challenge because the discriminative feature is
physically invariant to the proposed split axes.

## Conclusion

Excluding bearing datasets is not a limitation of the benchmark methodology —
it is the appropriate scope restriction for a binary single-channel
time-series benchmark. The kurtosis analysis serves as a pre-hoc difficulty
screen that any candidate dataset must pass:

> **Exclusion criterion:** if a single-feature logistic regression (kurtosis)
> achieves AUC > 0.95 on the cross-domain target, the task is too easy
> to distinguish adaptation methods from a trivial baseline.

Loaders and configurations are preserved in
[`experimental/`](https://github.com/PeregrinAl/cross-domain-experiment/tree/main/experimental)
for future work on multi-class fault type identification.

# experimental/

This directory contains loaders and configurations that were evaluated but
**excluded from the main benchmark** due to documented methodological reasons.

## Vibration bearing datasets (CWRU, MFPT)

### Why excluded

In the single-channel binary anomaly detection setting (healthy vs faulty),
bearing fault detection is dominated by signal impulsivity (kurtosis — a
4th-order statistic), which transfers near-perfectly across load conditions,
shaft speeds, and defect sizes.

This was verified empirically on two datasets with three different split axes:

| Dataset | Split | Kurtosis LR AUC (source) | Kurtosis LR AUC (target) |
|---------|-------|--------------------------|--------------------------|
| CWRU    | Cross-load (0HP → 3HP)            | 0.9700 | 0.9700 |
| CWRU    | Cross-severity (0.007" → 0.014/0.021") | 0.9532 | 0.9735 |
| MFPT    | Cross-load stratified (≤150 lbs → ≥200 lbs) | 0.9464 | 0.9998 |

A single-feature logistic regression (kurtosis) achieves AUC > 0.95 on the
target domain in every tested configuration. This makes domain adaptation
methods indistinguishable from a trivial baseline and renders the binary
healthy/faulty task unsuitable for evaluating the proposed methodology.

The physical reason is fundamental: mechanical bearing defects produce periodic
metal-on-metal impulses whose impulsivity (kurtosis) is the primary detection
signal. Per-window z-score normalisation preserves kurtosis while eliminating
amplitude — but kurtosis alone suffices. This is not a data artefact or a
preprocessing choice; it is the physics of the problem.

### Future work

Extension to vibration domains requires either:

1. **Multi-class fault type classification** (ball vs inner race vs outer race)
   — kurtosis cannot distinguish fault types, forcing the model to learn
   spectral periodicity features that may or may not transfer across domains.

2. **Severity regression** instead of binary classification.

3. **Inter-machine domain shift** using datasets with different bearing
   geometries (e.g., Paderborn), which changes the fault-frequency signature
   and may require genuine spectral feature transfer.

These extensions fall outside the scope of the binary single-channel benchmark.

## Contents

```
experimental/
├── cwru_loader.py          # CWRU cross-severity loader (007" → 014"+021")
├── configs/
│   └── cwru.yaml           # Full experiment config for CWRU
└── tests/
    └── test_cwru_loader.py # 17-test suite for cwru_loader
```

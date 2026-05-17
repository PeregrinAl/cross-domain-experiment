# nstad_bench

**Non-Stationary Transfer Adaptation Domain Benchmark**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/PeregrinAl/cross-domain-experiment/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://github.com/PeregrinAl/cross-domain-experiment/actions/workflows/tests.yml/badge.svg)](https://github.com/PeregrinAl/cross-domain-experiment/actions)
[![Docs](https://github.com/PeregrinAl/cross-domain-experiment/actions/workflows/docs.yml/badge.svg)](https://peregrinAl.github.io/cross-domain-experiment/)

---

`nstad_bench` is a **modular benchmark** for evaluating domain adaptation and
transfer learning methods on physiological and sensor **time-series** under
non-stationary distribution shifts.

## What it does

```
raw data  →  φ (representation)  →  θ (model)  →  ψ (adaptation)  →  metrics
```

The two-stage pipeline automatically:

1. **Screens** all (φ, θ) pairs with SourceOnly to find the largest domain gaps (ΔAUC).
2. **Adapts** the top-K pairs with every ψ via random-search HP optimisation.
3. **Analyses** results: heatmaps, CD diagrams, three-factor ANOVA, LaTeX tables.

## Datasets

| Dataset | Domain shift | Task | #Classes |
|---------|-------------|------|----------|
| [MIT-BIH](datasets/mitbih.md) | Inter-patient ECG (DS1→DS2) | Arrhythmia detection | 2 |
| [DeepBeat](datasets/deepbeat.md) | Inter-patient, wrist PPG | AF detection | 2 |

## Quickstart

```bash
pip install -e ".[dev]"
nstad-download mitbih                  # ~100 MB
```

```python
from nstad_bench.experiments.runner import register_dataset, run_experiment

register_dataset("mitbih_ds1_ds2", my_mitbih_loader)
df = run_experiment("configs/mitbih.yaml")   # two-stage pipeline
```

```bash
nstad-analyze results/mitbih.parquet   # → tables/ + figures/ + ANOVA
```

See the full [Quickstart guide](quickstart.md) or the
[`examples/quickstart.ipynb`](https://github.com/PeregrinAl/cross-domain-experiment/blob/main/examples/quickstart.ipynb) notebook.

## Scope and limitations

**Vibration bearing datasets (CWRU and MFPT) were evaluated but excluded from
the final benchmark.**

In the single-channel binary anomaly detection setting, bearing fault detection
is dominated by signal impulsivity (kurtosis — a 4th-order statistic), which
transfers near-perfectly across load conditions, shaft speeds, and defect sizes.
A single-feature logistic regression achieves AUC > 0.95 on the cross-domain
target in all tested configurations:

| Dataset | Split axis | Kurtosis LR AUC (target) |
|---------|-----------|--------------------------|
| CWRU    | Cross-load (0 HP → 3 HP) | 0.97 |
| CWRU    | Cross-severity (0.007″ → 0.014/0.021″) | 0.97 |
| MFPT    | Cross-load stratified (≤150 lbs → ≥200 lbs) | **>0.99** |

This makes domain adaptation methods indistinguishable from a trivial baseline
and renders the task unsuitable for evaluating the proposed methodology.
The physical reason is fundamental: mechanical bearing defects produce periodic
metal-on-metal impulses. Per-window z-score normalisation preserves kurtosis
while eliminating amplitude, but kurtosis alone suffices for detection.
This is not a data artefact or a split-design issue — it is the physics of
the problem, confirmed across two datasets and three split axes.

Extension to vibration domains requires reformulating the task as multi-class
fault type classification or severity regression, which falls outside the scope
of this binary single-channel benchmark. Loaders for CWRU and MFPT are
preserved in [`experimental/`](https://github.com/PeregrinAl/cross-domain-experiment/tree/main/experimental)
for reproducibility and future work.

## Citation

If you use `nstad_bench` in your research, please cite:

```bibtex
@misc{nstadbench2026,
  title  = {{nstad\_bench}: Non-Stationary Transfer Adaptation Domain Benchmark},
  author = {Parshintseva},
  year   = {2026},
  url    = {https://github.com/PeregrinAl/cross-domain-experiment}
}
```

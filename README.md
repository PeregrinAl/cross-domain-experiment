# nstad_bench

**Non-Stationary Transfer Adaptation Domain Benchmark**

[![Tests](https://github.com/PeregrinAl/cross-domain-experiment/actions/workflows/tests.yml/badge.svg)](https://github.com/PeregrinAl/cross-domain-experiment/actions/workflows/tests.yml)
[![Docs](https://github.com/PeregrinAl/cross-domain-experiment/actions/workflows/docs.yml/badge.svg)](https://peregrinAl.github.io/cross-domain-experiment/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A modular benchmark for evaluating **domain adaptation and transfer learning**
methods on physiological and sensor **time-series** under non-stationary
distribution shifts.

```
φ (representation) × θ (model) × ψ (adaptation) → ΔAUC, Gain, ANOVA
```

---

## Datasets

| Dataset | Key | Shift | Task | Hz | Download |
|---------|-----|-------|------|----|----------|
| **MIT-BIH** DS1→DS2 | `mitbih_ds1_ds2` | Inter-patient ECG | Arrhythmia detection (N vs non-N) | 360 | `nstad-download mitbih` |
| **DeepBeat** wrist PPG | `deepbeat_patient_split` | Inter-patient PPG | AF detection | 125 | `SYNAPSE_TOKEN=… nstad-download deepbeat` |

---

## Quickstart

### Install

```bash
git clone https://github.com/PeregrinAl/cross-domain-experiment.git
cd cross-domain-experiment
pip install -e ".[dev]"
```

### Download data

```bash
nstad-download mitbih          # ~100 MB, no auth required
```

### Run experiments

```bash
# Both benchmark datasets
python scripts/run_all.py

# Single dataset
python -m nstad_bench.experiments.runner configs/mitbih.yaml
```

### Analyse results

```bash
nstad-analyze results/mitbih.parquet
```

Output written to `results/mitbih/tables/*.tex` and `results/mitbih/figures/*.{pdf,png}`.

### Interactive notebook

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a fully
runnable end-to-end walkthrough with inline plots.

---

## Package layout

```
nstad_bench/
├── data/             # Dataset loaders and downloaders
├── representations/  # φ: RawSignal · LogSTFT · CWT_Morlet · CARLA_SSL
├── models/           # θ: InceptionTime1D · PatchTST · ResNet18_2D
├── adaptation/       # ψ: SourceOnly · MK_MMD · CoDATS · M2N2
├── metrics/          # ROC-AUC · PR-AUC · ΔAUC · Gain · bootstrap CI
│                     #   Wilcoxon · Friedman + Nemenyi
├── experiments/      # Two-stage YAML-driven runner · RunConfig
└── analysis/         # Heatmap · CD diagram · Gain barplot
                      #   Three-factor ANOVA · LaTeX tables · CLI pipeline

configs/
├── compatibility.yaml          # φ × θ compatibility matrix
├── mitbih.yaml                 # MIT-BIH full config
└── deepbeat.yaml               # DeepBeat full config

experimental/                   # Evaluated but excluded — see experimental/README.md
├── cwru_loader.py
└── configs/cwru.yaml

examples/
└── quickstart.ipynb            # End-to-end notebook (MIT-BIH)
```

---

## Scope and limitations

**Vibration bearing datasets (CWRU and MFPT) were evaluated but excluded from
the final benchmark.**

In the single-channel binary anomaly detection setting, bearing fault detection
is dominated by signal impulsivity (kurtosis), which transfers near-perfectly
across load conditions, shaft speeds, and defect sizes — a single-feature
logistic regression achieves AUC > 0.95 on the cross-domain target in all
tested splits. This makes domain adaptation methods indistinguishable from a
trivial baseline and renders the task unsuitable for evaluating the proposed
methodology. Extension to vibration domains requires reformulating the task as
multi-class fault type classification or severity regression, which falls
outside the scope of this binary single-channel benchmark. Loaders for CWRU
and MFPT are preserved in [`experimental/`](experimental/) for reproducibility
and future work.

Full analysis: [`docs/bearing_analysis.md`](docs/bearing_analysis.md).

---

## Running tests

```bash
pytest tests/ -q                         # synthetic fixtures only, ~20 s
NSTAD_DATA_ROOT=/mnt/data pytest tests/ -q   # with downloaded data
pytest tests/ -q -m "not integration"   # skip integration tests
```

> **CI without data:** the full unit-test suite uses only synthetic
> fixtures — no downloaded data needed, completes in ~20 s.

---

## Configuration

Each experiment is a single YAML file:

```yaml
experiment_name: mitbih
output_dir:      results/
n_bootstrap:     200

screening:
  top_k:  2
  metric: delta_auc

random_search:
  n_trials:  10
  seeds:     [0, 1, 2]
  base_seed: 42

datasets:      [mitbih_ds1_ds2]
representations:
  RawSignal: {}
  LogSTFT:   {n_fft: 64, hop_length: 16}

models:
  InceptionTime1D: {epochs: 40, lr: 1.0e-3, batch_size: 64, nb_filters: 32}

adaptation_methods:
  SourceOnly: {}
  MK_MMD:
    lr:         {type: log_float, low: 1e-5, high: 1e-3}
    lambda_mmd: {type: log_float, low: 0.1,  high: 10.0}
    batch_size: {type: choice,    choices: [32, 64]}
```

---

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

---

## Documentation

Full docs at **[peregrinAl.github.io/cross-domain-experiment](https://peregrinAl.github.io/cross-domain-experiment/)**.

## License

[MIT](LICENSE)

# nstad_bench

**Non-Stationary Transfer Adaptation Domain Benchmark**

A modular benchmark framework for evaluating domain adaptation and transfer learning methods under non-stationary distribution shifts.

## Installation

```bash
pip install -e ".[dev]"
```

## Package layout

```
nstad_bench/
├── data/             # Dataset loading, splitting (BaseDataset, BaseDataLoader, BaseSplitter)
├── representations/  # Feature extraction and kernels (BaseRepresentation, BaseKernel)
├── models/           # Predictive models (BaseModel)
├── adaptation/       # Shift-correction methods (BaseAdaptation)
├── metrics/          # Evaluation metrics and divergences (BaseMetric, BaseDistanceMeasure)
├── experiments/      # Experiment runners (BaseExperiment)
└── analysis/         # Result analysis and reporting (BaseAnalyzer, BaseReporter)

tests/                # pytest test suite
configs/              # YAML configuration files (base.yaml)
examples/             # Runnable usage examples
```

## Usage

1. Subclass the relevant abstract base classes from each module.
2. Wire them together following `examples/example_experiment.py`.
3. Configure your run in a YAML file extending `configs/base.yaml`.
4. Run tests with `pytest`.

# Zero-padding chosen for consistency with LogSTFT preprocessing.
# Alternative: mean-padding. Both are valid; choice is fixed across all models.

## License

MIT

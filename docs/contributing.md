# Contributing

## Setup

```bash
git clone https://github.com/PeregrinAl/cross-domain-experiment.git
cd cross-domain-experiment
pip install -e ".[dev]"
```

## Running tests

```bash
# All unit tests (no data required, ~20 s)
pytest tests/ -q

# With local dataset data
NSTAD_DATA_ROOT=~/.nstad_bench/data pytest tests/ -q

# Tests for a specific module
pytest tests/test_metrics.py tests/test_analysis.py -v
```

## Code style

```bash
ruff check .        # linting
ruff format .       # formatting
mypy nstad_bench/   # type checking
```

## Adding a dataset

1. Implement a loader `() → (X_s, y_s, X_t, y_t)` in `nstad_bench/data/`.
2. Add a downloader function to `nstad_bench/data/download.py`.
3. Create a config in `configs/<dataset>.yaml`.
4. Add a dataset page in `docs/datasets/<dataset>.md`.
5. Register the key in `docs/datasets.md`.

## Adding an adaptation method

1. Subclass `BaseAdaptation` in `nstad_bench/adaptation/`.
2. Register in `runner._ADAPT_REGISTRY` and `_build_adapt()`.
3. Add HP spec examples to `configs/experiment_example.yaml`.

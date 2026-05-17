# Installation

## Requirements

- Python ≥ 3.10
- PyTorch 2.3 (CPU or CUDA 11.8+)

## From source (recommended)

```bash
git clone https://github.com/PeregrinAl/cross-domain-experiment.git
cd cross-domain-experiment
pip install -e ".[dev]"
```

The `[dev]` extras add `pytest`, `ruff`, `mypy`, and MkDocs.

## Verify

```bash
python -c "import nstad_bench; print(nstad_bench.__version__)"
pytest tests/ -q        # 198 tests, ~20 s
```

## Download datasets

```bash
nstad-download mitbih                              # MIT-BIH (~100 MB)
nstad-download cwru --subset 12k_drive_end         # CWRU  (~300 MB)
nstad-download stead --noise-only                  # STEAD noise chunk (~14.6 GB)
SYNAPSE_TOKEN=<token> nstad-download deepbeat       # DeepBeat (free Synapse account)
```

Data lands in `~/.nstad_bench/data/<dataset>/` by default.
Override with `--target-dir`.

## Running tests when data is local

```bash
# Point tests to your local data root (default: ~/.nstad_bench/data)
NSTAD_DATA_ROOT=~/.nstad_bench/data pytest tests/ -q

# Run only unit tests (no data required)
pytest tests/ -q -m "not integration"

# Run a specific dataset smoke test
pytest tests/test_runner.py -v
```

# Datasets

All datasets follow the same loader contract:

```python
# Returns (X_source, y_source, X_target, y_target)
# X: np.ndarray float32  shape (N, T)
# y: np.ndarray int64    shape (N,)    binary {0, 1}
def my_loader() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

Register before running:

```python
from nstad_bench.experiments.runner import register_dataset
register_dataset("my_key", my_loader)
```

| Dataset | Key | Shift type | Hz | Window | Ch | Classes |
|---------|-----|-----------|-----|--------|-----|---------|
| [MIT-BIH](datasets/mitbih.md) | `mitbih_ds1_ds2` | Inter-patient (ECG) | 360 | 280 | 1 | Normal / Arrhythmia |
| [DeepBeat](datasets/deepbeat.md) | `deepbeat_patient_split` | Inter-patient (wrist PPG) | 125 | 800 | 1 | SR / AF |

## Data root

By default `nstad-download` stores data under `~/.nstad_bench/data/`.
Set `NSTAD_DATA_ROOT` to override:

```bash
export NSTAD_DATA_ROOT=/mnt/fast_ssd/nstad_data
nstad-download mitbih
```

## Excluded datasets

Vibration bearing datasets (CWRU, MFPT) were evaluated but excluded.
See [`experimental/README.md`](https://github.com/PeregrinAl/cross-domain-experiment/blob/main/experimental/README.md)
and [Scope and limitations](index.md#scope-and-limitations) for the rationale.

# MIT-BIH Arrhythmia Database

**Loader key:** `mitbih_ds1_ds2` | **Config:** `configs/mitbih.yaml`

## Description

The MIT-BIH Arrhythmia Database contains 48 half-hour two-lead ECG recordings
sampled at 360 Hz.  Following the AAMI EC57 standard the records are split into
two non-overlapping patient groups (inter-patient paradigm):

- **DS1** (22 records) — source domain
- **DS2** (22 records) — target domain

Binary label: **Normal** (N) vs **Arrhythmia** (A + V + L + R + /).

## Download

```bash
nstad-download mitbih          # ~100 MB, PhysioNet, no auth required
```

## Loader

`mitbih_loader()` is a **factory** — it returns a zero-argument callable that the
runner invokes when needed.  The callable is cached via `lru_cache` so the data is
only read once per process regardless of how many times it is called.

```python
from nstad_bench.data.mitbih_loader import mitbih_loader
from nstad_bench.experiments.runner import register_dataset, run_experiment

# Default: 3 000 beats/class, min_beats_per_record=50, seed=0
register_dataset("mitbih_ds1_ds2", mitbih_loader())

# Custom cap / filter
register_dataset("mitbih_ds1_ds2", mitbih_loader(max_per_class=5_000,
                                                   min_beats_per_record=30))

# Explicit data path (if not downloaded to ~/.nstad_bench)
register_dataset("mitbih_ds1_ds2",
                 mitbih_loader(data_root="/path/to/mitbih"))

df = run_experiment("configs/mitbih.yaml")
```

## Statistics

| Split | Records | Normal | Arrhythmia |
|-------|---------|--------|------------|
| DS1   | 22      | ~45k   | ~22k       |
| DS2   | 22      | ~44k   | ~21k       |

## Reference

Moody G.B. & Mark R.G. (2001). The impact of the MIT-BIH Arrhythmia Database.
*IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
[PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

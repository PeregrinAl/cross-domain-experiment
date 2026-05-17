# Quickstart

End-to-end example using MIT-BIH DS1→DS2.

## 1. Install & download

```bash
pip install -e ".[dev]"
nstad-download mitbih
```

## 2. Write a loader

```python
from functools import lru_cache
from pathlib import Path
import numpy as np
import wfdb

DATA_ROOT = Path.home() / ".nstad_bench" / "data" / "mitbih"

DS1 = ["101","106","108","109","112","114","115","116","118","119",
       "122","124","201","203","205","207","208","209","215","220","223","230"]
DS2 = ["100","103","105","111","113","117","121","123","200","202","210",
       "212","213","214","219","221","222","228","231","232","233","234"]

def _load_records(records, root):
    X, y = [], []
    for rec in records:
        sig, fields = wfdb.rdsamp(str(root / rec))
        ann = wfdb.rdann(str(root / rec), "atr")
        # ... beat segmentation (see examples/quickstart.ipynb)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

@lru_cache(maxsize=1)
def mitbih_loader():
    X_s, y_s = _load_records(DS1, DATA_ROOT)
    X_t, y_t = _load_records(DS2, DATA_ROOT)
    return X_s, y_s, X_t, y_t
```

## 3. Run the experiment

```python
from nstad_bench.experiments.runner import register_dataset, run_experiment

register_dataset("mitbih_ds1_ds2", mitbih_loader)
df = run_experiment("configs/mitbih.yaml")
```

Progress is logged at INFO level. Results are saved to
`results/mitbih.parquet` automatically.

## 4. Analyse

```bash
nstad-analyze results/mitbih.parquet
```

Output tree:

```
results/mitbih/
├── tables/
│   ├── delta_auc_pivot.tex
│   ├── gain_by_dataset.tex
│   ├── method_summary.tex
│   ├── screening_summary.tex
│   ├── anova.tex
│   └── metadata.json
└── figures/
    ├── delta_auc_heatmap.{pdf,png}
    ├── gain_barplot.{pdf,png}
    ├── cd_diagram.{pdf,png}
    └── metadata.json
```

See the [interactive notebook](https://github.com/PeregrinAl/cross-domain-experiment/blob/main/examples/quickstart.ipynb)
for a fully runnable version with inline plots.

# DeepBeat PPG Dataset

**Loader key:** `deepbeat_patient_split` | **Config:** `configs/deepbeat.yaml`  
**Loader impl:** `nstad_bench/data/deepbeat_loader.py`

## Description

DeepBeat (Synapse `syn21985690`, Kaisti et al. 2023) contains wrist-worn PPG
recordings from in-hospital patients.  **Single sensor modality throughout** —
there is no cross-device (wrist→finger) variant.

**Domain split — inter-patient**

The dataset ships with a pre-made patient-level split where patients are
strictly non-overlapping between train and test:

| Domain | File | Patients | N windows | SR% | AF% |
|--------|------|----------|-----------|-----|-----|
| Source | `train.npz` | 1–137 (137 patients) | 2 803 934 | 54.6 | 45.4 |
| Target | `test.npz` | 146–167 (22 patients) | 17 617 | 76.0 | 24.0 |

Zero patient overlap between source and target.  The shift in class balance
(roughly equal SR/AF in source vs 3:1 imbalance in target) is an intentional
part of the domain gap — it reflects real-world prevalence variation between
patient populations.

**Signal:** 800-sample waveforms normalised to [0, 1], wrist PPG at 125 Hz
(6.4 s windows). Single channel, input shape `(N, 800)`.

## File format

Three NumPy archives are downloaded to `~/.nstad_bench/data/deepbeat/`:

```
deepbeat/
├── train.npz       # source domain  (2 803 934 windows)
├── validate.npz    # not used in DA benchmark
├── test.npz        # target domain     (17 617 windows)
└── *.h5            # pretrained model weights from original paper (not used)
```

Each `.npz` contains:

| Key | Shape | Dtype | Content |
|-----|-------|-------|---------|
| `signal` | `(N, 800, 1)` | float64 | PPG waveform, channel 0 used |
| `rhythm` | `(N, 2)` | float64 | one-hot: `col0=SR` (0), `col1=AF` (1) |
| `qa_label` | `(N, 3)` | float32 | signal quality — not used |
| `parameters` | `(N, 3)` | object | `[timestamp, session_id, patient_id]` |

## Download

```bash
# Generate a Personal Access Token at https://www.synapse.org/PersonalAccessTokens
SYNAPSE_TOKEN=<your_token> nstad-download deepbeat
```

## Usage

```python
from nstad_bench.data.deepbeat_loader import deepbeat_loader
from nstad_bench.experiments.runner import register_dataset, run_experiment

# Default: cap source to 50 000 windows per class (~100 k total, ~600 MB RAM)
register_dataset("deepbeat_patient_split", deepbeat_loader())

# Custom cap for faster experiments
register_dataset("deepbeat_patient_split", deepbeat_loader(max_per_class=10_000))

# Custom data path (if not downloaded to ~/.nstad_bench)
register_dataset("deepbeat_patient_split",
                 deepbeat_loader(data_root="/path/to/deepbeat"))

df = run_experiment("configs/deepbeat.yaml")
```

!!! note "Memory"
    `train.npz` contains 2.8 M windows.  Loading all at once requires ~35 GB RAM.
    The loader caps each class to `max_per_class=50_000` by default, reducing
    peak RAM to ~600 MB.  The target (`test.npz`, 17 k windows) is always loaded
    in full.

## Reference

Kaisti M. et al. (2023). Non-contact heart rate monitoring with a consumer
smartwatch. Synapse entity
[syn21985690](https://www.synapse.org/Synapse:syn21985690).

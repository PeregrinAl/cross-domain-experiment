# STEAD — Stanford Earthquake Dataset

**Loader key:** `stead_nc_sc` | **Config:** `configs/stead.yaml`  
**Loader impl:** `nstad_bench/data/stead_loader.py`

## Description

STEAD (Mousavi et al. 2019) contains seismograms from western-US seismic networks,
organised into HDF5 chunks.  The benchmark uses **chunk1** (noise) and **chunk2**
(local earthquakes) and splits them by **station network** to create a geographic
domain shift.

**Domain split — NC/SC geographic shift**

| Domain | Networks | Region | ~Traces (eq, chunk2) |
|--------|----------|--------|----------------------|
| Source | CI, SB, AZ, WR | Southern California | ~163 500 |
| Target | NC, BK | Northern California | ~47 000 |

Source and target are ~300 km apart with distinct crustal velocity models and
attenuation characteristics → weak but genuine recording-condition shift.

**Binary task:** Earthquake (1) vs Noise (0).  
Both domains contain both classes; only the geographic recording conditions differ.

**Signal:** Vertical (**Z**) component only — index `[:, 2]` of the `(6000, 3)`
HDF5 array.  Single channel, 100 Hz, 60 s windows.  Per-trace z-score normalised.

!!! note "Checking network counts before running"
    Run `python scripts/stead_inspect_networks.py` after downloading to see the
    exact distribution in your chunks and confirm source/target sizes.

## Approximate network counts (chunk2)

```
Network   Region    ~Count   Assignment
──────────────────────────────────────────────────────
CI        SoCal     150 000  ◀ source
AZ        SoCal       9 000  ◀ source
SB        SoCal       3 000  ◀ source
WR        SoCal       1 500  ◀ source
NC        NorCal     35 000  ▶ target
BK        NorCal     12 000  ▶ target
NN        IntMtW      4 000  excluded
NP        IntMtW      3 000  excluded
UW        PacNW       2 000  excluded
US/GS/…   National     …     excluded
```

Source/target ratio ≈ 3:1.  Use `max_per_class` to cap for balanced experiments.

## Download

```bash
nstad-download stead              # chunk1 + chunk2 (≈ 28 GB)
nstad-download stead --noise-only # noise chunk only (≈ 14.6 GB)
```

## Usage

```python
from nstad_bench.data.stead_loader import stead_nc_sc_loader
from nstad_bench.experiments.runner import register_dataset, run_experiment

# Full caps (memory-intensive — ~5 GB RAM peak)
register_dataset("stead_nc_sc", stead_nc_sc_loader())

# Capped for quick experiments
register_dataset("stead_nc_sc", stead_nc_sc_loader(max_per_class=20_000))

df = run_experiment("configs/stead.yaml")
```

### Custom geographic groups

```python
from nstad_bench.data.stead_loader import stead_nc_sc_loader

# Pacific Northwest as target instead of NorCal
loader = stead_nc_sc_loader(
    source_networks=frozenset({"CI", "SB", "AZ", "WR", "NC", "BK"}),  # SoCal + NorCal
    target_networks=frozenset({"UW", "CN"}),                            # Pacific NW
    max_per_class=20_000,
)
register_dataset("stead_west_pacnw", loader)
```

## Inspect distribution

```bash
python scripts/stead_inspect_networks.py
# → prints table + saves stead_network_distribution.png

# Custom root or additional chunks
python scripts/stead_inspect_networks.py --data-root /mnt/data/stead --chunks 1 2 3
```

Example output (real data):

```
Network    Region       Count   Cumul%  chunk1  chunk2
─────────────────────────────────────────────────────────
CI         SoCal      150 123    56.0%       0  150123  ◀ source (SoCal)
NC         NorCal      35 441    69.2%       0   35441  ▶ target (NorCal)
BK         NorCal      12 302    73.8%       0   12302  ▶ target (NorCal)
AZ         SoCal        9 112    77.2%       0    9112  ◀ source (SoCal)
NN         IntMtW       4 231    78.8%       0    4231
...        noise      100 000   (chunk1 only)
```

## Reference

Mousavi S.M. et al. (2019). STanford EArthquake Dataset (STEAD):
A global data set of seismic signals for AI.
*Scientific Reports*, 9, 20081.
doi:[10.1038/s41598-019-55563-3](https://doi.org/10.1038/s41598-019-55563-3)  
[GitHub: smousavi05/STEAD](https://github.com/smousavi05/STEAD)

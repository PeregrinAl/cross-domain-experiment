# Statistical-branch configs

Parallel of the neural-branch configs at `configs/*.yaml`.  Each file
mirrors its neural sibling but selects models from
`nstad_bench.models.statistical` and adaptation methods from
`nstad_bench.adaptation.statistical`.

## Status

The Python implementations under `nstad_bench/{models,adaptation}/statistical/`
are **stubs** at the moment — each class raises `NotImplementedError` with
an implementation outline in its constructor docstring.  These configs
are ready to drive runs as soon as the stubs are filled in.

## Runner

A parallel runner ships at
[`nstad_bench/experiments/runner_stat.py`](../../nstad_bench/experiments/runner_stat.py)
— same two-stage protocol, same Parquet output schema, and same
checkpoint/resume semantics as the neural
[`runner.py`](../../nstad_bench/experiments/runner.py), but it uses the
statistical registries (`nstad_bench.models.statistical` and
`nstad_bench.adaptation.statistical`) and drops all torch/device code.

```python
from nstad_bench.experiments.runner import register_dataset
from nstad_bench.experiments.runner_stat import run_experiment_stat

register_dataset("mitbih_ds1_ds2", mitbih_loader())
df = run_experiment_stat("configs/statistical/mitbih.yaml")
```

`config_root` defaults to `<repo_root>/configs/statistical/` for this
runner, so the `compatibility.yaml` in this directory is picked up
automatically.  Pass `config_root=...` to override.

The dataset registry is shared with the neural runner — register a
loader once via `nstad_bench.experiments.runner.register_dataset` and
either runner can consume it.

## Files

| File | Purpose |
|------|---------|
| `compatibility.yaml`        | φ × θ compatibility matrix for statistical models |
| `experiment_example.yaml`   | Fully annotated template |
| `mitbih.yaml`               | MIT-BIH DS1→DS2 — statistical models + DA |
| `deepbeat.yaml`             | DeepBeat patient split — statistical models + DA |
| `stead.yaml`                | STEAD NC↔SC — statistical models + DA |
| `smoke_minimal.yaml`        | 30-second smoke test |

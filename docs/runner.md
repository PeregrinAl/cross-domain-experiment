# Experiment runner

## Two-stage pipeline

```
Stage 1 — Screening
  All (φ, θ) × SourceOnly × datasets × seeds
  → rank by mean ΔAUC → keep top-K pairs

Stage 2 — Adaptation
  Top-K pairs × all ψ × random-search HP trials × seeds
  → collect metrics → save Parquet
```

## YAML config

```yaml
experiment_name: my_experiment
output_dir:      results/
n_bootstrap:     200

screening:
  top_k:  2
  metric: delta_auc   # delta_auc | roc_auc | source_roc_auc

random_search:
  n_trials:  10
  seeds:     [0, 1, 2]
  base_seed: 42

datasets:
  - my_dataset_key

representations:
  RawSignal: {}
  LogSTFT:
    n_fft: 128

models:
  InceptionTime1D:
    epochs: 30
    lr:     1.0e-3
    batch_size: 64

adaptation_methods:
  SourceOnly: {}
  MK_MMD:
    lr:         {type: log_float, low: 1e-5, high: 1e-3}
    lambda_mmd: {type: log_float, low: 0.1,  high: 10.0}
    batch_size: {type: choice, choices: [32, 64]}
```

## Output schema

Long-format Parquet, one row per **(run × metric)**:

| Column | Type | Description |
|--------|------|-------------|
| `config_hash` | str | SHA-256 prefix — unique run identifier |
| `dataset` | str | Dataset key |
| `phi` | str | Representation name |
| `theta` | str | Model name |
| `psi` | str | Adaptation method |
| `seed` | int | Random seed |
| `metric_name` | str | `roc_auc` / `pr_auc` / `delta_auc` / `gain` / `source_roc_auc` |
| `metric_value` | float | Point estimate |
| `metric_ci_lower` | float | Bootstrap 95% CI lower (NaN for scalar metrics) |
| `metric_ci_upper` | float | Bootstrap 95% CI upper |

## Dry run

```python
df = run_experiment("configs/mitbih.yaml", dry_run=True)
# logs queue size without executing any run
```

## CLI analysis

```bash
nstad-analyze results/mitbih.parquet
nstad-analyze results/mitbih.parquet -o paper/outputs --alpha 0.01
nstad-analyze results/mitbih.parquet --no-plots --formats pdf
```

# Adaptation methods (ψ)

| Name | Type | HP space | Reference |
|------|------|----------|-----------|
| `SourceOnly` | Baseline (no adaptation) | — | — |
| `MK_MMD` | Feature alignment (MMD) | `n_epochs`, `lr`, `lambda_mmd`, `batch_size` | Gretton et al. 2012 |
| `CoDATS` | Adversarial DA | `n_epochs`, `lr`, `lr_disc`, `lambda_domain`, `batch_size` | Wilson et al. 2020 |
| `M2N2` | Normalisation flow | `n_steps`, `lr`, `batch_size` | — |

!!! note "No target labels"
    All methods receive only **unlabelled** target samples `X_target` during
    adaptation. `CoDATS` uses domain labels `{0=source, 1=target}` — these are
    **not** class labels and do not constitute leakage.

## HP search spec

```yaml
# log-uniform float (sampled in log10 space)
lr: {type: log_float, low: 1.0e-5, high: 1.0e-3}

# uniform float
lambda: {type: float, low: 0.1, high: 10.0}

# uniform integer
n_epochs: {type: int, low: 5, high: 30}

# categorical
batch_size: {type: choice, choices: [32, 64, 128]}
```

HP trials are **deterministic**: seed derived from `sha256(f"{psi}_{trial}_{base_seed}")`.

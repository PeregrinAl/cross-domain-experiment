# Models (θ)

| Name | Input | Parameters | Reference |
|------|-------|-----------|-----------|
| `InceptionTime1D` | `(N, C, T)` 1-D | `nb_filters`, `depth` | Fawaz et al. 2020 |
| `PatchTST` | `(N, 1, T)` 1-D | `d_model`, `n_heads`, `n_layers` | Nie et al. 2023 |
| `ResNet18_2D` | `(N, 1, F, T)` 2-D | — | He et al. 2016 |

`in_channels` and `seq_len` are **auto-detected** from data shape and should
not be specified in the config.

## Training keys (shared across all models)

```yaml
epochs:     30      # training epochs
lr:         1.0e-3  # initial learning rate (AdamW)
batch_size: 64      # mini-batch size
```

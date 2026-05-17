# Representations (φ)

| Name | Output shape | Compatible models | Description |
|------|-------------|-------------------|-------------|
| `RawSignal` | `(N, T)` | InceptionTime1D, PatchTST | Identity transform + per-sample z-score |
| `LogSTFT` | `(N, F, T)` | ResNet18_2D | Log-power Short-Time Fourier Transform spectrogram |
| `CWT_Morlet` | `(N, F, T)` | ResNet18_2D | Continuous Wavelet Transform (Morlet wavelet) |
| `CARLA_SSL` | `(N, D)` | InceptionTime1D, PatchTST | Self-supervised ECG embeddings (CARLA pretrained) |

Compatibility is enforced by `configs/compatibility.yaml`.  Incompatible pairs
are silently skipped — no error is raised.

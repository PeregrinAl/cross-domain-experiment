# Analysis & reporting

## One-command pipeline

```bash
nstad-analyze results/mitbih.parquet
```

Reads Parquet → writes `results/mitbih/tables/*.tex` + `results/mitbih/figures/*.{pdf,png}`.
Re-running adds a timestamp suffix (`mitbih_20260517_2030`) — no overwrites.

## Python API

```python
from nstad_bench.analysis import analyze_experiment

report = analyze_experiment(
    "results/mitbih.parquet",
    output_root = "paper/",   # optional; default = same dir as Parquet
    alpha       = 0.05,
    figures     = True,
    tables      = True,
    anova       = True,
)
print(report)
```

## Tables

| File | Content |
|------|---------|
| `delta_auc_pivot.tex` | (φ×θ) × dataset heatmap of mean ΔAUC |
| `gain_by_dataset.tex` | ψ × dataset pivot of mean Gain |
| `method_summary.tex` | Per-method mean ± CI for all metrics |
| `screening_summary.tex` | (φ,θ) ranked by mean ΔAUC |
| `anova.tex` | Three-factor ANOVA with partial η² |
| `metadata.json` | Source file, git hash, library versions, timestamp |

## Figures

| File | Content |
|------|---------|
| `delta_auc_heatmap.{pdf,png}` | Seaborn heatmap |
| `gain_barplot.{pdf,png}` | Grouped bar chart with CI error bars |
| `cd_diagram.{pdf,png}` | Critical Difference diagram (Demšar 2006) |
| `metadata.json` | Same provenance record as in tables/ |

## ANOVA

Three-factor OLS with Type II SS:

$$\Delta\text{AUC} \sim C(\varphi) + C(\theta) + C(\psi) + C(\varphi):C(\theta) + C(\varphi):C(\psi) + C(\theta):C(\psi)$$

Effect size: partial η² = SS_effect / (SS_effect + SS_residual).

```python
from nstad_bench.analysis import run_anova, effect_size_ranking

result = run_anova(df)
print(result)                         # ANOVA table + interpretation
print(effect_size_ranking(result))    # sorted by partial η²
result.to_latex("tables/anova.tex")   # booktabs + significance stars
```

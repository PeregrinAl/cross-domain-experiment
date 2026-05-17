# Metrics

## Scores

| Metric | Description |
|--------|-------------|
| `roc_auc` | ROC-AUC on target domain with 95% bootstrap CI |
| `pr_auc` | Precision-Recall AUC with 95% bootstrap CI |
| `source_roc_auc` | ROC-AUC on source training data (proxy for source performance) |
| `delta_auc` | `source_roc_auc − roc_auc` — measures domain gap |
| `gain` | `roc_auc_adapted − roc_auc_SourceOnly` — measures adaptation benefit |

!!! warning "Point-adjust F1 is not implemented"
    Point-adjust F1 is methodologically unsound for our binary classification
    setting and is intentionally excluded.

## Bootstrap CI

```python
from nstad_bench.metrics import bootstrap_ci, roc_auc

ci = bootstrap_ci(roc_auc, y_true, y_score, n_bootstrap=1000, confidence=0.95)
print(ci.estimate, ci.lower, ci.upper)
```

Degenerate resamples (single-class) are skipped automatically.

## Statistical tests

```python
from nstad_bench.metrics import wilcoxon_test, friedman_nemenyi

# Paired comparison
result = wilcoxon_test(scores_a, scores_b)
print(result.p_value, result.is_significant())

# Multiple comparison
result = friedman_nemenyi(scores_matrix, method_names=["A", "B", "C"])
print(result.significant_pairs())
```

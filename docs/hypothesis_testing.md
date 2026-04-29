[Back to README](../README.md)

# Hypothesis Testing (Significance Evaluation)

RadEval includes paired approximate randomization (AR) testing to determine whether differences between systems are statistically significant.

## Statistical Background

**Approximate Randomization** is a non-parametric test that makes no assumptions about score distributions, making it well-suited for evaluation metrics that may not follow normal distributions.

The procedure:
1. Compute the observed score difference (delta) between two systems
2. For each of N trials, randomly swap outputs between the systems at the sample level and recompute the delta
3. The p-value is the proportion of trials where the randomized delta >= the observed delta
4. If p < 0.05, reject the null hypothesis that the systems perform equally

## Basic Usage

```python
from radeval import RadEval, compare_systems

evaluator = RadEval(metrics=["bleu"])

signatures, scores = compare_systems(
    systems={
        'baseline': baseline_reports,
        'improved': improved_reports,
    },
    metrics={
        'bleu': lambda hyps, refs: evaluator(refs, hyps)['bleu'],
    },
    references=reference_reports,
    n_samples=10000,          # Number of randomization trials
)
```

The first system is treated as the baseline. Each subsequent system is compared pairwise against it.

## Full Example

### Step 1: Define systems and references

```python
from radeval import RadEval, compare_systems

references = [
    "No acute cardiopulmonary process.",
    "No radiographic findings to suggest pneumonia.",
    "Mild cardiomegaly with clear lung fields.",
    "Small pleural effusion on the right side.",
    "Status post cardiac surgery with stable appearance.",
]

systems = {
    'baseline': [
        "No acute findings.",
        "No pneumonia.",
        "Mild cardiomegaly, clear lungs.",
        "Small right pleural effusion.",
        "Post-cardiac surgery, stable."
    ],
    'improved': [
        "No acute cardiopulmonary process.",
        "No radiographic findings suggesting pneumonia.",
        "Mild cardiomegaly with clear lung fields bilaterally.",
        "Small pleural effusion present on the right side.",
        "Status post cardiac surgery with stable appearance."
    ],
}
```

### Step 2: Define metrics

Each metric function receives `(hyps, refs)` and must return a scalar (or a tuple/list/dict -- see below).

```python
bleu_evaluator = RadEval(metrics=["bleu"])
rouge_evaluator = RadEval(metrics=["rouge"])

metrics = {
    'bleu': lambda hyps, refs: bleu_evaluator(refs, hyps)['bleu'],
    'rouge1': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge1'],
}
```

You can also add custom metrics:

```python
def word_count_metric(hyps, refs):
    return sum(len(r.split()) for r in hyps) / len(hyps)

metrics['word_count'] = word_count_metric
```

### Step 3: Run the test

```python
signatures, scores = compare_systems(
    systems=systems,
    metrics=metrics,
    references=references,
    n_samples=10000,
    significance_level=0.05,
    print_results=True,
)
```

Output:

```
================================================================================
PAIRED SIGNIFICANCE TEST RESULTS
================================================================================
System                                             bleu         rouge1
----------------------------------------------------------------------
Baseline: baseline                              0.0000         0.6652
----------------------------------------------------------------------
improved                                      0.6874         0.9531
                                           (p=0.0000)*    (p=0.0800)
----------------------------------------------------------------------
- Significance level: 0.05
- '*' indicates significant difference (p < significance level)
```

### Step 4: Inspect results programmatically

```python
baseline_name = list(systems.keys())[0]
for system_name in systems:
    if system_name == baseline_name:
        continue
    for metric_name in metrics:
        pval = scores[system_name].get(f"{metric_name}_pvalue")
        if pval is not None and pval < 0.05:
            print(f"{system_name} vs {baseline_name}: {metric_name} is significant (p={pval:.4f})")
```

## Metric Function Contract

Each callable in the `metrics` dict must accept `(hyps, refs)` and return one of:

| Return type | How the scalar is extracted |
|------------|----------------------------|
| `float` or `int` | Used directly |
| `tuple` or `list` | `result[0]` |
| `dict` with `'score'` key | `result['score']` |
| `dict` without `'score'` key | First value (fragile -- prefer explicit) |

## Performance Considerations

Each randomization trial recomputes the **full corpus-level metric** from scratch. This means:

| `n_samples` | With BLEU/ROUGE | With RadGraph/GREEN |
|-------------|-----------------|---------------------|
| 1,000 | Seconds | Hours |
| 10,000 | Seconds | Impractical |

For heavy model-based metrics, consider:
- Using lightweight proxy metrics for significance testing
- Precomputing per-sample scores and writing a custom averaging function
- Using a smaller `n_samples` (1,000 minimum recommended)

## Interpreting Results

- **p < 0.05**: Statistically significant difference (marked with `*`)
- **p >= 0.05**: No significant evidence of difference
- Look for consistent significance across multiple metrics, not just one
- Statistical significance does not imply clinical significance -- use domain judgment

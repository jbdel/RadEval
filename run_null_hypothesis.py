#!/usr/bin/env python3

from RadEval import RadEval
from utils import compare_systems


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
    'poor': [
        "Normal.",
        "OK.",
        "Heart big.",
        "Some fluid.",
        "Surgery done."
    ]
}

def word_count_metric(hyps, refs):
    return sum(len(report.split()) for report in hyps) / len(hyps)

# Set up metrics
evaluator = RadEval(
    do_bleu=True, 
    do_rouge=True, 
    do_bertscore=True,
    do_radgraph=True,
    do_chexbert=True
)

metrics = {
    'bleu': lambda hyps, refs: evaluator.bleu_scorer(refs, hyps)[0],
    'rouge1': lambda hyps, refs: evaluator.rouge_scorers["rouge1"](refs, hyps)[0],
    'rouge2': lambda hyps, refs: evaluator.rouge_scorers["rouge2"](refs, hyps)[0],
    'rougeL': lambda hyps, refs: evaluator.rouge_scorers["rougeL"](refs, hyps)[0],
    'bertscore': lambda hyps, refs: evaluator.bertscore_scorer(refs, hyps)[0],
    'radgraph': lambda hyps, refs: evaluator.radgraph_scorer(refs, hyps)[0],
    'chexbert': lambda hyps, refs: evaluator.chexbert_scorer(refs, hyps)[0],
    'word_count': word_count_metric,
}

print("Running tests...")
signatures, scores = compare_systems(
    systems=systems,
    metrics=metrics,
    references=references,
    n_samples=50,
    significance_level=0.05,
    print_results=True
)

# Significance testing
print("\nSignificant differences (p < 0.05):")
baseline_name = list(systems.keys())[0]

for system_name in systems.keys():
    if system_name == baseline_name:
        continue
        
    significant_metrics = []
    for metric_name in metrics.keys():
        pvalue_key = f"{metric_name}_pvalue"
        if pvalue_key in scores[system_name]:
            p_val = scores[system_name][pvalue_key]
            if p_val < 0.05:
                significant_metrics.append(metric_name)
    
    if significant_metrics:
        print(f"  {system_name} vs {baseline_name}: {', '.join(significant_metrics)}")
    else:
        print(f"  {system_name} vs {baseline_name}: No significant differences")
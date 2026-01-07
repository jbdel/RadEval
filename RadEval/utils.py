# ---------------------------------------------------------------
# This file includes code adapted from:
# https://github.com/jbdel/RadEval/blob/null-hypothesis/utils.py
# Original author: Justin Xu
# ---------------------------------------------------------------
import re
import nltk
import random
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict

nltk.download("punkt_tab", quiet=True)

def clean_numbered_list(text):
    """
    Clean a report if it's a numbered list by:
    1. Adding proper spacing between numbered items
    2. Removing the numbered list markers
    3. Adding spaces after periods between sentences
    """
    # First, separate numbered items that are stuck together without spaces
    # Example: "textx.2. text2" -> "texty. 2. text2"
    text = re.sub(r'\.(\d+\.)', r'. \1', text)
    
    # Handle patterns where there's no period between numbered entries
    # Example: "1. item1 2. item2" -> "1. item1. 2. item2"
    text = re.sub(r'(\d+\.\s*[^.]+?)\s+(?=\d+\.)', r'\1. ', text)
    
    # Then remove the numbered list markers
    # But avoid removing decimal numbers in measurements like "3.5 cm"
    text = re.sub(r'(?<!\d)\d+\.\s*', '', text)
    
    # Add spaces after periods between sentences if missing
    # Example: "sentence1.sentence2" -> "sentence1. sentence2"
    # But don't split decimal numbers like "3.5 cm"
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)
    return nltk.sent_tokenize(text)

class PairedTest:
    """
    Paired significance testing for comparing radiology report generation systems.
    
    Supports paired approximate randomization (AR).
    """
    
    def __init__(self, 
                 systems: Dict[str, List[str]], 
                 metrics: Dict[str, Callable],
                 references: Optional[List[str]],
                 n_samples: int = 10000,
                 n_jobs: int = 1,
                 seed: int = 12345):
        """
        Args:
            systems: Dictionary mapping system names to their generated reports
            metrics: Dictionary mapping metric names to metric functions
            references: List of reference reports
            n_samples: Number of resampling trials (default: 10000)
            n_jobs: Number of parallel jobs (default: 1)
            seed: Random seed for reproducibility
        """
        self.systems = systems
        self.metrics = metrics
        self.references = references
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.seed = seed
        
        random.seed(seed)
        
        if not systems:
            raise ValueError("At least one system is required")
        
        system_lengths = [len(outputs) for outputs in systems.values()]
        if len(set(system_lengths)) > 1:
            raise ValueError("All systems must have the same number of outputs")
        
        if references and len(references) != system_lengths[0]:
            raise ValueError("References must have same length as system outputs")
        
        self.n_instances = system_lengths[0]
        
    def __call__(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
        """
        Run the paired significance test.
        
        Returns:
            Tuple of (signatures, scores) where:
            - signatures: Dict mapping metric names to signature strings
            - scores: Dict mapping system names to metric scores and p-values
        """
        # Calculate baseline scores for all systems and metrics
        baseline_scores = self._calculate_baseline_scores()
        
        # Get baseline system (first system)
        baseline_name = list(self.systems.keys())[0]
        
        scores = {}
        signatures = {}
        
        # Calculate scores and p-values for each system
        for system_name in self.systems.keys():
            scores[system_name] = {}
            
            for metric_name in self.metrics.keys():
                score = baseline_scores[system_name][metric_name]
                scores[system_name][metric_name] = score
                
                if system_name != baseline_name:
                    p_value = self._calculate_p_value(
                        baseline_name, system_name, metric_name, baseline_scores
                    )
                    scores[system_name][f'{metric_name}_pvalue'] = p_value
        
        for metric_name in self.metrics.keys():
            signatures[metric_name] = f"{metric_name}|{'ar'}:{self.n_samples}|seed:{self.seed}"
        
        return signatures, scores
    
    def _calculate_baseline_scores(self) -> Dict[str, Dict[str, float]]:
        """Calculate baseline scores for all systems and metrics."""
        scores = defaultdict(dict)
        
        for system_name, outputs in self.systems.items():
            for metric_name, metric_func in self.metrics.items():
                if self.references:
                    score = metric_func(outputs, self.references)
                else:
                    score = metric_func(outputs)
                
                if isinstance(score, dict):
                    if 'score' in score:
                        scores[system_name][metric_name] = score['score']
                    else:
                        scores[system_name][metric_name] = list(score.values())[0]
                elif isinstance(score, (tuple, list)):
                    scores[system_name][metric_name] = score[0]
                else:
                    scores[system_name][metric_name] = score
        
        return scores
    
    def _calculate_p_value(self, 
                          baseline_name: str, 
                          system_name: str, 
                          metric_name: str,
                          baseline_scores: Dict[str, Dict[str, float]]) -> float:
        """Calculate p-value using AR test"""
        
        baseline_outputs = self.systems[baseline_name]
        system_outputs = self.systems[system_name]
        metric_func = self.metrics[metric_name]
        
        baseline_score = baseline_scores[baseline_name][metric_name]
        system_score = baseline_scores[system_name][metric_name]
        original_delta = abs(system_score - baseline_score)
        
        return self._approximate_randomization_test(
            baseline_outputs, system_outputs, metric_func, original_delta
        )
    
    def _approximate_randomization_test(self, 
                                      baseline_outputs: List[str],
                                      system_outputs: List[str],
                                      metric_func: Callable,
                                      original_delta: float) -> float:
        """
        Perform AR test.
        
        For each trial, randomly swap outputs between systems and calculate
        the score difference. P-value is the proportion of trials where
        the randomized delta >= original delta.
        """
        count_greater = 0
        
        for _ in range(self.n_samples):
            randomized_baseline = []
            randomized_system = []
            
            for i in range(self.n_instances):
                if random.random() < 0.5:
                    # Don't swap
                    randomized_baseline.append(baseline_outputs[i])
                    randomized_system.append(system_outputs[i])
                else:
                    # Swap
                    randomized_baseline.append(system_outputs[i])
                    randomized_system.append(baseline_outputs[i])
            
            if self.references:
                rand_baseline_score = metric_func(randomized_baseline, self.references)
                rand_system_score = metric_func(randomized_system, self.references)
            else:
                rand_baseline_score = metric_func(randomized_baseline)
                rand_system_score = metric_func(randomized_system)
            
            if isinstance(rand_baseline_score, dict):
                rand_baseline_score = rand_baseline_score.get('score', list(rand_baseline_score.values())[0])
            elif isinstance(rand_baseline_score, (tuple, list)):
                rand_baseline_score = rand_baseline_score[0]
                
            if isinstance(rand_system_score, dict):
                rand_system_score = rand_system_score.get('score', list(rand_system_score.values())[0])
            elif isinstance(rand_system_score, (tuple, list)):
                rand_system_score = rand_system_score[0]
            
            rand_delta = abs(rand_system_score - rand_baseline_score)
            
            if rand_delta >= original_delta:
                count_greater += 1
        
        return count_greater / self.n_samples
    

def print_significance_results(scores: Dict[str, Dict[str, float]], 
                             signatures: Dict[str, str],
                             baseline_name: str,
                             significance_level: float = 0.05):
    """    
    Args:
        scores: Dictionary of system scores and p-values
        signatures: Dictionary of metric signatures
        baseline_name: Name of the baseline system
        significance_level: Significance threshold (default: 0.05)
    """
    assert baseline_name in scores, f"Baseline system '{baseline_name}' not found in scores."
    
    metric_names = [name for name in signatures.keys()]
    system_names = list(scores.keys())
    
    print("=" * 80)
    print("PAIRED SIGNIFICANCE TEST RESULTS")
    print("=" * 80)
    
    header = f"{'System':<40}"
    for metric in metric_names:
        header += f"{metric:>15}"
    print(header)
    print("-" * len(header))
    
    baseline_row = f"Baseline: {baseline_name:<32}"
    for metric in metric_names:
        score = scores[baseline_name][metric]
        baseline_row += f"{score:>12.4f}   "
    print(baseline_row)
    print("-" * len(header))
    
    for system_name in system_names:
        if system_name == baseline_name:
            continue
            
        system_row = f"{system_name:<40}"
        for metric in metric_names:
            score = scores[system_name].get(metric, 0.0)
            if isinstance(score, float):
                system_row += f"{score:>12.4f}   "
            else:
                system_row += f"{str(score):>12}   "
        print(system_row)
        
        # P-value row
        pvalue_row = " " * 40
        for metric in metric_names:            
            pvalue_key = f"{metric}_pvalue"
            if pvalue_key in scores[system_name]:
                p_val = scores[system_name][pvalue_key]
                significance_marker = "*" if p_val < significance_level else ""
                pvalue_row += f"(p={p_val:.4f}){significance_marker:<2}".rjust(15)
            else:
                pvalue_row += " " * 15
        print(pvalue_row)
        print("-" * len(header))
    
    # Footer
    print(f"- Significance level: {significance_level}")
    print("- '*' indicates significant difference (p < significance level)")
    print("- Null hypothesis: systems are essentially the same")
    print("- Significant results suggest systems are meaningfully different\n")
    
    print("METRIC SIGNATURES:")
    for metric, signature in signatures.items():
        print(f"- {metric}: {signature}")


def compare_systems(systems: Dict[str, List[str]],
                   metrics: Dict[str, Callable],
                   references: Optional[List[str]] = None,
                   n_samples: int = 10000,
                   significance_level: float = 0.05,
                   seed: int = 12345,
                   print_results: bool = True) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """    
    Args:
        systems: Dictionary mapping system names to their generated reports
        metrics: Dictionary mapping metric names to metric functions
        references: Optional list of reference reports
        n_samples: Number of resampling trials
        significance_level: Significance threshold for printing results
        seed: Random seed for reproducibility
        print_results: Whether to print formatted results
    
    Returns:
        Tuple of (signatures, scores)
    
    Example:
        ```python
        systems = {
            'baseline_model': baseline_reports,
            'new_model': new_model_reports,
            'other_model': other_model_reports
        }
        
        metrics = {
            'bleu': lambda hyp, ref: bleu_score(hyp, ref),
            'rouge': lambda hyp, ref: rouge_score(hyp, ref),
            'bertscore': lambda hyp, ref: bert_score(hyp, ref)
            'custom_metric': lambda hyp, ref: custom_metric(hyp, ref)
        }
        
        signatures, scores = compare_systems(
            systems, metrics, references, 
            n_samples=10000
        )
        ```
    """
    
    paired_test = PairedTest(
        systems=systems,
        metrics=metrics, 
        references=references,
        n_samples=n_samples,
        seed=seed
    )
    
    signatures, scores = paired_test()
    
    if print_results:
        baseline_name = list(systems.keys())[0]
        print_significance_results(scores, signatures, baseline_name, significance_level)
    
    return signatures, scores

def multilabel_prf_per_sample(y_true, y_pred, eps=1e-8, aggregate="mean", include="union"):
    """
    Per-sample aggregated precision/recall/f1 for multi-label classification,
    by aggregating per-label scores within each sample.

    Args:
        y_true, y_pred: array-like, shape (N, K), values {0,1}
        aggregate:
            - "mean":   average over selected labels
            - "weighted": weight by per-sample label support (y_true) over selected labels
        include:
            - "union":  labels where (y_true==1) OR (y_pred==1) for that sample
            - "true":   labels where (y_true==1) for that sample only

    Returns:
        sample_precision: np.ndarray shape (N,)
        sample_recall:    np.ndarray shape (N,)
        sample_f1:        np.ndarray shape (N,)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # per-sample, per-label confusion terms: each is (N, K) with 0/1 values
    tp = ((y_true == 1) & (y_pred == 1)).astype(int)
    fp = ((y_true == 0) & (y_pred == 1)).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).astype(int)

    # per-sample, per-label P/R/F1 (still (N, K))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)

    # choose which labels to include in each sample’s aggregation
    if include == "true":
        mask = (y_true == 1)
    elif include == "union":
        mask = (y_true == 1) | (y_pred == 1)
    else:
        raise ValueError("include must be 'union' or 'true'")

    # aggregate per sample
    sample_precision = np.zeros(y_true.shape[0], dtype=float)
    sample_recall = np.zeros(y_true.shape[0], dtype=float)
    sample_f1 = np.zeros(y_true.shape[0], dtype=float)

    for i in range(y_true.shape[0]):
        m = mask[i]
        if not np.any(m):
            # no relevant labels in this sample:
            # you can choose 1.0 or 0.0; 1.0 avoids penalising empty-empty cases.
            sample_precision[i] = 1.0
            sample_recall[i] = 1.0
            sample_f1[i] = 1.0
            continue

        if aggregate == "mean":
            sample_precision[i] = float(p[i][m].mean())
            sample_recall[i] = float(r[i][m].mean())
            sample_f1[i] = float(f1[i][m].mean())

        elif aggregate == "weighted":
            # weight by true-label support within the sample (gives more weight to labels present in y_true)
            w = y_true[i][m].astype(float)
            if w.sum() == 0:
                # if include="union" and this sample has only predicted labels (no true labels),
                # fallback to mean
                sample_precision[i] = float(p[i][m].mean())
                sample_recall[i] = float(r[i][m].mean())
                sample_f1[i] = float(f1[i][m].mean())
            else:
                sample_precision[i] = float((p[i][m] * w).sum() / (w.sum() + eps))
                sample_recall[i] = float((r[i][m] * w).sum() / (w.sum() + eps))
                sample_f1[i] = float((f1[i][m] * w).sum() / (w.sum() + eps))
        else:
            raise ValueError("aggregate must be 'mean' or 'weighted'")

    return sample_precision, sample_recall, sample_f1
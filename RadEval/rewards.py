"""TRL-compatible reward function wrappers for RadEval metrics.

TRL is NOT a dependency — only the interface convention is adopted.
"""
import warnings
from typing import Callable, Optional


def make_reward_fn(
    metric: str,
    reference_column: str = "ground_truth",
    score_transform: Optional[Callable[[float], float]] = None,
    **metric_kwargs,
):
    """Wrap a RadEval metric as a TRL-compatible reward function.

    Args:
        metric: metric name (e.g. "bertscore", "bleu", "f1chexbert")
        reference_column: dataset column with reference reports
        score_transform: optional fn to normalize/scale each score
        **metric_kwargs: forwarded to the metric adapter's __init__

    Returns:
        Callable with signature (completions, **kwargs) -> list[float]
    """
    from .metrics._registry import get_metric_class

    cls = get_metric_class(metric)

    if getattr(cls, 'is_api_based', False):
        warnings.warn(
            f"'{metric}' makes API calls per sample — very slow and expensive "
            f"as an RL reward. Consider fast local metrics: bleu, rouge, "
            f"bertscore, radeval_bertscore, f1chexbert, f1radbert_ct.",
            UserWarning, stacklevel=2,
        )

    scorer = cls(**metric_kwargs)
    key = scorer.metric_keys()[0]

    def reward_fn(completions, **kwargs):
        refs = kwargs[reference_column]
        result = scorer.compute(refs=refs, hyps=completions, per_sample=True)
        values = result[key]
        if score_transform is not None:
            values = [score_transform(v) for v in values]
        return values

    return reward_fn

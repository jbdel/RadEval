"""RadEval-facing adapter for the NoduleEval metric.

Wraps NoduleEvalScore and emits per-dimension output keys in the shape
expected by RadEval.compute_scores() across all three output modes
(default, per_sample, detailed).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._base import MetricBase

logger = logging.getLogger(__name__)


# Primary aggregate metric keys (one value each in default/detailed mode;
# per-row lists in per_sample mode).
_AGGREGATE_KEYS: tuple[str, ...] = (
    "nodule_eval_detection_precision",
    "nodule_eval_detection_recall",
    "nodule_eval_detection_f1",
    "nodule_eval_size_accuracy",
    "nodule_eval_size_exact_match",
    "nodule_eval_size_mae_mm",
    "nodule_eval_size_mape",
    "nodule_eval_type_accuracy",
    "nodule_eval_location_accuracy",
    "nodule_eval_noun_accuracy",
    "nodule_eval_uncertainty_accuracy",
    "nodule_eval_composite",
)

# Mapping from output key -> DataFrame column in results_df.
_COL_MAP: dict[str, str] = {
    "nodule_eval_detection_precision": "detection_precision",
    "nodule_eval_detection_recall":    "detection_recall",
    "nodule_eval_detection_f1":        "detection_f1",
    "nodule_eval_size_accuracy":       "size_accuracy",
    "nodule_eval_size_exact_match":    "size_exact_match",
    "nodule_eval_size_mae_mm":         "size_mae_mm",
    "nodule_eval_size_mape":           "size_mape",
    "nodule_eval_type_accuracy":       "type_accuracy",
    "nodule_eval_location_accuracy":   "location_accuracy",
    "nodule_eval_noun_accuracy":       "noun_accuracy",
    "nodule_eval_uncertainty_accuracy": "uncertainty_accuracy",
    "nodule_eval_composite":           "composite",
}


def _mean_skip_none(values) -> float | None:
    clean = [v for v in values
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _std_skip_none(values) -> float | None:
    clean = [v for v in values
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return 0.0 if clean else None
    return float(np.std(clean))


class NoduleEvalMetric(MetricBase):
    name = "nodule_eval"
    display_name = "NoduleEval"
    is_api_based = True

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        max_concurrent: int = 50,
        cache_dir: str | None = None,  # accepted but unused (no local caching)
        **kwargs,
    ):
        from .nodule_eval import NoduleEvalScore
        self._scorer = NoduleEvalScore(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, "cost_tracker", None)

    def metric_keys(self, detailed: bool = False) -> list[str]:
        keys = list(_AGGREGATE_KEYS)
        if detailed:
            keys.extend([k + "_std" for k in _AGGREGATE_KEYS])
        return keys

    def compute(
        self,
        refs: list[str],
        hyps: list[str],
        per_sample: bool = False,
        detailed: bool = False,
        on_progress=None,
    ) -> dict[str, Any]:
        """Run scoring and return output in the requested mode."""
        _, _, _, results_df = self._scorer(
            refs, hyps, on_sample_done=on_progress,
        )

        out: dict[str, Any] = {}
        for out_key, col in _COL_MAP.items():
            series = results_df[col].tolist() if col in results_df.columns else []

            if per_sample:
                out[out_key] = series
            else:
                mean = _mean_skip_none(series)
                out[out_key] = round(mean, 4) if mean is not None else None
                if detailed:
                    std = _std_skip_none(series)
                    out[out_key + "_std"] = round(std, 4) if std is not None else None

        return out

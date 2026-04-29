"""NoduleEval scorer — LLM-as-judge metric for pulmonary nodules section.

Takes full clean_findings strings (ref, hyp), extracts the PULMONARY NODULES:
segment from each, sends them to a judge LLM, parses the structured JSON
response, and computes per-row metrics deterministically.

Supported providers: openai, gemini.
"""
from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from .._llm_base import LLMMetricBase
from .prompt_parts import SYSTEM_MSG, build_prompt
from .utils import (
    compute_per_row_metrics,
    empty_row_result,
    extract_pn_segment,
    parse_json_response,
    validate_response,
)

logger = logging.getLogger(__name__)


# Output keys that are valid floats (skip per-row values that are None when aggregating).
_METRIC_KEYS: tuple[str, ...] = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "size_accuracy",
    "size_exact_match",
    "size_mae_mm",
    "size_mape",
    "type_accuracy",
    "location_accuracy",
    "noun_accuracy",
    "uncertainty_accuracy",
    "composite",
)


def _mean_skip_none(values: list) -> Optional[float]:
    """Mean of non-None numeric values; None if list empty after filtering."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _std_skip_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return 0.0 if clean else None
    return float(np.std(clean))


class NoduleEvalScore(LLMMetricBase):
    """Nodule-focused LLM-as-judge scorer.

    Usage:
        scorer = NoduleEvalScore(provider="gemini", model_name="gemini-2.5-flash")
        mean_composite, std_composite, per_row_scores, results_df = scorer(refs, hyps)

    `per_row_scores` is a list of composite scores (one per input pair).
    `results_df` is a DataFrame with one row per sample and one column per metric.

    Default provider is gemini (passes self-consistency test); gpt-4o-mini is
    supported as a fallback but has a known self-consistency defect — see
    /fss/jb/RadEval/logs/nodule_eval_pilot/pilot_summary.md.
    """

    SUPPORTED_PROVIDERS: ClassVar[set[str]] = {"openai", "gemini"}

    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        max_concurrent: int = 50,
    ):
        if model_name is None:
            model_name = (
                self.DEFAULT_OPENAI_MODEL if provider == "openai"
                else self.DEFAULT_GEMINI_MODEL
            )
        super().__init__(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
        )

    # ------------------------------------------------------------------
    # LLMMetricBase interface
    # ------------------------------------------------------------------

    def _build_request(self, ref: str, hyp: str, **kwargs) -> dict[str, Any]:
        """Build the provider-specific request payload.

        ref and hyp are full clean_findings strings; we extract the
        PULMONARY NODULES: segment before prompting.
        """
        ref_pn = extract_pn_segment(ref)
        hyp_pn = extract_pn_segment(hyp)
        prompt = build_prompt(ref_pn, hyp_pn)

        if self.provider == "openai":
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "seed": 42,
                "response_format": {"type": "json_object"},
            }
        elif self.provider == "gemini":
            from google.genai import types
            return {
                "contents": prompt,
                "config": types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    system_instruction=SYSTEM_MSG,
                ),
            }
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    def _parse_response(self, raw: str) -> dict:
        """Parse raw LLM response -> per-row metric dict."""
        data = parse_json_response(raw)
        validate_response(data)
        return compute_per_row_metrics(data)

    # ------------------------------------------------------------------
    # Override _evaluate_one to short-circuit both-empty rows.
    # ------------------------------------------------------------------

    def _short_circuit(self, ref: str, hyp: str) -> Optional[dict]:
        """Return a pre-computed result if the row doesn't need an LLM call."""
        ref_pn = extract_pn_segment(ref)
        hyp_pn = extract_pn_segment(hyp)
        if not ref_pn and not hyp_pn:
            return empty_row_result(both_empty=True)
        return None

    def _evaluate_one(self, ref, hyp, max_retries=2, **kwargs):
        # Short-circuit for both-empty rows (no LLM call needed).
        sc = self._short_circuit(ref, hyp)
        if sc is not None:
            return sc
        try:
            return super()._evaluate_one(ref, hyp, max_retries=max_retries, **kwargs)
        except RuntimeError as e:
            logger.error("NoduleEval: all attempts failed for one sample: %s", e)
            return self._nan_fallback()

    async def _evaluate_one_async(self, ref, hyp, max_retries=2, **kwargs):
        sc = self._short_circuit(ref, hyp)
        if sc is not None:
            return sc
        try:
            return await super()._evaluate_one_async(ref, hyp, max_retries=max_retries, **kwargs)
        except RuntimeError as e:
            logger.error("NoduleEval: all async attempts failed for one sample: %s", e)
            return self._nan_fallback()

    @staticmethod
    def _nan_fallback() -> dict:
        """Per-row result to use when the LLM call fails persistently."""
        return {
            **{k: None for k in _METRIC_KEYS},
            "n_reference": 0, "n_predicted": 0, "n_matched": 0,
            "n_false_findings": 0, "n_missing_findings": 0,
            "n_size_pairs": 0,
            "n_size_errors": 0, "n_type_errors": 0, "n_location_errors": 0,
            "n_noun_errors": 0, "n_uncertainty_errors": 0,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, results, refs, hyps):
        """Aggregate per-row metric dicts into (mean_composite, std_composite,
        per_sample_composite, results_df).

        Returns the tuple expected by RadEval's adapter (same shape as CRIMSON).
        The DataFrame carries every per-row metric + count for downstream use
        (the adapter pulls specific keys into the RadEval output dict).
        """
        # Composite per-row; used as the "primary" score.
        composite_per_row = [r.get("composite") for r in results]
        mean_composite = _mean_skip_none(composite_per_row) or 0.0
        std_composite = _std_skip_none(composite_per_row) or 0.0

        # Build one-row-per-sample DataFrame.
        rows = []
        for ref, hyp, r in zip(refs, hyps, results):
            rows.append({
                "reference": ref,
                "prediction": hyp,
                **{k: r.get(k) for k in _METRIC_KEYS},
                "n_reference": r.get("n_reference", 0),
                "n_predicted": r.get("n_predicted", 0),
                "n_matched": r.get("n_matched", 0),
                "n_false_findings": r.get("n_false_findings", 0),
                "n_missing_findings": r.get("n_missing_findings", 0),
                "n_size_pairs": r.get("n_size_pairs", 0),
                "n_size_errors": r.get("n_size_errors", 0),
                "n_type_errors": r.get("n_type_errors", 0),
                "n_location_errors": r.get("n_location_errors", 0),
                "n_noun_errors": r.get("n_noun_errors", 0),
                "n_uncertainty_errors": r.get("n_uncertainty_errors", 0),
            })
        results_df = pd.DataFrame(rows)

        return mean_composite, std_composite, composite_per_row, results_df


# Public alias matching RadEval's metric-naming style.
class NoduleEval(NoduleEvalScore):
    """Alias of NoduleEvalScore."""

"""HopprCrimsonCT: CT-adapted CRIMSON clinical significance scoring.

Reuses the CRIMSON scoring engine (weighted error penalty) with CT-specific
prompts for finding extraction, significance classification, and error taxonomy.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..crimson.crimson import (
    CRIMSONScore,
    _extract_json_str,
    _validate_crimson_response,
)
from .prompt_parts_ct import build_prompt_ct

logger = logging.getLogger(__name__)

_SYSTEM_MSG = (
    "You are a radiology AI assistant specialized in CT report evaluation. "
    "You compare predicted CT findings against reference findings and identify "
    "errors. Always respond with valid JSON only."
)


class HopprCrimsonCT(CRIMSONScore):
    """CRIMSON scorer adapted for CT reports with CT-specific prompts.

    Inherits scoring logic from CRIMSONScore, overrides prompt building.
    """

    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        provider="openai",
        model_name=None,
        openai_api_key=None,
        gemini_api_key=None,
        batch_size=1,
        max_concurrent=50,
    ):
        super().__init__(
            provider=provider,
            model_name=model_name or self.DEFAULT_OPENAI_MODEL,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )

    def _build_evaluation_prompt(self, reference_findings, predicted_findings,
                                 patient_context=None, include_guidelines=True):
        """Override to use CT-adapted prompts."""
        return build_prompt_ct(
            reference_findings,
            predicted_findings,
            patient_context=patient_context,
            include_significance_examples=include_guidelines,
            include_attribute_guidelines=include_guidelines,
            include_context_guidelines=include_guidelines,
        )


class CRIMSON_CT:
    """RadEval-facing wrapper for HopprCrimsonCT.

    Matches the interface expected by RadEval.compute_scores():
    __call__(refs, hyps, on_sample_done) -> (mean, std, scores, df)
    """

    def __init__(
        self,
        provider="openai",
        model_name=None,
        openai_api_key=None,
        gemini_api_key=None,
        batch_size=1,
        max_concurrent=50,
    ):
        self.scorer = HopprCrimsonCT(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )
        self.cost_tracker = getattr(self.scorer, "cost_tracker", None)

    def __call__(
        self, refs: List[str], hyps: List[str], on_sample_done=None,
    ) -> Tuple[float, float, list, pd.DataFrame]:
        return self.scorer(refs, hyps, on_sample_done=on_sample_done)

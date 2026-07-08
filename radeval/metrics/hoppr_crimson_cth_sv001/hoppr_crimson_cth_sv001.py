"""hoppr_crimson_cth_sv001: CRIMSON-style factual-correctness judge for CT
head/neck reports.

Scores the PRESENCE axis of a fixed 10-finding schema using a Gemini judge and
the CRIMSON [-1, 1] scoring formula, weighting findings by clinical
significance. Modifier accuracy is handled by the sibling
``hoppr_modifier_cth_sv001`` metric.

Score version: sv001.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from .._llm_base import LLMMetricBase
from .prompt import SYSTEM_MSG, build_prompt

logger = logging.getLogger(__name__)

try:
    from google.genai import types
    GEMINI_TYPES_AVAILABLE = True
except ImportError:
    types = None
    GEMINI_TYPES_AVAILABLE = False


# CRIMSON significance weights (shared with the CT/CXR CRIMSON family).
SIGNIFICANCE_WEIGHTS = {
    "urgent": 1.0,
    "actionable_not_urgent": 0.5,
    "not_actionable_not_urgent": 0.25,
    "benign_expected": 0.0,
}
_DEFAULT_WEIGHT = 0.25


def _extract_json_str(text: str) -> str:
    """Best-effort extraction of a JSON object from a model response."""
    t = (text or "").strip()

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if match:
        t = match.group(1).strip()
    else:
        match = re.search(r"```(?:json)?\s*([\s\S]*)", t)
        if match:
            t = match.group(1).strip()

    if not (t.startswith("{") and t.endswith("}")):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start:end + 1]

    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return t


class HopprCrimsonCthSv001(LLMMetricBase):
    """Factual-correctness (presence-axis) judge for CT head/neck reports.

    ``__call__(refs, hyps)`` returns ``(mean, std, scores, results_df)``.
    """

    SUPPORTED_PROVIDERS: ClassVar[set[str]] = {"openai", "gemini"}

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        provider: str = "gemini",
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        max_concurrent: int = 50,
    ):
        super().__init__(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # Retained after a run for failure-mode analysis.
        self.results_df: Optional[pd.DataFrame] = None
        self.per_sample_results: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # LLMMetricBase interface
    # ------------------------------------------------------------------

    def _build_request(self, ref: str, hyp: str, **kwargs) -> dict[str, Any]:
        prompt = build_prompt(ref, hyp)
        if self.provider == "openai":
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
        if not GEMINI_TYPES_AVAILABLE:
            raise ImportError(
                "google-genai is not installed. "
                "Install it with: pip install google-genai"
            )
        return {
            "contents": prompt,
            "config": types.GenerateContentConfig(
                temperature=self.temperature,
                maxOutputTokens=self.max_output_tokens,
                systemInstruction=SYSTEM_MSG,
                responseMimeType="application/json",
            ),
        }

    def _parse_response(self, raw: str) -> dict:
        cleaned = _extract_json_str(raw)
        try:
            evaluation = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model did not return valid JSON. Raw: {raw[:500]}") from e
        if not isinstance(evaluation, dict) or "errors" not in evaluation:
            raise ValueError(f"Malformed judge response: {raw[:500]}")
        evaluation.setdefault("reference_findings", [])
        evaluation.setdefault("predicted_findings", [])
        evaluation.setdefault("matched_findings", [])
        errors = evaluation["errors"]
        errors.setdefault("false_findings", [])
        errors.setdefault("missing_findings", [])
        errors.setdefault("presence_state_mismatches", [])
        return self._calculate_score(evaluation)

    # ------------------------------------------------------------------
    # Scoring (CRIMSON [-1, 1] formula, presence-only)
    # ------------------------------------------------------------------

    def _calculate_score(self, evaluation: dict) -> dict:
        errors = evaluation.get("errors", {})
        matched = evaluation.get("matched_findings", [])
        ref_findings = evaluation.get("reference_findings", [])
        pred_findings = evaluation.get("predicted_findings", [])

        ref_weight_by_id = {
            r.get("id"): SIGNIFICANCE_WEIGHTS.get(
                r.get("clinical_significance", ""), _DEFAULT_WEIGHT)
            for r in ref_findings
        }
        pred_weight_by_id = {
            p.get("id"): SIGNIFICANCE_WEIGHTS.get(
                p.get("clinical_significance", ""), _DEFAULT_WEIGHT)
            for p in pred_findings
        }

        false_ids = errors.get("false_findings", [])
        missing_ids = errors.get("missing_findings", [])
        presence_mismatches = errors.get("presence_state_mismatches", [])

        E_false = sum(pred_weight_by_id.get(f_id, 0.0) for f_id in false_ids)
        E_miss = sum(ref_weight_by_id.get(m_id, 0.0) for m_id in missing_ids)

        # A presence-state mismatch on a matched finding erodes partial credit,
        # analogous to a "significant" attribute error in CRIMSON.
        mismatch_by_ref_id: dict[str, int] = {}
        for m in presence_mismatches:
            rid = m.get("ref_id")
            if rid is not None:
                mismatch_by_ref_id[rid] = mismatch_by_ref_id.get(rid, 0) + 1
        _MISMATCH_WEIGHT = 0.5

        N_G = sum(ref_weight_by_id.values())
        if N_G == 0 and not ref_findings:
            N_G = len(matched) + E_miss

        E_penalty = E_false

        matched_ref_ids: set = set()
        correct = 0.0
        for m in matched:
            rid = m.get("ref_id")
            if rid in matched_ref_ids:
                continue
            matched_ref_ids.add(rid)
            base_weight = ref_weight_by_id.get(rid, 0.0)
            n_mismatch = mismatch_by_ref_id.get(rid, 0)
            if n_mismatch == 0:
                correct += base_weight
            else:
                sum_err = n_mismatch * _MISMATCH_WEIGHT
                denom = base_weight + sum_err
                credit_factor = base_weight / denom if denom > 0 else 0.0
                correct += base_weight * credit_factor

        errors_more_than_correct = E_penalty - correct

        if N_G == 0:
            score = 1.0 if (E_penalty == 0 and E_miss == 0) else -(
                E_penalty + E_miss + 1)
        else:
            score = (correct - E_penalty) / N_G

        if score >= 0:
            final = score
        elif errors_more_than_correct > 0:
            final = -1 * errors_more_than_correct / (1 + errors_more_than_correct)
        else:
            final = 0.0

        return {
            "raw_evaluation": evaluation,
            "error_counts": {
                "false_findings": len(false_ids),
                "missing_findings": len(missing_ids),
                "presence_state_mismatches": len(presence_mismatches),
            },
            "weighted_error_counts": {
                "false_findings": E_false,
                "missing_findings": E_miss,
            },
            "metrics": {
                "N_G": N_G,
                "E_penalty": E_penalty,
                "correct": correct,
                "S": score,
            },
            "crimson_cth_score": round(float(final), 4),
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, results: list[dict], refs: list[str], hyps: list[str],
    ) -> tuple:
        scores = [r["crimson_cth_score"] for r in results]

        valid = [s for s in scores if s is not None and not np.isnan(s)]
        mean = float(np.mean(valid)) if valid else float("nan")
        std = float(np.std(valid)) if len(valid) > 1 else 0.0

        rows = []
        for ref, hyp, result in zip(refs, hyps, results):
            counts = result.get("error_counts", {})
            rows.append({
                "reference": ref,
                "prediction": hyp,
                "crimson_cth_score": result.get("crimson_cth_score"),
                "false_findings": counts.get("false_findings", 0),
                "missing_findings": counts.get("missing_findings", 0),
                "presence_state_mismatches": counts.get(
                    "presence_state_mismatches", 0),
            })
        results_df = pd.DataFrame(rows)

        self.results_df = results_df
        self.per_sample_results = results
        return mean, std, scores, results_df

    # ------------------------------------------------------------------
    # Failure-mode summary (summed error-type totals over all calls)
    # ------------------------------------------------------------------

    def error_totals(self) -> dict[str, int]:
        """Sum each error-type count across the last run's samples."""
        if self.per_sample_results is None:
            return {}
        totals = {
            "false_findings": 0,
            "missing_findings": 0,
            "presence_state_mismatches": 0,
        }
        for r in self.per_sample_results:
            for k, v in r.get("error_counts", {}).items():
                totals[k] = totals.get(k, 0) + int(v)
        return totals

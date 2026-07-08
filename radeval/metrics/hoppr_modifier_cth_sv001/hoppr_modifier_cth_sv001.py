"""hoppr_modifier_cth_sv001: modifier-accuracy judge for CT head/neck reports.

For the fixed 10-finding schema, scores how accurately the predicted report
captures the non-presence modifiers (severity / localization / laterality /
region / sinus subsite) of findings both reports agree are present. Presence is
out of scope (owned by ``hoppr_crimson_cth_sv001``).

Score = correct_modifier_slots / total_applicable_modifier_slots, in [0, 1],
micro-averaged across all agreed-present findings in a report. Samples with no
agreed-present findings score NaN and are excluded from the mean.

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


# A verdict counts as a scored slot; "correct" is the only one that earns credit.
_VERDICTS = {"correct", "incorrect", "pred_missing", "pred_extra"}

# Map each verdict onto the error-type bucket for failure-mode aggregation.
_MODIFIER_TYPES = ("severity", "localization", "laterality", "region", "sinus")


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


class HopprModifierCthSv001(LLMMetricBase):
    """Modifier-accuracy judge for CT head/neck reports.

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
        if not isinstance(evaluation, dict):
            raise ValueError(f"Malformed judge response: {raw[:500]}")
        evaluation.setdefault("agreed_present_findings", [])
        evaluation.setdefault("modifier_verdicts", [])
        return self._calculate_score(evaluation)

    # ------------------------------------------------------------------
    # Scoring (micro-averaged modifier accuracy, [0, 1])
    # ------------------------------------------------------------------

    def _calculate_score(self, evaluation: dict) -> dict:
        verdicts = evaluation.get("modifier_verdicts", []) or []

        scored = [v for v in verdicts if v.get("verdict") in _VERDICTS]
        total_slots = len(scored)
        correct = sum(1 for v in scored if v.get("verdict") == "correct")

        # Per-modifier-type error counts (any non-correct verdict is an error).
        type_errors = {f"{mt}_errors": 0 for mt in _MODIFIER_TYPES}
        verdict_counts = {
            "correct": 0, "incorrect": 0, "pred_missing": 0, "pred_extra": 0,
        }
        for v in scored:
            verdict = v.get("verdict")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            if verdict != "correct":
                mod = v.get("modifier", "")
                key = f"{mod}_errors"
                if key in type_errors:
                    type_errors[key] += 1

        if total_slots == 0:
            # No agreed-present findings with applicable modifiers -> undefined.
            score = float("nan")
        else:
            score = correct / total_slots

        return {
            "raw_evaluation": evaluation,
            "n_agreed_present": len(
                evaluation.get("agreed_present_findings", [])),
            "total_slots": total_slots,
            "correct_slots": correct,
            "error_counts": {
                **type_errors,
                "incorrect": verdict_counts["incorrect"],
                "pred_missing": verdict_counts["pred_missing"],
                "pred_extra": verdict_counts["pred_extra"],
            },
            "modifier_cth_score": (
                round(float(score), 4) if not np.isnan(score) else float("nan")),
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, results: list[dict], refs: list[str], hyps: list[str],
    ) -> tuple:
        scores = [r["modifier_cth_score"] for r in results]

        valid = [s for s in scores if s is not None and not np.isnan(s)]
        mean = float(np.mean(valid)) if valid else float("nan")
        std = float(np.std(valid)) if len(valid) > 1 else 0.0

        rows = []
        for ref, hyp, result in zip(refs, hyps, results):
            counts = result.get("error_counts", {})
            rows.append({
                "reference": ref,
                "prediction": hyp,
                "modifier_cth_score": result.get("modifier_cth_score"),
                "n_agreed_present": result.get("n_agreed_present", 0),
                "total_slots": result.get("total_slots", 0),
                "correct_slots": result.get("correct_slots", 0),
                "severity_errors": counts.get("severity_errors", 0),
                "localization_errors": counts.get("localization_errors", 0),
                "laterality_errors": counts.get("laterality_errors", 0),
                "region_errors": counts.get("region_errors", 0),
                "sinus_errors": counts.get("sinus_errors", 0),
                "pred_missing": counts.get("pred_missing", 0),
                "pred_extra": counts.get("pred_extra", 0),
            })
        results_df = pd.DataFrame(rows)

        self.results_df = results_df
        self.per_sample_results = results
        return mean, std, scores, results_df

    # ------------------------------------------------------------------
    # Failure-mode summary
    # ------------------------------------------------------------------

    def error_totals(self) -> dict[str, int]:
        """Sum each per-modifier-type error count across the last run."""
        if self.per_sample_results is None:
            return {}
        totals: dict[str, int] = {}
        for r in self.per_sample_results:
            for k, v in r.get("error_counts", {}).items():
                totals[k] = totals.get(k, 0) + int(v)
        return totals

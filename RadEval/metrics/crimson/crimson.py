"""CRIMSON: LLM-based clinical radiology report scoring metric.

Evaluates radiology report quality by comparing predicted findings against
reference findings, using an LLM to identify errors (false findings, missing
findings, attribute errors) and weighting them by clinical significance.

Supports OpenAI API and HuggingFace (MedGemma) backends.

Reference: https://arxiv.org/abs/2603.06183
Code: https://github.com/rajpurkarlab/CRIMSON
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .prompt_parts import build_prompt as _build_evaluation_prompt_fn

logger = logging.getLogger(__name__)

_SYSTEM_MSG = (
    "You are an expert radiology evaluator that assesses "
    "the accuracy of radiology reports."
)


def _extract_json_str(text: str) -> str:
    """Extract JSON from LLM response, handling markdown fences and trailing commas."""
    t = text.strip()
    md = re.search(r"```(?:json)?\s*\n?(.*?)```", t, re.DOTALL)
    if md:
        t = md.group(1).strip()
    else:
        start = t.find("{")
        end_brace = t.rfind("}")
        if start >= 0 and end_brace > start:
            t = t[start:end_brace + 1]
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return t


def _validate_crimson_response(data: dict) -> None:
    """Validate the LLM response has the expected CRIMSON structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    for key in ("reference_findings", "predicted_findings", "matched_findings", "errors"):
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in CRIMSON response")
    errors = data["errors"]
    if not isinstance(errors, dict):
        raise ValueError(f"'errors' must be a dict, got {type(errors).__name__}")


class CRIMSONScore:
    """CRIMSON scorer with OpenAI or HuggingFace backend."""

    DEFAULT_HF_MODEL = "CRIMSONScore/medgemma-4b-it-crimson"
    DEFAULT_OPENAI_MODEL = "gpt-5.2"

    def __init__(
        self,
        api="hf",
        model_name=None,
        api_key=None,
        device=None,
        batch_size=1,
    ):
        self.api = api
        self.batch_size = batch_size

        if api == "openai":
            from openai import OpenAI
            self.model_name = model_name or self.DEFAULT_OPENAI_MODEL
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not found. Pass api_key or set OPENAI_API_KEY.")
            self.client = OpenAI(api_key=resolved_key)
            from .._llm import CostTracker
            self.cost_tracker = CostTracker(self.model_name)

        elif api in ("huggingface", "hf"):
            import torch
            import transformers
            self.model_name = model_name or self.DEFAULT_HF_MODEL
            self.torch_dtype = torch.bfloat16
            logger.info("Loading HuggingFace model: %s", self.model_name)
            self.pipe = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                dtype=self.torch_dtype,
                device_map="auto",
            )
            logger.info("Model loaded.")
        else:
            raise ValueError(
                f"Unsupported api: {api}. Use 'openai', 'huggingface', or 'hf'.")

    def _chat_completion(self, prompt: str) -> str:
        if self.api == "openai":
            from .._llm import call_openai
            return call_openai(
                self.client, self.model_name,
                [
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                cost_tracker=self.cost_tracker,
                seed=42,
                response_format={"type": "json_object"},
            )

        elif self.api in ("huggingface", "hf"):
            messages = [
                {"role": "system", "content": _SYSTEM_MSG},
                {"role": "user", "content": prompt + "\nPlease respond with valid JSON only."},
            ]
            outputs = self.pipe(
                messages,
                max_new_tokens=4096,
                do_sample=False,
                batch_size=self.batch_size,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            if not response:
                logger.warning("Empty response from HF pipeline. Raw: %s", outputs)
            return response

    def _build_evaluation_prompt(self, reference_findings, predicted_findings,
                                 patient_context=None, include_guidelines=True):
        return _build_evaluation_prompt_fn(
            reference_findings,
            predicted_findings,
            patient_context=patient_context,
            include_significance_examples=include_guidelines,
            include_attribute_guidelines=include_guidelines,
            include_context_guidelines=include_guidelines,
        )

    def evaluate(self, reference_findings, predicted_findings,
                 patient_context=None, include_guidelines=True):
        """Evaluate a single pair and return CRIMSON result dict."""
        prompt = self._build_evaluation_prompt(
            reference_findings, predicted_findings,
            patient_context, include_guidelines=include_guidelines)

        raw = self._chat_completion(prompt)
        cleaned = _extract_json_str(raw)
        try:
            evaluation = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse CRIMSON response as JSON: {e}\nResponse: {raw}")

        _validate_crimson_response(evaluation)
        return self._calculate_crimson(evaluation)

    def _calculate_crimson(self, evaluation):
        """Calculate CRIMSON score from parsed evaluation. Science-critical -- do not alter."""
        errors = evaluation.get("errors", {})
        matched = evaluation.get("matched_findings", [])
        reference_findings_list = evaluation.get("reference_findings", [])
        predicted_findings_list = evaluation.get("predicted_findings", [])

        significance_weights = {
            "urgent": 1.0,
            "actionable_not_urgent": 0.5,
            "not_actionable_not_urgent": 0.25,
            "benign_expected": 0.0,
        }
        attribute_severity_weights = {
            "significant": 0.5,
            "negligible": 0.0,
        }

        def calculate_weighted_count(error_list, weights=significance_weights,
                                     key="clinical_significance"):
            return sum(weights.get(error.get(key, "benign_expected"), 0.0)
                       for error in error_list)

        ref_weight_by_id = {
            ref["id"]: significance_weights.get(ref.get("clinical_significance", "benign_expected"), 0.0)
            for ref in reference_findings_list
        }
        pred_weight_by_id = {
            pred["id"]: significance_weights.get(pred.get("clinical_significance", "benign_expected"), 0.0)
            for pred in predicted_findings_list
        }

        E_false = sum(pred_weight_by_id.get(f_id, 0.0)
                      for f_id in errors.get("false_findings", []))
        E_miss = sum(ref_weight_by_id.get(m_id, 0.0)
                     for m_id in errors.get("missing_findings", []))

        attr_errors = errors.get("attribute_errors", [])
        n_location = sum(1 for e in attr_errors if "location" in e.get("error_types", []))
        n_severity = sum(1 for e in attr_errors if "severity" in e.get("error_types", []))
        n_descriptor = sum(1 for e in attr_errors if "descriptor" in e.get("error_types", []))
        n_measurement = sum(1 for e in attr_errors if "measurement" in e.get("error_types", []))
        n_certainty = sum(1 for e in attr_errors if "certainty" in e.get("error_types", []))
        n_unspecific = sum(1 for e in attr_errors if "unspecific" in e.get("error_types", []))
        n_overinterpretation = sum(1 for e in attr_errors if "overinterpretation" in e.get("error_types", []))
        n_temporal = sum(1 for e in attr_errors if "temporal" in e.get("error_types", []))

        attr_errors_by_ref_id = {}
        for err in attr_errors:
            ref_id = err.get("ref_id")
            if ref_id is not None:
                attr_errors_by_ref_id.setdefault(ref_id, []).append(err)

        N_G = calculate_weighted_count(reference_findings_list)
        if N_G == 0 and not reference_findings_list:
            N_G = len(matched) + E_miss

        E_penalty = E_false

        matched_ref_ids = set()
        correct = 0.0
        for m in matched:
            ref_id = m.get("ref_id")
            if ref_id in matched_ref_ids:
                continue
            matched_ref_ids.add(ref_id)
            base_weight = ref_weight_by_id.get(ref_id, 0.0)

            finding_attr_errors = attr_errors_by_ref_id.get(ref_id, [])
            if not finding_attr_errors:
                correct += base_weight
            else:
                sum_error_weights = sum(
                    attribute_severity_weights.get(err.get("severity", "negligible"), 0.0)
                    for err in finding_attr_errors)
                denom = base_weight + sum_error_weights
                credit_factor = base_weight / denom if denom > 0 else 0.0
                correct += base_weight * credit_factor

        errors_more_than_correct = E_penalty - correct

        if N_G == 0:
            S = 1.0 if E_penalty == 0 and E_miss == 0 else -(E_penalty + E_miss + 1)
        else:
            S = (correct - E_penalty) / N_G

        if S >= 0:
            crimson = S
        else:
            if errors_more_than_correct > 0:
                crimson = -1 * errors_more_than_correct / (1 + errors_more_than_correct)
            else:
                crimson = 0

        return {
            "raw_evaluation": evaluation,
            "error_counts": {
                "false_findings": len(errors.get("false_findings", [])),
                "missing_findings": len(errors.get("missing_findings", [])),
                "attribute_errors": len(attr_errors),
                "location_errors": n_location,
                "severity_errors": n_severity,
                "descriptor_errors": n_descriptor,
                "measurement_errors": n_measurement,
                "certainty_errors": n_certainty,
                "unspecific_errors": n_unspecific,
                "overinterpretation_errors": n_overinterpretation,
                "temporal_errors": n_temporal,
            },
            "weighted_error_counts": {
                "false_findings": E_false,
                "missing_findings": E_miss,
                "attribute_errors": calculate_weighted_count(
                    attr_errors, attribute_severity_weights, "severity"),
            },
            "metrics": {
                "N_G": N_G,
                "E_penalty": E_penalty,
                "correct": correct,
                "errors_more_than_correct": errors_more_than_correct,
                "S": S,
            },
            "crimson_score": round(crimson, 4),
        }

    def __call__(self, refs, hyps, patient_contexts=None,
                 include_guidelines=True, on_sample_done=None):
        """Compute CRIMSON across multiple report pairs.

        Returns:
            (mean, std, crimson_scores, results_df)
        """
        if not (isinstance(refs, list) and isinstance(hyps, list)):
            raise TypeError("refs and hyps must be of type list")
        if len(refs) != len(hyps):
            raise ValueError("refs and hyps lists don't have the same size")

        if patient_contexts is None:
            patient_contexts = [None] * len(refs)
        elif len(patient_contexts) != len(refs):
            raise ValueError("patient_contexts must have same size as refs/hyps")

        pair_results = []
        for ref, hyp, context in zip(refs, hyps, patient_contexts):
            pair_results.append(self.evaluate(
                reference_findings=ref,
                predicted_findings=hyp,
                patient_context=context,
                include_guidelines=include_guidelines,
            ))
            if on_sample_done:
                on_sample_done()

        crimson_scores = [r["crimson_score"] for r in pair_results]
        mean = float(np.mean(crimson_scores)) if crimson_scores else 0.0
        std = float(np.std(crimson_scores)) if len(crimson_scores) > 1 else 0.0

        rows = []
        for ref, hyp, result in zip(refs, hyps, pair_results):
            counts = result.get("error_counts", {})
            rows.append({
                "reference": ref,
                "prediction": hyp,
                "crimson_score": result.get("crimson_score", 0.0),
                **{k: counts.get(k, 0) for k in (
                    "false_findings", "missing_findings", "attribute_errors",
                    "location_errors", "severity_errors", "descriptor_errors",
                    "measurement_errors", "certainty_errors", "unspecific_errors",
                    "overinterpretation_errors", "temporal_errors",
                )},
            })

        results_df = pd.DataFrame(rows)
        return mean, std, crimson_scores, results_df


class CRIMSON(CRIMSONScore):
    """Alias matching RadEval metric naming style."""

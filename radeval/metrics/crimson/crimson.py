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
import re
from typing import Any, ClassVar, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._llm_base import LLMMetricBase
from .prompt_parts import build_prompt as _build_evaluation_prompt_fn
from .utils import parse_json_response as _parse_json_response_robust

logger = logging.getLogger(__name__)

_SYSTEM_MSG = (
    "You are an expert radiology evaluator that assesses "
    "the accuracy of radiology reports."
)


# ---------------------------------------------------------------------------
# JSON helpers (module-level, reused by HopprCrimsonCT)
# ---------------------------------------------------------------------------

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


def _repair_truncated_json(text: str) -> Optional[str]:
    """Try to close a truncated JSON object so it can be parsed.

    Walks through the string tracking bracket/brace depth and string
    state, then appends the missing closing characters.  Returns *None*
    if the result still cannot be parsed.
    """
    in_string = False
    escape = False
    stack: list[str] = []
    closer = {"{": "}", "[": "]"}

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in closer:
            stack.append(closer[ch])
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        text += '"'

    text = re.sub(r",\s*$", "", text)
    text += "".join(reversed(stack))

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


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


# ---------------------------------------------------------------------------
# CRIMSONScore – main scorer
# ---------------------------------------------------------------------------

class CRIMSONScore(LLMMetricBase):
    """CRIMSON scorer with OpenAI or HuggingFace backend."""

    SUPPORTED_PROVIDERS: ClassVar[set[str]] = {"openai", "hf"}

    DEFAULT_HF_MODEL = "rajpurkarlab/medgemma-4b-it-crimson"
    DEFAULT_OPENAI_MODEL = "gpt-5.2"
    DEFAULT_MAX_NEW_TOKENS = 8192

    def __init__(
        self,
        provider="hf",
        model_name=None,
        openai_api_key=None,
        gemini_api_key=None,
        device=None,
        batch_size=1,
        max_concurrent=50,
        cache_dir=None,
    ):
        resolved_model = model_name or (
            self.DEFAULT_OPENAI_MODEL if provider == "openai"
            else self.DEFAULT_HF_MODEL
        )

        super().__init__(
            provider=provider,
            model_name=resolved_model,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
        )

        self.batch_size = batch_size
        self.cache_dir = cache_dir

        if provider in ("huggingface", "hf"):
            self._init_hf_pipeline()

    def _init_hf_pipeline(self):
        import torch
        import transformers

        self.torch_dtype = torch.bfloat16
        logger.info("Loading HuggingFace model: %s", self.model_name)

        model_kwargs = {"torch_dtype": self.torch_dtype}
        try:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2.")
        except ImportError:
            pass
        if self.cache_dir:
            model_kwargs["cache_dir"] = self.cache_dir

        self.pipe = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs=model_kwargs,
            device_map="auto",
        )
        # Left-pad for efficient batch generation with decoder-only models
        if self.pipe.tokenizer.padding_side != "left":
            self.pipe.tokenizer.padding_side = "left"
        # If the model has a generation_config.json, use it instead of
        # injecting our own generation kwargs at inference time.
        _from_mc = getattr(
            self.pipe.model.generation_config, "_from_model_config", None
        )
        # In transformers v5, `_from_model_config` may be None instead of bool.
        # Treat any falsy value (None/False) as "no custom generation config".
        self._has_generation_config = bool(_from_mc is False)
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # LLMMetricBase interface
    # ------------------------------------------------------------------

    def _build_request(self, ref: str, hyp: str, **kwargs) -> dict[str, Any]:
        patient_context = kwargs.get("patient_context")
        include_guidelines = kwargs.get("include_guidelines", True)
        prompt = self._build_evaluation_prompt(
            ref, hyp, patient_context, include_guidelines)

        if self.provider == "openai":
            return {
                "messages": [
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "seed": 42,
                "response_format": {"type": "json_object"},
            }
        else:
            return {"prompt": prompt}

    def _parse_response(self, raw: str) -> dict:
        cleaned = _extract_json_str(raw)
        try:
            evaluation = _parse_json_response_robust(cleaned)
        except (ValueError, json.JSONDecodeError):
            repaired = _repair_truncated_json(cleaned)
            if repaired is not None:
                logger.warning("Repaired truncated CRIMSON JSON.")
                evaluation = json.loads(repaired)
            else:
                raise ValueError(
                    f"Failed to parse CRIMSON JSON: {raw[:500]}")

        _validate_crimson_response(evaluation)
        return self._calculate_crimson(evaluation)

    def _aggregate(
        self, results: list[dict], refs: list[str], hyps: list[str],
    ) -> tuple:
        crimson_scores = [r["crimson_score"] for r in results]
        valid = [s for s in crimson_scores
                 if not (isinstance(s, float) and np.isnan(s))]
        n_failed = len(crimson_scores) - len(valid)
        if n_failed:
            logger.warning(
                "CRIMSON: %d/%d samples failed and were excluded.",
                n_failed, len(crimson_scores))
        mean = float(np.mean(valid)) if valid else 0.0
        std = float(np.std(valid)) if len(valid) > 1 else 0.0

        rows = []
        for ref, hyp, result in zip(refs, hyps, results):
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

    # ------------------------------------------------------------------
    # HF-specific chat completion (overrides base for local model)
    # ------------------------------------------------------------------

    def _chat_completion(self, request: dict[str, Any]) -> str:
        if self.provider in ("huggingface", "hf"):
            return self._hf_generate(request["prompt"])
        return super()._chat_completion(request)

    def _hf_generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]

        if self._has_generation_config:
            outputs = self.pipe(
                messages,
                generation_config=self.pipe.model.generation_config,
                repetition_penalty=1.1,
            )
        else:
            outputs = self.pipe(
                messages,
                max_new_tokens=self.DEFAULT_MAX_NEW_TOKENS,
                max_length=None,
                do_sample=False,
                repetition_penalty=1.1,
            )
        response = outputs[0]["generated_text"][-1]["content"]
        if not response:
            logger.warning("Empty response from HF pipeline. Raw: %s", outputs)
        return response

    # ------------------------------------------------------------------
    # Override _evaluate_one for NaN fallback on persistent failure
    # ------------------------------------------------------------------

    def _evaluate_one(self, ref, hyp, max_retries=2, **kwargs):
        try:
            return super()._evaluate_one(ref, hyp, max_retries=max_retries, **kwargs)
        except RuntimeError:
            logger.error(
                "CRIMSON: all attempts failed – returning NaN score.")
            return self._nan_fallback()

    async def _evaluate_one_async(self, ref, hyp, max_retries=2, **kwargs):
        try:
            return await super()._evaluate_one_async(
                ref, hyp, max_retries=max_retries, **kwargs)
        except RuntimeError:
            logger.error(
                "CRIMSON: all async attempts failed – returning NaN score.")
            return self._nan_fallback()

    # ------------------------------------------------------------------
    # Override __call__ to support patient_contexts
    # ------------------------------------------------------------------

    def __call__(self, refs, hyps, patient_contexts=None,
                 include_guidelines=True, on_sample_done=None):
        """Compute CRIMSON across multiple report pairs.

        Uses async concurrency for OpenAI, sequential for HF.

        Returns:
            (mean, std, crimson_scores, results_df)
        """
        if not isinstance(refs, list) or not isinstance(hyps, list):
            raise TypeError("refs and hyps must be of type list")
        if len(refs) != len(hyps):
            raise ValueError("refs and hyps lists don't have the same size")

        if patient_contexts is None:
            patient_contexts = [None] * len(refs)
        elif len(patient_contexts) != len(refs):
            raise ValueError(
                "patient_contexts must have same size as refs/hyps")

        if self.provider == "openai":
            results = self._run_concurrent_crimson(
                refs, hyps, patient_contexts, include_guidelines,
                on_sample_done)
        else:
            results = []
            for ref, hyp, ctx in zip(refs, hyps, patient_contexts):
                results.append(self._evaluate_one(
                    ref, hyp,
                    patient_context=ctx,
                    include_guidelines=include_guidelines))
                if on_sample_done:
                    on_sample_done()

        return self._aggregate(results, refs, hyps)

    def _run_concurrent_crimson(self, refs, hyps, patient_contexts,
                                include_guidelines, on_sample_done):
        """Run CRIMSON evaluations concurrently (passes patient_context per sample)."""
        import asyncio

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _sem_eval(ref, hyp, ctx):
            async with sem:
                result = await self._evaluate_one_async(
                    ref, hyp,
                    patient_context=ctx,
                    include_guidelines=include_guidelines)
                if on_sample_done:
                    on_sample_done()
                return result

        async def _gather():
            tasks = [_sem_eval(r, h, c)
                     for r, h, c in zip(refs, hyps, patient_contexts)]
            return list(await asyncio.gather(*tasks))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(lambda: asyncio.run(_gather())).result()
        return asyncio.run(_gather())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nan_fallback():
        """Return a result dict with NaN score for unparseable responses."""
        return {
            "raw_evaluation": {},
            "error_counts": {k: 0 for k in (
                "false_findings", "missing_findings", "attribute_errors",
                "location_errors", "severity_errors", "descriptor_errors",
                "measurement_errors", "certainty_errors", "unspecific_errors",
                "overinterpretation_errors", "temporal_errors",
            )},
            "weighted_error_counts": {
                "false_findings": 0, "missing_findings": 0,
                "attribute_errors": 0,
            },
            "metrics": {
                "N_G": 0, "E_penalty": 0, "correct": 0,
                "errors_more_than_correct": 0, "S": 0,
            },
            "crimson_score": float("nan"),
        }

    def _build_evaluation_prompt(self, reference_findings, predicted_findings,
                                 patient_context=None, include_guidelines=True):
        # For the MedGemmaCRIMSON, exclude guidelines. The model was trained without them.
        if self.provider in ("huggingface", "hf") and self.model_name == self.DEFAULT_HF_MODEL:
            include_guidelines = False
        return _build_evaluation_prompt_fn(
            reference_findings,
            predicted_findings,
            patient_context=patient_context,
            include_significance_examples=include_guidelines,
            include_attribute_guidelines=include_guidelines,
            include_context_guidelines=include_guidelines,
        )

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
            return sum(weights.get(error.get(key, ""), 0.25)
                       for error in error_list)

        ref_weight_by_id = {
            ref["id"]: significance_weights.get(ref.get("clinical_significance", ""), 0.25)
            for ref in reference_findings_list
        }
        pred_weight_by_id = {
            pred["id"]: significance_weights.get(pred.get("clinical_significance", ""), 0.25)
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
                    attribute_severity_weights.get(err.get("severity", ""), 0.25)
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


class CRIMSON(CRIMSONScore):
    """Alias matching RadEval metric naming style."""

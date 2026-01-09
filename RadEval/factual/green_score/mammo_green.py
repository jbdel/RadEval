# mammo_green.py
"""
MammoGREEN: Mammography-specific GREEN score using LLM as judge.

Adapted from Mammo-FM style implementation for RadEval.

Supports OpenAI and Google Gemini models as judges.

Implements 5 clinically significant error categories for mammography:
  (a) false_finding
  (b) missing_finding
  (c) mischaracterization
  (d) wrong_location_laterality
  (e) incorrect_birads

GREEN score:
    matched_findings / (matched_findings + sum(significant_errors))

Dependencies:
  pip install openai google-generativeai pydantic tenacity

Env:
  export OPENAI_API_KEY="..."  # for OpenAI models
  export GOOGLE_API_KEY="..."  # for Gemini models

Usage:
  from RadEval.factual.green_score import MammoGREEN

  refs = [...]
  hyps = [...]

  # OpenAI models (auto-detected from model name)
  mg = MammoGREEN(model_name="gpt-4o-mini")

  # Gemini models (auto-detected from model name)
  mg = MammoGREEN(model_name="gemini-1.5-flash")

  # Explicit provider specification (recommended for clarity)
  mg = MammoGREEN(model_name="gpt-4o-mini", provider="openai")
  mg = MammoGREEN(model_name="gemini-1.5-flash", provider="gemini")

  mean, std, green_scores, results_df = mg(refs, hyps)
"""

from __future__ import annotations

import json
import os
import time
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI

# Optional Gemini support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


# -----------------------------
# Provider Detection
# -----------------------------

SUPPORTED_PROVIDERS = ("openai", "gemini")


def _detect_provider(model_name: str) -> str:
    """
    Detect the provider based on model name.

    Returns "gemini" for models starting with "gemini", otherwise "openai".
    """
    model_lower = model_name.lower()
    if model_lower.startswith("gemini"):
        return "gemini"
    # Default to OpenAI for gpt-*, o1-*, etc.
    return "openai"


# -----------------------------
# Schema
# -----------------------------

class MammoGreenOutput(BaseModel):
    matched_findings: int = Field(ge=0)
    significant_errors: Dict[str, int]
    insignificant_errors: int = Field(ge=0)

    def validate_keys(self) -> None:
        required = {
            "false_finding",
            "missing_finding",
            "mischaracterization",
            "wrong_location_laterality",
            "incorrect_birads",
        }
        if set(self.significant_errors.keys()) != required:
            raise ValueError(
                f"significant_errors keys must be exactly: {sorted(required)}; "
                f"got: {sorted(self.significant_errors.keys())}"
            )
        for k, v in self.significant_errors.items():
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"significant_errors[{k}] must be a non-negative int")


# -----------------------------
# Prompt (Mammo-FM-style, strict JSON)
# -----------------------------

MAMMO_GREEN_SYSTEM_PROMPT = """You are a breast imaging report evaluator for screening/diagnostic mammography.
Compare a GENERATED mammography report to a REFERENCE report and count matched findings and errors.

Definitions:
- matched_findings: number of clinically equivalent mammography findings present in BOTH reports.
  A match requires agreement on the finding and its laterality/location and major characterization.
  Paraphrases count as matches.

Clinically significant errors (count each instance):
- false_finding: GENERATED reports a finding not present in REFERENCE.
- missing_finding: GENERATED omits a finding present in REFERENCE.
- mischaracterization: GENERATED describes a correct finding but with incorrect characterization
  (e.g., size, margins, stability, suspiciousness, calcification morphology/distribution).
- wrong_location_laterality: GENERATED describes correct finding but wrong laterality or location.
- incorrect_birads: GENERATED BI-RADS category/assessment differs from REFERENCE.

Clinically insignificant errors:
- insignificant_errors: stylistic/wording issues with no clinical impact.

Output:
Return ONLY valid JSON matching exactly this schema (no markdown, no explanation, no extra keys):
{
  "matched_findings": int,
  "significant_errors": {
    "false_finding": int,
    "missing_finding": int,
    "mischaracterization": int,
    "wrong_location_laterality": int,
    "incorrect_birads": int
  },
  "insignificant_errors": int
}
All counts must be non-negative integers.
"""


# -----------------------------
# Scoring
# -----------------------------

def mammo_green_score(matched_findings: int, significant_errors: Dict[str, int]) -> float:
    sig_sum = sum(int(v) for v in significant_errors.values())
    denom = matched_findings + sig_sum
    return float(matched_findings / denom) if denom > 0 else 0.0


# -----------------------------
# OpenAI judge call + parsing
# -----------------------------

class JudgeError(RuntimeError):
    pass


def _extract_json_str(text: str) -> str:
    """Best-effort extraction if the model accidentally wraps JSON in prose."""
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


@dataclass
class OpenAIJudgeConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_output_tokens: int = 300


@dataclass
class GeminiJudgeConfig:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 300


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(JudgeError),
)
def _judge_one_openai(
    client: OpenAI,
    cfg: OpenAIJudgeConfig,
    reference: str,
    hypothesis: str,
) -> Dict[str, Any]:
    """Judge using OpenAI API."""
    user_msg = f"REFERENCE_REPORT:\n{reference}\n\nGENERATED_REPORT:\n{hypothesis}\n"

    try:
        resp = client.chat.completions.create(
            model=cfg.model_name,
            messages=[
                {"role": "system", "content": MAMMO_GREEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
        )
    except Exception as e:
        raise JudgeError(f"OpenAI call failed: {e}") from e

    raw = resp.choices[0].message.content if resp.choices else None
    if raw is None:
        raise JudgeError("Could not read response content from OpenAI")

    return _parse_judge_response(raw)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(JudgeError),
)
def _judge_one_gemini(
    model,  # genai.GenerativeModel
    cfg: GeminiJudgeConfig,
    reference: str,
    hypothesis: str,
) -> Dict[str, Any]:
    """Judge using Google Gemini API."""
    user_msg = f"{MAMMO_GREEN_SYSTEM_PROMPT}\n\nREFERENCE_REPORT:\n{reference}\n\nGENERATED_REPORT:\n{hypothesis}\n"

    try:
        generation_config = genai.types.GenerationConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )
        resp = model.generate_content(
            user_msg,
            generation_config=generation_config,
        )
    except Exception as e:
        raise JudgeError(f"Gemini call failed: {e}") from e

    raw = resp.text if resp else None
    if raw is None:
        raise JudgeError("Could not read response content from Gemini")

    return _parse_judge_response(raw)


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Parse and validate judge response JSON."""
    raw_json = _extract_json_str(raw)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise JudgeError(f"Model did not return valid JSON. Raw head: {raw[:200]}") from e

    # Validate schema strictly
    try:
        parsed = MammoGreenOutput(**data)
        parsed.validate_keys()
    except (ValidationError, ValueError) as e:
        raise JudgeError(f"JSON schema invalid: {e}. Raw head: {raw[:200]}") from e

    return data


# -----------------------------
# Main class
# -----------------------------

class MammoGREEN:
    """
    Mammography-GREEN metric using LLM judge.

    Supports both OpenAI (gpt-*) and Google Gemini (gemini-*) models.

    Compatible with RadEval interface.

    __call__(refs, hyps) returns:
      (mean, std, green_scores, results_df)

    Where results_df contains:
      - reference: original reference text
      - prediction: original hypothesis text
      - green_score: per-sample GREEN score
      - matched_findings: count of matched findings
      - false_finding, missing_finding, mischaracterization,
        wrong_location_laterality, incorrect_birads: error counts
      - insignificant_errors: count of insignificant errors
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        output_dir: str = ".",
        batch_size: int = 8,
        max_output_tokens: int = 300,
        temperature: float = 0.0,
        sleep_s: float = 0.0,
        compute_summary_stats: bool = True,
        api_key: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.batch_size = int(batch_size)
        self.sleep_s = float(sleep_s)
        self.compute_summary_stats = bool(compute_summary_stats)
        self.model_name = model_name

        # Determine provider: explicit parameter takes precedence, else auto-detect
        if provider is not None:
            if provider.lower() not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported provider: '{provider}'. "
                    f"Supported providers are: {SUPPORTED_PROVIDERS}"
                )
            self.provider = provider.lower()
        else:
            self.provider = _detect_provider(model_name)

        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-generativeai is not installed. "
                    "Install it with: pip install google-generativeai"
                )
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY is not set (or pass api_key=...). "
                    "For Gemini models, set GOOGLE_API_KEY environment variable."
                )
            genai.configure(api_key=key)
            self.gemini_model = genai.GenerativeModel(model_name)
            self.cfg = GeminiJudgeConfig(
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            self.client = None  # Not used for Gemini
        else:
            # Default to OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("OPENAI_API_KEY is not set (or pass api_key=...).")
            self.client = OpenAI(api_key=key)
            self.cfg = OpenAIJudgeConfig(
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            self.gemini_model = None  # Not used for OpenAI

        # For compatibility / introspection
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Mischaracterization of a finding",
            "(d) Misidentification of a finding's location/laterality",
            "(e) Incorrect BI-RADS category/assessment",
        ]

        # Outputs
        self.prompts: Optional[List[str]] = None
        self.completions: Optional[List[str]] = None
        self.green_scores: Optional[List[float]] = None
        self.error_counts: Optional[List[Dict[str, Any]]] = None

    def __call__(self, refs: List[str], hyps: List[str]):
        """
        Compute MammoGREEN scores for reference/hypothesis pairs.

        Args:
            refs: List of reference mammography reports
            hyps: List of generated/candidate mammography reports

        Returns:
            Tuple of (mean, std, green_scores, results_df)
            - mean: Mean GREEN score across all samples
            - std: Standard deviation of GREEN scores
            - green_scores: List of per-sample GREEN scores
            - results_df: DataFrame with detailed results
        """
        if len(refs) != len(hyps):
            raise ValueError(f"refs and hyps must have same length. Got {len(refs)} vs {len(hyps)}")

        error_counts: List[Dict[str, Any]] = []
        green_scores: List[float] = []

        # Process sequentially
        for i, (r, h) in enumerate(zip(refs, hyps)):
            if self.provider == "gemini":
                data = _judge_one_gemini(self.gemini_model, self.cfg, r, h)
            else:
                data = _judge_one_openai(self.client, self.cfg, r, h)
            mg = MammoGreenOutput(**data)
            mg.validate_keys()

            score = mammo_green_score(mg.matched_findings, mg.significant_errors)
            error_counts.append(data)
            green_scores.append(score)

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        self.error_counts = error_counts
        self.green_scores = green_scores

        # Compute statistics
        mean = float(np.mean(green_scores)) if green_scores else 0.0
        std = float(np.std(green_scores)) if len(green_scores) > 1 else 0.0

        # Build results DataFrame (RadEval-compatible format)
        results_df = self._build_results_df(refs, hyps, green_scores, error_counts)

        return mean, std, green_scores, results_df

    def _build_results_df(
        self,
        refs: List[str],
        hyps: List[str],
        green_scores: List[float],
        error_counts: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Build a pandas DataFrame with detailed results."""
        rows = []
        for i, (ref, hyp, score, errors) in enumerate(zip(refs, hyps, green_scores, error_counts)):
            row = {
                "reference": ref,
                "prediction": hyp,
                "green_score": score,
                "matched_findings": errors["matched_findings"],
                "false_finding": errors["significant_errors"]["false_finding"],
                "missing_finding": errors["significant_errors"]["missing_finding"],
                "mischaracterization": errors["significant_errors"]["mischaracterization"],
                "wrong_location_laterality": errors["significant_errors"]["wrong_location_laterality"],
                "incorrect_birads": errors["significant_errors"]["incorrect_birads"],
                "insignificant_errors": errors["insignificant_errors"],
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _summary(self, scores: List[float], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics across all samples."""
        sig_keys = [
            "false_finding",
            "missing_finding",
            "mischaracterization",
            "wrong_location_laterality",
            "incorrect_birads",
        ]
        sig_sums = {k: 0 for k in sig_keys}
        matched_sum = 0
        insig_sum = 0

        for e in errors:
            matched_sum += int(e["matched_findings"])
            insig_sum += int(e["insignificant_errors"])
            for k in sig_keys:
                sig_sums[k] += int(e["significant_errors"][k])

        return {
            "n": len(scores),
            "mean_green": statistics.mean(scores) if scores else 0.0,
            "stdev_green": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "min_green": min(scores) if scores else 0.0,
            "max_green": max(scores) if scores else 0.0,
            "total_matched_findings": matched_sum,
            "total_insignificant_errors": insig_sum,
            "total_significant_errors_by_type": sig_sums,
            "total_significant_errors": sum(sig_sums.values()),
        }

    def score(self, refs: List[str], hyps: List[str]) -> Dict[str, Any]:
        """
        Alternative interface returning a dictionary (original mammo_green.py style).

        Returns:
            Dict with keys: green_scores, error_counts, summary (optional)
        """
        mean, std, green_scores, results_df = self(refs, hyps)

        out: Dict[str, Any] = {
            "green_scores": green_scores,
            "error_counts": self.error_counts,
        }

        if self.compute_summary_stats:
            out["summary"] = self._summary(green_scores, self.error_counts)

        return out


# -----------------------------
# CLI demo
# -----------------------------
if __name__ == "__main__":
    refs = [
        (
            "Bilateral digital mammography demonstrates scattered areas of fibroglandular density. "
            "There is a spiculated mass in the upper outer quadrant of the right breast measuring approximately 1.5 cm. "
            "Associated pleomorphic calcifications are present. "
            "No suspicious findings are seen in the left breast. "
            "BI-RADS 5."
        ),
        (
            "Bilateral digital mammography shows heterogeneously dense breasts. "
            "No suspicious mass, calcifications, or architectural distortion are identified in either breast. "
            "BI-RADS 1."
        ),
    ]

    hyps = [
        (
            "The breasts demonstrate scattered fibroglandular densities. "
            "There is an irregular mass in the upper outer quadrant of the right breast measuring approximately 1.4 cm "
            "with associated calcifications. "
            "The left breast is unremarkable. "
            "BI-RADS 4."
        ),
        (
            "Bilateral mammography demonstrates heterogeneously dense breast tissue. "
            "No suspicious masses or calcifications are identified. "
            "BI-RADS 1."
        ),
    ]

    # Requires OPENAI_API_KEY env var or pass api_key parameter
    mg = MammoGREEN(
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    mean, std, green_scores, results_df = mg(refs, hyps)

    print("GREEN scores:", green_scores)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    print("\nResults DataFrame:")
    print(results_df.to_string())

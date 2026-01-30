# mammo_green.py
"""
MammoGREEN: Mammography-specific GREEN score using LLM as judge.

Adapted from Mammo-FM style implementation for RadEval.

Supports OpenAI and Google Gemini models as judges.

Implements 6 clinically significant error categories for mammography:
  (a) false_finding
  (b) missing_finding
  (c) mischaracterization
  (d) wrong_location_laterality
  (e) incorrect_birads
  (f) incorrect_breast_density

GREEN score:
    matched_findings / (matched_findings + sum(significant_errors))

Dependencies:
  pip install openai google-genai pydantic tenacity

Env:
  export OPENAI_API_KEY="..."  # for OpenAI models
  export GOOGLE_API_KEY="..."  # for Gemini models

Usage:
  from RadEval.factual.green_score import MammoGREEN

  refs = [...]
  hyps = [...]

  # OpenAI models (auto-detected from model name)
  mg = MammoGREEN(model_name="gpt-4o")

  # Gemini models (auto-detected from model name)
  mg = MammoGREEN(model_name="gemini-2.5-flash")

  # Explicit provider specification
  mg = MammoGREEN(model_name="gpt-4o", provider="openai")
  mg = MammoGREEN(model_name="gemini-2.5-flash", provider="gemini")

  mean, std, green_scores, results_df = mg(refs, hyps)
"""

from __future__ import annotations

import json
import os
import re
import time
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI

# Optional Gemini support (new SDK)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GEMINI_AVAILABLE = False


# -----------------------------
# Constants
# -----------------------------

SUPPORTED_PROVIDERS = ("openai", "gemini")

SIGNIFICANT_ERROR_KEYS = (
    "false_finding",
    "missing_finding",
    "mischaracterization",
    "wrong_location_laterality",
    "incorrect_birads",
    "incorrect_breast_density",
)


# -----------------------------
# Provider Detection
# -----------------------------

def _detect_provider(model_name: str) -> str:
    """
    Detect the provider based on model name.

    Returns "gemini" for models starting with "gemini", otherwise "openai".
    """
    model_lower = model_name.lower()
    if model_lower.startswith("gemini"):
        return "gemini"
    return "openai"


# -----------------------------
# Schema
# -----------------------------

class MammoGreenOutput(BaseModel):
    """Schema for validated MammoGREEN judge output."""
    matched_findings: int = Field(ge=0)
    significant_errors: Dict[str, int]
    insignificant_errors: int = Field(ge=0)

    def validate_keys(self) -> None:
        """Validate that significant_errors contains exactly the required keys."""
        required = set(SIGNIFICANT_ERROR_KEYS)
        if set(self.significant_errors.keys()) != required:
            raise ValueError(
                f"significant_errors keys must be exactly: {sorted(required)}; "
                f"got: {sorted(self.significant_errors.keys())}"
            )
        for k, v in self.significant_errors.items():
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"significant_errors[{k}] must be a non-negative int")


# -----------------------------
# Prompt (Mammo-FM-style with BIRADS and benign handling)
# -----------------------------

MAMMO_GREEN_SYSTEM_PROMPT = """You are a breast imaging report evaluator for screening/diagnostic mammography.
Compare a GENERATED mammography report to a REFERENCE report and count matched findings and errors.
Scope: What counts as a finding
- Findings are counted at the lesion-group or clinically actionable item level.
- Include: masses, asymmetries/focal asymmetries, architectural distortion, calcifications
  (by group), skin or nipple changes, suspicious lymph nodes, implant rupture/complications,
  and explicitly stated actionable recommendations (additional imaging, ultrasound,
  short-interval follow-up, biopsy).
- Exclude purely stylistic content, boilerplate language, or redundant narrative restatement.
Definitions:
- matched_findings: number of clinically equivalent findings present in BOTH reports.
  A match requires agreement on:
  - lesion group or actionable item,
  - laterality,
  - region/location at the level of specificity provided by the REFERENCE,
  - major characterization (lesion type and overall suspiciousness).
  Paraphrases count as matches.
- Do NOT require agreement on exact lesion counts or depth to award a matched finding.
Important benign-handling rules:
- Statements indicating absence or normality (e.g., "no suspicious findings",
  "unremarkable", "within normal limits", "benign exam") are benign defaults.
- Do NOT count benign/negative statements as false_finding or missing_finding
  unless they directly contradict a positive abnormal or actionable finding
  explicitly stated in the REFERENCE.
- If the REFERENCE BI-RADS is 1 or 2 and does NOT describe any suspicious or BI-RADS 3+ finding,
  then a GENERATED statement like "no suspicious findings" MUST be counted as
  1 matched_findings for overall benign/negative impression, even if the REFERENCE
  does not explicitly say "no suspicious findings".
- If both reports indicate absence of suspicious findings, this counts as a matched finding.
Recommendations handling:
- Actionable recommendations (e.g., diagnostic views, ultrasound, biopsy,
  short-interval follow-up) are treated as clinically actionable findings.
- If the REFERENCE includes an actionable recommendation and the GENERATED omits it:
  missing_finding += 1.
- If the GENERATED includes an actionable recommendation not present in the REFERENCE:
  false_finding += 1.
- If the recommendation is present but incorrect in intent or intensity
  (e.g., biopsy vs short-interval follow-up): mischaracterization += 1.
Comparisons / priors (STRICTLY DISALLOWED):
- The GENERATED report MUST NOT include any comparison-to-prior-exams language or
  temporal change/stability claims. The model must not assume priors were available.
- If ANY comparison language is present (e.g., "compared to prior", "stable",
  "unchanged", "no interval change", "new since prior", "interval change",
  "previously seen", or specific prior dates):
    false_finding += 1 (maximum 1 per case for comparison language).
- Do not count comparison language toward matched_findings.
- Do not generate additional mischaracterization or missing_finding penalties
  solely due to comparison language.
- Negative statements about additional lesion categories not mentioned in the REFERENCE (e.g., skin thickening, lymph nodes) MUST NOT be counted as mischaracterization unless they directly contradict a positive finding in the REFERENCE.
Multiplicity / number claims (STRICT, CAPPED):
- The GENERATED report MUST NOT invent or over-specify the number of lesions or groups
  (e.g., number of masses, number of calcification groups, number of asymmetries).
- Any explicit number or multiplicity language ("two", "three", "multiple", "several",
  "numerous", enumerated lesions) is a multiplicity claim.
- Score multiplicity errors with a stability cap: for each breast (left/right) and
  each lesion type (mass, calcifications, asymmetry, distortion), count at most ONE
  multiplicity-related significant error.
Multiplicity scoring:
- REFERENCE indicates one lesion/group and GENERATED indicates >1:
  false_finding += 1.
- REFERENCE indicates multiple lesions/groups and GENERATED indicates fewer or one:
  missing_finding += 1.
- Both indicate multiple but disagree on number, or GENERATED is more specific than REFERENCE:
  mischaracterization += 1.
- Do not award additional matched_findings for extra lesions/groups unless the REFERENCE
  clearly describes them with consistent laterality/location.
- Still award matched_findings for lesion presence when laterality and major
  characterization match, even if multiplicity differs.
Location specificity (STRICT – NO INVENTED DETAIL):
- The GENERATED report MUST NOT introduce greater location specificity than explicitly
  stated in the REFERENCE.
- The GENERATED report may be less specific than the REFERENCE, but never more specific.
Location handling:
- Laterality must match exactly.
- Region may be described using quadrant or clock-face.
- Quadrant and clock-face are compatible ONLY when the GENERATED report is not more
  specific than the REFERENCE.
Specific rules:
- If the REFERENCE uses quadrant only and the GENERATED uses clock-face or adds depth:
  mischaracterization += 1 (invented specificity).
- If the REFERENCE uses clock-face and the GENERATED uses quadrant:
  allow as a match if compatible.
- If both specify region and they conflict (e.g., inner vs outer, clock-face outside
  the implied quadrant):
  wrong_location_laterality += 1.
- Depth (anterior/middle/posterior):
  - If REFERENCE does not specify depth and GENERATED does:
    mischaracterization += 1.
  - If both specify depth and disagree:
    mischaracterization += 1.
  - If GENERATED omits depth present in the REFERENCE:
    do NOT penalize.
- Central, retroareolar, and subareolar locations are considered equivalent.
Descriptor expectation (REFERENCE-DRIVEN):
- Enforce descriptor-first rules ONLY when the REFERENCE explicitly characterizes
  a lesion (e.g., calcification morphology/distribution, mass shape/margins/density).
- If the REFERENCE does NOT include lesion descriptors and only states lesion presence,
  triage language, or a global negative assessment, the GENERATED report is NOT required
  to include descriptors and should not be penalized for their absence.
- The GENERATED report MUST NOT introduce descriptors or diagnostic certainty that
  exceed the level of detail present in the REFERENCE.
Diagnosis labels and certainty (STRICT – NO INVENTED DIAGNOSES):
- The GENERATED report MUST NOT assign specific pathologic or ultrasound-dependent
  diagnoses that cannot be established on mammography alone.
- Disallowed labels include (unless the REFERENCE explicitly states them):
  "cyst", "fibroadenoma", "hamartoma", "phyllodes", "papilloma", or similar entities.
- If the GENERATED assigns a disallowed diagnosis when the REFERENCE describes only
  an imaging finding (e.g., "mass"):
  mischaracterization += 1.
- Probabilistic phrasing (e.g., "likely cyst", "probable fibroadenoma") is still
  considered invented diagnostic specificity and should be penalized as mischaracterization.
Calcification assessment (DESCRIPTOR-FIRST, REFERENCE-DRIVEN):
- Apply calcification descriptor requirements ONLY if the REFERENCE describes
  calcification morphology and/or distribution.
- If descriptors are present in the REFERENCE:
  - Assessment terms such as "benign", "probably benign", or "suspicious" MUST be
    supported by compatible morphology/distribution.
- If descriptors are NOT present in the REFERENCE:
  - The GENERATED report MUST NOT introduce calcification descriptors or diagnostic
    certainty beyond generic triage or negative language.
Scoring:
- Unsupported or contradictory assessment language:
  mischaracterization += 1 (and incorrect_birads += 1 if applicable).
Clinically significant errors (count each instance unless otherwise specified):
- false_finding: GENERATED reports a positive abnormal finding or actionable recommendation
  not present in REFERENCE, including:
  - invented lesions,
  - invented recommendations,
  - disallowed comparison language (max 1 per case),
  - multiplicity overcall (subject to caps).
- missing_finding: GENERATED omits a positive abnormal or clinically actionable finding
  present in REFERENCE, including multiplicity undercall (subject to caps).
  Omission of benign/incidental findings (BI-RADS 2) should NOT be counted as missing_finding
  if the BI-RADS assessment and management remain correct.
- mischaracterization: GENERATED describes a correct finding or recommendation
  but with incorrect or invented characterization (e.g., size, margins,
  unjustified numeric precision, invented location detail, unsupported diagnostic labels,
  unsupported assessment language, calcification morphology/distribution,
  or management intent).
- wrong_location_laterality: GENERATED assigns incorrect laterality or clearly incompatible
  region to an otherwise correct finding.
- incorrect_birads: GENERATED BI-RADS category differs from REFERENCE.
  Count at most ONE incorrect_birads per case.
- incorrect_breast_density: GENERATED breast density assessment differs from REFERENCE
  in a clinically meaningful way.
  Minor wording differences or equivalent categories (A=fatty, B=scattered,
  C=heterogeneously dense, D=extremely dense) should NOT be penalized.
Error precedence and double-counting rules:
- When the same lesion or recommendation exists in both reports:
  prefer mischaracterization or wrong_location_laterality over
  false_finding or missing_finding.
- Use false_finding and missing_finding only when findings or recommendations
  are truly absent or newly introduced.
- A single lesion or recommendation should not generate multiple significant
  error types unless it clearly represents distinct clinical errors.
Clinically insignificant errors:
- insignificant_errors: stylistic, formatting, or wording issues with no clinical impact.
Output:
Return ONLY valid JSON matching exactly this schema (no markdown, no explanation, no extra keys):
{
  "matched_findings": int,
  "significant_errors": {
    "false_finding": int,
    "missing_finding": int,
    "mischaracterization": int,
    "wrong_location_laterality": int,
    "incorrect_birads": int,
    "incorrect_breast_density": int
  },
  "insignificant_errors": int
}
All counts must be non-negative integers.
"""


# -----------------------------
# Scoring
# -----------------------------

def mammo_green_score(matched_findings: int, significant_errors: Dict[str, int]) -> float:
    """
    Compute the MammoGREEN score.

    Score = matched_findings / (matched_findings + sum(significant_errors))

    Returns 0.0 if denominator is zero.
    """
    sig_sum = sum(int(v) for v in significant_errors.values())
    denom = matched_findings + sig_sum
    return float(matched_findings / denom) if denom > 0 else 0.0


# -----------------------------
# Error Handling
# -----------------------------

class JudgeError(RuntimeError):
    """Exception raised when judge call fails or returns invalid response."""
    pass


# -----------------------------
# JSON Extraction Utilities
# -----------------------------

def _extract_json_str(text: str) -> str:
    """
    Best-effort extraction of JSON from model response.

    Handles markdown code blocks and surrounding prose.
    """
    t = (text or "").strip()

    # Handle markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(code_block_pattern, t)
    if match:
        t = match.group(1).strip()
    else:
        # Handle incomplete code blocks (no closing ```)
        incomplete_pattern = r'```(?:json)?\s*([\s\S]*)'
        match = re.search(incomplete_pattern, t)
        if match:
            t = match.group(1).strip()

    # Extract JSON object
    if not (t.startswith("{") and t.endswith("}")):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]

    # Fix trailing commas before closing braces/brackets (common LLM JSON error)
    t = re.sub(r',(\s*[}\]])', r'\1', t)

    return t


# -----------------------------
# Judge Configurations
# -----------------------------

@dataclass
class OpenAIJudgeConfig:
    """Configuration for OpenAI judge."""
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_output_tokens: int = 8192


@dataclass
class GeminiJudgeConfig:
    """Configuration for Gemini judge."""
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 8192*2


# -----------------------------
# OpenAI Judge Implementation
# -----------------------------

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
    """Judge a single reference/hypothesis pair using OpenAI API."""
    user_msg = f"REFERENCE_REPORT:\n{reference}\n\nGENERATED_REPORT:\n{hypothesis}"

    try:
        resp = client.chat.completions.create(
            model=cfg.model_name,
            messages=[
                {"role": "system", "content": MAMMO_GREEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_output_tokens,
        )
    except Exception as e:
        raise JudgeError(f"OpenAI call failed: {e}") from e

    if not resp.choices:
        raise JudgeError("OpenAI returned no choices")

    choice = resp.choices[0]
    raw = choice.message.content

    if raw is None:
        raise JudgeError("Could not read response content from OpenAI")

    # Check for truncation
    if choice.finish_reason == "length":
        raw_preview = raw[:200] if raw else "(empty)"
        raise JudgeError(
            f"Response truncated due to max_tokens limit. "
            f"Increase max_output_tokens. Raw head: {raw_preview}"
        )

    return _parse_judge_response(raw)


# -----------------------------
# Gemini Judge Implementation
# -----------------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(JudgeError),
)
def _judge_one_gemini(
    client,  # genai.Client
    cfg: GeminiJudgeConfig,
    reference: str,
    hypothesis: str,
) -> Dict[str, Any]:
    """Judge a single reference/hypothesis pair using Google Gemini API."""
    user_msg = f"REFERENCE_REPORT:\n{reference}\n\nGENERATED_REPORT:\n{hypothesis}"

    try:
        config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            topP=0.95,
            topK=40,
            maxOutputTokens=cfg.max_output_tokens,
            systemInstruction=MAMMO_GREEN_SYSTEM_PROMPT,
        )

        response = client.models.generate_content(
            model=cfg.model_name,
            contents=user_msg,
            config=config,
        )
    except Exception as e:
        raise JudgeError(f"Gemini call failed: {e}") from e

    # Extract response text - handle different response formats
    raw = None
    try:
        raw = response.text
    except Exception:
        pass

    # Fallback: access candidates directly
    if not raw:
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    raw = candidate.content.parts[0].text
        except Exception:
            pass

    if not raw:
        debug_info = ""
        try:
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    debug_info = f" finish_reason={candidate.finish_reason}"
                if hasattr(candidate, 'safety_ratings'):
                    debug_info += f" safety_ratings={candidate.safety_ratings}"
        except Exception:
            pass
        raise JudgeError(f"Model returned empty response.{debug_info}")

    # Check for truncation (Gemini uses MAX_TOKENS or MAXTOKEN as finish reason)
    try:
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason and str(finish_reason).upper() in ("MAX_TOKENS", "MAXTOKEN", "LENGTH"):
                raw_preview = raw[:200] if raw else "(empty)"
                raise JudgeError(
                    f"Response truncated due to max_tokens limit. "
                    f"Increase max_output_tokens. Raw head: {raw_preview}"
                )
    except JudgeError:
        raise
    except Exception:
        pass  # Ignore errors in truncation check

    return _parse_judge_response(raw)


# -----------------------------
# Response Parsing
# -----------------------------

def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Parse and validate judge response JSON."""
    raw_json = _extract_json_str(raw)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raw_preview = raw[:500] if raw else "(empty)"
        raise JudgeError(f"Model did not return valid JSON. Raw: {raw_preview}") from e

    # Validate schema
    try:
        parsed = MammoGreenOutput(**data)
        parsed.validate_keys()
    except (ValidationError, ValueError) as e:
        raw_preview = raw[:200] if raw else "(empty)"
        raise JudgeError(f"JSON schema invalid: {e}. Raw head: {raw_preview}") from e

    return data


# -----------------------------
# Main Class
# -----------------------------

class MammoGREEN:
    """
    Mammography-GREEN metric using LLM judge.

    Supports both OpenAI (gpt-4o, gpt-5.x) and Google Gemini (gemini-2.5-*) models.

    Compatible with RadEval interface.

    __call__(refs, hyps) returns:
      (mean, std, green_scores, results_df)

    Where results_df contains:
      - reference: original reference text
      - prediction: original hypothesis text
      - green_score: per-sample GREEN score
      - matched_findings: count of matched findings
      - false_finding, missing_finding, mischaracterization,
        wrong_location_laterality, incorrect_birads, incorrect_breast_density: error counts
      - insignificant_errors: count of insignificant errors
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        provider: Optional[str] = None,
        output_dir: str = ".",
        batch_size: int = 8,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        sleep_s: float = 0.0,
        compute_summary_stats: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize MammoGREEN scorer.

        Args:
            model_name: Model identifier. Supported models:
                - OpenAI: gpt-4o, gpt-4o-mini, gpt-5.2-2025-12-11, gpt-5-mini-2025-08-07
                - Gemini: gemini-2.5-flash, gemini-2.5-pro
            provider: Explicit provider ("openai" or "gemini"). Auto-detected if None.
            output_dir: Directory for output files.
            batch_size: Batch size for processing.
            max_output_tokens: Maximum tokens in judge response.
            temperature: Sampling temperature (0.0 for deterministic).
            sleep_s: Sleep duration between API calls.
            compute_summary_stats: Whether to compute summary statistics.
            api_key: API key. Falls back to environment variables if not provided.
        """
        self.output_dir = output_dir
        self.batch_size = int(batch_size)
        self.sleep_s = float(sleep_s)
        self.compute_summary_stats = bool(compute_summary_stats)
        self.model_name = model_name

        # Determine provider
        if provider is not None:
            if provider.lower() not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported provider: '{provider}'. "
                    f"Supported providers are: {SUPPORTED_PROVIDERS}"
                )
            self.provider = provider.lower()
        else:
            self.provider = _detect_provider(model_name)

        # Initialize provider-specific client
        if self.provider == "gemini":
            self._init_gemini_client(api_key, model_name, temperature, max_output_tokens)
        else:
            self._init_openai_client(api_key, model_name, temperature, max_output_tokens)

        # Metadata for introspection
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
            "(f) Incorrect breast density assessment",
        ]

        # Outputs
        self.prompts: Optional[List[str]] = None
        self.completions: Optional[List[str]] = None
        self.green_scores: Optional[List[float]] = None
        self.error_counts: Optional[List[Dict[str, Any]]] = None

    def _init_gemini_client(
        self,
        api_key: Optional[str],
        model_name: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        """Initialize Gemini client and configuration."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai is not installed. "
                "Install it with: pip install google-genai"
            )

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set (or pass api_key=...). "
                "For Gemini models, set GOOGLE_API_KEY environment variable."
            )

        self.gemini_client = genai.Client(api_key=key)
        self.cfg = GeminiJudgeConfig(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.openai_client = None

    def _init_openai_client(
        self,
        api_key: Optional[str],
        model_name: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        """Initialize OpenAI client and configuration."""
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set (or pass api_key=...).")

        self.openai_client = OpenAI(api_key=key)
        self.cfg = OpenAIJudgeConfig(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.gemini_client = None

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

        for r, h in zip(refs, hyps):
            if self.provider == "gemini":
                data = _judge_one_gemini(self.gemini_client, self.cfg, r, h)
            else:
                data = _judge_one_openai(self.openai_client, self.cfg, r, h)

            mg = MammoGreenOutput(**data)
            mg.validate_keys()

            score = mammo_green_score(mg.matched_findings, mg.significant_errors)
            error_counts.append(data)
            green_scores.append(score)

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        self.error_counts = error_counts
        self.green_scores = green_scores

        mean = float(np.mean(green_scores)) if green_scores else 0.0
        std = float(np.std(green_scores)) if len(green_scores) > 1 else 0.0

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
        for ref, hyp, score, errors in zip(refs, hyps, green_scores, error_counts):
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
                "incorrect_breast_density": errors["significant_errors"]["incorrect_breast_density"],
                "insignificant_errors": errors["insignificant_errors"],
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _summary(self, scores: List[float], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics across all samples."""
        sig_sums = {k: 0 for k in SIGNIFICANT_ERROR_KEYS}
        matched_sum = 0
        insig_sum = 0

        for e in errors:
            matched_sum += int(e["matched_findings"])
            insig_sum += int(e["insignificant_errors"])
            for k in SIGNIFICANT_ERROR_KEYS:
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
        Alternative interface returning a dictionary.

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
# CLI Demo
# -----------------------------

if __name__ == "__main__":
    demo_refs = [
        (
            "Bilateral mammography demonstrates scattered fibroglandular density. "
            "There is a benign calcification at the upper outer left breast. "
            "BI-RADS 2."
        ),
    ]

    demo_hyps = [
        (
            "Bilateral mammography shows scattered fibroglandular tissue. "
            "No suspicious findings are identified. "
            "BI-RADS 1."
        ),
    ]

    # Example usage (requires OPENAI_API_KEY or GOOGLE_API_KEY)
    scorer = MammoGREEN(
        model_name="gpt-4o",
        temperature=0.0,
    )

    mean, std, scores, results_df = scorer(demo_refs, demo_hyps)

    print("GREEN scores:", scores)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    print("\nResults DataFrame:")
    print(results_df.to_string())

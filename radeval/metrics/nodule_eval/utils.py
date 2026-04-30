"""Utility helpers for the nodule_eval metric.

Responsibilities:
    - `extract_pn_segment(clean_findings)`: pull the PULMONARY NODULES: segment from
      a full clean_findings string. Returns "" if not present.
    - `extract_json_str(raw)`: strip markdown fences / repair truncated JSON from
      an LLM response.
    - `compute_per_row_metrics(parsed)`: deterministic scoring from the LLM's
      validated JSON output. Emits all per-row scores the metric exposes.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

# All 8 section headers used by lv004/lv005 clean_findings format.
# Order matters for the "next header" lookup regex: any of these terminates
# the PULMONARY NODULES: segment.
_ALL_HEADERS = [
    "LUNGS AND AIRWAYS",
    "PULMONARY NODULES",
    "MEDIASTINUM",
    "HEART AND GREAT VESSELS",
    "UPPER ABDOMEN",
    "BONES",
    "SUPPORT DEVICES",
    "SOFT TISSUES",
]
_OTHER_HEADERS = [h for h in _ALL_HEADERS if h != "PULMONARY NODULES"]

# Matches `PULMONARY NODULES: <content>` up to the next top-level header or EOS.
# `\s` covers the space / newline before the next header.
_PN_SEGMENT_RE = re.compile(
    r"PULMONARY NODULES:\s*(.*?)"
    r"(?=\s(?:" + "|".join(re.escape(h) for h in _OTHER_HEADERS) + r"):|$)",
    re.DOTALL,
)


def extract_pn_segment(clean_findings: str) -> str:
    """Extract the content of the PULMONARY NODULES: section.

    Returns the section body (without the `PULMONARY NODULES: ` prefix) or
    an empty string if the section is absent.
    """
    if not clean_findings or not isinstance(clean_findings, str):
        return ""
    m = _PN_SEGMENT_RE.search(clean_findings)
    if not m:
        return ""
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# JSON cleanup
# ---------------------------------------------------------------------------

def extract_json_str(text: str) -> str:
    """Strip markdown fences and trailing commas from an LLM JSON response."""
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


def parse_json_response(raw: str) -> dict:
    """Parse a JSON response. Raises ValueError on persistent failure."""
    cleaned = extract_json_str(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse nodule_eval JSON: {e}. Raw: {raw[:400]}"
        )


def validate_response(data: dict) -> None:
    """Validate the LLM response has the expected structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    for key in (
        "reference_nodules",
        "predicted_nodules",
        "matched_pairs",
        "false_findings",
        "missing_findings",
    ):
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in nodule_eval response")


# ---------------------------------------------------------------------------
# Per-row scoring (deterministic, pure-Python from parsed JSON)
# ---------------------------------------------------------------------------

# CRIMSON-inspired attribute severity weight for the composite score.
_ATTR_ERROR_WEIGHT = 0.5

# Per-nodule base weight — every nodule counts equally in this metric
# (no CRIMSON-style significance weighting; the LLM doesn't classify them).
_NODULE_BASE_WEIGHT = 1.0


def _is_numeric(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def compute_per_row_metrics(parsed: dict) -> dict[str, Any]:
    """Deterministic per-row scoring from the parsed LLM JSON.

    Returns a dict with all per-row metric values plus error counts. Never
    raises (any ambiguity resolves to NaN or 0 depending on the field).

    Key behaviors:
    - Detection P/R/F1 derived from counts of matched_pairs vs false/missing.
    - Size accuracy / exact_match / MAE / MAPE computed ONLY over matched
      pairs where BOTH sides have a numeric size. Fields that don't apply
      (e.g. no matches with both numeric sizes) come back as None.
    - Type / location / noun / uncertainty accuracy computed over ALL matched
      pairs; the LLM's flags (e.g. `type_error`) are used directly.
    - Composite follows CRIMSON's S-shaped formula:
        S = (correct - penalty) / N_G,   where
        correct = sum over matched pairs of base_weight * credit_factor
        penalty = sum of false-finding weights
        N_G     = sum over reference nodules of base_weight
      crimson_score = S if S >= 0 else -d/(1+d), d = penalty - correct
    """
    ref_list = parsed.get("reference_nodules") or []
    pred_list = parsed.get("predicted_nodules") or []
    matched = parsed.get("matched_pairs") or []
    false_findings = parsed.get("false_findings") or []
    missing_findings = parsed.get("missing_findings") or []

    n_ref = len(ref_list)
    n_pred = len(pred_list)
    n_matched = len(matched)
    n_false = len(false_findings)
    n_miss = len(missing_findings)

    # ------------------------------------------------------------------
    # Detection P/R/F1
    # ------------------------------------------------------------------
    # Precision: TP / (TP + FP). Undefined when there are no predicted nodules.
    # Recall:    TP / (TP + FN). Undefined when there are no reference nodules.
    precision = None
    recall = None
    f1 = None
    if n_pred > 0:
        precision = n_matched / (n_matched + n_false) if (n_matched + n_false) > 0 else 0.0
    if n_ref > 0:
        recall = n_matched / (n_matched + n_miss) if (n_matched + n_miss) > 0 else 0.0
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision is not None and recall is not None:
        f1 = 0.0

    # ------------------------------------------------------------------
    # Size metrics (over matched pairs with both numeric sizes)
    # ------------------------------------------------------------------
    size_acc_hits = size_exact_hits = 0
    size_samples = 0
    abs_errors_mm: list[float] = []
    pct_errors: list[float] = []

    for m in matched:
        ref_sz = m.get("ref_size_mm")
        pred_sz = m.get("pred_size_mm")
        if not (_is_numeric(ref_sz) and _is_numeric(pred_sz)):
            continue
        size_samples += 1
        # Tolerance flag comes from the LLM's `size_error` field; invert to accuracy.
        if not m.get("size_error", False):
            size_acc_hits += 1
        # Exact match: either LLM's flag, or compute from the numeric fields.
        exact_flag = m.get("size_exact_match")
        if exact_flag is None:
            exact_flag = (float(ref_sz) == float(pred_sz))
        if exact_flag:
            size_exact_hits += 1
        # Distances
        err = abs(float(pred_sz) - float(ref_sz))
        abs_errors_mm.append(err)
        if float(ref_sz) > 0:
            pct_errors.append(err / float(ref_sz))

    size_accuracy = (size_acc_hits / size_samples) if size_samples else None
    size_exact_match = (size_exact_hits / size_samples) if size_samples else None
    size_mae_mm = (sum(abs_errors_mm) / len(abs_errors_mm)) if abs_errors_mm else None
    size_mape = (sum(pct_errors) / len(pct_errors)) if pct_errors else None

    # ------------------------------------------------------------------
    # Type / location / noun / uncertainty accuracy (over all matched pairs)
    # ------------------------------------------------------------------
    def _acc(field_err: str) -> Optional[float]:
        if not matched:
            return None
        hits = sum(1 for m in matched if not m.get(field_err, False))
        return hits / len(matched)

    type_accuracy = _acc("type_error")
    location_accuracy = _acc("location_error")
    noun_accuracy = _acc("noun_error")
    uncertainty_accuracy = _acc("uncertainty_error")

    # ------------------------------------------------------------------
    # CRIMSON-style composite
    # ------------------------------------------------------------------
    #   correct: each matched pair credited by base_weight * credit_factor
    #            where credit_factor = base / (base + sum(attr_error_weights))
    #   penalty: sum of base_weight over false-finding ids
    #   N_G: sum of base_weight over all reference nodules
    N_G = n_ref * _NODULE_BASE_WEIGHT
    penalty = n_false * _NODULE_BASE_WEIGHT

    correct = 0.0
    for m in matched:
        base = _NODULE_BASE_WEIGHT
        n_attr_errors = sum(
            1 for k in ("size_error", "type_error", "location_error",
                        "noun_error", "uncertainty_error")
            if m.get(k, False)
        )
        sum_attr_weights = n_attr_errors * _ATTR_ERROR_WEIGHT
        denom = base + sum_attr_weights
        credit_factor = base / denom if denom > 0 else 0.0
        correct += base * credit_factor

    if N_G == 0 and n_pred == 0:
        # Both empty = perfect (neither side made claims). Composite = 1.
        composite = 1.0
        S = 1.0
    elif N_G == 0:
        # Ref is empty but pred isn't: pure false-finding penalty.
        S = -(penalty + 1)
        composite = -penalty / (1 + penalty) if penalty > 0 else 0.0
    else:
        S = (correct - penalty) / N_G
        if S >= 0:
            composite = S
        else:
            d = penalty - correct
            composite = -d / (1 + d) if d > 0 else 0.0

    return {
        # Detection
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        # Size
        "size_accuracy": size_accuracy,
        "size_exact_match": size_exact_match,
        "size_mae_mm": size_mae_mm,
        "size_mape": size_mape,
        "n_size_pairs": size_samples,
        # Attribute accuracies
        "type_accuracy": type_accuracy,
        "location_accuracy": location_accuracy,
        "noun_accuracy": noun_accuracy,
        "uncertainty_accuracy": uncertainty_accuracy,
        # Composite
        "composite": composite,
        # Raw counts (for detailed mode / diagnostics)
        "n_reference": n_ref,
        "n_predicted": n_pred,
        "n_matched": n_matched,
        "n_false_findings": n_false,
        "n_missing_findings": n_miss,
        # Raw attribute error counts
        "n_size_errors": sum(1 for m in matched if m.get("size_error")),
        "n_type_errors": sum(1 for m in matched if m.get("type_error")),
        "n_location_errors": sum(1 for m in matched if m.get("location_error")),
        "n_noun_errors": sum(1 for m in matched if m.get("noun_error")),
        "n_uncertainty_errors": sum(1 for m in matched if m.get("uncertainty_error")),
    }


def empty_row_result(both_empty: bool = True) -> dict[str, Any]:
    """Per-row result for a row where neither ref nor hyp has a PN section.

    Both sides empty -> composite=1.0 (perfect "agreement on no nodules"),
    all other metric fields None so they don't pollute aggregates.
    """
    if both_empty:
        composite = 1.0
    else:
        composite = None  # caller handles the asymmetric empty case upstream
    return {
        "detection_precision": None,
        "detection_recall": None,
        "detection_f1": None,
        "size_accuracy": None,
        "size_exact_match": None,
        "size_mae_mm": None,
        "size_mape": None,
        "n_size_pairs": 0,
        "type_accuracy": None,
        "location_accuracy": None,
        "noun_accuracy": None,
        "uncertainty_accuracy": None,
        "composite": composite,
        "n_reference": 0,
        "n_predicted": 0,
        "n_matched": 0,
        "n_false_findings": 0,
        "n_missing_findings": 0,
        "n_size_errors": 0,
        "n_type_errors": 0,
        "n_location_errors": 0,
        "n_noun_errors": 0,
        "n_uncertainty_errors": 0,
    }

"""Prompt construction for hoppr_modifier_cth_sv001 (modifier-accuracy judge).

For the 10 CT head/neck findings, this judge scores ONLY the non-presence
modifiers (severity, localization, laterality, region, sinus subsite). It is
told to first identify findings that BOTH reports assert as present, then
compare the applicable modifiers on each such finding.

Presence (present / absent / uncertain / not changed / not reported) is
explicitly out of scope - presence errors are scored by the sibling
``hoppr_crimson_cth_sv001`` metric, so scoring presence here would double-count.
"""
from __future__ import annotations

from ..hoppr_crimson_cth_sv001.cth_schema import (
    FINDING_DEFINITIONS,
    MODIFIER_MAP,
    MULTI_VALUED_MODIFIERS,
)

SYSTEM_MSG = (
    "You are an expert neuroradiology evaluator. For a fixed set of head/neck "
    "CT findings, you compare the descriptive MODIFIERS (severity, "
    "localization, laterality, region, sinus subsite) of a predicted report "
    "against a reference report. You never judge whether a finding is present "
    "or absent - only its modifiers. Always respond with valid JSON only."
)


def _modifier_catalog() -> str:
    """Render the per-finding applicable-modifier catalog for the prompt."""
    lines = []
    for finding, mods in MODIFIER_MAP.items():
        parts = []
        for mod, values in mods.items():
            multi = " (multiple may apply)" if mod in MULTI_VALUED_MODIFIERS else ""
            parts.append(f"{mod} [{'/'.join(values)}]{multi}")
        lines.append(f"  - {finding}: {'; '.join(parts)}")
    return "\n".join(lines)


_TASK = """\
TASK

STEP 1 - Agreed-present findings.
Identify the findings that BOTH the reference AND the predicted report assert as
PRESENT (positively described, not absent, not merely uncertain, not just
mentioned as normal). Only these findings are eligible for modifier scoring.
Ignore findings only one report describes - those presence errors are handled
by a separate metric.

STEP 2 - Compare applicable modifiers.
For each agreed-present finding, compare ONLY the modifiers listed for it in the
catalog above. For each applicable modifier slot emit a verdict:
- "correct": reference and predicted agree (after normalization).
- "incorrect": both reports state the modifier but they disagree
  (e.g. reference severe vs predicted mild; reference left vs predicted right).
- "pred_missing": the reference states the modifier but the prediction is
  silent about it.
- "pred_extra": the prediction states the modifier but the reference is silent
  about it.
Rules:
- Apply the severity-normalization rules before comparing severity.
- For Atrophy, only score "laterality" when localization is focal in the
  reference.
- Multi-valued modifiers (sinus subsite, calvarial region): compare as sets.
  "correct" if the sets match; "incorrect" if they overlap but differ;
  "pred_missing"/"pred_extra" if one side is silent.
- NEVER emit a verdict about presence. Presence is out of scope."""

_OUTPUT_FORMAT = """\
OUTPUT FORMAT
Return ONLY valid JSON in exactly this structure:
{
  "agreed_present_findings": ["Atrophy", "Sinus Disease"],
  "modifier_verdicts": [
    {"finding": "Atrophy", "modifier": "severity", "verdict": "correct", "reference_value": "mild", "predicted_value": "mild"},
    {"finding": "Atrophy", "modifier": "localization", "verdict": "incorrect", "reference_value": "focal", "predicted_value": "diffuse"},
    {"finding": "Sinus Disease", "modifier": "laterality", "verdict": "pred_missing", "reference_value": "right", "predicted_value": null}
  ]
}
If there are no agreed-present findings, return
{"agreed_present_findings": [], "modifier_verdicts": []}."""


def build_prompt(reference: str, predicted: str) -> str:
    """Assemble the full user prompt for one (reference, predicted) pair."""
    return "\n\n".join(
        [
            "OBJECTIVE\nScore how accurately the predicted head/neck CT report "
            "captures the descriptive MODIFIERS of findings both reports agree "
            "are present. Do not score presence.",
            FINDING_DEFINITIONS,
            "APPLICABLE MODIFIERS PER FINDING (only these are scored):\n"
            + _modifier_catalog(),
            f"Reference Report (Ground Truth):\n{reference}",
            f"Predicted Report (Candidate):\n{predicted}",
            _TASK,
            _OUTPUT_FORMAT,
        ]
    )

"""Prompt construction for hoppr_crimson_cth_sv001 (factual-correctness judge).

The judge scores the PRESENCE axis of the 10 CT head/neck findings: it extracts
the positive (present or uncertain) findings from both the reference and the
predicted report, matches them, and classifies presence errors into
false_findings / missing_findings, plus flags presence-state disagreements.

Modifier accuracy (severity, laterality, etc.) is intentionally NOT scored here
- that is the job of the sibling ``hoppr_modifier_cth_sv001`` metric.

The score reuses the CRIMSON [-1, 1] formula, weighting each finding by its
clinical significance.
"""
from __future__ import annotations

from .cth_schema import FINDING_DEFINITIONS, FINDINGS

SYSTEM_MSG = (
    "You are an expert neuroradiology evaluator. You compare a predicted head/"
    "neck CT report against a reference (ground-truth) report and identify "
    "factual errors at the level of whether each of a fixed set of findings is "
    "present. Always respond with valid JSON only, no markdown, no commentary."
)

# Significance levels mirror CRIMSON so the scorer weights are shared.
_SIGNIFICANCE_LEVELS = """\
CLINICAL SIGNIFICANCE LEVELS
Assign each positive finding one clinical_significance level, reflecting how
much an error involving it would matter for this head/neck CT domain:
- "urgent": a finding whose omission or fabrication would change acute
  management (e.g. acute-appearing Technical Limitation rendering the study
  nondiagnostic; large/new Encephalomalacia implying a significant prior
  insult).
- "actionable_not_urgent": clinically meaningful but not emergent (e.g.
  Encephalomalacia, moderate/severe Atrophy out of proportion to age,
  moderate/severe Sinus Disease with air-fluid level, Mastoid Effusion,
  Post-Surgical Calvarium newly described).
- "not_actionable_not_urgent": commonly incidental, documented but rarely
  acted on (e.g. mild White Matter Disease, mild Intracranial Atherosclerosis,
  mild Sinus Disease / retention cyst, age-appropriate Atrophy).
- "benign_expected": post-surgical/implant hardware that is stable and
  expected (e.g. Lens Replacement / pseudophakia, Scleral Buckle, stable
  chronic changes).
When unsure, default to "not_actionable_not_urgent"."""

_TASK = """\
TASK (perform in TWO steps)

STEP 1 - Extract positive findings.
For BOTH the reference and the predicted report, list every finding that the
report asserts as PRESENT or UNCERTAIN (i.e. positively described or hedged as
possibly present). Do NOT list findings the report calls absent, normal, or
does not mention. Each listed finding MUST be one of the 10 findings by its
exact name. Assign reference findings sequential IDs R1, R2, ... and predicted
findings P1, P2, .... Give each a clinical_significance level. If the same
finding name appears once per report, it is a single entry.

STEP 2 - Match and classify presence errors.
- matched_findings: pair each reference finding with the predicted finding of
  the SAME finding name, when both reports positively describe it.
- false_findings: predicted findings (by Pk id) with NO matching reference
  finding (the model asserted a finding the reference does not support).
- missing_findings: reference findings (by Rk id) with NO matching predicted
  finding (the model omitted a finding present in the reference).
- presence_state_mismatches: for a MATCHED finding, note when the presence
  STATE differs meaningfully, specifically reference "present" vs predicted
  "uncertain" or vice versa. (Do NOT report severity/laterality/other-modifier
  differences here - those are out of scope for this metric.)

Only assess PRESENCE. Ignore severity, localization, laterality, region, and
sinus subsite entirely."""

_OUTPUT_FORMAT = """\
OUTPUT FORMAT
Return ONLY valid JSON in exactly this structure:
{
  "reference_findings": [
    {"id": "R1", "finding": "<one of the 10 finding names>", "presence": "present|uncertain", "clinical_significance": "urgent|actionable_not_urgent|not_actionable_not_urgent|benign_expected"}
  ],
  "predicted_findings": [
    {"id": "P1", "finding": "<one of the 10 finding names>", "presence": "present|uncertain", "clinical_significance": "urgent|actionable_not_urgent|not_actionable_not_urgent|benign_expected"}
  ],
  "matched_findings": [
    {"ref_id": "R1", "pred_id": "P1"}
  ],
  "errors": {
    "false_findings": ["P2"],
    "missing_findings": ["R2"],
    "presence_state_mismatches": [
      {"ref_id": "R1", "pred_id": "P1", "reference_presence": "present", "predicted_presence": "uncertain", "explanation": "<brief>"}
    ]
  }
}"""


def build_prompt(reference: str, predicted: str) -> str:
    """Assemble the full user prompt for one (reference, predicted) pair."""
    findings_list = "\n".join(f"  - {name}" for name in FINDINGS)
    return "\n\n".join(
        [
            "OBJECTIVE\nEvaluate the factual accuracy of a predicted head/neck "
            "CT report against a reference report, restricted to whether each "
            "of the following 10 findings is present:",
            findings_list,
            FINDING_DEFINITIONS,
            f"Reference Report (Ground Truth):\n{reference}",
            f"Predicted Report (Candidate):\n{predicted}",
            _TASK,
            _SIGNIFICANCE_LEVELS,
            _OUTPUT_FORMAT,
        ]
    )

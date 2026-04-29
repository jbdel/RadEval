"""Prompt parts for the NoduleEval LLM-as-judge metric.

Compact, composable sections following the CRIMSON / MammoGREEN pattern.
`build_prompt(ref_segment, hyp_segment)` assembles the full user prompt.

Design notes:
    - Prompt stays under ~1.5K tokens (static portion ~1K + ~2x500 dynamic).
    - All scoring is deterministic in Python from the JSON output — the LLM
      only extracts structure and flags attribute errors.
    - Size tolerance rules mirror CRIMSON_CT's measurement guidance.
"""

SYSTEM_MSG = (
    "You are a radiology AI evaluator specializing in pulmonary nodule comparison. "
    "You compare predicted nodule descriptions against reference descriptions and "
    "identify matches, misses, false findings, and attribute errors. "
    "Always respond with strictly valid JSON only."
)


OBJECTIVE = """\
Objective:

Compare a PREDICTED list of pulmonary nodules / masses to a REFERENCE list. For
each side, parse every sentence into a structured nodule object, then determine
matches, false findings, and attribute errors. This is the sole evaluation of
positive pulmonary focal lesions; you do NOT consider any other anatomy.
Negated nodule statements (e.g. "No pulmonary nodules.") do not appear in these
inputs; ignore them if seen."""


TEMPLATE_REF = """\
Input format:

Each side is a sequence of sentences produced by a templated generator. The
target template is:

    There is a [size] [type] nodule in the [location].
    There is a [size] [type] mass in the [location].
    There are multiple [size] [type] nodules in the [location].
    There is possibly a [size] [type] nodule in the [location].  (existence uncertain)

Any slot may be omitted if not stated. Examples:
    "There is an 8 mm solid nodule in the right upper lobe."
    "There is a 4 mm nodule in the left lower lobe."  (no type)
    "There is a solid nodule in the right middle lobe." (no size)
    "There is a 3.2 cm mass in the right lower lobe."
    "There are multiple calcified nodules in the bilateral lungs."
    "There is possibly a 5 mm nodule in the lingula."

The predicted side may deviate from the template — parse it robustly."""


NODULE_FIELDS = """\
Nodule fields to extract per sentence:
    - id:         "R1", "R2", ... for reference; "P1", "P2", ... for predicted
    - size_mm:    numeric in millimeters if stated; otherwise null.
                  Convert cm -> mm (e.g. "2.3 cm" -> 23). For ranges
                  ("5-7 mm"), use the LARGER value. For dimensions
                  ("15 x 12 mm"), use the largest dimension.
    - type:       one of "solid", "part-solid", "ground-glass", "calcified";
                  otherwise null. Normalize "densely calcified" -> "calcified".
    - location:   standardized lobe or region. Use exactly one of:
                  "right upper lobe", "right middle lobe", "right lower lobe",
                  "left upper lobe", "lingula", "left lower lobe",
                  "right lung", "left lung", "bilateral lungs", "lung".
    - noun:       "nodule" or "mass" (based on what the source uses).
                  Clusters ("multiple nodules", "nodules") count as "nodule".
    - uncertain:  true if the sentence expresses existence uncertainty
                  ("possibly", "questionable", "cannot exclude"), else false.
    - text:       the exact source sentence."""


MATCHING_CRITERIA = """\
Matching criteria:

Two nodules match across ref/pred if BOTH conditions hold:
1. Compatible LOCATION — same lobe, or one is a parent of the other
   (e.g. "right upper lobe" matches "right lung", "bilateral lungs" matches
   any single lobe on either side IF the predicted count doesn't wildly
   contradict). Different specific lobes (e.g. "right upper lobe" vs
   "right lower lobe") do NOT match.
2. Compatible SIZE — both within 50% of each other, or at least one side
   has null size (in which case location alone determines the match).

Edge cases:
- A cluster sentence ("There are multiple ...") may match a single-nodule
  cluster on the other side (preserving the count category) or a group of
  individual nodule sentences in the same location. Use your clinical
  judgment; prefer one-to-one matches when possible.
- If several predictions fit one reference, match the closest (smallest
  size delta, then identical type, then identical noun) and flag the rest
  as false findings.
- Mass-vs-nodule is NOT a match criterion (both nouns can match each other
  if location+size compatible), but `noun_error` is flagged when they differ."""


ERROR_TAXONOMY = """\
Per matched pair, flag each of these BOOLEANS independently:

- size_error: true if sizes are present on BOTH sides AND the difference
    exceeds the clinical tolerance:
      * ref_size < 6 mm:  tolerance +/- 2 mm
      * 6 mm <= ref_size < 30 mm: tolerance +/- 4 mm
      * ref_size >= 30 mm (masses): tolerance +/- 20% of ref_size; always
        a size_error if the pred value crosses the 30 mm nodule-vs-mass
        boundary in either direction.
    If either side lacks a numeric size, size_error = false (we can't score).
- size_exact_match: true ONLY if both sides have a numeric size AND
    pred_size_mm == ref_size_mm exactly. false otherwise (including null).
- type_error: true if both sides have a non-null type AND they disagree
    (e.g. "solid" vs "ground-glass"). If either side's type is null, false.
- location_error: true if the locations disagree at the lobe level, after
    allowing the parent-hierarchy equivalence above.
- noun_error: true if one side says "nodule" and the other says "mass".
- uncertainty_error: true if one side marks `uncertain: true` and the
    other marks `false`.

For false_findings (predicted items not matched to any reference) and
missing_findings (reference items not matched to any prediction), emit only
the id(s)."""


OUTPUT_FORMAT = """\
Output format:

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "type": "solid", "location": "right upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 9, "type": "solid", "location": "right upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 9 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {
            "ref_id": "R1", "pred_id": "P1",
            "ref_size_mm": 8, "pred_size_mm": 9,
            "size_error": false, "size_exact_match": false,
            "type_error": false, "location_error": false,
            "noun_error": false, "uncertainty_error": false,
            "notes": "size within 4 mm tolerance for 8 mm nodule"
        }
    ],
    "false_findings": ["P2"],
    "missing_findings": ["R3"]
}

`ref_size_mm` and `pred_size_mm` must be numeric (int or float) when both
sides have a size; null otherwise. `size_error` and `size_exact_match` must
both be `false` when either size is null.

Do NOT include any fields beyond those listed. Do NOT add commentary outside
the JSON."""


FEW_SHOT_EXAMPLES = """\
Example 1 (perfect match, size within tolerance, same type and location):

REFERENCE NODULES:
There is an 8 mm solid nodule in the right upper lobe.

PREDICTED NODULES:
There is a 9 mm solid nodule in the right upper lobe.

Expected JSON:
{
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "type": "solid", "location": "right upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 9, "type": "solid", "location": "right upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 9 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 8, "pred_size_mm": 9,
         "size_error": false, "size_exact_match": false,
         "type_error": false, "location_error": false,
         "noun_error": false, "uncertainty_error": false,
         "notes": "within 4 mm tolerance"}
    ],
    "false_findings": [],
    "missing_findings": []
}


Example 2 (miss + false finding + matched pair with type error and size error):

REFERENCE NODULES:
There is a 5 mm solid nodule in the right upper lobe. There is a 15 mm nodule in the left lower lobe.

PREDICTED NODULES:
There is a 4 mm ground-glass nodule in the right upper lobe. There is an 8 mm nodule in the left upper lobe.

Expected JSON:
{
    "reference_nodules": [
        {"id": "R1", "size_mm": 5, "type": "solid", "location": "right upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 5 mm solid nodule in the right upper lobe."},
        {"id": "R2", "size_mm": 15, "type": null, "location": "left lower lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 15 mm nodule in the left lower lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 4, "type": "ground-glass", "location": "right upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 4 mm ground-glass nodule in the right upper lobe."},
        {"id": "P2", "size_mm": 8, "type": null, "location": "left upper lobe",
         "noun": "nodule", "uncertain": false,
         "text": "There is an 8 mm nodule in the left upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 5, "pred_size_mm": 4,
         "size_error": false, "size_exact_match": false,
         "type_error": true, "location_error": false,
         "noun_error": false, "uncertainty_error": false,
         "notes": "solid vs ground-glass"}
    ],
    "false_findings": ["P2"],
    "missing_findings": ["R2"]
}"""


IMPORTANT_NOTES = """\
Important notes:
- Only the PULMONARY NODULES content is passed to you. Do not invent other
  findings.
- Parse sizes carefully. "3.2 cm" -> 32. "2 mm micronodule" -> 2.
- Empty input for reference or prediction -> return the remaining side
  with all of its items in missing_findings or false_findings respectively,
  and an empty matched_pairs list.
- If both inputs are empty, return four empty lists plus empty finding lists.
- Deterministic behavior: do not rely on natural language judgment calls
  beyond the explicit criteria above. All scoring is computed from your
  JSON in downstream Python code."""


def build_prompt(ref_segment: str, hyp_segment: str) -> str:
    """Build the full user prompt.

    Parameters
    ----------
    ref_segment : str
        The PULMONARY NODULES: section content from the reference report.
        Can be empty string if the reference has no nodules.
    hyp_segment : str
        Same but for the predicted/hypothesis report.

    Returns
    -------
    str
        Complete prompt ready for LLM consumption.
    """
    ref_body = ref_segment.strip() or "(none)"
    hyp_body = hyp_segment.strip() or "(none)"

    sections = [
        OBJECTIVE,
        TEMPLATE_REF,
        NODULE_FIELDS,
        MATCHING_CRITERIA,
        ERROR_TAXONOMY,
        OUTPUT_FORMAT,
        FEW_SHOT_EXAMPLES,
        IMPORTANT_NOTES,
        f"REFERENCE NODULES:\n{ref_body}",
        f"PREDICTED NODULES:\n{hyp_body}",
    ]
    return "\n\n".join(sections)

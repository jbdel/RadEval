"""
Composable prompt parts for CRIMSON evaluation.

"""

OBJECTIVE = """\
Objective:

Evaluate the accuracy of predicted chest X-ray (CXR) findings compared to reference (ground truth) findings.
Only evaluate positive findings, not normal findings. Focus on clinical accuracy."""

TASK_STEPS = """\
Task:
Perform this evaluation in TWO STEPS:

STEP 1: Extract All Positive Findings
For BOTH reference and predicted reports, identify each ABNORMAL/POSITIVE finding exactly as written, including all descriptors, locations, severity terms, and measurements. List each finding as it appears in the original text.
If a single sentence contains multiple distinct findings, split them into separate entries (e.g., "bilateral pleural effusions and bibasilar atelectasis" should be two findings: "bilateral pleural effusions" and "bibasilar atelectasis").
When splitting, carry over shared qualifiers (certainty terms like "suggestive of", "consistent with", "suspicious for", as well as shared descriptors like location or severity) to EACH resulting finding so no context is lost.
For each finding (both reference and predicted), assign a clinical_significance level.
Assign each reference finding a sequential ID starting with "R" (R1, R2, R3, ...) and each predicted finding a sequential ID starting with "P" (P1, P2, P3, ...).

IMPORTANT: Include all abnormal findings in the reference_findings and predicted_findings lists (not just clinically significant ones). Do not include normal findings.

STEP 2: Compare and Classify Errors
Using the assigned IDs, match findings between reference and predicted, then identify errors in these 3 categories."""

CLINICAL_SIGNIFICANCE_LEVELS = """\
Clinical significance levels (used for findings):
    - "urgent": Requires immediate action / life-threatening
    - "actionable_not_urgent": Would change treatment plan but not immediately critical
    - "not_actionable_not_urgent": Minor discrepancy with minimal clinical impact (but still worth noting)
    - "benign_expected": Benign/expected finding with no clinical relevance (age-related changes)"""

SIGNIFICANCE_APPLICATION = """\
Apply these significance levels as follows:
- reference_findings and predicted_findings: classify each FINDING's clinical significance
- missing_findings (b) and false_findings (a): Do NOT include clinical_significance.
  Their weights are derived automatically from reference_findings and predicted_findings respectively."""

ATTRIBUTE_SEVERITY_LEVELS = """\
Attribute error severity (used ONLY for attribute_errors):
    - "significant": The attribute difference affects treatment or could lead to mismanagement
    - "negligible": The attribute difference is clinically insignificant and would not affect treatment"""

ATTRIBUTE_ERROR_INSTRUCTIONS = """\
For attribute_errors, also specify which error types apply from: ["location", "severity", "descriptor", "measurement", "certainty", "unspecific", "overinterpretation", "temporal"]
For each attribute_error, provide the ref_id and pred_id of the matched pair, and a brief "explanation" describing what the discrepancy is.

IMPORTANT: A matched finding pair can appear in attribute_errors MULTIPLE TIMES if it has multiple distinct issues.
Each attribute error should be a separate entry with its own severity classification and explanation.
For example, if a finding has both a location error and a severity error, create two separate entries with the same ref_id and pred_id."""

ERROR_CATEGORIES = """\
Error categories:
    a) False findings: Positive findings in predicted that are NOT present in reference
    b) Missing findings: Positive findings in reference but MISSING from predicted
    c) Attribute errors: Matched finding has incorrect or incomplete attributes including:
        - Location/laterality: Errors in anatomical location or side specification (left vs right lung, different lobes). 
        - Severity/extent: Errors in the degree or size characterization of the finding (small vs large, mild vs severe, minimal vs moderate)
        - Morphological descriptors: Errors in the shape, appearance, or characteristics of the finding (well-defined vs irregular margins, solid vs ground-glass density)
        - Measurements: Errors in quantitative size specifications (3 cm vs 1.5 cm for lesions, nodule diameter discrepancies)
        - Certainty: Errors in the confidence level of the finding (definite finding vs "possible" or "suspicious for")
        - Unspecific: Predicted finding is too vague or non-specific compared to the reference (reference says "pneumothorax" but predicted says "lucency")
        - Overinterpretation: Predicted finding makes an unsupported diagnostic leap beyond the imaging finding (reference says "opacity" but predicted says "mass")
        - Temporal/comparison: Errors in temporal descriptors or comparison with prior studies (missing "new", "worsening", "resolved", "stable", "unchanged")"""

MATCHING_CRITERIA = """\
Matching criteria:
- A finding matches if the core finding is present in both (e.g., "pleural effusion", "pneumothorax")
- A finding still matches even if location differs (location mismatch is an attribute error, not a match failure)
- One reference finding CAN match multiple predicted findings (one-to-many). For example, if R1 is "bilateral pleural effusions" and the predicted report splits it into P1 "left pleural effusion" and P2 "right pleural effusion", then both {{R1, P1}} and {{R1, P2}} are valid matches.
- For matched findings, check all attributes (location, severity, descriptors, measurements, certainty, unspecific, overinterpretation, temporal)
- If any attributes differ, add separate entries to attribute_errors for each distinct issue, referencing both the reference ID (ref_id) and predicted ID (pred_id)
- Any reference finding whose ID does not appear in matched_findings MUST be in missing_findings
- Any predicted finding whose ID does not appear in matched_findings MUST be in false_findings"""

OUTPUT_FORMAT = """\
Output Format:
Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
    "reference_findings": [
        {{"id": "R1", "finding": "<finding text>", "clinical_significance": "urgent|actionable_not_urgent|not_actionable_not_urgent|benign_expected"}},
        ...
    ],
    "predicted_findings": [
        {{"id": "P1", "finding": "<finding text>", "clinical_significance": "urgent|actionable_not_urgent|not_actionable_not_urgent|benign_expected"}},
        ...
    ],
    "matched_findings": [
        {{"ref_id": "R1", "pred_id": "P1"}},
        {{"ref_id": "R1", "pred_id": "P2"}},
        ...
    ],
    "errors": {{
        "false_findings": [
            "P3",
            ...
        ],
        "missing_findings": [
            "R2",
            ...
        ],
        "attribute_errors": [
            {{"ref_id": "R1", "pred_id": "P1", "severity": "significant|negligible", "error_types": ["location|severity|descriptor|measurement|certainty|unspecific|overinterpretation|temporal"], "explanation": "<brief description of the discrepancy>"}},
            ...
        ]
    }}
}}"""

CONTEXT_GUIDELINES = """\
Context guidelines:

Age-appropriate findings:
- Elderly patients (≥65): Expected degenerative changes such as aortic calcification, vascular tortuosity, degenerative spine changes, unfolding of the aorta, and sternal wire/median sternotomy findings (if chronic) should be classified under benign_expected — UNLESS the finding is directly related to the clinical indication.

Indication-finding concordance:
- A finding's clinical significance depends on whether it is EXPECTED given the indication and age, or UNEXPECTED and therefore more clinically important.
- The SAME finding can be urgent, actionable, or incidental depending on the clinical context:
    - Pleural effusion in a known heart failure patient with dyspnea: actionable_not_urgent (expected clinical course)
    - Pleural effusion in a young patient with palpitations: actionable_not_urgent or urgent (unexpected, needs workup)
    - Rib fractures in a young child with chest pain: urgent (concern for non-accidental injury)
    - Rib fractures in an adult post-RTA: actionable_not_urgent (trauma workup, possible hemothorax)
    - Rib fractures in an elderly patient with known old trauma: not_actionable_not_urgent or benign_expected (chronic/expected)
- When the indication directly explains a finding (e.g., "known heart failure" + cardiomegaly), the finding may be less actionable than the same finding without a known cause.
- When the indication does NOT explain a finding, it is generally more clinically significant and warrants higher classification.

Post-procedural, post-surgical context, and expectation of complication:
- If the indication references a recent procedure (line placement, intubation, chest tube insertion, post-CABG, post-thoracotomy), expected post-procedural findings should generally be classified as not_actionable_not_urgent:
- When a complication is clinically anticipated or known to be a high-probability outcome, its presence carries lower urgency than the same complication appearing unexpectedly:
    - Post-intubation aspiration (known risk): actionable_not_urgent even if new
    - Pneumothorax after central line placement (known procedural risk): actionable_not_urgent if small and anticipated
    - Mediastinal free air (Pneumomediastinum) after recent sternotomy or esophageal procedure: actionable_not_urgent if small and stable (expected post-surgical finding, but requires follow-up imaging)
    - New pleural effusion after cardiac surgery (expected post-op reaction): not_actionable_not_urgent if small and stable
- However, post-procedural COMPLICATIONS and UNEXPECTED findings remain at their usual or higher severity:
    - Effusion in a young, otherwise healthy patient: actionable_not_urgent (unexpected, needs workup)
    - Mediastinal widening without known prior dissection or trauma: urgent

Pediatric considerations:
- In young children, thymic shadow is normal and should not be classified as cardiomegaly or mediastinal widening.
- Unexpected findings in children (fractures, effusions) without a clear traumatic mechanism should be weighted higher due to concern for non-accidental injury.
- Lobar atelectasis in a young child may represent foreign body aspiration and should be classified as urgent, especially if clinical context suggests acute onset or no other explanation.
- Pediatric cardiac and mediastinal silhouette norms differ from adults — use age-appropriate thresholds.

Trauma context:
- When the indication mentions trauma, fall, MVC/RTA, or assault, fractures and pneumothorax should be classified at least actionable_not_urgent.
- Associated findings such as pleural effusion in trauma should raise concern for hemothorax (actionable_not_urgent or urgent depending on size).
- In elderly patients with a history of old/remote trauma, stable chronic findings (healed fractures, old deformities) are not_actionable_not_urgent or benign_expected.
"""

IMPORTANT_NOTES = """\
IMPORTANT NOTES:
- If a category has no items, use an empty array [].
- A single reference finding ID (R1, R2, ...) may appear in multiple matched_findings pairs (one-to-many matching). Any reference ID that does not appear in ANY matched pair MUST be in missing_findings.
- Every predicted finding ID (P1, P2, ...) must appear in EXACTLY ONE place: either in matched_findings or in false_findings.
- A missing finding CANNOT also appear in attribute_errors. If a reference finding is completely absent from the predicted findings, its ID goes ONLY in missing_findings.
- If a finding is classified as unspecific (in attribute_errors with "unspecific" error type), it is NOT a missing finding. Unspecific findings are PRESENT in the predicted report but too vague - they should appear in predicted_findings list and NOT in missing_findings."""

SIGNIFICANCE_EXAMPLES = """\
Examples of typical classifications (use guidelines and clinical judgment for patient-specific context):

Urgent findings (require immediate action):
    - Pneumothorax (especially tension pneumothorax)
    - Mediastinal widening 
    - Pneumomediastinum
    - Pneumoperitoneum
    - Mediastinal shift
    - Foreign Bodies 
    - Large pericardial effusion / Cardiomegaly
    - Malpositioned devices (e.g., endotracheal tube in right mainstem bronchus, misplaced central venous catheter, malpositioned chest tube)

Actionable non-urgent findings (change treatment but not emergent):
    - Moderate pleural effusion
    - Pulmonary nodules requiring follow-up
    - Pneumonia / Consolidation / Airspace disease or opacity / Infiltrates
    - Pulmonary edema (prominent perihilar / bronchial / vascular markings)
    - Mild to moderate cardiomegaly
    - Small non-tension pneumothorax
    - Masses
    - Fractures
    - Adenopathy

Not actionable, not urgent findings (minor clinical discrepancy, but still documented):
    - Stable chronic findings (old granulomas, chronic scarring)
    - Atelectasis
    - Trace pleural thickening / effusion 
    - Osteopenia
    - Appropriately positioned support devices and lines (NG tube, ETT, central lines, chest tubes in correct position) 

Benign/expected findings (no clinical relevance, age-related changes):
    - Age-appropriate atherosclerotic calcifications (e.g., aortic calcification in elderly)
    - Age-appropriate vascular tortuosity
    - Age-appropriate degenerative spine changes (in elderly patients)
    - Incidental stable benign findings (healed rib fracture, granuloma mentioned as "unchanged")
    - Expected physiologic findings (gastric air, bowel gas pattern)"""

ATTRIBUTE_SEVERITY_GUIDELINES = """\
Specific guidelines for attribute error severity:

Location/laterality errors:
    - NOT an error: If the reference finding does not specify a location, the predicted report adding location detail is acceptable and should NOT be flagged as a location error.
    - Significant: Wrong lung (left vs right), non-adjacent lobes (e.g., right upper lobe vs right lower lobe), non-adjacent zones (e.g., upper zone vs lower zone)
    - Negligible: Adjacent/touching lobes (e.g., right upper vs right middle, right middle vs right lower, left upper vs left lower), adjacent zones (e.g., upper vs mid zone, mid vs lower zone), minor positional differences within the same lobe (e.g., "apical" vs "lateral")

Severity/extent errors:
    - Significant: Changes clinical urgency or management (e.g., "small" vs "large", "mild" vs "severe", "minimal" vs "moderate", "simple" vs "tension")
    - Negligible: Stylistic differences that do not affect management (e.g., "small" vs "tiny")

Morphological descriptor errors:
    - Significant: Changes diagnostic considerations or management (e.g., "well-defined" vs "irregular margins", "solid" vs "ground-glass density", "simple" vs "complex")
    - Negligible: Stylistic differences that do not affect interpretation (e.g., "opacity" vs "opacification", "density" vs "opacification")

Measurement errors:
    - For nodules (< 3 cm): Significant if difference exceeds margin of error (for nodules < 6mm use 2mm margin; for nodules ≥ 6mm use 4mm margin). Negligible if within margin (e.g., "4mm" vs "5mm", "8mm" vs "11mm").
    - For masses (≥ 3 cm): Significant if difference exceeds 20% of the reference size. Negligible if within 20%. However, if the measurement error reclassifies a mass as a nodule (crossing below 3 cm), it is always significant regardless of percentage.

Certainty errors:
    - Significant: Adding or removing hedging language that changes management decisions (e.g., definite "pneumothorax" vs "possible pneumothorax" when clinical urgency would differ)
    - Negligible: Minor differences in hedging that do not affect management (e.g., "likely" vs "probable", "suggestive of" vs "compatible with")

Unspecific and Overinterpretation errors:
    Unspecific means the candidate is vaguer than the reference, overinterpretation means the candidate is more specific/diagnostic than the reference.
    - Significant: The specificity gap changes clinical management — either the vague term would not prompt the same workup, or the more specific term makes an unsupported diagnostic leap
        - "pneumothorax" and "lucency"
        - "pneumonia" and "opacity"
        - "mass" or "tumor" or "malignancy" and "opacity"
        - "pleural effusion" and "costophrenic angle blunting"
        - "consolidation" and "pulmonary edema" or "hemorrhage"
        - "lucency" and "pneumothorax" or "emphysema" or "cavitation"
        - "ground-glass opacity" and "pulmonary hemorrhage" or "early ARDS"
        - "nodule" and "granuloma" or "carcinoma"
        - "mediastinal widening" and "aortic dissection" or "lymphoma" or "mediastinal mass"
        - "pleural thickening" and "mesothelioma" or "empyema"
    - Negligible: The terms are commonly accepted as interchangeable, describe the same pathophysiological process at a slightly different point on its spectrum, and management would not meaningfully change
        - "consolidation" and "pneumonia"
        - "infiltrate" and "pneumonia"
        - "airspace opacity" and "airspace disease"
    - Note: If the overinterpreted or unspecific finding is in the same location as the reference finding, treat it as a MATCH with an attribute error, NOT as a false finding

Temporal/comparison errors:
    - Significant: Missing or incorrect "new", "worsening", "increasing", "resolved", or falsely adding these when not indicated (changes clinical urgency)
    - Negligible: Missing "stable", "unchanged", "chronic", "longstanding", or minor phrasing differences
    - Note: Do not penalize absence of temporal descriptors if no prior study is referenced in either report"""

def build_prompt(
    reference_findings,
    predicted_findings,
    patient_context=None,
    include_significance_examples=True,
    include_attribute_guidelines=True,
    include_context_guidelines=True,
):
    """Build the full evaluation prompt from composable parts.

    Args:
        reference_findings: Ground truth findings (string or list of strings)
        predicted_findings: Predicted findings (string or list of strings)
        patient_context: Optional dict with patient context (age, indication, etc.)
        include_significance_examples: Include examples of typical classifications
        include_attribute_guidelines: Include specific guidelines for attribute severity
        include_context_guidelines: Include context guidelines (only used when patient_context is provided)

    Returns:
        Formatted prompt string
    """
    # Convert lists to strings if needed
    if isinstance(reference_findings, list):
        reference_findings = " ".join(reference_findings)
    if isinstance(predicted_findings, list):
        predicted_findings = " ".join(predicted_findings)

    # Patient context block
    context_str = ""
    if patient_context:
        context_str = "Patient Context:\n"
        for key, value in patient_context.items():
            context_str += f"    {key}: {value}\n"

    # Assemble prompt sections
    sections = [
        OBJECTIVE,
        context_str,
        f"Reference Findings (Ground Truth):\n{reference_findings}",
        f"Predicted Findings (Candidate):\n{predicted_findings}",
        TASK_STEPS,
        CLINICAL_SIGNIFICANCE_LEVELS,
    ]

    if include_significance_examples:
        sections.append(SIGNIFICANCE_EXAMPLES)

    sections.append(SIGNIFICANCE_APPLICATION)
    sections.append(ATTRIBUTE_SEVERITY_LEVELS)

    sections.append(ERROR_CATEGORIES)
    sections.append(MATCHING_CRITERIA)
    sections.append(ATTRIBUTE_ERROR_INSTRUCTIONS)

    if include_attribute_guidelines:
        sections.append(ATTRIBUTE_SEVERITY_GUIDELINES)

    if include_context_guidelines and patient_context:
        sections.append(CONTEXT_GUIDELINES)

    sections.append(OUTPUT_FORMAT)
    sections.append(IMPORTANT_NOTES)

    return "\n\n".join(s for s in sections if s)

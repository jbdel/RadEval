"""CT-adapted prompt parts for CRIMSON evaluation.

Replaces CXR-specific sections from the original CRIMSON prompt with CT equivalents.
Sections not listed here (TASK_STEPS, CLINICAL_SIGNIFICANCE_LEVELS, etc.) are
imported unchanged from the original prompt_parts module.
"""

OBJECTIVE_CT = """\
Objective:

Evaluate the accuracy of predicted CT findings compared to reference (ground truth) findings.
Only evaluate positive findings, not normal findings. Focus on clinical accuracy."""

SIGNIFICANCE_EXAMPLES_CT = """\
Examples of typical classifications (use guidelines and clinical judgment for patient-specific context):

Urgent findings (require immediate action):
    - Pulmonary embolism (main or lobar arteries)
    - Aortic dissection / intramural hematoma / aortic rupture
    - Tension pneumothorax
    - Active hemorrhage / contrast extravasation
    - Bowel ischemia / pneumatosis intestinalis / portal venous gas
    - Free air (pneumoperitoneum) without prior surgery
    - Large pericardial effusion with tamponade physiology
    - Malpositioned devices (e.g., central venous catheter tip in wrong vessel)
    - High-grade airway obstruction
    - Cauda equina compression (spine CT)

Actionable non-urgent findings (change treatment but not emergent):
    - Pulmonary nodules requiring follow-up (Lung-RADS 3/4)
    - Solid organ lesions requiring characterization (liver, kidney, adrenal)
    - Moderate-to-large pleural / pericardial effusion
    - Pneumonia / consolidation / ground-glass opacities
    - Lymphadenopathy (short axis > 1 cm)
    - Fractures (vertebral, rib, pelvic)
    - Bowel obstruction (dilated loops, transition point)
    - Masses (lung, mediastinal, abdominal)
    - Pulmonary embolism (segmental / subsegmental)
    - Deep vein thrombosis
    - Moderate-to-severe stenosis (vascular)

Not actionable, not urgent findings (minor clinical discrepancy, but still documented):
    - Stable chronic findings (old granulomas, chronic scarring, fibrosis)
    - Subsegmental atelectasis
    - Small simple cysts (renal, hepatic)
    - Mild bronchiectasis
    - Trace pleural effusion / thickening
    - Osteopenia / mild degenerative changes
    - Diverticulosis without diverticulitis
    - Small non-obstructing renal calculi

Benign/expected findings (no clinical relevance, age-related changes):
    - Age-appropriate atherosclerotic calcifications (coronary, aortic)
    - Age-appropriate degenerative spine changes (in elderly patients)
    - Incidental stable benign findings (healed fracture, granuloma, simple cyst mentioned as unchanged)
    - Expected physiologic findings (bowel gas pattern, adrenal limb thickness)
    - Cholelithiasis without cholecystitis"""

CONTEXT_GUIDELINES_CT = """\
Context guidelines:

Age-appropriate findings:
- Elderly patients (>=65): Expected degenerative changes such as atherosclerotic calcifications, degenerative disc disease, facet arthropathy, vascular tortuosity, and benign prostatic hypertrophy should be classified under benign_expected -- UNLESS the finding is directly related to the clinical indication.

Indication-finding concordance:
- A finding's clinical significance depends on whether it is EXPECTED given the indication and age, or UNEXPECTED and therefore more clinically important.
- The SAME finding can be urgent, actionable, or incidental depending on the clinical context:
    - Pulmonary embolism in a post-operative patient with suspected PE: urgent (confirmation of suspected diagnosis)
    - Pulmonary embolism as incidental finding on staging CT: actionable_not_urgent
    - Liver lesion in a patient with known metastatic disease: not_actionable_not_urgent (known disease)
    - Liver lesion in a patient scanned for abdominal pain: actionable_not_urgent (needs characterization)
    - Adrenal nodule in a cancer staging CT: actionable_not_urgent (needs follow-up)
    - Adrenal nodule in elderly patient with no cancer history: not_actionable_not_urgent (likely adenoma)
- When the indication directly explains a finding, the finding may be less actionable than the same finding without a known cause.
- When the indication does NOT explain a finding, it is generally more clinically significant and warrants higher classification.

Post-procedural and post-surgical context:
- If the indication references a recent procedure or surgery, expected post-procedural findings should generally be classified as not_actionable_not_urgent:
    - Post-operative free fluid (small amount after abdominal surgery): not_actionable_not_urgent
    - Post-lobectomy volume loss and surgical changes: benign_expected
    - Post-stent graft endoleak (type II, small): not_actionable_not_urgent (requires surveillance)
- However, post-procedural COMPLICATIONS remain at their usual or higher severity:
    - Anastomotic leak after bowel surgery: urgent
    - Post-operative abscess: actionable_not_urgent or urgent depending on size/location
    - New pneumothorax after lung biopsy: actionable_not_urgent if small, urgent if large

Oncology staging context:
- When the indication is staging or restaging, findings should be interpreted in the context of treatment response vs progression.
- New or enlarging lesions suggest progression (actionable_not_urgent).
- Stable or shrinking lesions suggest response (not_actionable_not_urgent).
- Incidental findings unrelated to the primary malignancy still require appropriate classification.

Trauma context:
- When the indication mentions trauma, fall, MVC/RTA, or assault, fractures, solid organ lacerations, and hemorrhage should be classified at least actionable_not_urgent.
- Active contrast extravasation in trauma: urgent.
- In elderly patients with known prior trauma, stable chronic findings (healed fractures, old hematomas) are not_actionable_not_urgent or benign_expected."""

ATTRIBUTE_SEVERITY_GUIDELINES_CT = """\
Specific guidelines for attribute error severity:

Location/laterality errors:
    - NOT an error: If the reference finding does not specify a location, the predicted report adding location detail is acceptable and should NOT be flagged as a location error.
    - Significant: Wrong organ (liver vs spleen), wrong lung (left vs right), wrong kidney, wrong lobe (right upper vs right lower), wrong vertebral level by >1 level, wrong vascular territory
    - Negligible: Adjacent segments within the same organ (e.g., liver segment 6 vs 7), adjacent vertebral levels (e.g., L4 vs L5), minor positional differences within the same lobe

Severity/extent errors:
    - Significant: Changes clinical urgency or management (e.g., "small" vs "large", "mild" vs "severe", "minimal" vs "moderate", "partial" vs "complete obstruction", "low-grade" vs "high-grade stenosis")
    - Negligible: Stylistic differences that do not affect management (e.g., "small" vs "tiny", "trace" vs "minimal")

Morphological descriptor errors:
    - Significant: Changes diagnostic considerations (e.g., "solid" vs "cystic", "enhancing" vs "non-enhancing", "simple" vs "complex", "fat-containing" vs "solid", "calcified" vs "non-calcified")
    - Negligible: Stylistic differences that do not affect interpretation (e.g., "opacity" vs "density", "hypodense" vs "low-attenuation")

Measurement errors:
    - For nodules (< 3 cm): Significant if difference exceeds margin of error (for nodules < 6mm use 2mm margin; for nodules >= 6mm use 4mm margin). Negligible if within margin.
    - For masses (>= 3 cm): Significant if difference exceeds 20% of the reference size. Negligible if within 20%. However, if the measurement error reclassifies a mass as a nodule (crossing below 3 cm), it is always significant.
    - For organ measurements: Significant if crossing a clinical threshold (e.g., aortic diameter crossing 5 cm aneurysm threshold)

Certainty errors:
    - Significant: Adding or removing hedging that changes management (e.g., definite "pulmonary embolism" vs "possible pulmonary embolism")
    - Negligible: Minor hedging differences (e.g., "likely" vs "probable", "suggestive of" vs "compatible with")

Unspecific and Overinterpretation errors:
    Unspecific means the candidate is vaguer than the reference, overinterpretation means the candidate is more specific/diagnostic than the reference.
    - Significant: The specificity gap changes clinical management
        - "filling defect" and "pulmonary embolism"
        - "hypodense lesion" and "metastasis" or "abscess"
        - "ground-glass opacity" and "COVID pneumonia" or "pulmonary hemorrhage"
        - "lymph node" and "lymphoma" or "metastatic lymphadenopathy"
        - "consolidation" and "pneumonia" vs "hemorrhage" vs "infarct"
        - "bone lesion" and "metastasis" or "myeloma"
        - "adrenal nodule" and "adrenal adenoma" or "adrenal metastasis"
        - "aortic dilation" and "aortic dissection"
    - Negligible: Commonly accepted interchangeable terms
        - "consolidation" and "pneumonia"
        - "ground-glass opacity" and "ground-glass attenuation"
        - "airspace disease" and "airspace opacity"
    - Note: If the overinterpreted or unspecific finding is in the same location as the reference finding, treat it as a MATCH with an attribute error, NOT as a false finding

Temporal/comparison errors:
    - Significant: Missing or incorrect "new", "worsening", "increasing", "resolved", or falsely adding these when not indicated (changes clinical urgency)
    - Negligible: Missing "stable", "unchanged", "chronic", "longstanding", or minor phrasing differences
    - Note: Do not penalize absence of temporal descriptors if no prior study is referenced in either report"""


def build_prompt_ct(
    reference_findings,
    predicted_findings,
    patient_context=None,
    include_significance_examples=True,
    include_attribute_guidelines=True,
    include_context_guidelines=True,
):
    """Build the full CT evaluation prompt from composable parts.

    Uses CT-specific sections for OBJECTIVE, SIGNIFICANCE_EXAMPLES,
    CONTEXT_GUIDELINES, and ATTRIBUTE_SEVERITY_GUIDELINES. All other sections
    are reused from the original CRIMSON prompt_parts.
    """
    from ..crimson.prompt_parts import (
        TASK_STEPS,
        CLINICAL_SIGNIFICANCE_LEVELS,
        SIGNIFICANCE_APPLICATION,
        ATTRIBUTE_SEVERITY_LEVELS,
        ATTRIBUTE_ERROR_INSTRUCTIONS,
        ERROR_CATEGORIES,
        MATCHING_CRITERIA,
        OUTPUT_FORMAT,
        IMPORTANT_NOTES,
    )

    if isinstance(reference_findings, list):
        reference_findings = " ".join(reference_findings)
    if isinstance(predicted_findings, list):
        predicted_findings = " ".join(predicted_findings)

    context_str = ""
    if patient_context:
        context_str = "Patient Context:\n"
        for key, value in patient_context.items():
            context_str += f"    {key}: {value}\n"

    sections = [
        OBJECTIVE_CT,
        context_str,
        f"Reference Findings (Ground Truth):\n{reference_findings}",
        f"Predicted Findings (Candidate):\n{predicted_findings}",
        TASK_STEPS,
        CLINICAL_SIGNIFICANCE_LEVELS,
    ]

    if include_significance_examples:
        sections.append(SIGNIFICANCE_EXAMPLES_CT)

    sections.append(SIGNIFICANCE_APPLICATION)
    sections.append(ATTRIBUTE_SEVERITY_LEVELS)
    sections.append(ERROR_CATEGORIES)
    sections.append(MATCHING_CRITERIA)
    sections.append(ATTRIBUTE_ERROR_INSTRUCTIONS)

    if include_attribute_guidelines:
        sections.append(ATTRIBUTE_SEVERITY_GUIDELINES_CT)

    if include_context_guidelines and patient_context:
        sections.append(CONTEXT_GUIDELINES_CT)

    sections.append(OUTPUT_FORMAT)
    sections.append(IMPORTANT_NOTES)

    return "\n\n".join(s for s in sections if s)

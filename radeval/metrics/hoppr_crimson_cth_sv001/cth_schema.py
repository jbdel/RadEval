"""Shared CT head/neck finding schema for the CTH LLM-judge metrics.

This module is the single source of truth for the fixed, closed set of 10
CT head/neck findings and their clinically relevant modifiers. It is consumed
by both CTH judge metrics:

  - ``hoppr_crimson_cth_sv001``  (factual correctness, presence axis)
  - ``hoppr_modifier_cth_sv001`` (modifier accuracy, everything except presence)

Keeping the schema here (inside a private metric folder) means it is stripped
from the public repo along with the metrics that use it.

Two constants matter for prompt construction:

  * ``FINDINGS`` — ordered list of the 10 finding names.
  * ``MODIFIER_MAP`` — for each finding, which non-presence modifiers apply and
    their allowed values. Presence is deliberately absent from this map: the
    presence axis (present / absent / uncertain / not changed / not reported)
    is owned by the CRIMSON-style factual-correctness metric, so the modifier
    metric never scores it.

The long ``FINDING_DEFINITIONS`` block is the verbatim clinical guidance that is
injected into both judge prompts so the model classifies findings and
normalizes modifiers consistently.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# The fixed, closed set of findings
# ---------------------------------------------------------------------------

FINDINGS: list[str] = [
    "Atrophy",
    "White Matter Disease",
    "Sinus Disease",
    "Post-Surgical Calvarium",
    "Intracranial Atherosclerosis",
    "Lens Replacement",
    "Scleral Buckle",
    "Technical Limitation",
    "Encephalomalacia",
    "Mastoid Effusion",
]

# ---------------------------------------------------------------------------
# Non-presence modifiers per finding.
#
# Presence (present / absent / uncertain / not changed / not reported) is
# intentionally excluded: it belongs to the factual-correctness metric.
#
# Each modifier maps to its allowed value set. ``severity`` / ``localization`` /
# ``laterality`` are single-valued; ``sinus`` / ``region`` are multi-valued
# (multiple values may apply and are compared as sets).
# ---------------------------------------------------------------------------

SEVERITY_VALUES = ["mild", "moderate", "severe"]
LOCALIZATION_VALUES = ["diffuse", "focal"]
LATERALITY_VALUES = ["left", "right", "bilateral"]
LATERALITY_VALUES_WITH_MIDLINE = ["left", "right", "bilateral", "midline"]
SINUS_VALUES = ["maxillary", "ethmoid", "frontal", "sphenoid"]
CALVARIAL_REGION_VALUES = ["frontal", "parietal", "temporal", "occipital"]

MODIFIER_MAP: dict[str, dict[str, list[str]]] = {
    "Atrophy": {
        "severity": SEVERITY_VALUES,
        "localization": LOCALIZATION_VALUES,
        # laterality applies only when localization == focal
        "laterality": LATERALITY_VALUES,
    },
    "White Matter Disease": {
        "severity": SEVERITY_VALUES,
    },
    "Sinus Disease": {
        "severity": SEVERITY_VALUES,
        "laterality": LATERALITY_VALUES,
        "sinus": SINUS_VALUES,
    },
    "Post-Surgical Calvarium": {
        "laterality": LATERALITY_VALUES_WITH_MIDLINE,
        "region": CALVARIAL_REGION_VALUES,
    },
    "Intracranial Atherosclerosis": {
        "severity": SEVERITY_VALUES,
    },
    "Lens Replacement": {
        "laterality": LATERALITY_VALUES,
    },
    "Scleral Buckle": {
        "laterality": LATERALITY_VALUES,
    },
    "Technical Limitation": {
        "severity": SEVERITY_VALUES,
    },
    "Encephalomalacia": {
        "laterality": LATERALITY_VALUES,
    },
    "Mastoid Effusion": {
        "laterality": LATERALITY_VALUES,
    },
}

# Multi-valued modifiers are compared as set overlap rather than exact scalar
# equality.
MULTI_VALUED_MODIFIERS = {"sinus", "region"}


def modifiers_for(finding: str) -> dict[str, list[str]]:
    """Return the modifier -> allowed-values map for one finding (empty if none)."""
    return MODIFIER_MAP.get(finding, {})


# ---------------------------------------------------------------------------
# Verbatim clinical definitions injected into both judge prompts.
# ---------------------------------------------------------------------------

FINDING_DEFINITIONS = """\
CT HEAD/NECK FINDING DEFINITIONS (closed set of 10 findings)

You MUST only reason about these 10 findings. Do not invent findings outside
this list. Match report language to the closest of these findings.

1. Atrophy
   Loss of brain parenchymal volume: "volume loss", "atrophy", "prominent/
   exaggerated sulci", "ex vacuo ventricular enlargement", "ventriculomegaly
   secondary to volume loss". May be diffuse (global) or focal/regional
   (frontotemporal, cerebellar, hippocampal). Ventricular enlargement ALONE is
   ambiguous (ex vacuo vs hydrocephalus) - attribute to Atrophy only when framed
   as volume-related or paired with sulcal prominence.
   Modifiers: severity (mild/moderate/severe); localization (diffuse/focal);
   laterality (left/right/bilateral) when localization=focal. "generalized/
   global/diffuse" -> diffuse; named region/territory -> focal. Use "bilateral"
   for symmetric regional patterns (bifrontal, frontotemporal without a side).

2. White Matter Disease
   Hypoattenuation of cerebral white matter, typically chronic small vessel
   ischemic disease: "white matter hypodensity", "small vessel ischemic
   disease", "chronic microvascular ischemic changes", "leukoaraiosis",
   "periventricular/subcortical white matter disease". Excludes acute
   infarction, non-ischemic demyelination, focal encephalomalacia. Treated as
   diffuse/bilateral; laterality is NOT a modifier.
   Modifiers: severity (mild/moderate/severe). "minimal/scattered/mild" ->
   mild; "moderate/patchy-confluent" -> moderate; "advanced/extensive/
   confluent/severe" -> severe.

3. Sinus Disease
   Inflammatory/fluid abnormality of paranasal sinuses: "mucosal thickening",
   "sinus opacification", "air-fluid level", "retention cyst", "polypoid
   mucosal disease". The sinuses are often only partially imaged on head CT so
   omission is common.
   Modifiers: severity (mild/moderate/severe); laterality (left/right/
   bilateral); sinus subsite (maxillary/ethmoid/frontal/sphenoid, multiple may
   apply). "minimal/mild mucosal thickening" -> mild; "moderate thickening /
   air-fluid level / partial opacification" -> moderate; "complete/near-complete
   opacification" -> severe.

4. Post-Surgical Calvarium
   Evidence of prior cranial surgery: "craniotomy", "craniectomy", "burr hole",
   "cranioplasty", "bone flap", "calvarial fixation hardware", "post-surgical
   changes of the calvarium". Distinguish surgical defects from traumatic
   fractures (a fracture without stated surgical history does NOT qualify).
   Modifiers: laterality (left/right/bilateral/midline - use midline for vertex/
   sagittal-spanning or bifrontal defects with no side stated); region
   (frontal/parietal/temporal/occipital, multiple may apply).

5. Intracranial Atherosclerosis
   Atherosclerotic calcification of intracranial arteries (cavernous ICA,
   vertebrobasilar): "cavernous carotid calcification", "vertebrobasilar
   calcification", "intracranial atherosclerotic calcification", "vascular
   calcification". Non-contrast CT shows only mural calcification, NOT luminal
   stenosis - do not infer from "stenosis" without calcification. Treated as
   bilateral/systemic; laterality is NOT a modifier.
   Modifiers: severity (mild/moderate/severe). "trace/minimal/mild" -> mild;
   "moderate" -> moderate; "extensive/heavy/severe" -> severe.

6. Lens Replacement
   Intraocular lens (IOL) implant / pseudophakia, typically post-cataract:
   "intraocular lens", "lens implant/prosthesis", "pseudophakia", "post-cataract
   surgery changes". Orbits often only partially imaged.
   Modifiers: laterality (left/right/bilateral).

7. Scleral Buckle
   Surgically implanted band/buckle encircling the globe for retinal detachment
   repair: "scleral buckle", "encircling scleral band", "retinal detachment
   repair hardware". Incidental on head CT, only assessable when globes imaged.
   Modifiers: laterality (left/right/bilateral).

8. Technical Limitation
   Any stated factor degrading diagnostic quality: patient motion, beam-
   hardening/streak artifact, metallic artifact, limited field of view,
   "suboptimal study", "evaluation limited by [artifact]". Counts as present if
   it limits assessment of any region.
   Modifiers: severity (mild/moderate/severe). "mild/minimal/slight" -> mild;
   "moderate / limits evaluation of a region" -> moderate; "severe / markedly
   degraded / nondiagnostic / uninterpretable" -> severe.

9. Encephalomalacia
   Chronic parenchymal tissue loss/softening, sequela of prior infarct,
   hemorrhage, trauma, or surgery: "encephalomalacia", "encephalomalacic
   change", "gliosis with volume loss", "chronic infarct with encephalomalacia".
   Excludes acute/subacute ischemic hypodensity (chronicity framing required).
   Modifiers: laterality (left/right/bilateral).

10. Mastoid Effusion
    Fluid/soft-tissue opacification of the mastoid air cells: "mastoid
    effusion", "fluid in the mastoid air cells", "mastoid air cell
    opacification". Does not require frank mastoiditis. Chronic sclerosis/
    underpneumatization is NOT effusion.
    Modifiers: laterality (left/right/bilateral).

SEVERITY NORMALIZATION (apply before comparing severity):
- minimal / mild / slight / trace / scattered -> mild
- moderate / patchy-confluent -> moderate
- marked / advanced / extensive / confluent / heavy / severe / nondiagnostic ->
  severe
- For a range (e.g. "mild-to-moderate"), assign the HIGHER class."""

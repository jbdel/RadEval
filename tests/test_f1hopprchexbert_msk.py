"""Tests for HopprF1CheXbertMSK (v2) using real MSK report examples.

The ground-truth labels use 4-way coding:
  0 = definitely absent, 1 = not reported, 2 = uncertain, 3 = definitely present
Binary mapping: {0,1} -> negative, {2,3} -> positive.
"""
import os
import pytest

from radeval.metrics.f1hopprchexbert_msk import HopprF1CheXbertMSK

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/f1hopprchexbert_msk"
)

if HopprF1CheXbertMSK is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CheXbertMSK not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CheXbertMSK(checkpoint_dir=_CKPT_DIR)


# -- Real examples from the MSK v2 dataset ---------------------------------
# Condition order (20 heads, matches training __init__.py):
#   0  acute_fracture
#   1  healed_or_chronic_fracture
#   2  pathologic_fracture
#   3  osteoarthritis_or_degenerative_joint_disease
#   4  dislocation_or_subluxation
#   5  joint_effusion
#   6  osteopenia
#   7  soft_tissue_swelling_or_mass
#   8  bone_lesion
#   9  hardware_or_implanted_device
#  10  degenerative_disc_disease
#  11  scoliosis_or_spinal_deformity
#  12  erosive_or_inflammatory_arthropathy
#  13  osteonecrosis_or_avascular_necrosis
#  14  periosteal_reaction
#  15  chondrocalcinosis
#  16  soft_tissue_calcification
#  17  osteomyelitis
#  18  malalignment_or_deformity
#  19  spondylolisthesis

# Study 00000142: lumbar spine with OA/DJD, degenerative disc disease, scoliosis
SAMPLE_LUMBAR_OA_DDD_SCOLIOSIS = (
    "The vertebral body heights are maintained. There is no evidence of acute "
    "fracture. No osseous lesions are identified. The oblique views demonstrate "
    "no evidence of pars defect. There is a normal lumbar lordosis without "
    "significant subluxation. There is slight levoscoliosis centered at L2-3. "
    "Marginal osteophytosis. Moderate degenerative changes of L2-3 and L4-5. "
    "Other discs are normal in height."
)
# osteoarthritis_or_degenerative_joint_disease=1, degenerative_disc_disease=1,
# scoliosis_or_spinal_deformity=1
GT_BINARY_LUMBAR = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# Study 000002d6: right knee with OA, subluxation, joint effusion
SAMPLE_KNEE_OA_EFFUSION = (
    "There is mild tricompartmental degenerative disease with slight lateral "
    "subluxation of the tibial plateau relative to the femoral condyles. There "
    "is a moderate suprapatellar joint effusion with mild patellar spurring. "
    "There is normal intercondylar spine prominence."
)
# osteoarthritis_or_degenerative_joint_disease=1, dislocation_or_subluxation=1,
# joint_effusion=1
GT_BINARY_KNEE = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Study 00001663: left toes with acute fracture, soft tissue swelling, deformity
SAMPLE_TOE_FRACTURE_SWELLING = (
    "There is a nondisplaced fracture involving the base of the proximal second "
    "phalanx extending into the metatarsophalangeal joint. No dislocations are "
    "seen. Remaining joint spaces are intact. There is a nondisplaced fracture "
    "involving the base of the proximal second phalanx extending into the "
    "metatarsophalangeal joint. There is mild-to-moderate hallux valgus "
    "deformity of the first digit with mild soft tissue swelling compatible "
    "with bunion formation. There is mild soft tissue swelling compatible with "
    "bunion formation. Soft tissue swelling."
)
# acute_fracture=1, soft_tissue_swelling_or_mass=1, malalignment_or_deformity=1
GT_BINARY_TOE = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

REAL_REFS = [
    SAMPLE_LUMBAR_OA_DDD_SCOLIOSIS,
    SAMPLE_KNEE_OA_EFFUSION,
    SAMPLE_TOE_FRACTURE_SWELLING,
]
REAL_GT_BINARY = [GT_BINARY_LUMBAR, GT_BINARY_KNEE, GT_BINARY_TOE]


# -- Tests: model output structure ------------------------------------------

class TestHopprF1CheXbertMSKDirect:

    def test_returns_correct_tuple(self, scorer):
        accuracy, per_sample, report = scorer(REAL_REFS, REAL_REFS)
        assert isinstance(accuracy, float)
        assert isinstance(per_sample, list)
        assert len(per_sample) == len(REAL_REFS)
        assert isinstance(report, dict)

    def test_report_has_all_conditions(self, scorer):
        _, _, report = scorer(REAL_REFS, REAL_REFS)
        for label in scorer.LABELS:
            assert label in report, f"Missing condition: {label}"
        assert scorer.NO_FINDING in report

    def test_report_has_aggregate_keys(self, scorer):
        _, _, report = scorer(REAL_REFS, REAL_REFS)
        for key in ("micro avg", "macro avg", "weighted avg"):
            assert key in report
            for field in ("precision", "recall", "f1-score", "support"):
                assert field in report[key]

    def test_identical_reports_perfect_accuracy(self, scorer):
        accuracy, per_sample, _ = scorer(REAL_REFS, REAL_REFS)
        assert accuracy == 1.0
        assert all(s == 1.0 for s in per_sample)

    def test_validation_errors(self, scorer):
        with pytest.raises(TypeError):
            scorer("not a list", REAL_REFS)
        with pytest.raises(ValueError):
            scorer(REAL_REFS[:1], REAL_REFS)


# -- Tests: model predictions match ground truth ----------------------------

class TestHopprF1CheXbertMSKPredictions:
    """Verify the model correctly identifies conditions from real MSK reports."""

    def test_lumbar_oa_ddd_scoliosis(self, scorer):
        """Report with OA/DJD, degenerative disc disease, scoliosis."""
        y_pred = scorer._predict_label_matrix([SAMPLE_LUMBAR_OA_DDD_SCOLIOSIS])
        pred_binary = y_pred[0, :20].tolist()
        assert pred_binary == GT_BINARY_LUMBAR, (
            f"Expected {GT_BINARY_LUMBAR}, got {pred_binary}")

    def test_knee_oa_effusion(self, scorer):
        """Report with OA, subluxation, and joint effusion."""
        y_pred = scorer._predict_label_matrix([SAMPLE_KNEE_OA_EFFUSION])
        pred_binary = y_pred[0, :20].tolist()
        assert pred_binary == GT_BINARY_KNEE, (
            f"Expected {GT_BINARY_KNEE}, got {pred_binary}")

    def test_toe_fracture_swelling(self, scorer):
        """Report with acute fracture, soft tissue swelling, deformity."""
        y_pred = scorer._predict_label_matrix([SAMPLE_TOE_FRACTURE_SWELLING])
        pred_binary = y_pred[0, :20].tolist()
        assert pred_binary == GT_BINARY_TOE, (
            f"Expected {GT_BINARY_TOE}, got {pred_binary}")

    def test_no_finding_column(self, scorer):
        """When all 20 conditions are negative, no_finding should be 1."""
        y_pred = scorer._predict_label_matrix(
            ["Normal musculoskeletal exam. No acute findings."])
        all_negative = y_pred[0, :20].sum() == 0
        no_finding = y_pred[0, 20]
        if all_negative:
            assert no_finding == 1


# -- Tests: RadEval integration ---------------------------------------------

class TestHopprF1CheXbertMSKViaRadEval:

    def test_basic_output(self):
        from radeval import RadEval
        evaluator = RadEval(metrics=["f1hopprchexbert_msk"], show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_msk_accuracy" in results
        assert results["f1hopprchexbert_msk_accuracy"] == 1.0

    def test_details_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_msk"], detailed=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_msk_accuracy" in results
        assert "f1hopprchexbert_msk_label_scores_f1" in results
        assert isinstance(results["f1hopprchexbert_msk_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_msk"], per_sample=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)

        assert "f1hopprchexbert_msk_sample_acc" in results
        assert isinstance(results["f1hopprchexbert_msk_sample_acc"], list)
        assert len(results["f1hopprchexbert_msk_sample_acc"]) == len(REAL_REFS)

        assert all(s == 1.0 for s in results["f1hopprchexbert_msk_sample_acc"])

        assert "f1hopprchexbert_msk" not in results
        assert "f1hopprchexbert_msk_accuracy" not in results

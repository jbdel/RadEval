"""Tests for HopprF1CheXbertCT using real CT report examples from the test set.

The ground-truth labels use 4-way coding:
  0 = definitely absent, 1 = not reported, 2 = uncertain, 3 = definitely present
Binary mapping: {0,1} -> negative, {2,3} -> positive.
"""
import os
import pytest

from RadEval.metrics.f1hopprchexbert_ct import HopprF1CheXbertCT

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/hoppr_f1chexbert_ct"
)

if HopprF1CheXbertCT is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CheXbertCT not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CheXbertCT(checkpoint_dir=_CKPT_DIR)


# ── Real examples from the test set ──────────────────────────────────────
# Condition order (16 heads):
#   0  acute_aortic_injury
#   1  acute_pulmonary_embolism
#   2  acute_rib_fracture
#   3  acute_vertebral_fracture
#   4  aortic_aneurysm
#   5  aortic_atherosclerosis
#   6  aortic_valve_calcification
#   7  cardiomegaly
#   8  chronic_pulmonary_embolism
#   9  chronic_vertebral_compression_fracture
#  10  copd_emphysema
#  11  lung_nodule_or_mass
#  12  pleural_effusion
#  13  pneumothorax
#  14  air_space_opacity
#  15  prior_myocardial_infarction

SAMPLE_ATHEROSCLEROSIS_EMPHYSEMA = (
    "Non-Pulmonary Findings: \n \n"
    "Coronary and aortic calcific atherosclerosis. Minimal thoracic spondylosis.\n\n"
    "Pulmonary Findings: Centrilobular emphysema. Biapical pleural scarring.\n\n"
    "Nodules: No suspicious noncalcified pulmonary nodule identified."
)
GT_BINARY_15 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

SAMPLE_CARDIOMEGALY_OPACITY = (
    "There are nonspecific mediastinal lymph nodes.  The heart is borderline enlarged. "
    "There is mild bilateral cylindrical bronchiectasis. There are some lingular and "
    "right middle lobe atelectatic changes. \n \n"
    "On a limited exam of the upper abdomen, the visualized portion of the liver, "
    "spleen and adrenals are unremarkable."
)
GT_BINARY_85 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

SAMPLE_ATHEROSCLEROSIS_NODULE = (
    "Pleura: Normal. No pleural effusion or thickening.\n\n"
    "Mediastinum And Hila: Normal.\n\n"
    "Chest Wall And Lower Neck: Normal.\n\n"
    "Vessels: There is coronary artery calcification. "
    "There is a small amount of calcified plaque in the aorta.\n\n"
    "Bones: Degenerative changes are seen in the spine. \n \n"
    "Cholelithiasis is visualized."
)
GT_BINARY_195 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

REAL_REFS = [
    SAMPLE_ATHEROSCLEROSIS_EMPHYSEMA,
    SAMPLE_CARDIOMEGALY_OPACITY,
    SAMPLE_ATHEROSCLEROSIS_NODULE,
]
REAL_GT_BINARY = [GT_BINARY_15, GT_BINARY_85, GT_BINARY_195]


# ── Tests: model output structure ────────────────────────────────────────

class TestHopprF1CheXbertCTDirect:

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


# ── Tests: model predictions match ground truth ──────────────────────────

class TestHopprF1CheXbertCTPredictions:
    """Verify the model correctly identifies conditions from real CT reports."""

    def test_atherosclerosis_and_emphysema(self, scorer):
        """Report with aortic atherosclerosis and centrilobular emphysema."""
        y_pred = scorer._predict_label_matrix([SAMPLE_ATHEROSCLEROSIS_EMPHYSEMA])
        pred_binary = y_pred[0, :16].tolist()
        assert pred_binary == GT_BINARY_15, (
            f"Expected {GT_BINARY_15}, got {pred_binary}")

    def test_cardiomegaly_and_opacity(self, scorer):
        """Report with borderline cardiomegaly and atelectatic changes."""
        y_pred = scorer._predict_label_matrix([SAMPLE_CARDIOMEGALY_OPACITY])
        pred_binary = y_pred[0, :16].tolist()
        assert pred_binary == GT_BINARY_85, (
            f"Expected {GT_BINARY_85}, got {pred_binary}")

    def test_atherosclerosis_and_nodule(self, scorer):
        """Report with aortic calcified plaque and coronary calcification."""
        y_pred = scorer._predict_label_matrix([SAMPLE_ATHEROSCLEROSIS_NODULE])
        pred_binary = y_pred[0, :16].tolist()
        assert pred_binary == GT_BINARY_195, (
            f"Expected {GT_BINARY_195}, got {pred_binary}")

    def test_no_finding_column(self, scorer):
        """When all 16 conditions are negative, no_finding should be 1."""
        y_pred = scorer._predict_label_matrix(
            ["Normal CT chest. No acute findings."])
        all_negative = y_pred[0, :16].sum() == 0
        no_finding = y_pred[0, 16]
        if all_negative:
            assert no_finding == 1


# ── Tests: RadEval integration ───────────────────────────────────────────

class TestHopprF1CheXbertCTViaRadEval:

    def test_basic_output(self):
        from RadEval import RadEval
        evaluator = RadEval(metrics=["f1hopprchexbert_ct"], show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_ct_accuracy" in results
        assert results["f1hopprchexbert_ct_accuracy"] == 1.0

    def test_details_output(self):
        from RadEval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_ct"], detailed=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_ct_accuracy" in results
        assert "f1hopprchexbert_ct_label_scores_f1" in results
        assert isinstance(results["f1hopprchexbert_ct_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from RadEval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_ct"], per_sample=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)

        assert "f1hopprchexbert_ct_sample_acc" in results
        assert isinstance(results["f1hopprchexbert_ct_sample_acc"], list)
        assert len(results["f1hopprchexbert_ct_sample_acc"]) == len(REAL_REFS)

        assert all(s == 1.0 for s in results["f1hopprchexbert_ct_sample_acc"])

        assert "f1hopprchexbert_ct" not in results
        assert "f1hopprchexbert_ct_accuracy" not in results

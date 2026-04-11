"""Tests for HopprF1CheXbertAbd using real abdomen report examples.

The ground-truth labels use 4-way coding:
  0 = definitely absent, 1 = not reported, 2 = uncertain, 3 = definitely present
Binary mapping: {0,1} -> negative, {2,3} -> positive.
"""
import os
import pytest

from RadEval.metrics.f1hopprchexbert_abd import HopprF1CheXbertAbd

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/f1hopprchexbert_abd"
)

if HopprF1CheXbertAbd is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CheXbertAbd not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CheXbertAbd(checkpoint_dir=_CKPT_DIR)


# -- Real examples from the abdomen dataset ---------------------------------
# Condition order (13 heads, matches training __init__.py):
#   0  small_bowel_obstruction
#   1  large_bowel_obstruction_or_pseudo_obstruction
#   2  volvulus
#   3  ileus
#   4  pneumoperitoneum
#   5  fecal_loading
#   6  urinary_tract_calculus
#   7  foreign_body
#   8  hardware_or_implanted_device
#   9  vascular_calcification
#  10  pancreatic_calcification
#  11  gallstones
#  12  pneumatosis_or_portal_venous_gas

# Study 0000587a: Sitzmarks, fecal loading, surgical clips
SAMPLE_SITZMARKS = (
    "The bowel gas pattern is normal. 10 Sitzmarks are noted in the descending "
    "colon. A single Sitzmark is noted in the ascending colon. There is a marked "
    "amount of stool in the colon. There is no evidence of urinary calculi. The "
    "osseous structures demonstrate age-appropriate degenerative change of the "
    "lumbar spine. There are surgical clips in the right upper pelvis."
)
# fecal_loading=1, foreign_body=1, hardware_or_implanted_device=1
GT_BINARY_SITZMARKS = [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]

# Study 000323e7: kidney calculi and ureteral stent
SAMPLE_KIDNEY = (
    "There is no visceromegaly. Renal outlines are normal. Pulverized calculi "
    "within the collecting structures of the mid and lower pole of the right "
    "kidney are identified. There is no left nephrolithiasis. Bones are normal. "
    "There is a right ureteral stent in place with coils in the right renal "
    "pelvis and urinary bladder."
)
# urinary_tract_calculus=1, hardware_or_implanted_device=1
GT_BINARY_KIDNEY = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

# Study 0003a91b: stool burden and hip prosthesis
SAMPLE_STOOL = (
    "Nonobstructive bowel gas pattern. Moderate stool burden. No radiopaque "
    "urinary calculi. Advanced multilevel lumbar spondylosis. Right hip "
    "prosthesis partially imaged."
)
# fecal_loading=1, hardware_or_implanted_device=1
GT_BINARY_STOOL = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

REAL_REFS = [
    SAMPLE_SITZMARKS,
    SAMPLE_KIDNEY,
    SAMPLE_STOOL,
]
REAL_GT_BINARY = [GT_BINARY_SITZMARKS, GT_BINARY_KIDNEY, GT_BINARY_STOOL]


# -- Tests: model output structure ------------------------------------------

class TestHopprF1CheXbertAbdDirect:

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

class TestHopprF1CheXbertAbdPredictions:
    """Verify the model correctly identifies conditions from real abdomen reports."""

    def test_sitzmarks_fecal_foreign(self, scorer):
        """Report with fecal loading, foreign body (Sitzmarks), and surgical clips."""
        y_pred = scorer._predict_label_matrix([SAMPLE_SITZMARKS])
        pred_binary = y_pred[0, :13].tolist()
        assert pred_binary == GT_BINARY_SITZMARKS, (
            f"Expected {GT_BINARY_SITZMARKS}, got {pred_binary}")

    def test_kidney_calculi_stent(self, scorer):
        """Report with urinary tract calculus and ureteral stent."""
        y_pred = scorer._predict_label_matrix([SAMPLE_KIDNEY])
        pred_binary = y_pred[0, :13].tolist()
        assert pred_binary == GT_BINARY_KIDNEY, (
            f"Expected {GT_BINARY_KIDNEY}, got {pred_binary}")

    def test_stool_prosthesis(self, scorer):
        """Report with fecal loading and hip prosthesis."""
        y_pred = scorer._predict_label_matrix([SAMPLE_STOOL])
        pred_binary = y_pred[0, :13].tolist()
        assert pred_binary == GT_BINARY_STOOL, (
            f"Expected {GT_BINARY_STOOL}, got {pred_binary}")

    def test_no_finding_column(self, scorer):
        """When all 13 conditions are negative, no_finding should be 1."""
        y_pred = scorer._predict_label_matrix(
            ["Normal abdomen. No acute findings."])
        all_negative = y_pred[0, :13].sum() == 0
        no_finding = y_pred[0, 13]
        if all_negative:
            assert no_finding == 1


# -- Tests: RadEval integration ---------------------------------------------

class TestHopprF1CheXbertAbdViaRadEval:

    def test_basic_output(self):
        from RadEval import RadEval
        evaluator = RadEval(do_f1hopprchexbert_abd=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_abd_accuracy" in results
        assert results["f1hopprchexbert_abd_accuracy"] == 1.0

    def test_details_output(self):
        from RadEval import RadEval
        evaluator = RadEval(
            do_f1hopprchexbert_abd=True, do_details=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_abd_accuracy" in results
        assert "f1hopprchexbert_abd_label_scores_f1" in results
        assert isinstance(results["f1hopprchexbert_abd_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from RadEval import RadEval
        evaluator = RadEval(
            do_f1hopprchexbert_abd=True, do_per_sample=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)

        assert "f1hopprchexbert_abd_sample_acc" in results
        assert isinstance(results["f1hopprchexbert_abd_sample_acc"], list)
        assert len(results["f1hopprchexbert_abd_sample_acc"]) == len(REAL_REFS)

        assert all(s == 1.0 for s in results["f1hopprchexbert_abd_sample_acc"])

        assert "f1hopprchexbert_abd" not in results
        assert "f1hopprchexbert_abd_accuracy" not in results

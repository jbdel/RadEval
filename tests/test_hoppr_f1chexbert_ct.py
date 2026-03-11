import os
import pytest

from RadEval.metrics.hoppr_f1chexbert_ct import HopprF1CheXbertCT

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/"
    "hoppr_f1chexbert_ct"
)

if HopprF1CheXbertCT is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CheXbertCT not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CheXbertCT(checkpoint_dir=_CKPT_DIR)


CT_REFS = [
    "No acute aortic injury. No pulmonary embolism. Small bilateral pleural effusions. Cardiomegaly.",
    "Clear lungs. No pleural effusion. No pneumothorax.",
    "Large right-sided pneumothorax. Acute rib fractures on the right.",
]

CT_HYPS = [
    "No acute aortic injury. Small bilateral pleural effusions. Mild cardiomegaly.",
    "Clear lungs. No pleural effusion or pneumothorax.",
    "Large right-sided pneumothorax. Acute rib fractures on the right.",
]


class TestHopprF1CheXbertCTDirect:
    """Test the metric class directly."""

    def test_returns_correct_tuple(self, scorer):
        accuracy, per_sample, report = scorer(CT_HYPS, CT_REFS)
        assert isinstance(accuracy, float)
        assert isinstance(per_sample, list)
        assert len(per_sample) == len(CT_REFS)
        assert isinstance(report, dict)

    def test_report_has_all_conditions(self, scorer):
        _, _, report = scorer(CT_HYPS, CT_REFS)
        for label in scorer.LABELS:
            assert label in report, f"Missing condition: {label}"
        assert scorer.NO_FINDING in report

    def test_report_has_aggregate_keys(self, scorer):
        _, _, report = scorer(CT_HYPS, CT_REFS)
        for key in ("micro avg", "macro avg", "weighted avg"):
            assert key in report
            for field in ("precision", "recall", "f1-score", "support"):
                assert field in report[key]

    def test_identical_reports_perfect_accuracy(self, scorer):
        same = ["Large right-sided pneumothorax. Acute rib fractures."]
        accuracy, per_sample, _ = scorer(same, same)
        assert accuracy == 1.0
        assert per_sample == [1.0]

    def test_per_sample_length_matches_input(self, scorer):
        _, per_sample, _ = scorer(CT_HYPS, CT_REFS)
        assert len(per_sample) == len(CT_REFS)

    def test_accuracy_in_valid_range(self, scorer):
        accuracy, _, _ = scorer(CT_HYPS, CT_REFS)
        assert 0.0 <= accuracy <= 1.0

    def test_validation_errors(self, scorer):
        with pytest.raises(TypeError):
            scorer("not a list", CT_REFS)
        with pytest.raises(ValueError):
            scorer(CT_HYPS[:1], CT_REFS)


class TestHopprF1CheXbertCTViaRadEval:
    """Test integration through the RadEval interface."""

    def test_basic_output(self):
        from RadEval import RadEval
        evaluator = RadEval(do_hoppr_f1chexbert_ct=True, show_progress=False)
        results = evaluator(refs=CT_REFS, hyps=CT_HYPS)
        assert "hoppr_f1chexbert_ct_accuracy" in results
        assert "hoppr_f1chexbert_ct_micro avg_f1-score" in results
        assert "hoppr_f1chexbert_ct_macro avg_f1-score" in results
        assert "hoppr_f1chexbert_ct_weighted_f1" in results

    def test_details_output(self):
        from RadEval import RadEval
        evaluator = RadEval(
            do_hoppr_f1chexbert_ct=True, do_details=True, show_progress=False)
        results = evaluator(refs=CT_REFS, hyps=CT_HYPS)
        assert "hoppr_f1chexbert_ct" in results
        detail = results["hoppr_f1chexbert_ct"]
        assert "hoppr_f1chexbert_ct_accuracy" in detail
        assert "sample_scores" in detail
        assert "label_scores_f1-score" in detail

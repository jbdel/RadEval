import pytest
from RadEval import RadEval


@pytest.fixture(scope="module")
def evaluator():
    return RadEval(do_srrbert=True)


@pytest.fixture(scope="module")
def evaluator_details():
    return RadEval(do_srrbert=True, do_details=True)


class TestSRRBertPerfectMatch:
    """Identical ref/hyp pairs should yield perfect agreement."""

    refs = [
        "Small bilateral pleural effusions.",
        "Stable cardiomegaly. No pulmonary edema.",
    ]
    hyps = refs[:]

    def test_weighted_scores_are_perfect(self, evaluator):
        results = evaluator(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_f1"] == 1.0
        assert results["srrbert_weighted_precision"] == 1.0
        assert results["srrbert_weighted_recall"] == 1.0


class TestSRRBertCompletelyDifferent:
    """Reports with no clinical overlap should score zero."""

    refs = [
        "No acute cardiopulmonary process.",
        "Clear lungs bilaterally. No pleural effusion.",
    ]
    hyps = [
        "Large left pleural effusion with compressive atelectasis.",
        "Severe pulmonary edema. Bilateral pleural effusions.",
    ]

    def test_weighted_scores_are_zero(self, evaluator):
        results = evaluator(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_f1"] == 0.0
        assert results["srrbert_weighted_precision"] == 0.0
        assert results["srrbert_weighted_recall"] == 0.0


class TestSRRBertNumberedLists:
    """Numbered-list reports test sentence parsing and partial overlap.

    Pair 1: hyp drops finding 3 (edema) → partial match
    Pair 2: hyp adds a pneumothorax finding instead of negating it → mismatch
    """

    refs = [
        "1. Cardiomegaly. 2. Bilateral pleural effusions. 3. Pulmonary edema.",
        "1. Right lower lobe atelectasis. 2. No pneumothorax.",
    ]
    hyps = [
        "1. Cardiomegaly. 2. Bilateral pleural effusions.",
        "1. Right lower lobe atelectasis. 2. Small right pneumothorax.",
    ]

    def test_weighted_scores(self, evaluator):
        results = evaluator(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_f1"] == pytest.approx(0.6, abs=1e-4)
        assert results["srrbert_weighted_precision"] == pytest.approx(0.6, abs=1e-4)
        assert results["srrbert_weighted_recall"] == pytest.approx(0.6, abs=1e-4)


class TestSRRBertMixedStyles:
    """Mix of single-sentence, multi-sentence, and numbered-list reports.

    Pair 1: hyp adds pleural effusions not in ref
    Pair 2: hyp drops atelectasis present in ref
    Pair 3: identical "no acute findings"
    """

    refs = [
        "Mild pulmonary edema.",
        "1. Stable cardiomegaly. 2. Left pleural effusion. 3. Right basilar atelectasis.",
        "No acute findings.",
    ]
    hyps = [
        "Mild pulmonary edema with bilateral pleural effusions.",
        "Stable cardiomegaly. Left pleural effusion.",
        "No acute findings.",
    ]

    def test_weighted_scores(self, evaluator):
        results = evaluator(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_f1"] == pytest.approx(0.7333, abs=1e-4)
        assert results["srrbert_weighted_precision"] == pytest.approx(0.7, abs=1e-4)
        assert results["srrbert_weighted_recall"] == pytest.approx(0.8, abs=1e-4)

    def test_precision_lte_recall(self, evaluator):
        """Hyp adds an extra finding (pair 1), so precision should be <= recall."""
        results = evaluator(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_precision"] <= results["srrbert_weighted_recall"]


class TestSRRBertDetails:
    """Validate the structure and content of do_details=True output."""

    refs = [
        "Mild pulmonary edema.",
        "1. Stable cardiomegaly. 2. Left pleural effusion. 3. Right basilar atelectasis.",
        "No acute findings.",
    ]
    hyps = [
        "Mild pulmonary edema with bilateral pleural effusions.",
        "Stable cardiomegaly. Left pleural effusion.",
        "No acute findings.",
    ]

    def test_flat_keys(self, evaluator_details):
        results = evaluator_details(refs=self.refs, hyps=self.hyps)
        assert "srrbert_weighted_f1" in results
        assert "srrbert_weighted_precision" in results
        assert "srrbert_weighted_recall" in results
        assert "srrbert_label_scores" in results
        assert "srrbert" not in results

    def test_detail_weighted_means(self, evaluator_details):
        results = evaluator_details(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_f1"] == pytest.approx(0.7333, abs=1e-4)
        assert results["srrbert_weighted_precision"] == pytest.approx(0.7, abs=1e-4)
        assert results["srrbert_weighted_recall"] == pytest.approx(0.8, abs=1e-4)

    def test_label_scores_expected_labels(self, evaluator_details):
        results = evaluator_details(refs=self.refs, hyps=self.hyps)
        label_scores = results["srrbert_label_scores"]

        for label in ("Edema (Present)", "Cardiomegaly (Present)", "No Finding"):
            assert label in label_scores, f"Expected label '{label}' missing"

        for label, info in label_scores.items():
            for field in ("f1-score", "precision", "recall", "support"):
                assert field in info, f"Missing '{field}' in label '{label}'"

    def test_matched_labels_have_perfect_f1(self, evaluator_details):
        """Labels present in both ref and hyp for all their samples should have F1 = 1."""
        results = evaluator_details(refs=self.refs, hyps=self.hyps)
        label_scores = results["srrbert_label_scores"]
        assert label_scores["Edema (Present)"]["f1-score"] == 1.0
        assert label_scores["Cardiomegaly (Present)"]["f1-score"] == 1.0
        assert label_scores["No Finding"]["f1-score"] == 1.0

    def test_missed_label_has_zero_f1(self, evaluator_details):
        """Atelectasis is in the ref but missing from the hyp → F1 = 0."""
        results = evaluator_details(refs=self.refs, hyps=self.hyps)
        label_scores = results["srrbert_label_scores"]
        assert label_scores["Atelectasis (Present)"]["f1-score"] == 0.0
        assert label_scores["Atelectasis (Present)"]["support"] == 1.0


# ── Per-sample mode ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def evaluator_per_sample():
    return RadEval(do_srrbert=True, do_per_sample=True)


class TestSRRBertPerSample:
    """Validate do_per_sample=True output structure and values."""

    refs = [
        "Mild pulmonary edema.",
        "1. Stable cardiomegaly. 2. Left pleural effusion. 3. Right basilar atelectasis.",
        "No acute findings.",
    ]
    hyps = [
        "Mild pulmonary edema with bilateral pleural effusions.",
        "Stable cardiomegaly. Left pleural effusion.",
        "No acute findings.",
    ]

    def test_flat_keys(self, evaluator_per_sample):
        results = evaluator_per_sample(refs=self.refs, hyps=self.hyps)
        for key in ("srrbert_weighted_f1", "srrbert_weighted_precision", "srrbert_weighted_recall"):
            assert key in results
        assert "srrbert" not in results

    def test_values_are_lists(self, evaluator_per_sample):
        results = evaluator_per_sample(refs=self.refs, hyps=self.hyps)
        for key in ("srrbert_weighted_f1", "srrbert_weighted_precision", "srrbert_weighted_recall"):
            assert isinstance(results[key], list)
            assert len(results[key]) == len(self.refs)

    def test_per_sample_ordering(self, evaluator_per_sample):
        """Pair 3 (identical) should score highest."""
        results = evaluator_per_sample(refs=self.refs, hyps=self.hyps)
        sample_f1 = results["srrbert_weighted_f1"]
        assert sample_f1[2] > sample_f1[1] > sample_f1[0]

    def test_identical_pair_is_perfect(self, evaluator_per_sample):
        results = evaluator_per_sample(refs=self.refs, hyps=self.hyps)
        assert results["srrbert_weighted_f1"][2] == pytest.approx(1.0, abs=1e-6)
        assert results["srrbert_weighted_precision"][2] == pytest.approx(1.0, abs=1e-6)
        assert results["srrbert_weighted_recall"][2] == pytest.approx(1.0, abs=1e-6)

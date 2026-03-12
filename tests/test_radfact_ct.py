"""Tests for RadFact-CT metric.

Integration tests require OPENAI_API_KEY and are marked with @pytest.mark.integration.
Run with: OPENAI_API_KEY=... pytest tests/test_radfact_ct.py -s -m integration
"""
import os
import csv
import pytest

from RadEval.metrics.radfact_ct import RadFactCT

_HAS_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))

if RadFactCT is None:
    pytest.skip("RadFactCT not available", allow_module_level=True)


def _load_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


class TestRadFactCTUnit:
    """Unit tests that don't require an API key."""

    def test_import(self):
        assert RadFactCT is not None

    def test_missing_api_key(self):
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(EnvironmentError):
                RadFactCT(api_key=None)
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old

    def test_validation_errors(self):
        if not _HAS_API_KEY:
            pytest.skip("No API key")
        scorer = RadFactCT()
        with pytest.raises(TypeError):
            scorer("not a list", ["ref"])
        with pytest.raises(ValueError):
            scorer(["hyp1", "hyp2"], ["ref"])


@pytest.mark.integration
class TestRadFactCTIntegration:
    """Integration tests that call the LLM API."""

    @pytest.fixture(scope="class")
    def scorer(self):
        if not _HAS_API_KEY:
            pytest.skip("No OPENAI_API_KEY")
        return RadFactCT(model_name="gpt-4o-mini")

    @pytest.fixture(scope="class")
    def scorer_filter(self):
        if not _HAS_API_KEY:
            pytest.skip("No OPENAI_API_KEY")
        return RadFactCT(model_name="gpt-4o-mini", filter_negatives=True)

    def test_cynthia_csv_radfact_plus_minus(self, scorer):
        """Validate against expected values from RadFact authors (3-sample CSV)."""
        rows = _load_csv("tests/fixtures/radfact_ct_microsoft_original.csv")
        hyps = [r["prediction"] for r in rows]
        refs = [r["target"] for r in rows]
        agg, _ = scorer(hyps, refs)
        assert agg["logical_precision"] == pytest.approx(66.67, abs=5)
        assert agg["logical_recall"] == pytest.approx(83.33, abs=5)
        assert agg["logical_f1"] == pytest.approx(74.07, abs=5)

    def test_cynthia_csv_radfact_plus(self, scorer_filter):
        """Validate RadFact+ against expected values."""
        rows = _load_csv("tests/fixtures/radfact_ct_microsoft_original.csv")
        hyps = [r["prediction"] for r in rows]
        refs = [r["target"] for r in rows]
        agg, _ = scorer_filter(hyps, refs)
        assert agg["logical_precision"] == pytest.approx(66.67, abs=5)
        assert agg["logical_recall"] == pytest.approx(66.67, abs=5)
        assert agg["logical_f1"] == pytest.approx(66.67, abs=5)

    def test_identical_reports(self, scorer):
        """Identical ref/hyp should yield perfect scores."""
        same = ["The lungs are clear. No pleural effusion. Heart size is normal."]
        agg, details = scorer(same, same)
        assert agg["logical_precision"] == pytest.approx(100.0, abs=5)
        assert agg["logical_recall"] == pytest.approx(100.0, abs=5)
        assert agg["logical_f1"] == pytest.approx(100.0, abs=5)

    def test_completely_different_reports(self, scorer):
        """Completely unrelated reports should score low."""
        hyps = ["Large right pneumothorax requiring urgent chest tube placement."]
        refs = ["The liver appears normal. No renal calculi identified."]
        agg, details = scorer(hyps, refs)
        assert agg["logical_f1"] < 50

    def test_output_structure(self, scorer):
        """Check aggregate and per-sample output format."""
        hyps = ["Mild cardiomegaly. No pleural effusion."]
        refs = ["Heart size is mildly enlarged. Lungs are clear."]
        agg, details = scorer(hyps, refs)

        assert "logical_precision" in agg
        assert "logical_recall" in agg
        assert "logical_f1" in agg
        assert isinstance(details, list)
        assert len(details) == 1
        d = details[0]
        assert "hyp_phrases" in d
        assert "ref_phrases" in d
        assert "hyp_evidenced" in d
        assert "ref_evidenced" in d
        assert isinstance(d["hyp_phrases"], list)
        assert len(d["hyp_phrases"]) > 0

    def test_filter_negatives_reduces_phrases(self, scorer, scorer_filter):
        """RadFact+ should have fewer phrases than RadFact+/-."""
        text = "No pneumothorax. No pleural effusion. Mild cardiomegaly."
        from RadEval.metrics.radfact_ct.radfact_ct import report_to_phrases, filter_negatives
        all_phrases = report_to_phrases(
            scorer.client, scorer.model_name, text, scorer.temperature)
        filtered = filter_negatives(
            scorer_filter.client, scorer_filter.model_name,
            all_phrases, scorer_filter.temperature)
        assert len(filtered) <= len(all_phrases)

    def test_via_radeval_interface(self):
        """Test integration through RadEval."""
        if not _HAS_API_KEY:
            pytest.skip("No OPENAI_API_KEY")
        from RadEval import RadEval
        evaluator = RadEval(do_radfact_ct=True, show_progress=False)
        results = evaluator(
            refs=["The lungs are clear. Heart size is normal."],
            hyps=["The lungs are clear. Heart size is normal."],
        )
        assert "radfact_ct_f1" in results
        assert results["radfact_ct_f1"] >= 80

    def test_via_radeval_details(self):
        """do_details returns same flat keys as default for radfact_ct."""
        if not _HAS_API_KEY:
            pytest.skip("No OPENAI_API_KEY")
        from RadEval import RadEval
        evaluator = RadEval(
            do_radfact_ct=True, do_details=True, show_progress=False)
        results = evaluator(
            refs=["Mild cardiomegaly."],
            hyps=["Heart is mildly enlarged."],
        )
        assert "radfact_ct_precision" in results
        assert "radfact_ct_recall" in results
        assert "radfact_ct_f1" in results
        assert isinstance(results["radfact_ct_precision"], (int, float))

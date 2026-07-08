"""Tests for hoppr_crimson_cth_sv001 (CRIMSON-style CT head/neck factual judge).

Unit tests mock the Gemini judge (no live API). Integration tests require
GEMINI_API_KEY / GOOGLE_API_KEY and are marked with @pytest.mark.integration.
"""
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from radeval.metrics.hoppr_crimson_cth_sv001 import HopprCrimsonCthSv001

if HopprCrimsonCthSv001 is None:
    pytest.skip("HopprCrimsonCthSv001 not available", allow_module_level=True)

_HAS_API_KEY = bool(
    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

REFS_IDENTICAL = [
    "Mild generalized volume loss appropriate for age. Scattered periventricular "
    "white matter hypodensities consistent with chronic microvascular ischemia.",
]
HYPS_IDENTICAL = REFS_IDENTICAL[:]

REFS_DIFFERENT = [
    "Encephalomalacia in the right MCA territory consistent with prior infarct.",
]
HYPS_DIFFERENT = [
    "Study markedly degraded by patient motion, nondiagnostic.",
]

mock_eval_identical = {
    "reference_findings": [
        {"id": "R1", "finding": "Atrophy", "presence": "present",
         "clinical_significance": "not_actionable_not_urgent"},
        {"id": "R2", "finding": "White Matter Disease", "presence": "present",
         "clinical_significance": "not_actionable_not_urgent"},
    ],
    "predicted_findings": [
        {"id": "P1", "finding": "Atrophy", "presence": "present",
         "clinical_significance": "not_actionable_not_urgent"},
        {"id": "P2", "finding": "White Matter Disease", "presence": "present",
         "clinical_significance": "not_actionable_not_urgent"},
    ],
    "matched_findings": [
        {"ref_id": "R1", "pred_id": "P1"},
        {"ref_id": "R2", "pred_id": "P2"},
    ],
    "errors": {
        "false_findings": [],
        "missing_findings": [],
        "presence_state_mismatches": [],
    },
}

mock_eval_different = {
    "reference_findings": [
        {"id": "R1", "finding": "Encephalomalacia", "presence": "present",
         "clinical_significance": "actionable_not_urgent"},
    ],
    "predicted_findings": [
        {"id": "P1", "finding": "Technical Limitation", "presence": "present",
         "clinical_significance": "urgent"},
    ],
    "matched_findings": [],
    "errors": {
        "false_findings": ["P1"],
        "missing_findings": ["R1"],
        "presence_state_mismatches": [],
    },
}


def _make_scorer(**kwargs):
    """Construct a scorer with the Gemini client patched out."""
    with patch("google.genai.Client"):
        return HopprCrimsonCthSv001(
            provider="gemini", gemini_api_key="test-key", **kwargs)


class TestHopprCrimsonCthUnit:

    def test_import(self):
        assert HopprCrimsonCthSv001 is not None

    def test_prompt_contains_all_findings(self):
        from radeval.metrics.hoppr_crimson_cth_sv001.prompt import build_prompt
        from radeval.metrics.hoppr_crimson_cth_sv001.cth_schema import FINDINGS
        prompt = build_prompt("ref", "pred")
        for finding in FINDINGS:
            assert finding in prompt
        # presence-only: it should not ask the model to score modifiers
        assert "Only assess PRESENCE" in prompt

    def test_scoring_identical_perfect(self):
        scorer = _make_scorer()
        result = scorer._calculate_score(mock_eval_identical)
        assert result["crimson_cth_score"] == 1.0

    def test_scoring_different_negative(self):
        scorer = _make_scorer()
        result = scorer._calculate_score(mock_eval_different)
        assert result["crimson_cth_score"] < 0
        assert result["error_counts"]["false_findings"] == 1
        assert result["error_counts"]["missing_findings"] == 1

    def test_presence_mismatch_erodes_credit(self):
        ev = json.loads(json.dumps(mock_eval_identical))
        ev["errors"]["presence_state_mismatches"] = [
            {"ref_id": "R1", "pred_id": "P1",
             "reference_presence": "present", "predicted_presence": "uncertain",
             "explanation": "hedged"},
        ]
        scorer = _make_scorer()
        result = scorer._calculate_score(ev)
        assert 0 < result["crimson_cth_score"] < 1.0

    def test_mock_run_identical(self):
        scorer = _make_scorer()
        scorer._chat_completion_async = AsyncMock(
            return_value=json.dumps(mock_eval_identical))
        mean, std, scores, df = scorer(REFS_IDENTICAL, HYPS_IDENTICAL)
        assert mean == 1.0
        assert len(scores) == 1
        assert "crimson_cth_score" in df.columns

    def test_mock_run_different(self):
        scorer = _make_scorer()
        scorer._chat_completion_async = AsyncMock(
            return_value=json.dumps(mock_eval_different))
        mean, std, scores, df = scorer(REFS_DIFFERENT, HYPS_DIFFERENT)
        assert mean < 0

    def test_error_totals(self):
        scorer = _make_scorer()
        scorer._chat_completion_async = AsyncMock(
            return_value=json.dumps(mock_eval_different))
        scorer(REFS_DIFFERENT, HYPS_DIFFERENT)
        totals = scorer.error_totals()
        assert totals["false_findings"] == 1
        assert totals["missing_findings"] == 1

    def test_unsupported_provider_raises(self):
        with pytest.raises(NotImplementedError, match="does not support"):
            HopprCrimsonCthSv001(provider="hf", gemini_api_key="test-key")

    def test_radeval_integration_mock(self):
        from radeval import RadEval
        with patch("google.genai.Client"):
            evaluator = RadEval(
                metrics=["hoppr_crimson_cth_sv001"],
                gemini_api_key="test-key",
                detailed=True,
                show_progress=False,
            )
            evaluator.hoppr_crimson_cth_sv001_scorer._chat_completion_async = (
                AsyncMock(return_value=json.dumps(mock_eval_identical)))
            results = evaluator(refs=REFS_IDENTICAL, hyps=HYPS_IDENTICAL)
            assert results["hoppr_crimson_cth_sv001"] == 1.0
            assert "hoppr_crimson_cth_sv001_std" in results
            assert "hoppr_crimson_cth_sv001_total_missing_findings" in results


@pytest.mark.integration
class TestHopprCrimsonCthIntegration:
    """Real API tests -- require GEMINI_API_KEY / GOOGLE_API_KEY."""

    @pytest.fixture
    def api_key(self):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY / GOOGLE_API_KEY not set")
        return key

    def test_identical_reports(self, api_key):
        scorer = HopprCrimsonCthSv001(gemini_api_key=api_key)
        mean, std, scores, df = scorer(REFS_IDENTICAL, HYPS_IDENTICAL)
        assert mean >= 0.5

    def test_different_reports(self, api_key):
        scorer = HopprCrimsonCthSv001(gemini_api_key=api_key)
        mean, std, scores, df = scorer(REFS_DIFFERENT, HYPS_DIFFERENT)
        assert mean < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

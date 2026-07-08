"""Tests for hoppr_modifier_cth_sv001 (CT head/neck modifier-accuracy judge).

Unit tests mock the Gemini judge (no live API). Integration tests require
GEMINI_API_KEY / GOOGLE_API_KEY and are marked with @pytest.mark.integration.
"""
import json
import math
import os
from unittest.mock import AsyncMock, patch

import pytest

from radeval.metrics.hoppr_modifier_cth_sv001 import HopprModifierCthSv001

if HopprModifierCthSv001 is None:
    pytest.skip("HopprModifierCthSv001 not available", allow_module_level=True)

_HAS_API_KEY = bool(
    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

REFS = [
    "Moderate confluent small vessel ischemic change. Mild mucosal thickening "
    "of the right maxillary sinus.",
]
HYPS = [
    "Mild white matter hypodensity. Mucosal thickening of the left maxillary "
    "sinus.",
]

mock_eval_all_correct = {
    "agreed_present_findings": ["White Matter Disease", "Sinus Disease"],
    "modifier_verdicts": [
        {"finding": "White Matter Disease", "modifier": "severity",
         "verdict": "correct", "reference_value": "moderate",
         "predicted_value": "moderate"},
        {"finding": "Sinus Disease", "modifier": "severity",
         "verdict": "correct", "reference_value": "mild",
         "predicted_value": "mild"},
        {"finding": "Sinus Disease", "modifier": "laterality",
         "verdict": "correct", "reference_value": "right",
         "predicted_value": "right"},
    ],
}

mock_eval_mixed = {
    "agreed_present_findings": ["White Matter Disease", "Sinus Disease"],
    "modifier_verdicts": [
        {"finding": "White Matter Disease", "modifier": "severity",
         "verdict": "incorrect", "reference_value": "moderate",
         "predicted_value": "mild"},
        {"finding": "Sinus Disease", "modifier": "severity",
         "verdict": "correct", "reference_value": "mild",
         "predicted_value": "mild"},
        {"finding": "Sinus Disease", "modifier": "laterality",
         "verdict": "incorrect", "reference_value": "right",
         "predicted_value": "left"},
        {"finding": "Sinus Disease", "modifier": "sinus",
         "verdict": "correct", "reference_value": "maxillary",
         "predicted_value": "maxillary"},
    ],
}

mock_eval_none_agreed = {
    "agreed_present_findings": [],
    "modifier_verdicts": [],
}


def _make_scorer(**kwargs):
    with patch("google.genai.Client"):
        return HopprModifierCthSv001(
            provider="gemini", gemini_api_key="test-key", **kwargs)


class TestHopprModifierCthUnit:

    def test_import(self):
        assert HopprModifierCthSv001 is not None

    def test_prompt_excludes_presence(self):
        from radeval.metrics.hoppr_modifier_cth_sv001.prompt import build_prompt
        prompt = build_prompt("ref", "pred")
        assert "Presence is out of scope" in prompt
        # catalog should list applicable modifiers per finding
        assert "severity [mild/moderate/severe]" in prompt
        assert "sinus [maxillary/ethmoid/frontal/sphenoid]" in prompt

    def test_scoring_all_correct(self):
        scorer = _make_scorer()
        result = scorer._calculate_score(mock_eval_all_correct)
        assert result["modifier_cth_score"] == 1.0
        assert result["total_slots"] == 3
        assert result["correct_slots"] == 3

    def test_scoring_mixed(self):
        scorer = _make_scorer()
        result = scorer._calculate_score(mock_eval_mixed)
        # 2 correct of 4 slots
        assert result["modifier_cth_score"] == 0.5
        assert result["error_counts"]["severity_errors"] == 1
        assert result["error_counts"]["laterality_errors"] == 1

    def test_no_agreed_present_is_nan(self):
        scorer = _make_scorer()
        result = scorer._calculate_score(mock_eval_none_agreed)
        assert math.isnan(result["modifier_cth_score"])

    def test_nan_excluded_from_mean(self):
        scorer = _make_scorer()
        # first sample scores, second has no agreed-present findings
        responses = [
            json.dumps(mock_eval_all_correct),
            json.dumps(mock_eval_none_agreed),
        ]

        async def _fake(request):
            return responses.pop(0)

        scorer._chat_completion_async = AsyncMock(side_effect=_fake)
        mean, std, scores, df = scorer(REFS * 2, HYPS * 2)
        assert mean == 1.0  # nan sample excluded
        assert math.isnan(scores[1])

    def test_mock_run_mixed(self):
        scorer = _make_scorer()
        scorer._chat_completion_async = AsyncMock(
            return_value=json.dumps(mock_eval_mixed))
        mean, std, scores, df = scorer(REFS, HYPS)
        assert mean == 0.5
        assert "severity_errors" in df.columns

    def test_error_totals(self):
        scorer = _make_scorer()
        scorer._chat_completion_async = AsyncMock(
            return_value=json.dumps(mock_eval_mixed))
        scorer(REFS, HYPS)
        totals = scorer.error_totals()
        assert totals["severity_errors"] == 1
        assert totals["laterality_errors"] == 1

    def test_radeval_integration_mock(self):
        from radeval import RadEval
        with patch("google.genai.Client"):
            evaluator = RadEval(
                metrics=["hoppr_modifier_cth_sv001"],
                gemini_api_key="test-key",
                detailed=True,
                show_progress=False,
            )
            evaluator.hoppr_modifier_cth_sv001_scorer._chat_completion_async = (
                AsyncMock(return_value=json.dumps(mock_eval_mixed)))
            results = evaluator(refs=REFS, hyps=HYPS)
            assert results["hoppr_modifier_cth_sv001"] == 0.5
            assert "hoppr_modifier_cth_sv001_total_severity_errors" in results

    def test_unsupported_provider_raises(self):
        with pytest.raises(NotImplementedError, match="does not support"):
            HopprModifierCthSv001(provider="hf", gemini_api_key="test-key")


@pytest.mark.integration
class TestHopprModifierCthIntegration:
    """Real API tests -- require GEMINI_API_KEY / GOOGLE_API_KEY."""

    @pytest.fixture
    def api_key(self):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY / GOOGLE_API_KEY not set")
        return key

    def test_runs(self, api_key):
        scorer = HopprModifierCthSv001(gemini_api_key=api_key)
        mean, std, scores, df = scorer(REFS, HYPS)
        assert (math.isnan(mean)) or (0.0 <= mean <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

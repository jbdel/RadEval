"""Tests for HopprCrimsonCT (CT-adapted CRIMSON).

Integration tests require OPENAI_API_KEY and are marked with @pytest.mark.integration.
"""
import json
import math
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from RadEval.metrics.hoppr_crimson_ct import CRIMSON_CT

if CRIMSON_CT is None:
    pytest.skip("CRIMSON_CT not available", allow_module_level=True)

_HAS_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))

REFS_IDENTICAL = [
    "Centrilobular emphysema. Coronary artery calcifications. No pulmonary nodule.",
]
HYPS_IDENTICAL = REFS_IDENTICAL[:]

REFS_DIFFERENT = [
    "Large right pleural effusion with compressive atelectasis.",
]
HYPS_DIFFERENT = [
    "Normal liver parenchyma. No biliary dilation.",
]

REFS_PARTIAL = [
    "Right lower lobe pneumonia. Small right pleural effusion. Coronary calcifications.",
]
HYPS_PARTIAL = [
    "Right lower lobe pneumonia. Coronary calcifications.",
]

mock_eval_identical = {
    "reference_findings": [
        {"id": "R1", "finding": "centrilobular emphysema",
         "clinical_significance": "not_actionable_not_urgent"},
        {"id": "R2", "finding": "coronary artery calcifications",
         "clinical_significance": "benign_expected"},
    ],
    "predicted_findings": [
        {"id": "P1", "finding": "centrilobular emphysema",
         "clinical_significance": "not_actionable_not_urgent"},
        {"id": "P2", "finding": "coronary artery calcifications",
         "clinical_significance": "benign_expected"},
    ],
    "matched_findings": [
        {"ref_id": "R1", "pred_id": "P1"},
        {"ref_id": "R2", "pred_id": "P2"},
    ],
    "errors": {
        "false_findings": [],
        "missing_findings": [],
        "attribute_errors": [],
    },
}

mock_eval_different = {
    "reference_findings": [
        {"id": "R1", "finding": "large right pleural effusion",
         "clinical_significance": "actionable_not_urgent"},
        {"id": "R2", "finding": "compressive atelectasis",
         "clinical_significance": "actionable_not_urgent"},
    ],
    "predicted_findings": [
        {"id": "P1", "finding": "normal liver parenchyma",
         "clinical_significance": "benign_expected"},
    ],
    "matched_findings": [],
    "errors": {
        "false_findings": ["P1"],
        "missing_findings": ["R1", "R2"],
        "attribute_errors": [],
    },
}


def _mock_openai_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    return resp


class TestHopprCrimsonCTUnit:

    def test_import(self):
        assert CRIMSON_CT is not None

    def test_prompt_uses_ct_objective(self):
        from RadEval.metrics.hoppr_crimson_ct.prompt_parts_ct import build_prompt_ct
        prompt = build_prompt_ct("ref findings", "pred findings")
        assert "CT findings" in prompt
        assert "chest X-ray" not in prompt

    def test_prompt_has_ct_examples(self):
        from RadEval.metrics.hoppr_crimson_ct.prompt_parts_ct import build_prompt_ct
        prompt = build_prompt_ct("ref", "pred")
        assert "Pulmonary embolism" in prompt
        assert "Bowel ischemia" in prompt
        assert "Aortic dissection" in prompt

    def test_mock_identical_reports(self):
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                _mock_openai_response(json.dumps(mock_eval_identical)))
            mock_class.return_value = mock_client

            scorer = CRIMSON_CT(provider="openai", openai_api_key="test-key")
            scorer.scorer._chat_completion_async = AsyncMock(
                return_value=json.dumps(mock_eval_identical))
            mean, std, scores, df = scorer(REFS_IDENTICAL, HYPS_IDENTICAL)
            assert mean == 1.0

    def test_mock_different_reports(self):
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                _mock_openai_response(json.dumps(mock_eval_different)))
            mock_class.return_value = mock_client

            scorer = CRIMSON_CT(provider="openai", openai_api_key="test-key")
            scorer.scorer._chat_completion_async = AsyncMock(
                return_value=json.dumps(mock_eval_different))
            mean, std, scores, df = scorer(REFS_DIFFERENT, HYPS_DIFFERENT)
            assert mean <= 0

    def test_radeval_integration_mock(self):
        from RadEval import RadEval
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                _mock_openai_response(json.dumps(mock_eval_identical)))
            mock_class.return_value = mock_client

            evaluator = RadEval(
                metrics={"hoppr_crimson_ct": {"provider": "openai"}},
                openai_api_key="test-key",
                show_progress=False,
            )
            evaluator.hoppr_crimson_ct_scorer.scorer._chat_completion_async = AsyncMock(
                return_value=json.dumps(mock_eval_identical))
            results = evaluator(refs=REFS_IDENTICAL, hyps=HYPS_IDENTICAL)
            assert "hoppr_crimson_ct" in results
            assert results["hoppr_crimson_ct"] == 1.0

    def test_radeval_per_sample_mock(self):
        from RadEval import RadEval
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                _mock_openai_response(json.dumps(mock_eval_identical)))
            mock_class.return_value = mock_client

            evaluator = RadEval(
                metrics={"hoppr_crimson_ct": {"provider": "openai"}},
                openai_api_key="test-key",
                per_sample=True,
                show_progress=False,
            )
            evaluator.hoppr_crimson_ct_scorer.scorer._chat_completion_async = AsyncMock(
                return_value=json.dumps(mock_eval_identical))
            results = evaluator(refs=REFS_IDENTICAL, hyps=HYPS_IDENTICAL)
            assert isinstance(results["hoppr_crimson_ct"], list)
            assert len(results["hoppr_crimson_ct"]) == 1

    def test_radeval_details_mock(self):
        from RadEval import RadEval
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                _mock_openai_response(json.dumps(mock_eval_identical)))
            mock_class.return_value = mock_client

            evaluator = RadEval(
                metrics={"hoppr_crimson_ct": {"provider": "openai"}},
                openai_api_key="test-key",
                detailed=True,
                show_progress=False,
            )
            evaluator.hoppr_crimson_ct_scorer.scorer._chat_completion_async = AsyncMock(
                return_value=json.dumps(mock_eval_identical))
            results = evaluator(refs=REFS_IDENTICAL, hyps=HYPS_IDENTICAL)
            assert isinstance(results["hoppr_crimson_ct"], float)
            assert "hoppr_crimson_ct_std" in results

    def test_unsupported_provider_raises(self):
        """Test that unsupported provider raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support"):
            CRIMSON_CT(provider="gemini", openai_api_key="test-key")


@pytest.mark.integration
class TestHopprCrimsonCTIntegration:
    """Real API tests -- require OPENAI_API_KEY."""

    @pytest.fixture
    def api_key(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            pytest.skip("OPENAI_API_KEY not set")
        return key

    def test_identical_reports(self, api_key):
        scorer = CRIMSON_CT(provider="openai", openai_api_key=api_key)
        mean, std, scores, df = scorer(REFS_IDENTICAL, HYPS_IDENTICAL)
        assert mean >= 0.5

    def test_different_reports(self, api_key):
        scorer = CRIMSON_CT(provider="openai", openai_api_key=api_key)
        mean, std, scores, df = scorer(REFS_DIFFERENT, HYPS_DIFFERENT)
        assert mean < 0.5

    def test_partial_match(self, api_key):
        scorer = CRIMSON_CT(provider="openai", openai_api_key=api_key)
        mean, std, scores, df = scorer(REFS_PARTIAL, HYPS_PARTIAL)
        assert -1.0 <= mean <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

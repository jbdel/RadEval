"""Integration tests for the nodule_eval metric.

Uses mocked LLM responses so tests are deterministic and don't hit the API.
Four scenarios:
    1. Perfect match (ref == hyp, one nodule)
    2. Size tolerance error (8 mm vs 20 mm — outside ± 4 mm tolerance)
    3. Size exact-match-only difference (8 mm vs 9 mm — within tolerance
       but not exactly equal)
    4. Complete miss (ref has a nodule, hyp has none — different section entirely)

Run with:
    pytest tests/test_nodule_eval.py -s
"""
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Clean-findings fragments used in the tests. Must start with PULMONARY NODULES:
# per our schema (lv004/lv005).
_CF_WITH_8MM = (
    "PULMONARY NODULES: There is an 8 mm solid nodule in the right upper lobe. "
    "LUNGS AND AIRWAYS: No consolidation."
)
_CF_WITH_9MM = (
    "PULMONARY NODULES: There is a 9 mm solid nodule in the right upper lobe. "
    "LUNGS AND AIRWAYS: No consolidation."
)
_CF_WITH_20MM = (
    "PULMONARY NODULES: There is a 20 mm solid nodule in the right upper lobe. "
    "LUNGS AND AIRWAYS: No consolidation."
)
_CF_NO_NODULE = "LUNGS AND AIRWAYS: No consolidation. No pleural effusion."


# ---------------------------------------------------------------------------
# Mock LLM responses (one JSON per scenario) — these are what the judge LLM
# would return on the corresponding input, assuming it parses correctly.
# ---------------------------------------------------------------------------

_RESP_PERFECT = {
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 8, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 8, "pred_size_mm": 8,
         "size_error": False, "size_exact_match": True,
         "type_error": False, "location_error": False,
         "noun_error": False, "uncertainty_error": False,
         "notes": "exact match"}
    ],
    "false_findings": [],
    "missing_findings": [],
}

_RESP_SIZE_TOLERANCE_ERR = {
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 20, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is a 20 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 8, "pred_size_mm": 20,
         "size_error": True, "size_exact_match": False,
         "type_error": False, "location_error": False,
         "noun_error": False, "uncertainty_error": False,
         "notes": "12 mm difference exceeds 4 mm tolerance"}
    ],
    "false_findings": [],
    "missing_findings": [],
}

_RESP_SIZE_INEXACT_ONLY = {
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 9, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is a 9 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 8, "pred_size_mm": 9,
         "size_error": False, "size_exact_match": False,
         "type_error": False, "location_error": False,
         "noun_error": False, "uncertainty_error": False,
         "notes": "within 4 mm tolerance, not exact"}
    ],
    "false_findings": [],
    "missing_findings": [],
}

_RESP_COMPLETE_MISS = {
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "type": "solid",
         "location": "right upper lobe", "noun": "nodule", "uncertain": False,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [],
    "matched_pairs": [],
    "false_findings": [],
    "missing_findings": ["R1"],
}


def _mock_openai_response(content):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 100
    response.usage.total_tokens = 200
    return response


class TestNoduleEvalUnit:
    """Pure-Python tests on the utils (no LLM / no imports of the scorer)."""

    def test_extract_pn_segment_present(self):
        from RadEval.metrics.nodule_eval.utils import extract_pn_segment
        cf = (
            "LUNGS AND AIRWAYS: Clear. "
            "PULMONARY NODULES: There is an 8 mm solid nodule in the right upper lobe. "
            "MEDIASTINUM: No adenopathy."
        )
        assert (extract_pn_segment(cf)
                == "There is an 8 mm solid nodule in the right upper lobe.")

    def test_extract_pn_segment_at_start(self):
        from RadEval.metrics.nodule_eval.utils import extract_pn_segment
        cf = (
            "PULMONARY NODULES: There is a 5 mm nodule in the lingula. "
            "LUNGS AND AIRWAYS: Emphysema."
        )
        assert extract_pn_segment(cf) == "There is a 5 mm nodule in the lingula."

    def test_extract_pn_segment_absent(self):
        from RadEval.metrics.nodule_eval.utils import extract_pn_segment
        cf = "LUNGS AND AIRWAYS: Clear. MEDIASTINUM: No adenopathy."
        assert extract_pn_segment(cf) == ""

    def test_scoring_perfect_match(self):
        from RadEval.metrics.nodule_eval.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_PERFECT)
        assert m["detection_f1"] == 1.0
        assert m["size_accuracy"] == 1.0
        assert m["size_exact_match"] == 1.0
        assert m["size_mae_mm"] == 0.0
        assert m["size_mape"] == 0.0
        assert m["composite"] == 1.0

    def test_scoring_size_tolerance_error(self):
        from RadEval.metrics.nodule_eval.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_SIZE_TOLERANCE_ERR)
        assert m["detection_f1"] == 1.0       # matched 1/1, no false/miss
        assert m["size_accuracy"] == 0.0      # outside tolerance
        assert m["size_exact_match"] == 0.0
        assert m["size_mae_mm"] == 12.0
        assert abs(m["size_mape"] - 1.5) < 1e-9
        # 1 matched with 1 attr error: credit = 1 / (1 + 0.5) = 0.6667
        assert abs(m["composite"] - 0.6667) < 0.01

    def test_scoring_size_exact_match_only(self):
        from RadEval.metrics.nodule_eval.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_SIZE_INEXACT_ONLY)
        assert m["detection_f1"] == 1.0
        assert m["size_accuracy"] == 1.0   # within tolerance
        assert m["size_exact_match"] == 0.0   # but not exact
        assert m["size_mae_mm"] == 1.0
        assert abs(m["size_mape"] - 0.125) < 1e-9
        assert m["composite"] == 1.0

    def test_scoring_complete_miss(self):
        from RadEval.metrics.nodule_eval.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_COMPLETE_MISS)
        assert m["detection_recall"] == 0.0
        assert m["detection_precision"] is None
        assert m["detection_f1"] is None
        assert m["size_accuracy"] is None   # no matched pairs
        assert m["composite"] == 0.0


class TestNoduleEvalIntegration:
    """Integration tests with a mocked OpenAI client."""

    @pytest.fixture
    def mock_openai_client(self):
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI") as mock_async_class:
            mock_sync = MagicMock()
            mock_class.return_value = mock_sync
            mock_async = MagicMock()
            mock_async_class.return_value = mock_async
            yield mock_sync, mock_async

    def test_import(self):
        from RadEval.metrics.nodule_eval import NoduleEvalScore
        from RadEval.metrics.nodule_eval.adapter import NoduleEvalMetric
        assert NoduleEvalScore is not None
        assert NoduleEvalMetric is not None

    def test_invalid_provider(self):
        from RadEval.metrics.nodule_eval import NoduleEvalScore
        with pytest.raises(NotImplementedError, match="does not support"):
            NoduleEvalScore(provider="invalid", openai_api_key="x")

    def test_initialization_with_api_key(self, mock_openai_client):
        from RadEval.metrics.nodule_eval import NoduleEvalScore
        scorer = NoduleEvalScore(
            provider="openai", openai_api_key="test-key")
        assert scorer.provider == "openai"
        assert scorer.model_name == NoduleEvalScore.DEFAULT_OPENAI_MODEL

    def test_both_empty_short_circuit(self, mock_openai_client):
        """Rows with no PN section on either side should skip the LLM call."""
        from RadEval.metrics.nodule_eval import NoduleEvalScore
        scorer = NoduleEvalScore(provider="openai", openai_api_key="x")

        refs = [_CF_NO_NODULE]
        hyps = [_CF_NO_NODULE]

        # Mock the async chat completion — if it IS called, the test should
        # notice via the call count later.
        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response("{}")
        )

        mean, std, per_sample, df = scorer(refs, hyps)
        assert per_sample[0] == 1.0   # both-empty -> composite=1.0
        # Must NOT have called the LLM.
        mock_async.chat.completions.create.assert_not_called()

    def test_full_pipeline_perfect_match(self, mock_openai_client):
        """End-to-end with mocked LLM returning the perfect-match JSON."""
        from RadEval.metrics.nodule_eval import NoduleEvalScore
        scorer = NoduleEvalScore(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        mean, std, per_sample, df = scorer([_CF_WITH_8MM], [_CF_WITH_8MM])
        assert per_sample[0] == 1.0
        assert mean == 1.0
        # DataFrame sanity
        assert df.iloc[0]["detection_f1"] == 1.0
        assert df.iloc[0]["size_exact_match"] == 1.0

    def test_full_pipeline_size_tolerance_err(self, mock_openai_client):
        from RadEval.metrics.nodule_eval import NoduleEvalScore
        scorer = NoduleEvalScore(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_SIZE_TOLERANCE_ERR))
        )

        mean, std, per_sample, df = scorer([_CF_WITH_8MM], [_CF_WITH_20MM])
        # Composite reduced by 1 attribute error
        assert abs(per_sample[0] - 0.6667) < 0.01
        assert df.iloc[0]["size_mae_mm"] == 12.0
        assert df.iloc[0]["size_accuracy"] == 0.0

    def test_adapter_default_mode(self, mock_openai_client):
        """Verify NoduleEvalMetric.compute() returns aggregate keys."""
        from RadEval.metrics.nodule_eval.adapter import NoduleEvalMetric
        metric = NoduleEvalMetric(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        out = metric.compute([_CF_WITH_8MM], [_CF_WITH_8MM],
                             per_sample=False, detailed=False)
        assert "nodule_eval_detection_f1" in out
        assert "nodule_eval_size_mae_mm" in out
        assert out["nodule_eval_composite"] == 1.0

    def test_adapter_per_sample_mode(self, mock_openai_client):
        from RadEval.metrics.nodule_eval.adapter import NoduleEvalMetric
        metric = NoduleEvalMetric(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        out = metric.compute([_CF_WITH_8MM], [_CF_WITH_8MM],
                             per_sample=True, detailed=False)
        assert isinstance(out["nodule_eval_composite"], list)
        assert out["nodule_eval_composite"] == [1.0]

    def test_adapter_detailed_mode(self, mock_openai_client):
        from RadEval.metrics.nodule_eval.adapter import NoduleEvalMetric
        metric = NoduleEvalMetric(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        out = metric.compute([_CF_WITH_8MM], [_CF_WITH_8MM],
                             per_sample=False, detailed=True)
        assert "nodule_eval_composite" in out
        assert "nodule_eval_composite_std" in out

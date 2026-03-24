import inspect
import json
import math
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# run with:
# pytest tests/test_crimson.py -s
# For integration tests with real API:
# OPENAI_API_KEY=<key> pytest tests/test_crimson.py -s -m integration

epsilon = 1e-5

refs = [
    "Moderate right pleural effusion.",
    "Small left apical pneumothorax.",
]

hyps = [
    "Moderate right pleural effusion.",
    "Large right apical pneumothorax.",
]


mock_evaluations = [
    {
        "reference_findings": [
            {
                "id": "R1",
                "finding": "moderate right pleural effusion",
                "clinical_significance": "actionable_not_urgent",
            }
        ],
        "predicted_findings": [
            {
                "id": "P1",
                "finding": "moderate right pleural effusion",
                "clinical_significance": "actionable_not_urgent",
            }
        ],
        "matched_findings": [{"ref_id": "R1", "pred_id": "P1"}],
        "errors": {
            "false_findings": [],
            "missing_findings": [],
            "attribute_errors": [],
        },
    },
    {
        "reference_findings": [
            {
                "id": "R1",
                "finding": "small left apical pneumothorax",
                "clinical_significance": "actionable_not_urgent",
            }
        ],
        "predicted_findings": [
            {
                "id": "P1",
                "finding": "large right pleural effusion",
                "clinical_significance": "urgent",
            }
        ],
        "matched_findings": [],
        "errors": {
            "false_findings": ["P1"],
            "missing_findings": ["R1"],
            "attribute_errors": [],
        },
    },
]

# First sample: perfect match -> 1.0
# Second sample: weighted false finding dominates -> -0.5
expected_scores = [1.0, -0.5]
expected_mean = 0.25
expected_std = 0.75


def create_mock_openai_response(content):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestCrimsonUnit:
    """Unit tests with mocked backends (no API calls or model loading)."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mocked OpenAI client."""
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI") as mock_async_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client
            mock_async_client = MagicMock()
            mock_async_class.return_value = mock_async_client
            yield mock_client

    @pytest.fixture
    def mock_hf_pipeline(self):
        """Create a mocked HuggingFace pipeline."""
        with patch("transformers.pipeline") as mock_pipeline:
            mock_pipe = MagicMock()
            mock_pipeline.return_value = mock_pipe
            yield mock_pipe

    def test_import(self):
        """Test that CRIMSON can be imported."""
        from RadEval.metrics.crimson import CRIMSON, CRIMSONScore
        assert CRIMSON is not None
        assert CRIMSONScore is not None

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        from RadEval.metrics.crimson import CRIMSONScore
        with pytest.raises(NotImplementedError, match="does not support"):
            CRIMSONScore(provider="invalid")

    def test_openai_initialization_without_api_key(self):
        """Test that OpenAI initialization fails without API key."""
        from RadEval.metrics.crimson import CRIMSONScore

        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(EnvironmentError):
                CRIMSONScore(provider="openai")
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_openai_initialization_with_api_key(self, mock_openai_client):
        """Test that OpenAI initialization succeeds with API key."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="openai", openai_api_key="test-key")
        assert scorer is not None
        assert scorer.provider == "openai"
        assert scorer.model_name == scorer.DEFAULT_OPENAI_MODEL

    def test_hf_initialization_default_model(self, mock_hf_pipeline):
        """Test that HF initialization uses correct default model."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="hf")
        assert scorer.model_name == scorer.DEFAULT_HF_MODEL
        assert scorer.provider == "hf"

    def test_hf_single_evaluate(self, mock_hf_pipeline):
        """Test single evaluate with mocked HF backend."""
        from RadEval.metrics.crimson import CRIMSONScore

        mock_hf_pipeline.return_value = [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": json.dumps(mock_evaluations[0]),
                    }
                ]
            }
        ]

        scorer = CRIMSONScore(provider="hf", model_name="mock-crimson-model")
        result = scorer._evaluate_one(refs[0], hyps[0])

        assert result["crimson_score"] == 1.0
        assert result["error_counts"]["attribute_errors"] == 0

    def test_openai_single_evaluate(self, mock_openai_client):
        """Test single evaluate with mocked OpenAI backend."""
        from RadEval.metrics.crimson import CRIMSONScore

        mock_openai_client.chat.completions.create.return_value = create_mock_openai_response(
            json.dumps(mock_evaluations[0])
        )

        scorer = CRIMSONScore(provider="openai", openai_api_key="test-key")
        result = scorer._evaluate_one(refs[0], hyps[0])

        assert isinstance(result, dict)
        assert result["crimson_score"] == 1.0
        assert result["error_counts"]["false_findings"] == 0
        assert result["error_counts"]["missing_findings"] == 0

    def test_call_interface(self, mock_openai_client):
        """Test the __call__ interface returns correct format."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="openai", openai_api_key="test-key")
        scorer._chat_completion_async = AsyncMock(side_effect=[
            json.dumps(mock_evaluations[0]),
            json.dumps(mock_evaluations[1]),
        ])
        mean, std, scores, results_df = scorer(refs, hyps)

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert isinstance(scores, list)
        assert isinstance(results_df, pd.DataFrame)

        assert len(scores) == len(refs)
        assert len(results_df) == len(refs)

    def test_computed_scores(self, mock_openai_client):
        """Test that computed scores match expected values."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="openai", openai_api_key="test-key")
        scorer._chat_completion_async = AsyncMock(side_effect=[
            json.dumps(mock_evaluations[0]),
            json.dumps(mock_evaluations[1]),
        ])
        mean, std, scores, results_df = scorer(refs, hyps)

        for i, (actual, expected) in enumerate(zip(scores, expected_scores)):
            assert math.isclose(actual, expected, rel_tol=epsilon), (
                f"Score mismatch at index {i}: actual={actual}, expected={expected}"
            )

        assert math.isclose(mean, expected_mean, rel_tol=epsilon), \
            f"Mean mismatch: actual={mean}, expected={expected_mean}"
        assert math.isclose(std, expected_std, rel_tol=epsilon), \
            f"Std mismatch: actual={std}, expected={expected_std}"

    def test_dataframe_columns(self, mock_openai_client):
        """Test that results DataFrame has expected columns."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="openai", openai_api_key="test-key")
        scorer._chat_completion_async = AsyncMock(side_effect=[
            json.dumps(mock_evaluations[0]),
            json.dumps(mock_evaluations[1]),
        ])
        _, _, _, results_df = scorer(refs, hyps)

        expected_columns = [
            "reference",
            "prediction",
            "crimson_score",
            "false_findings",
            "missing_findings",
            "attribute_errors",
            "location_errors",
            "severity_errors",
            "descriptor_errors",
            "measurement_errors",
            "certainty_errors",
            "unspecific_errors",
            "overinterpretation_errors",
            "temporal_errors",
        ]
        for col in expected_columns:
            assert col in results_df.columns, f"Missing column: {col}"

    def test_mismatched_lengths(self, mock_openai_client):
        """Test that mismatched refs/hyps lengths raise error."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="openai", openai_api_key="test-key")
        with pytest.raises(ValueError):
            scorer(refs[:1], hyps)

    def test_unsupported_provider_raises(self):
        """Test that unsupported provider raises NotImplementedError."""
        from RadEval.metrics.crimson import CRIMSONScore
        with pytest.raises(NotImplementedError, match="does not support"):
            CRIMSONScore(provider="gemini")


class TestCrimsonRadEvalIntegration:
    """Test CRIMSON integration with RadEval main class."""

    def test_radeval_initialization(self):
        """Test that RadEval has CRIMSON constructor parameters."""
        from RadEval import RadEval

        sig = inspect.signature(RadEval.__init__)
        assert "do_crimson" in sig.parameters
        assert "crimson_api" in sig.parameters
        assert "openai_api_key" in sig.parameters
        assert "gemini_api_key" in sig.parameters
        assert "crimson_batch_size" in sig.parameters
        assert "crimson_max_concurrent" in sig.parameters

    def test_radeval_with_crimson_openai(self):
        """Test RadEval with CRIMSON using mocked OpenAI backend."""
        from RadEval import RadEval

        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                create_mock_openai_response(json.dumps(mock_evaluations[0])),
                create_mock_openai_response(json.dumps(mock_evaluations[1])),
            ]
            mock_class.return_value = mock_client

            evaluator = RadEval(
                do_crimson=True,
                crimson_api="openai",
                openai_api_key="test-key",
                show_progress=False,
            )
            evaluator.crimson_scorer._chat_completion_async = AsyncMock(side_effect=[
                json.dumps(mock_evaluations[0]),
                json.dumps(mock_evaluations[1]),
            ])
            results = evaluator(refs=refs, hyps=hyps)

        assert "crimson" in results
        assert math.isclose(results["crimson"], expected_mean, rel_tol=epsilon)

    @pytest.mark.integration
    def test_radeval_with_crimson_hf(self):
        """Test RadEval with CRIMSON using real HF backend (requires GPU)."""
        from RadEval import RadEval

        evaluator = RadEval(
            do_crimson=True,
            crimson_api="hf",
            show_progress=False,
        )
        results = evaluator(refs=refs[:1], hyps=hyps[:1])
        assert "crimson" in results

    def test_radeval_with_crimson_details(self):
        """Test RadEval with CRIMSON in details mode."""
        from RadEval import RadEval

        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                create_mock_openai_response(json.dumps(mock_evaluations[0])),
                create_mock_openai_response(json.dumps(mock_evaluations[1])),
            ]
            mock_class.return_value = mock_client

            evaluator = RadEval(
                do_crimson=True,
                crimson_api="openai",
                openai_api_key="test-key",
                do_details=True,
                show_progress=False,
            )
            evaluator.crimson_scorer._chat_completion_async = AsyncMock(side_effect=[
                json.dumps(mock_evaluations[0]),
                json.dumps(mock_evaluations[1]),
            ])
            results = evaluator(refs=refs, hyps=hyps)

        assert "crimson" in results
        assert isinstance(results["crimson"], float)
        assert "crimson_std" in results
        assert isinstance(results["crimson_std"], float)
        assert math.isclose(results["crimson"], expected_mean, rel_tol=epsilon)

    def test_radeval_with_crimson_per_sample(self):
        """Test RadEval with CRIMSON in per-sample mode."""
        from RadEval import RadEval

        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                create_mock_openai_response(json.dumps(mock_evaluations[0])),
                create_mock_openai_response(json.dumps(mock_evaluations[1])),
            ]
            mock_class.return_value = mock_client

            evaluator = RadEval(
                do_crimson=True,
                crimson_api="openai",
                openai_api_key="test-key",
                do_per_sample=True,
                show_progress=False,
            )
            evaluator.crimson_scorer._chat_completion_async = AsyncMock(side_effect=[
                json.dumps(mock_evaluations[0]),
                json.dumps(mock_evaluations[1]),
            ])
            results = evaluator(refs=refs, hyps=hyps)

        assert "crimson" in results
        assert isinstance(results["crimson"], list)
        assert len(results["crimson"]) == len(refs)
        for i, (actual, exp) in enumerate(zip(results["crimson"], expected_scores)):
            assert math.isclose(actual, exp, rel_tol=epsilon)


@pytest.mark.integration
class TestCrimsonIntegration:
    """Integration tests that require a real OpenAI API key.

    Run with: OPENAI_API_KEY=sk-... pytest tests/test_crimson.py -s -m integration
    """

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            pytest.skip("OPENAI_API_KEY not set")
        return key

    def test_real_api_call(self, api_key):
        """Test with real OpenAI API call."""
        from RadEval.metrics.crimson import CRIMSONScore

        scorer = CRIMSONScore(provider="openai", openai_api_key=api_key)
        mean, std, scores, results_df = scorer(refs, hyps)

        assert len(scores) == len(refs)
        assert len(results_df) == len(refs)

        for i in range(len(refs)):
            assert results_df["false_findings"].iloc[i] >= 0
            assert results_df["missing_findings"].iloc[i] >= 0
            assert results_df["attribute_errors"].iloc[i] >= 0

        print(f"\nReal API test results:")
        print(f"  Mean score: {mean:.4f}")
        print(f"  Std: {std:.4f}")
        for i in range(len(refs)):
            print(f"  Sample {i+1}: score={scores[i]:.4f}")

    def test_radeval_with_crimson(self, api_key):
        """Test CRIMSON through RadEval interface with real API."""
        from RadEval import RadEval

        evaluator = RadEval(
            do_crimson=True,
            crimson_api="openai",
            openai_api_key=api_key,
        )

        results = evaluator(refs=refs, hyps=hyps)

        assert "crimson" in results
        print(f"\nRadEval integration test result:")
        print(f"  crimson score: {results['crimson']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

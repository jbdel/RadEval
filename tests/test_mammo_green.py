import os
import math
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# run with:
# pytest tests/test_mammo_green.py -s
# For integration tests with real API:
# OPENAI_API_KEY=<key> pytest tests/test_mammo_green.py -s -m integration
# GOOGLE_API_KEY=<key> pytest tests/test_mammo_green.py -s -m integration

epsilon = 1e-5

# Test data - mammography reports
refs = [
    "Bilateral mammography demonstrates scattered fibroglandular density. There is a benign calcification at the upper outer left breast. BI-RADS 2.",
    "Bilateral digital mammography demonstrates scattered areas of fibroglandular density. There is a spiculated mass in the upper outer quadrant of the right breast measuring approximately 1.5 cm. Associated pleomorphic calcifications are present. No suspicious findings are seen in the left breast. BI-RADS 5.",
    "Bilateral digital mammography shows heterogeneously dense breasts. No suspicious mass, calcifications, or architectural distortion are identified in either breast. BI-RADS 1.",
]

hyps = [
    "Bilateral mammography shows scattered fibroglandular tissue. No suspicious findings are identified. BI-RADS 1.",
    "The breasts demonstrate scattered fibroglandular densities. There is an irregular mass in the upper outer quadrant of the right breast measuring approximately 1.4 cm with associated calcifications. The left breast is unremarkable. BI-RADS 4.",
    "Bilateral mammography demonstrates heterogeneously dense breast tissue. No suspicious masses or calcifications are identified. BI-RADS 1.",
]

# Mock response data for unit tests (includes incorrect_breast_density)
mock_judge_responses = [
    {
        "matched_findings": 1,
        "significant_errors": {
            "false_finding": 0,
            "missing_finding": 1,
            "mischaracterization": 0,
            "wrong_location_laterality": 0,
            "incorrect_birads": 1,
            "incorrect_breast_density": 0,
        },
        "insignificant_errors": 0,
    },
    {
        "matched_findings": 3,
        "significant_errors": {
            "false_finding": 0,
            "missing_finding": 0,
            "mischaracterization": 1,
            "wrong_location_laterality": 0,
            "incorrect_birads": 1,
            "incorrect_breast_density": 0,
        },
        "insignificant_errors": 0,
    },
    {
        "matched_findings": 2,
        "significant_errors": {
            "false_finding": 0,
            "missing_finding": 0,
            "mischaracterization": 0,
            "wrong_location_laterality": 0,
            "incorrect_birads": 0,
            "incorrect_breast_density": 0,
        },
        "insignificant_errors": 0,
    },
]

# Expected scores based on mock responses
# Score = matched / (matched + sum(significant_errors))
# [0]: 1 / (1 + 2) = 0.333...
# [1]: 3 / (3 + 2) = 0.6
# [2]: 2 / (2 + 0) = 1.0
expected_scores = [1/3, 0.6, 1.0]
expected_mean = sum(expected_scores) / len(expected_scores)


def create_mock_openai_response(content: str):
    """Create a mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestMammoGreenUnit:
    """Unit tests with mocked OpenAI API."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mocked OpenAI client."""
        with patch('RadEval.factual.green_score.mammo_green.OpenAI') as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client
            yield mock_client

    def test_import(self):
        """Test that MammoGREEN can be imported."""
        from RadEval.factual.green_score import MammoGREEN
        assert MammoGREEN is not None

    def test_import_from_factual(self):
        """Test that MammoGREEN can be imported from factual."""
        from RadEval.factual import MammoGREEN
        assert MammoGREEN is not None

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        from RadEval.factual.green_score import MammoGREEN

        # Remove env var if present
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(EnvironmentError):
                MammoGREEN(model_name="gpt-4o")
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_gemini_initialization_without_api_key(self):
        """Test that Gemini initialization fails without API key (or ImportError if not installed)."""
        from RadEval.factual.green_score import MammoGREEN
        from RadEval.factual.green_score.mammo_green import GEMINI_AVAILABLE

        # Remove env var if present
        old_gemini_key = os.environ.pop("GEMINI_API_KEY", None)
        old_google_key = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            if GEMINI_AVAILABLE:
                with pytest.raises(EnvironmentError):
                    MammoGREEN(model_name="gemini-2.5-flash")
            else:
                # If google-genai is not installed, ImportError is raised
                with pytest.raises(ImportError):
                    MammoGREEN(model_name="gemini-2.5-flash")
        finally:
            if old_gemini_key:
                os.environ["GEMINI_API_KEY"] = old_gemini_key
            if old_google_key:
                os.environ["GOOGLE_API_KEY"] = old_google_key

    def test_provider_detection(self):
        """Test that provider is correctly detected based on model name."""
        from RadEval.factual.green_score.mammo_green import _detect_provider

        # OpenAI models
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("gpt-4o-mini") == "openai"
        assert _detect_provider("gpt-5.2-2025-12-11") == "openai"
        assert _detect_provider("gpt-5-mini-2025-08-07") == "openai"
        assert _detect_provider("o1-preview") == "openai"

        # Gemini models (2.5+ only)
        assert _detect_provider("gemini-2.5-flash") == "gemini"
        assert _detect_provider("gemini-2.5-pro") == "gemini"
        assert _detect_provider("Gemini-2.5-Flash") == "gemini"  # Case insensitive

    def test_explicit_provider_parameter(self, mock_openai_client):
        """Test that explicit provider parameter overrides auto-detection."""
        from RadEval.factual.green_score import MammoGREEN

        # Explicit OpenAI provider
        scorer = MammoGREEN(model_name="custom-model", provider="openai", api_key="test-key")
        assert scorer.provider == "openai"

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        from RadEval.factual.green_score import MammoGREEN

        with pytest.raises(ValueError, match="Unsupported provider"):
            MammoGREEN(model_name="gpt-4o", provider="invalid", api_key="test-key")

    def test_initialization_with_api_key(self, mock_openai_client):
        """Test that initialization succeeds with API key."""
        from RadEval.factual.green_score import MammoGREEN

        scorer = MammoGREEN(model_name="gpt-4o", api_key="test-key")
        assert scorer is not None
        assert scorer.model_name == "gpt-4o"

    def test_score_computation(self):
        """Test the mammo_green_score function."""
        from RadEval.factual.green_score.mammo_green import mammo_green_score

        # Test case 1: 1 match, 2 errors -> 1/3
        matched = 1
        errors = {
            "false_finding": 0,
            "missing_finding": 1,
            "mischaracterization": 0,
            "wrong_location_laterality": 0,
            "incorrect_birads": 1,
            "incorrect_breast_density": 0,
        }
        score = mammo_green_score(matched, errors)
        assert math.isclose(score, 1/3, rel_tol=epsilon)

        # Test case 2: 0 matches, 0 errors -> 0.0
        score = mammo_green_score(0, {k: 0 for k in errors})
        assert score == 0.0

        # Test case 3: 5 matches, 0 errors -> 1.0
        score = mammo_green_score(5, {k: 0 for k in errors})
        assert score == 1.0

    def test_json_extraction(self):
        """Test the JSON extraction helper."""
        from RadEval.factual.green_score.mammo_green import _extract_json_str

        # Clean JSON
        clean = '{"matched_findings": 1}'
        assert _extract_json_str(clean) == clean

        # JSON with surrounding text
        wrapped = 'Here is the result: {"matched_findings": 1} end'
        assert _extract_json_str(wrapped) == '{"matched_findings": 1}'

        # JSON with markdown code block (common LLM output)
        markdown = '```json\n{"matched_findings": 1}\n```'
        assert _extract_json_str(markdown) == '{"matched_findings": 1}'

        # JSON with trailing comma (common LLM error)
        trailing_comma = '{"matched_findings": 1,}'
        result = _extract_json_str(trailing_comma)
        assert result == '{"matched_findings": 1}'

    def test_output_schema_validation(self):
        """Test the MammoGreenOutput schema validation."""
        from RadEval.factual.green_score.mammo_green import MammoGreenOutput

        # Valid output with all 6 error types
        valid_data = {
            "matched_findings": 2,
            "significant_errors": {
                "false_finding": 0,
                "missing_finding": 1,
                "mischaracterization": 0,
                "wrong_location_laterality": 0,
                "incorrect_birads": 1,
                "incorrect_breast_density": 0,
            },
            "insignificant_errors": 0,
        }
        output = MammoGreenOutput(**valid_data)
        output.validate_keys()  # Should not raise

        # Missing key in significant_errors
        invalid_data = {
            "matched_findings": 2,
            "significant_errors": {
                "false_finding": 0,
                "missing_finding": 1,
                # missing other keys
            },
            "insignificant_errors": 0,
        }
        output = MammoGreenOutput(**invalid_data)
        with pytest.raises(ValueError):
            output.validate_keys()

    def test_call_interface(self, mock_openai_client):
        """Test the __call__ interface returns correct format."""
        import json
        from RadEval.factual.green_score import MammoGREEN

        # Setup mock to return test responses
        mock_openai_client.chat.completions.create.side_effect = [
            create_mock_openai_response(json.dumps(mock_judge_responses[i]))
            for i in range(len(refs))
        ]

        scorer = MammoGREEN(model_name="gpt-4o", api_key="test-key")
        mean, std, scores, results_df = scorer(refs, hyps)

        # Check return types
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert isinstance(scores, list)
        assert isinstance(results_df, pd.DataFrame)

        # Check lengths
        assert len(scores) == len(refs)
        assert len(results_df) == len(refs)

        # Check DataFrame columns (includes incorrect_breast_density)
        expected_columns = [
            "reference", "prediction", "green_score", "matched_findings",
            "false_finding", "missing_finding", "mischaracterization",
            "wrong_location_laterality", "incorrect_birads", "incorrect_breast_density",
            "insignificant_errors"
        ]
        for col in expected_columns:
            assert col in results_df.columns, f"Missing column: {col}"

    def test_score_method_interface(self, mock_openai_client):
        """Test the .score() method returns dict format."""
        import json
        from RadEval.factual.green_score import MammoGREEN

        # Setup mock
        mock_openai_client.chat.completions.create.side_effect = [
            create_mock_openai_response(json.dumps(mock_judge_responses[i]))
            for i in range(len(refs))
        ]

        scorer = MammoGREEN(model_name="gpt-4o", api_key="test-key")
        result = scorer.score(refs, hyps)

        # Check return type and keys
        assert isinstance(result, dict)
        assert "green_scores" in result
        assert "error_counts" in result
        assert "summary" in result

        # Check summary keys
        summary = result["summary"]
        assert "n" in summary
        assert "mean_green" in summary
        assert "total_significant_errors_by_type" in summary

        # Verify incorrect_breast_density is tracked in summary
        assert "incorrect_breast_density" in summary["total_significant_errors_by_type"]

    def test_computed_scores(self, mock_openai_client):
        """Test that computed scores match expected values."""
        import json
        from RadEval.factual.green_score import MammoGREEN

        # Setup mock
        mock_openai_client.chat.completions.create.side_effect = [
            create_mock_openai_response(json.dumps(mock_judge_responses[i]))
            for i in range(len(refs))
        ]

        scorer = MammoGREEN(model_name="gpt-4o", api_key="test-key")
        mean, std, scores, results_df = scorer(refs, hyps)

        # Check individual scores
        for i, (actual, expected) in enumerate(zip(scores, expected_scores)):
            assert math.isclose(actual, expected, rel_tol=epsilon), \
                f"Score mismatch at index {i}: actual={actual}, expected={expected}"

        # Check mean
        assert math.isclose(mean, expected_mean, rel_tol=epsilon), \
            f"Mean mismatch: actual={mean}, expected={expected_mean}"

    def test_mismatched_lengths(self, mock_openai_client):
        """Test that mismatched refs/hyps lengths raise error."""
        from RadEval.factual.green_score import MammoGREEN

        scorer = MammoGREEN(model_name="gpt-4o", api_key="test-key")

        with pytest.raises(ValueError):
            scorer(refs[:2], hyps)  # Different lengths

    def test_significant_error_keys_constant(self):
        """Test that SIGNIFICANT_ERROR_KEYS contains all 6 error types."""
        from RadEval.factual.green_score.mammo_green import SIGNIFICANT_ERROR_KEYS

        expected_keys = {
            "false_finding",
            "missing_finding",
            "mischaracterization",
            "wrong_location_laterality",
            "incorrect_birads",
            "incorrect_breast_density",
        }
        assert set(SIGNIFICANT_ERROR_KEYS) == expected_keys


class TestMammoGreenRadEvalIntegration:
    """Test MammoGREEN integration with RadEval main class."""

    def test_radeval_initialization(self):
        """Test that RadEval can be initialized with do_mammo_green."""
        # This test doesn't actually call the API, just checks initialization params
        from RadEval import RadEval

        # Check that the parameter exists (but don't initialize scorer without API key)
        import inspect
        sig = inspect.signature(RadEval.__init__)
        assert "do_mammo_green" in sig.parameters
        assert "mammo_green_model" in sig.parameters
        assert "mammo_green_api_key" in sig.parameters


@pytest.mark.integration
class TestMammoGreenIntegration:
    """Integration tests that require a real OpenAI API key.

    Run with: OPENAI_API_KEY=sk-... pytest tests/test_mammo_green.py -s -m integration
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
        from RadEval.factual.green_score import MammoGREEN

        # Use all samples
        test_refs = refs
        test_hyps = hyps

        scorer = MammoGREEN(
            model_name="gpt-5.2-2025-12-11", #"gpt-4o",
            api_key=api_key,
            temperature=0.0,
        )

        mean, std, scores, results_df = scorer(test_refs, test_hyps)

        # Basic sanity checks
        assert 0.0 <= mean <= 1.0
        assert len(scores) == len(refs)
        for score in scores:
            assert 0.0 <= score <= 1.0
        assert len(results_df) == len(refs)

        # Check that error counts are non-negative for all samples
        for i in range(len(refs)):
            assert results_df["matched_findings"].iloc[i] >= 0
            assert results_df["false_finding"].iloc[i] >= 0
            assert results_df["missing_finding"].iloc[i] >= 0
            assert results_df["incorrect_breast_density"].iloc[i] >= 0

        print(f"\nReal API test results:")
        print(f"  Mean score: {mean:.4f}")
        print(f"  Std: {std:.4f}")
        print(f"\n  Individual sample scores:")
        for i in range(len(refs)):
            print(f"    Sample {i+1}:")
            print(f"      Score: {scores[i]:.4f}")
            print(f"      Matched findings: {results_df['matched_findings'].iloc[i]}")
            print(f"      Missing findings: {results_df['missing_finding'].iloc[i]}")
            print(f"      Incorrect BI-RADS: {results_df['incorrect_birads'].iloc[i]}")
            print(f"      Incorrect breast density: {results_df['incorrect_breast_density'].iloc[i]}")

    def test_radeval_with_mammo_green(self, api_key):
        """Test MammoGREEN through RadEval interface."""
        from RadEval import RadEval

        # Use all pairs
        test_refs = refs
        test_hyps = hyps

        evaluator = RadEval(
            do_mammo_green=True,
            mammo_green_api_key=api_key,
            mammo_green_model="gpt-4o",
        )

        results = evaluator(refs=test_refs, hyps=test_hyps)

        assert "mammo_green" in results
        assert 0.0 <= results["mammo_green"] <= 1.0

        print(f"\nRadEval integration test result:")
        print(f"  mammo_green score: {results['mammo_green']}")
        if "mammo_green_scores" in results:
            print(f"  Individual scores: {results['mammo_green_scores']}")

    def test_real_gemini_api_call(self):
        """Test with real Gemini API call."""
        from RadEval.factual.green_score import MammoGREEN

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not set")

        # Use all samples
        test_refs = refs
        test_hyps = hyps

        scorer = MammoGREEN(
            model_name="gemini-2.5-flash",
            provider="gemini",
            api_key=key,
            temperature=0.0,
        )

        mean, std, scores, results_df = scorer(test_refs, test_hyps)

        # Basic sanity checks
        assert 0.0 <= mean <= 1.0
        assert len(scores) == len(refs)
        for score in scores:
            assert 0.0 <= score <= 1.0
        assert len(results_df) == len(refs)

        # Check that error counts are non-negative for all samples
        for i in range(len(refs)):
            assert results_df["matched_findings"].iloc[i] >= 0
            assert results_df["false_finding"].iloc[i] >= 0
            assert results_df["missing_finding"].iloc[i] >= 0
            assert results_df["incorrect_breast_density"].iloc[i] >= 0

        print(f"\nGemini API test results:")
        print(f"  Mean score: {mean:.4f}")
        print(f"  Std: {std:.4f}")
        print(f"\n  Individual sample scores:")
        for i in range(len(refs)):
            print(f"    Sample {i+1}:")
            print(f"      Score: {scores[i]:.4f}")
            print(f"      Matched findings: {results_df['matched_findings'].iloc[i]}")
            print(f"      Missing findings: {results_df['missing_finding'].iloc[i]}")
            print(f"      Incorrect BI-RADS: {results_df['incorrect_birads'].iloc[i]}")
            print(f"      Incorrect breast density: {results_df['incorrect_breast_density'].iloc[i]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

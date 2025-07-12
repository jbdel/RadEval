from RadEval.factual.green_score import GREEN
import numpy as np
import warnings

model_name = "StanfordAIMI/GREEN-radllama2-7b"

# Initialize GREEN scorer
green_scorer = GREEN(model_name, output_dir=".")

def test_green_scorer():
    """
    Test the GREEN scorer with a specific example to ensure consistent results.
    """
    # Test data
    refs = [
        "1.Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.Relatively lower lung volumes with no focal airspace consolidation appreciated.Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.No pleural effusions or pneumothoraces.",
    ]
    hyps = [
        "Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.",
    ]

    # Get results
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

    # Expected output based on the test run
    expected_mean = 0.4
    expected_std = 0.0
    expected_green_score_list = [0.4]
    expected_summary = "\n-------------GREEN-radllama2-7b----------------\n [Summary]: Green average 0.4 and standard deviation 0.0 \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n(a) False report of a finding in the candidate: 1.0. \n None \n\n(b) Missing a finding present in the reference: 0.0. \n  Lower lung volumes \n\n(c) Misidentification of a finding's anatomic location/position: 1.0. \n None \n\n(d) Misassessment of the severity of a finding: 1.0. \n None \n\n(e) Mentioning a comparison that isn't in the reference: 1.0. \n None \n\n(f) Omitting a comparison detailing a change from a prior study: 1.0. \n None \n\n----------------------------------\n"
    expected_df_shape = (1, 11)

    # Assertions
    assert abs(mean - expected_mean) < 1e-6, f"Expected mean {expected_mean}, got {mean}"
    assert abs(std - expected_std) < 1e-6, f"Expected std {expected_std}, got {std}"
    assert green_score_list == expected_green_score_list, f"Expected green_score_list {expected_green_score_list}, got {green_score_list}"
    assert summary == expected_summary, f"Summary mismatch. Expected:\n{expected_summary}\nGot:\n{summary}"
    
    # Additional checks
    assert result_df is not None and not result_df.empty, "Result dataframe should not be empty"
    assert result_df.shape == expected_df_shape, f"Expected shape {expected_df_shape}, got {result_df.shape}"

    # Check that result_df contains expected columns
    expected_columns = ['reference', 'predictions', 'green_analysis', 'green_score']
    for col in expected_columns:
        assert col in result_df.columns, f"Missing expected column: {col}"
    
    # Check specific values in the dataframe
    assert result_df.loc[0, 'reference'] == refs[0], "Reference text mismatch in dataframe"
    assert result_df.loc[0, 'predictions'] == hyps[0], "Prediction text mismatch in dataframe"
    assert result_df.loc[0, 'green_score'] == expected_green_score_list[0], "Green score mismatch in dataframe"

def test_green_scorer_basic():
    """
    Test the GREEN scorer with basic checks (less strict than full summary comparison).
    """
    # Test data
    refs = [
        "1.Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.Relatively lower lung volumes with no focal airspace consolidation appreciated.Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.No pleural effusions or pneumothoraces.",
    ]
    hyps = [
        "Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.",
    ]

    # Get results using the global green_scorer
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

    # Basic expected values
    expected_mean = 0.4
    expected_std = 0.0
    expected_green_score_list = [0.4]

    # Basic assertions
    assert abs(mean - expected_mean) < 1e-6, f"Expected mean {expected_mean}, got {mean}"
    assert abs(std - expected_std) < 1e-6, f"Expected std {expected_std}, got {std}"
    assert green_score_list == expected_green_score_list, f"Expected green_score_list {expected_green_score_list}, got {green_score_list}"
    
    # Check summary contains key elements
    assert summary is not None and len(summary) > 0, "Summary should not be empty"
    assert "GREEN-radllama2-7b" in summary, "Summary should contain model name"
    assert "Green average 0.4" in summary, "Summary should contain correct average"
    assert "standard deviation 0.0" in summary, "Summary should contain correct std"
    
    # Check dataframe
    assert result_df is not None and not result_df.empty, "Result dataframe should not be empty"
    assert result_df.shape == (1, 11), f"Expected shape (1, 11), got {result_df.shape}"
    
    # Check specific values in the dataframe
    assert result_df.loc[0, 'green_score'] == expected_green_score_list[0], "Green score mismatch in dataframe"


def test_green_scorer_multiple_examples():
    """
    Test the GREEN scorer with multiple examples.
    """
    # Test data with multiple examples
    refs = [
        "No acute cardiopulmonary process.",
        "Normal heart size and pulmonary vasculature.",
    ]
    hyps = [
        "No acute cardiopulmonary process.",
        "Normal heart size.",
    ]

    # Get results
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

    # Basic assertions
    assert isinstance(mean, (int, float)), "Mean should be a number"
    assert isinstance(std, (int, float)), "Std should be a number"
    assert isinstance(green_score_list, list), "Green score list should be a list"
    assert len(green_score_list) == len(refs), "Green score list length should match input length"
    assert result_df.shape[0] == len(refs), "Result dataframe rows should match input length"

    # Check value ranges
    assert 0 <= mean <= 1, "Mean should be between 0 and 1"
    assert std >= 0, "Std should be non-negative"

    for score in green_score_list:
        if score is not None:  # GREEN score can be None in some cases
            assert 0 <= score <= 1, f"Green score should be between 0 and 1, got {score}"

import pytest
from RadEval import RadEval
import json


def test_radeval():
    # Sample references and hypotheses
    refs = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "1.Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.Relatively lower lung volumes with no focal airspace consolidation appreciated.Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.No pleural effusions or pneumothoraces.",
        "1. Left PICC tip appears to terminate in the distal left brachiocephalic vein.2. Mild pulmonary vascular congestion.3. Interval improvement in aeration of the lung bases with residual streaky opacity likely reflective of atelectasis.Interval resolution of the left pleural effusion.",
        "No definite acute cardiopulmonary process.Enlarged cardiac silhouette could be accentuated by patient's positioning.",
        "Increased mild pulmonary edema and left basal atelectasis.",
    ]

    hyps = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.",
        "Relatively lower lung volumes with no focal airspace consolidation appreciated.",
        "Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.",
        "No pleural effusions or pneumothoraces.",
    ]

    # Instantiate RadEval with desired configurations
    evaluator = RadEval(do_radgraph=True,
                        do_green=False,
                        do_bleu=True,
                        do_rouge=True,
                        do_bertscore=True,
                        do_diseases=False)

    # Compute scores
    results = evaluator(refs=refs, hyps=hyps)

    # Expected result with pytest.approx for approximate comparison
    expected_result = {
        "radgraph_simple": pytest.approx(0.41111111111111115, 0.01),
        "radgraph_partial": pytest.approx(0.41111111111111115, 0.01),
        "radgraph_complete": pytest.approx(0.41414141414141414, 0.01),
        "bleu": pytest.approx(0.16681006823938177, 0.01),
        "bertscore": pytest.approx(0.6327474117279053, 0.01),
        "rouge1": pytest.approx(0.44681719607092746, 0.01),
        "rouge2": pytest.approx(0.4205128205128205, 0.01),
        "rougeL": pytest.approx(0.44681719607092746, 0.01),
    }

    # Compare computed results with expected results
    for key, expected_value in expected_result.items():
        assert key in results, f"Missing key in results: {key}"
        assert results[key] == expected_value, f"Mismatch for {key}: {results[key]} != {expected_value}"

    # Print results for debug (optional)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    pytest.main()
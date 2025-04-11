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
    evaluator = RadEval(do_diseases=True)

    # Compute scores
    results = evaluator(refs=refs, hyps=hyps)

    # Assert expected results with approximate values
    assert results['samples_avg_precision'] == pytest.approx(0.5)
    assert results['samples_avg_recall'] == pytest.approx(0.3888888888888889)
    assert results['samples_avg_f1-score'] == pytest.approx(0.4166666666666667)


if __name__ == "__main__":
    test_radeval()

import os
import logging
import pytest
import numpy as np

from RadEval.metrics.hoppr_f1chexbert import HopprF1CheXbert

logger = logging.getLogger(__name__)

_HOPPR_CHEXPERT_CKPT = "/fss/pranta_das/CheXbert/expermints_folder/25507/checkpoint_2.pth"

if HopprF1CheXbert is None or not os.path.exists(_HOPPR_CHEXPERT_CKPT):
    pytest.skip(
        "HopprCheXbert not available (missing module or checkpoint)",
        allow_module_level=True,
    )


def test_f1chexbert():
    f1chexbert = HopprF1CheXbert()
    accuracy, accuracy_not_averaged, class_report, class_report_5,_,_ = f1chexbert(
        hyps=['No pleural effusion. Normal heart size.',
              'Normal heart size.',
              'Increased mild pulmonary edema and left basal atelectasis.',
              'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
              'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
              ],
        refs=['No pleural effusions.',
              'Enlarged heart.',
              'No evidence of pneumonia. Stable cardiomegaly.',
              'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
              'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.'
              ])

    expected_output = (0.4, np.array([1., 0., 0., 1., 0.], dtype=np.float32),
                       {'acute_rib_fracture': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'air_space_opacity': {'precision': 0.5,
                        'recall': 0.5,
                        'f1-score': 0.5,
                        'support': 2.0},
                        'cardiomegaly': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 2.0},
                        'lung_nodule_or_mass': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 1.0},
                        'non_acute_rib_fracture': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'pleural_fluid': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'pneumothorax': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'pulmonary_artery_enlargement': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'atelectasis': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'bronchial_wall_thickening': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'bullous_disease': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'hilar_lymphadenopathy': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'hiatus_hernia': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'hyperinflation': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'implantable_electronic_device': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'intercostal_drain': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'interstitial_thickening': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'lobar_segmental_collapse': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'nonsurgical_internal_foreign_body': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'pacemaker_electronic_cardiac_device_or_wires': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'peribronchial_cuffing': {'precision': 1.0,
                        'recall': 1.0,
                        'f1-score': 1.0,
                        'support': 1.0},
                        'pulmonary_congestion_pulmonary_venous_congestion': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'shoulder_dislocation': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'subcutaneous_emphysema': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'tracheal_deviation': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'whole_lung_or_majority_collapse': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'no_finding': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'micro avg': {'precision': 0.4,
                        'recall': 0.3333333333333333,
                        'f1-score': 0.36363636363636365,
                        'support': 6.0},
                        'macro avg': {'precision': 0.05555555555555555,
                        'recall': 0.05555555555555555,
                        'f1-score': 0.05555555555555555,
                        'support': 6.0},
                        'weighted avg': {'precision': 0.3333333333333333,
                        'recall': 0.3333333333333333,
                        'f1-score': 0.3333333333333333,
                        'support': 6.0},
                        'samples avg': {'precision': 0.2,
                        'recall': 0.2,
                        'f1-score': 0.2,
                        'support': 6.0}},
                       {'cardiomegaly': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 2.0},
                        'air_space_opacity': {'precision': 0.5,
                        'recall': 0.5,
                        'f1-score': 0.5,
                        'support': 2.0},
                        'atelectasis': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'pleural_fluid': {'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0.0},
                        'micro avg': {'precision': 0.3333333333333333,
                        'recall': 0.25,
                        'f1-score': 0.2857142857142857,
                        'support': 4.0},
                        'macro avg': {'precision': 0.125,
                        'recall': 0.125,
                        'f1-score': 0.125,
                        'support': 4.0},
                        'weighted avg': {'precision': 0.25,
                        'recall': 0.25,
                        'f1-score': 0.25,
                        'support': 4.0},
                        'samples avg': {'precision': 0.2,
                        'recall': 0.2,
                        'f1-score': 0.2,
                        'support': 4.0}})

    assert accuracy == expected_output[0]
    assert np.array_equal(accuracy_not_averaged, expected_output[1])
    assert class_report == expected_output[2]
    assert class_report_5 == expected_output[3]


def test_f1hopprchexbert_per_sample():
    """do_per_sample returns flat per-sample accuracy lists via RadEval."""
    from RadEval import RadEval
    from tests.conftest import CHEXBERT_HYPS, CHEXBERT_REFS

    evaluator = RadEval(
        do_f1hopprchexbert=True, do_per_sample=True, show_progress=False)
    if not evaluator.do_f1hopprchexbert:
        pytest.skip("HopprF1CheXbert was disabled during init")

    results = evaluator(refs=CHEXBERT_REFS, hyps=CHEXBERT_HYPS)

    assert "f1hopprchexbert_sample_acc_5" in results
    assert "f1hopprchexbert_sample_acc_all" in results

    for key in ("f1hopprchexbert_sample_acc_5", "f1hopprchexbert_sample_acc_all"):
        assert isinstance(results[key], list)
        assert len(results[key]) == len(CHEXBERT_REFS)
        for val in results[key]:
            assert isinstance(val, (int, float))
            assert 0.0 <= val <= 1.0

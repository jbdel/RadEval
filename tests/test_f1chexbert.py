from RadEval.factual.f1chexbert import F1CheXbert
import numpy as np


def test_f1chexbert():
    f1chexbert = F1CheXbert()
    accuracy, accuracy_not_averaged, class_report, class_report_5 = f1chexbert(
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

    expected_output = (0.6, np.array([1., 0., 0., 1., 1.], dtype=np.float32),
                       {'Enlarged Cardiomediastinum': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Cardiomegaly': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 2.0},
                        'Lung Opacity': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1.0},
                        'Lung Lesion': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1.0},
                        'Edema': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Consolidation': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1.0},
                        'Pneumonia': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Atelectasis': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Pneumothorax': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Pleural Effusion': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Pleural Other': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Fracture': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Support Devices': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'No Finding': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.6666666666666666, 'support': 1.0},
                        'micro avg': {'precision': 0.4, 'recall': 0.3333333333333333, 'f1-score': 0.36363636363636365,
                                      'support': 6.0},
                        'macro avg': {'precision': 0.10714285714285714, 'recall': 0.14285714285714285,
                                      'f1-score': 0.11904761904761904, 'support': 6.0},
                        'weighted avg': {'precision': 0.25, 'recall': 0.3333333333333333,
                                         'f1-score': 0.27777777777777773, 'support': 6.0},
                        'samples avg': {'precision': 0.4, 'recall': 0.4, 'f1-score': 0.4, 'support': 6.0}},
                       {'Cardiomegaly': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 2.0},
                        'Edema': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Consolidation': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1.0},
                        'Atelectasis': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'Pleural Effusion': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
                        'micro avg': {'precision': 0.3333333333333333, 'recall': 0.3333333333333333,
                                      'f1-score': 0.3333333333333333, 'support': 3.0},
                        'macro avg': {'precision': 0.2, 'recall': 0.2, 'f1-score': 0.2, 'support': 3.0},
                        'weighted avg': {'precision': 0.3333333333333333, 'recall': 0.3333333333333333,
                                         'f1-score': 0.3333333333333333, 'support': 3.0},
                        'samples avg': {'precision': 0.2, 'recall': 0.2, 'f1-score': 0.2, 'support': 3.0}})

    assert accuracy == expected_output[0]
    assert np.array_equal(accuracy_not_averaged, expected_output[1])
    assert class_report == expected_output[2]
    assert class_report_5 == expected_output[3]

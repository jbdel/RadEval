from radeval.metrics.f1chexbert import F1CheXbert
import numpy as np
import torch.nn as nn

from radeval.metrics._chexbert_base import BaseCheXbertEvaluator


def test_f1chexbert():
    f1chexbert = F1CheXbert()
    accuracy, accuracy_not_averaged, class_report, class_report_5, _, _ = f1chexbert(
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


def test_f1chexbert_probe_sentences_expected_present():
    f1chexbert = F1CheXbert(device="cpu")

    probes = [
        ("Mild cardiomegaly is present.", "Cardiomegaly"),
        ("Patchy right lower lobe consolidation.", "Consolidation"),
        ("Patchy right air space opacity.", "Lung Opacity"),
        ("Small left pleural effusion is seen.", "Pleural Effusion"),
        ("Left apical pneumothorax is present.", "Pneumothorax"),
        ("Bibasilar subsegmental atelectatic change.", "Atelectasis"),
        ("Pulmonary edema is present.", "Edema"),
        ("A right upper lobe lung lesion is identified.", "Lung Lesion"),
        ("Acute displaced left rib fracture is present.", "Fracture"),
        ("Endotracheal tube and enteric tube are in place.", "Support Devices"),
        ("No focal airspace opacity. Mild right lower lobe pneumonia.", "Pneumonia"),
    ]

    sentences = [sentence for sentence, _ in probes]
    expected_diseases = [disease for _, disease in probes]

    preds = f1chexbert.get_labels(sentences, mode="rrg")
    assert len(preds) == len(expected_diseases)

    for sentence, expected, row in zip(sentences, expected_diseases, preds):
        present = [
            name for name, value in zip(f1chexbert.TARGET_NAMES, row) if value == 1
        ]
        assert expected in present, (
            f"Expected '{expected}' to be present for sentence '{sentence}', "
            f"but received present labels: {present}"
        )


def test_f1chexbert_per_sample():
    """do_per_sample returns flat per-sample accuracy lists."""
    from radeval import RadEval
    from tests.conftest import CHEXBERT_HYPS, CHEXBERT_REFS

    evaluator = RadEval(
        metrics=["f1chexbert"], per_sample=True, show_progress=False)
    results = evaluator(refs=CHEXBERT_REFS, hyps=CHEXBERT_HYPS)

    assert "f1chexbert_sample_acc_5" in results
    assert "f1chexbert_sample_acc_all" in results

    for key in ("f1chexbert_sample_acc_5", "f1chexbert_sample_acc_all"):
        assert isinstance(results[key], list)
        assert len(results[key]) == len(CHEXBERT_REFS)
        for val in results[key]:
            assert isinstance(val, (int, float))
            assert 0.0 <= val <= 1.0

    assert "f1chexbert_5_micro_f1" not in results
    assert "f1chexbert" not in results


class _StubCheXbertEvaluator(BaseCheXbertEvaluator):
    """CheXbert evaluator with canned labels, so no weights are needed."""

    CONDITION_NAMES = [
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices',
    ]
    NO_FINDING = 'No Finding'
    TOP5 = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',
            'Pleural Effusion']

    def __init__(self, label_table):
        nn.Module.__init__(self)
        self._init_evaluator(model=None, tokenizer=None, device="cpu",
                             batch_size=4)
        self._label_table = label_table

    def get_labels(self, reports, mode="rrg", on_batch_done=None):
        return [self._label_table[r] for r in reports]


def test_chexbert_per_sample_accuracy_values():
    """Guard the per-sample accuracy arithmetic against exact numpy ground truth.

    Regression test for #29: the exact-match accuracy and the per-sample label
    accuracies used to be computed via sklearn's private ``_check_targets``,
    which silently changed its tuple layout in scikit-learn 1.8. That both
    crashed ``pe_accuracy`` and made ``sample_acc_all`` return wrong numbers of
    the right shape, so type/range/length assertions could not catch it.
    """
    rng = np.random.default_rng(0)
    refs = [f"ref{i}" for i in range(6)]
    hyps = [f"hyp{i}" for i in range(6)]

    table = {r: rng.integers(0, 2, size=14).tolist() for r in refs}
    table.update({h: rng.integers(0, 2, size=14).tolist() for h in hyps})
    # force one exact match so the all-labels-correct path is exercised
    table[hyps[3]] = list(table[refs[3]])

    ev = _StubCheXbertEvaluator(table)
    _, pe_accuracy, _, _, sample_acc_full, sample_acc_5 = ev.forward(hyps, refs)

    ref_mat = np.array([table[r] for r in refs])
    hyp_mat = np.array([table[h] for h in hyps])
    match_full = ref_mat == hyp_mat
    match_5 = ref_mat[:, ev.top5_idx] == hyp_mat[:, ev.top5_idx]

    np.testing.assert_allclose(pe_accuracy, match_5.all(axis=1).astype(float))
    np.testing.assert_allclose(sample_acc_5, match_5.mean(axis=1))
    np.testing.assert_allclose(sample_acc_full, match_full.mean(axis=1))

    # the forced-identical pair must score a perfect 1.0 everywhere
    assert pe_accuracy[3] == 1.0
    assert sample_acc_5[3] == 1.0
    assert sample_acc_full[3] == 1.0

    assert len(sample_acc_5) == len(refs)
    assert len(sample_acc_full) == len(refs)

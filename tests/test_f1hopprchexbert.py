import os
import logging
import pytest
import numpy as np

from radeval.metrics.f1hopprchexbert import HopprF1CheXbert

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

    # Structural assertions: the model is deterministic on these inputs. We
    # pin the aggregate TOP5 accuracy and the per-sample match vector.
    #
    # Expected values below were verified by hand against the ref/hyp pairs
    # (see the comments in each column of the table that follows). They are
    # identical under transformers 4.57.3 and 5.6.2 — this test does NOT
    # depend on the transformers version.
    #
    # The TOP5 labels are ["cardiomegaly", "air_space_opacity", "atelectasis",
    # "pleural_fluid"]. Per-sample outcomes:
    #   0: ref "No pleural effusions."  / hyp "No pleural effusion. Normal heart
    #      size."  — both → none on TOP5 ⇒ MATCH (1)
    #   1: ref "Enlarged heart."  (cardiomegaly) / hyp "Normal heart size."
    #      (none) — semantic opposite ⇒ DIFFER (0)
    #   2: ref "No evidence of pneumonia. Stable cardiomegaly." (cardiomegaly)
    #      / hyp "Increased mild pulmonary edema and left basal atelectasis."
    #      (air_space_opacity, atelectasis) ⇒ DIFFER (0)
    #   3: ref == hyp (bronchiectasis case) ⇒ MATCH (1)
    #   4: ref (nodules/air_space_opacity) / hyp (no TOP5 labels) ⇒ DIFFER (0)
    assert accuracy == 0.4
    assert np.array_equal(
        accuracy_not_averaged, np.array([1., 0., 0., 1., 0.], dtype=np.float32))

    # All 27 conditions + no_finding + 4 summary rows present in full report
    for cond in HopprF1CheXbert.CONDITION_NAMES:
        assert cond in class_report
    assert "no_finding" in class_report
    for agg in ("micro avg", "macro avg", "weighted avg", "samples avg"):
        assert agg in class_report
        assert agg in class_report_5
    for top5 in HopprF1CheXbert.TOP5:
        assert top5 in class_report_5


def test_f1hopprchexbert_per_sample():
    """per_sample returns flat per-sample accuracy lists via RadEval."""
    from radeval import RadEval
    from tests.conftest import CHEXBERT_HYPS, CHEXBERT_REFS

    evaluator = RadEval(
        metrics=["f1hopprchexbert"], per_sample=True, show_progress=False)

    results = evaluator(refs=CHEXBERT_REFS, hyps=CHEXBERT_HYPS)

    assert "f1hopprchexbert_sample_acc_5" in results
    assert "f1hopprchexbert_sample_acc_all" in results

    for key in ("f1hopprchexbert_sample_acc_5", "f1hopprchexbert_sample_acc_all"):
        assert isinstance(results[key], list)
        assert len(results[key]) == len(CHEXBERT_REFS)
        for val in results[key]:
            assert isinstance(val, (int, float))
            assert 0.0 <= val <= 1.0

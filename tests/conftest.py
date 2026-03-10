"""Shared test fixtures and helpers."""
import pytest


def read_reports(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


CHEXBERT_HYPS = [
    'No pleural effusion. Normal heart size.',
    'Normal heart size.',
    'Increased mild pulmonary edema and left basal atelectasis.',
    'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
    'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
]

CHEXBERT_REFS = [
    'No pleural effusions.',
    'Enlarged heart.',
    'No evidence of pneumonia. Stable cardiomegaly.',
    'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
    'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.',
]


@pytest.fixture(scope="module")
def chexpert_reports():
    refs = read_reports("tests/utterances/preds_chexpert_impression.txt")
    hyps = read_reports("tests/utterances/refs_chexpert_impression.txt")
    return refs, hyps

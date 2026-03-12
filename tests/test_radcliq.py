"""RadCliQ-v1 tests validated against rajpurkarlab/CXR-Report-Metric reference.

The reference composite model (radcliq_v1.pkl) uses:
  - BERTScore: distilroberta-base, rescale_with_baseline=True, IDF from refs
  - BLEU: BLEU-2 (bigram)
  - RadGraph: (entity_f1 + relation_f1) / 2
  - Semantic embeddings: CheXbert [CLS] cosine similarity
  - Linear combination: scaler.transform(X) @ coefs  (with bias column)

Expected values computed by running both implementations side-by-side
and verifying np.allclose (max diff < 1.3e-8).
"""
import pytest
from RadEval import RadEval


REFS = [
    "No acute cardiopulmonary process.",
    "Small bilateral pleural effusions with bibasilar atelectasis.",
    "Mild cardiomegaly. No pleural effusion.",
]
HYPS = [
    "No acute findings.",
    "Bilateral pleural effusions and basilar atelectasis.",
    "Mild cardiomegaly with clear lungs.",
]

EXPECTED_RADCLIQ_PER_PAIR = [0.8346, 0.0416, 0.3987]


@pytest.fixture(scope="module")
def evaluator():
    return RadEval(do_radcliq=True, show_progress=False)


@pytest.fixture(scope="module")
def evaluator_details():
    return RadEval(do_radcliq=True, do_details=True, show_progress=False)


def test_radcliq_per_pair_scores(evaluator_details):
    """Per-pair RadCliQ-v1 scores match the reference implementation."""
    results = evaluator_details(refs=REFS, hyps=HYPS)
    sample_scores = results["radcliq_v1"]["sample_scores"]
    for i, (actual, expected) in enumerate(zip(sample_scores, EXPECTED_RADCLIQ_PER_PAIR)):
        assert actual == pytest.approx(expected, abs=0.01), (
            f"Pair {i}: {actual:.4f} != {expected:.4f}")


def test_radcliq_mean(evaluator):
    """Mean RadCliQ-v1 score matches 1/mean(per-pair)."""
    results = evaluator(refs=REFS, hyps=HYPS)
    expected_mean = 1.0 / (sum(EXPECTED_RADCLIQ_PER_PAIR) / len(EXPECTED_RADCLIQ_PER_PAIR))
    assert results["radcliq_v1"] == pytest.approx(expected_mean, abs=0.05)


def test_radcliq_details_structure(evaluator_details):
    results = evaluator_details(refs=REFS, hyps=HYPS)
    detail = results["radcliq_v1"]
    assert "mean_score" in detail
    assert "sample_scores" in detail
    assert len(detail["sample_scores"]) == len(REFS)


def test_radcliq_default_mode(evaluator):
    """Default (non-details) mode returns a single float."""
    results = evaluator(refs=REFS, hyps=HYPS)
    assert isinstance(results["radcliq_v1"], float)


# ── Per-sample mode tests ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def evaluator_per_sample():
    return RadEval(do_radcliq=True, do_per_sample=True, show_progress=False)


def test_radcliq_per_sample_structure(evaluator_per_sample):
    """do_per_sample returns a flat list under the same key."""
    results = evaluator_per_sample(refs=REFS, hyps=HYPS)
    assert isinstance(results["radcliq_v1"], list)
    assert len(results["radcliq_v1"]) == len(REFS)


def test_radcliq_per_sample_values(evaluator_per_sample):
    """Per-sample scores match the reference implementation."""
    results = evaluator_per_sample(refs=REFS, hyps=HYPS)
    for i, (actual, expected) in enumerate(
            zip(results["radcliq_v1"], EXPECTED_RADCLIQ_PER_PAIR)):
        assert actual == pytest.approx(expected, abs=0.01), (
            f"Pair {i}: {actual:.4f} != {expected:.4f}")

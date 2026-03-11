"""Tests for RadGraph-RadCliQ metric.

Validates that the standalone RadGraph-RadCliQ metric produces identical
scores to the RadGraph sub-metric computed inside RadCliQ-v1.
"""
import pytest
import numpy as np


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

EXPECTED_PER_PAIR = [0.2, 0.2727, 0.25]


@pytest.fixture(scope="module")
def standalone_scorer():
    from RadEval.metrics.radgraph_radcliq import RadGraphRadCliQ
    return RadGraphRadCliQ()


@pytest.fixture(scope="module")
def radcliq_scorer():
    from RadEval.metrics.RadCliQv1.radcliq import CompositeMetric
    return CompositeMetric()


def test_scores_match_radcliq_internal(standalone_scorer, radcliq_scorer):
    """Standalone scores must exactly match RadCliQ's internal RadGraph."""
    _, standalone_scores = standalone_scorer(HYPS, REFS)
    sub = radcliq_scorer._compute_sub_metrics(REFS, HYPS)
    radcliq_rg = sub["radgraph"]
    assert np.allclose(standalone_scores, radcliq_rg), (
        f"Standalone: {standalone_scores}\nRadCliQ internal: {radcliq_rg}")


def test_expected_per_pair_scores(standalone_scorer):
    """Per-pair scores match pre-computed expected values."""
    _, scores = standalone_scorer(HYPS, REFS)
    for i, (actual, expected) in enumerate(zip(scores, EXPECTED_PER_PAIR)):
        assert actual == pytest.approx(expected, abs=0.001), (
            f"Pair {i}: {actual:.4f} != {expected:.4f}")


def test_mean_score(standalone_scorer):
    mean, scores = standalone_scorer(HYPS, REFS)
    assert mean == pytest.approx(sum(scores) / len(scores), abs=1e-10)


def test_identical_reports_perfect(standalone_scorer):
    same = ["No acute cardiopulmonary process."]
    mean, scores = standalone_scorer(same, same)
    assert scores[0] == 1.0


def test_radeval_integration():
    from RadEval import RadEval
    ev = RadEval(do_radgraph_radcliq=True, show_progress=False)
    results = ev(refs=REFS, hyps=HYPS)
    assert "radgraph_radcliq" in results
    assert isinstance(results["radgraph_radcliq"], float)


def test_radeval_details():
    from RadEval import RadEval
    ev = RadEval(do_radgraph_radcliq=True, do_details=True, show_progress=False)
    results = ev(refs=REFS, hyps=HYPS)
    detail = results["radgraph_radcliq"]
    assert "mean_score" in detail
    assert "sample_scores" in detail
    assert len(detail["sample_scores"]) == len(REFS)


def test_validation_errors(standalone_scorer):
    with pytest.raises(TypeError):
        standalone_scorer("not a list", REFS)
    with pytest.raises(ValueError):
        standalone_scorer(HYPS[:1], REFS)

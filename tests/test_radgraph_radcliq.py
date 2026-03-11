"""Tests for RadGraph-RadCliQ metric, verified against the RadCliQ-v1 internal computation.

The per-pair radgraph scores from RadGraph-RadCliQ must exactly match
the radgraph sub-scores computed inside RadCliQ-v1's _compute_sub_metrics,
since they use the same model, entity/relation extraction, and F1 logic.
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


@pytest.fixture(scope="module")
def radcliq_radgraph_scores():
    """Get the radgraph sub-scores from RadCliQ-v1 as the ground truth."""
    from RadEval.metrics.RadCliQv1.radcliq import CompositeMetric
    composite = CompositeMetric()
    sub = composite._compute_sub_metrics(REFS, HYPS)
    return sub["radgraph"].tolist()


@pytest.fixture(scope="module")
def standalone_scorer():
    from RadEval.metrics.radgraph_radcliq import RadGraphRadCliQ
    return RadGraphRadCliQ()


def test_scores_match_radcliq_internal(standalone_scorer, radcliq_radgraph_scores):
    """Per-pair scores must be identical to RadCliQ's internal radgraph sub-scores."""
    mean, sample_scores = standalone_scorer(HYPS, REFS)
    assert np.allclose(sample_scores, radcliq_radgraph_scores, atol=1e-10), (
        f"Mismatch:\n  standalone: {sample_scores}\n  radcliq:    {radcliq_radgraph_scores}")


def test_mean_is_average_of_samples(standalone_scorer):
    mean, sample_scores = standalone_scorer(HYPS, REFS)
    assert mean == pytest.approx(sum(sample_scores) / len(sample_scores), abs=1e-10)


def test_identical_reports_perfect_score(standalone_scorer):
    same = ["No acute cardiopulmonary process."]
    mean, scores = standalone_scorer(same, same)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_output_structure(standalone_scorer):
    mean, scores = standalone_scorer(HYPS, REFS)
    assert isinstance(mean, float)
    assert isinstance(scores, list)
    assert len(scores) == len(REFS)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_validation_errors(standalone_scorer):
    with pytest.raises(TypeError):
        standalone_scorer("not a list", REFS)
    with pytest.raises(ValueError):
        standalone_scorer(HYPS[:1], REFS)


def test_via_radeval_interface():
    from RadEval import RadEval
    evaluator = RadEval(do_radgraph_radcliq=True, show_progress=False)
    results = evaluator(refs=REFS, hyps=HYPS)
    assert "radgraph_radcliq" in results
    assert isinstance(results["radgraph_radcliq"], float)
    assert 0.0 <= results["radgraph_radcliq"] <= 1.0


def test_via_radeval_details():
    from RadEval import RadEval
    evaluator = RadEval(
        do_radgraph_radcliq=True, do_details=True, show_progress=False)
    results = evaluator(refs=REFS, hyps=HYPS)
    detail = results["radgraph_radcliq"]
    assert "mean_score" in detail
    assert "sample_scores" in detail
    assert len(detail["sample_scores"]) == len(REFS)

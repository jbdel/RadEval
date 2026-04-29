"""End-to-end tests for the RadEval API surface."""
import pytest
from radeval import RadEval


# --- list syntax (primary) ---

def test_list_single_metric():
    evaluator = RadEval(metrics=["bleu"])
    result = evaluator(refs=["hello world"], hyps=["hello world"])
    assert "bleu" in result
    assert isinstance(result["bleu"], float)


def test_list_multiple_metrics():
    evaluator = RadEval(metrics=["bleu", "rouge"])
    result = evaluator(refs=["hello world"], hyps=["hello world"])
    assert "bleu" in result
    assert "rouge1" in result


def test_list_per_sample():
    evaluator = RadEval(metrics=["bleu"], per_sample=True)
    result = evaluator(refs=["a", "b"], hyps=["a", "b"])
    assert isinstance(result["bleu"], list)
    assert len(result["bleu"]) == 2


def test_list_detailed():
    evaluator = RadEval(metrics=["bleu"], detailed=True)
    result = evaluator(refs=["a b c d"], hyps=["a b c d"])
    assert "bleu_1" in result
    assert "bleu_2" in result


# --- dict syntax (for per-metric config) ---

def test_dict_syntax():
    evaluator = RadEval(metrics=["bleu", "rouge"])
    result = evaluator(refs=["hello world"], hyps=["hello world"])
    assert "bleu" in result and "rouge1" in result


# --- empty / none ---

def test_empty_list():
    evaluator = RadEval(metrics=[])
    assert evaluator(refs=["a"], hyps=["b"]) == {}


def test_empty_dict():
    evaluator = RadEval(metrics={})
    assert evaluator(refs=["a"], hyps=["b"]) == {}


def test_none_metrics():
    evaluator = RadEval(metrics=None)
    assert evaluator(refs=["a"], hyps=["b"]) == {}


# --- from_config ---

def test_from_config_list(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("metrics:\n  - bleu\n  - rouge\n")
    evaluator = RadEval.from_config(str(config))
    result = evaluator(refs=["hello"], hyps=["hello"])
    assert "bleu" in result and "rouge1" in result


def test_from_config_with_overrides(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "metrics:\n  - bleu\n  - rouge\n\noutput:\n  mode: per_sample\n")
    evaluator = RadEval.from_config(str(config))
    result = evaluator(refs=["hello", "world"], hyps=["hello", "world"])
    assert isinstance(result["bleu"], list)


# --- validation ---

def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        RadEval(metrics=["nonexistent"])


def test_type_validation():
    evaluator = RadEval(metrics=["bleu"])
    with pytest.raises(TypeError):
        evaluator(refs="not a list", hyps="not a list")


def test_length_validation():
    evaluator = RadEval(metrics=["bleu"])
    with pytest.raises(ValueError):
        evaluator(refs=["a"], hyps=["a", "b"])


def test_empty_input():
    evaluator = RadEval(metrics=["bleu"])
    assert evaluator(refs=[], hyps=[]) == {}


# --- trl reward adapter ---
# Reward-function unit tests live in tests/test_rewards.py.

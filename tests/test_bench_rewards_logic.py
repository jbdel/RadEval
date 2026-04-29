"""Pure-logic tests for scripts/bench_rewards.py.

No measurement, no model loads. Just exercises the pure helpers so
regressions in the glue code surface without a full benchmark run.
"""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_bench_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "bench_rewards.py"
    spec = importlib.util.spec_from_file_location("bench_rewards", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_rewards"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_bench_module()


def test_skip_record_shape(bench):
    r = bench.skip_record("bleu", None, "load-error:boom")
    assert r == {"metric": "bleu", "key": None, "skipped": "load-error:boom"}


def test_speed_record_rounding(bench):
    r = bench.speed_record(
        metric="bertscore",
        key="bertscore",
        cached_init_s=0.123456,
        warm_batch_s_median=0.002345,
        warm_per_sample_ms=0.1172589,
        peak_vram_mb_approx=512.987,
    )
    assert r["metric"] == "bertscore"
    assert r["key"] == "bertscore"
    assert r["cached_init_s"] == 0.1235
    assert r["warm_batch_s_median"] == 0.0023
    assert r["warm_per_sample_ms"] == 0.117
    assert r["peak_vram_mb_approx"] == 513.0
    assert "skipped" not in r


def test_speed_record_cpu_vram_none(bench):
    r = bench.speed_record(
        metric="bleu", key=None,
        cached_init_s=0.001, warm_batch_s_median=0.001,
        warm_per_sample_ms=0.05, peak_vram_mb_approx=None,
    )
    assert r["peak_vram_mb_approx"] is None


def test_divergence_row_shape(bench):
    entry = {"id": 1, "ref": "A.", "hyp": "B.", "narrative": "flip"}
    scores = {"bleu": 0.1, "bertscore": 0.9, "f1chexbert": 0.0}
    row = bench.divergence_row(entry, scores)
    assert row["id"] == 1
    assert row["ref"] == "A."
    assert row["hyp"] == "B."
    assert row["narrative"] == "flip"
    assert row["scores"] == scores


def test_validate_key_single_key_ok(bench):
    assert bench.validate_key("bleu", None, ["bleu"]) == "bleu"


def test_validate_key_single_key_drift(bench):
    # Adapter unexpectedly returns two keys for a metric we thought was single-key.
    assert bench.validate_key("bleu", None, ["bleu", "bleu_variant"]) is None


def test_validate_key_multi_key_ok(bench):
    assert bench.validate_key(
        "f1chexbert",
        "f1chexbert_sample_acc_5",
        ["f1chexbert_sample_acc_5", "f1chexbert_sample_acc_all"],
    ) == "f1chexbert_sample_acc_5"


def test_validate_key_multi_key_drift(bench):
    # Configured key vanished from the output.
    assert bench.validate_key(
        "f1chexbert",
        "f1chexbert_sample_acc_5",
        ["f1chexbert_sample_acc_new"],
    ) is None


def test_metric_plan_structure(bench):
    """Plan should be a list of (metric, optional_key) tuples, radcliq last."""
    plan = bench.METRIC_PLAN
    assert isinstance(plan, list) and len(plan) > 0
    assert all(len(entry) == 2 and isinstance(entry[0], str) for entry in plan)
    assert plan[-1][0] == "radcliq", "radcliq must be last (contamination guard)"


def test_gallery_metrics_structure(bench):
    """Gallery covers lexical/semantic/clinical spectrum."""
    names = [m for m, _ in bench.GALLERY_METRICS]
    assert "bleu" in names, "BLEU is the familiar lexical baseline"
    assert "bertscore" in names, "BERTScore is the headline semantic baseline"
    assert "f1chexbert" in names
    assert "radgraph" in names
    assert "radcliq" in names

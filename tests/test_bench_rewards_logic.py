"""Pure-logic tests for examples/bench_rewards.py.

No measurement, no model loads. Just exercises the pure helpers so
regressions in the glue code surface without a full benchmark run.
"""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_bench_module():
    path = Path(__file__).resolve().parent.parent / "examples" / "bench_rewards.py"
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
    """Plan is a list of (metric, optional_key, extra_kwargs) triples.
    radcliq must come before any API-backed or 7B-LLM metrics so its
    composite footprint can't contaminate their timing; the heavy LLM
    metrics (green, crimson, mammo_green, radfact_ct) are ordered at
    the end because they're minutes/hours per batch, not ms."""
    plan = bench.METRIC_PLAN
    assert isinstance(plan, list) and len(plan) > 0
    for entry in plan:
        assert len(entry) == 3, f"expected 3-tuple, got {entry!r}"
        assert isinstance(entry[0], str)
        assert isinstance(entry[2], dict), "extra_kwargs must be a dict"
    names = [m for m, _, _ in plan]
    # radcliq is the last LOCAL metric (before LLM/API tail).
    assert "radcliq" in names
    radcliq_idx = names.index("radcliq")
    for heavy in ("green", "crimson", "mammo_green", "radfact_ct"):
        if heavy in names:
            assert names.index(heavy) > radcliq_idx, (
                f"{heavy} must come after radcliq")


def test_gallery_metrics_structure(bench):
    """Gallery covers lexical/semantic/clinical spectrum."""
    names = [m for m, _ in bench.GALLERY_METRICS]
    assert "bleu" in names, "BLEU is the familiar lexical baseline"
    assert "bertscore" in names, "BERTScore is the headline semantic baseline"
    assert "f1chexbert" in names
    assert "radgraph" in names
    assert "radcliq" in names


# ---------- Integration-style: --dry-run structural check ----------


def test_dry_run_produces_expected_snapshot_structure(bench, tmp_path):
    """Drive the full orchestrator via `run_benchmark(dry_run=True)`.
    Exercises fixture loading, metric iteration, divergence row
    building, JSON writing. Skips every expensive step — no model
    loads, no real scoring. Protects against future script bit-rot
    (e.g., fixture path changes, JSON schema drift)."""
    import json

    out = tmp_path / "dry.json"
    bench.run_benchmark(out, dry_run=True)

    data = json.loads(out.read_text())

    # Top-level required keys.
    for field in ("run_ts", "env", "workload", "speed",
                  "divergence_fixture", "divergence"):
        assert field in data, f"missing top-level key: {field}"

    # Workload metadata.
    assert data["workload"]["n_samples"] == 20
    assert data["workload"]["fixture"].endswith("speed_workload.json")

    # Speed rows: one per metric in METRIC_PLAN, in declared order.
    expected_metrics = [m for m, _, _ in bench.METRIC_PLAN]
    observed_metrics = [r["metric"] for r in data["speed"]]
    assert observed_metrics == expected_metrics, (
        "speed rows must preserve METRIC_PLAN order (radcliq last)"
    )
    for row in data["speed"]:
        # In dry-run, every row is a successful record, not a skip.
        assert "skipped" not in row
        for field in ("metric", "key", "cached_init_s",
                      "warm_batch_s_median", "warm_per_sample_ms",
                      "peak_vram_mb_approx"):
            assert field in row, f"row {row['metric']} missing {field}"

    # Divergence rows: one per fixture entry, each scored by every
    # GALLERY_METRICS metric.
    assert len(data["divergence"]) == 8, "divergence fixture has 8 rows"
    gallery_names = [m for m, _ in bench.GALLERY_METRICS]
    for row in data["divergence"]:
        for field in ("id", "ref", "hyp", "narrative", "scores"):
            assert field in row
        assert set(row["scores"].keys()) == set(gallery_names)


def test_dry_run_skip_accounting_path(bench, tmp_path, monkeypatch):
    """Force a metric to fail during dry-run-adjacent pipeline by
    pointing `_fake_speed_record` at a `skip_record` for one named
    metric, and assert the JSON records it as skipped while the
    others continue."""
    import json

    real_fake = bench._fake_speed_record

    def _mocked(metric, key):
        if metric == "bertscore":
            return bench.skip_record(metric, key, "load-error:synthetic")
        return real_fake(metric, key)

    monkeypatch.setattr(bench, "_fake_speed_record", _mocked)

    out = tmp_path / "dry_with_skip.json"
    bench.run_benchmark(out, dry_run=True)
    data = json.loads(out.read_text())

    skipped = [r for r in data["speed"] if "skipped" in r]
    assert len(skipped) == 1 and skipped[0]["metric"] == "bertscore"
    assert "load-error:synthetic" in skipped[0]["skipped"]
    # Remaining rows are still successful, full set still present.
    names = [r["metric"] for r in data["speed"]]
    assert names == [m for m, _, _ in bench.METRIC_PLAN]

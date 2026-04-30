"""Benchmark RadEval reward metrics for speed and divergence.

One-shot documentation tool. Produces a JSON snapshot for
`docs/trl_rewards_benchmarks.md`. Not intended for CI or for
repeated automated reruns — the snapshot is pinned to one
RadEval release + one machine.

Usage:
    # First run: populates the HF cache. Ignore this output.
    python examples/bench_rewards.py --output /tmp/warmup.json

    # Second run: the canonical snapshot.
    python examples/bench_rewards.py \\
        --output docs/benchmarks/trl_rewards_260429.json

Execution model:
    Single in-process loop. Every metric loads, times, tears down
    (`del`, `gc.collect`, `torch.cuda.empty_cache`), then the next
    metric. `radcliq` runs LAST — it's the only composite metric
    (BERTScore + SembScore + RadGraph stacked) and the biggest
    contamination risk; running it last ensures its footprint
    can't pollute adjacent rows. No subprocess orchestration.

VRAM measurement is approximate by design:
    `torch.cuda.max_memory_allocated` only counts torch allocations
    and can be biased by allocator caching. The snapshot labels the
    column as `peak_vram_mb_approx` and the doc calls out the
    limitation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import radeval
import torch
import transformers


# ---------- metric lookup ----------

# (metric_name, key_or_none, extra_init_kwargs)
# Pre-validated `key=` values for per_sample=True output (probed against
# RadEval 2.2.0). If a future adapter drifts, the warm-up call inside the
# timing loop surfaces `"skipped": "key-drift"` with the observed keys.
METRIC_PLAN: list[tuple[str, str | None, dict]] = [
    # Cheap / fast local metrics.
    ("bleu", None, {}),
    ("rouge", "rouge1", {}),
    ("bertscore", None, {}),
    ("radeval_bertscore", None, {}),
    ("f1chexbert", "f1chexbert_sample_acc_5", {}),
    ("f1radbert_ct", "f1radbert_ct_sample_acc", {}),  # future-proof
    ("radgraph", "radgraph_partial", {}),
    ("ratescore", None, {}),
    ("srrbert", "srrbert_weighted_f1", {}),
    ("radgraph_radcliq", None, {}),
    ("temporal", None, {}),
    ("radcliq", None, {}),  # composite: BERTScore + SembScore + RadGraph
    # Heavy: 7B local LLM — minutes per 20-sample batch. Included for
    # scale calibration; the doc labels it as not online-RL-eligible.
    ("green", None, {}),
    # API-backed metrics: clearly not online-RL-eligible (one HTTP call
    # per sample) but included for honest scale comparison. Each requires
    # an env var (OPENAI_API_KEY / GEMINI_API_KEY); the benchmark skips
    # with "no-api-key" if the needed key isn't set.
    # CRIMSON defaults to provider="hf" (local) — override to openai
    # here so the "very slow API" framing in the doc is actually measured.
    ("crimson", None, {"provider": "openai", "model_name": "gpt-4o-mini"}),
    ("mammo_green", None, {}),  # defaults to gpt-4o-mini already
    ("radfact_ct", "radfact_ct_f1", {}),
]

# Gallery subset: 7 metrics spanning lexical → semantic → clinical
# → LLM-based clinical. The doc's divergence table renders a curated
# subset (5 of 7 columns, 5 of 8 rows) — see plan/06.md Findings for
# the selection analysis. The JSON snapshot keeps all 7 × 8 so users
# re-running the benchmark see the full picture.
#
# Doc-to-JSON mapping (authoritative, for anyone regenerating the
# rendered table from a new snapshot):
#   Rendered doc columns:  bleu, bertscore, radgraph, radcliq, crimson
#   Rendered doc rows:     JSON id=5 (exact match, shown as row 1),
#                          JSON id=1 (paraphrase, shown as row 2),
#                          JSON id=2 (negation flip, shown as row 3),
#                          JSON id=7 (severity flip, shown as row 4),
#                          JSON id=8 (opposite conclusion, shown as row 5).
#
# 3-tuple to match METRIC_PLAN shape: (metric, key, extra_init_kwargs).
# CRIMSON is forced to its openai provider so the gallery column
# reflects what the API judge returns, matching the "API-based
# clinical" row in the speed table.
GALLERY_METRICS: list[tuple[str, str | None, dict]] = [
    ("bleu", None, {}),
    ("bertscore", None, {}),
    ("f1chexbert", "f1chexbert_sample_acc_5", {}),
    ("radgraph", "radgraph_partial", {}),
    ("radcliq", None, {}),
    ("green", None, {}),
    ("crimson", None, {"provider": "openai", "model_name": "gpt-4o-mini"}),
]


# ---------- pure-logic helpers (unit-tested) ----------


def skip_record(metric: str, key: str | None, reason: str) -> dict[str, Any]:
    """Canonical shape for a skipped metric entry."""
    return {
        "metric": metric,
        "key": key,
        "skipped": reason,
    }


def speed_record(
    metric: str,
    key: str | None,
    cached_init_s: float,
    warm_batch_s_median: float,
    warm_per_sample_ms: float,
    peak_vram_mb_approx: float | None,
) -> dict[str, Any]:
    """Canonical shape for a successful speed row."""
    return {
        "metric": metric,
        "key": key,
        "cached_init_s": round(cached_init_s, 4),
        "warm_batch_s_median": round(warm_batch_s_median, 4),
        "warm_per_sample_ms": round(warm_per_sample_ms, 3),
        "peak_vram_mb_approx": (
            round(peak_vram_mb_approx, 1)
            if peak_vram_mb_approx is not None
            else None
        ),
    }


def divergence_row(
    entry: dict[str, Any],
    scores: dict[str, float | None],
) -> dict[str, Any]:
    """Canonical shape for a divergence-gallery row (entry + measured scores)."""
    return {
        "id": entry["id"],
        "ref": entry["ref"],
        "hyp": entry["hyp"],
        "narrative": entry["narrative"],
        "scores": scores,
    }


def validate_key(
    metric: str,
    expected_key: str | None,
    observed_keys: list[str],
) -> str | None:
    """Resolve which per-sample key to read from this metric's output.

    Returns the chosen key name, or None to indicate drift.
    None signals to the caller that the metric should be skipped with
    `"skipped": "key-drift"`.
    """
    if expected_key is None:
        # Single-key metric — let the warm-up call pick the only available key.
        if len(observed_keys) == 1:
            return observed_keys[0]
        # Drift: a metric that should be single-key now returns more.
        return None
    if expected_key in observed_keys:
        return expected_key
    return None


# ---------- environment + IO helpers ----------


def env_block() -> dict[str, Any]:
    """One-time environment metadata for the JSON snapshot."""
    gpu = "cpu"
    cuda = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda = torch.version.cuda
    return {
        "radeval_version": radeval.__version__ if hasattr(radeval, "__version__") else "2.2.0",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda": cuda,
        "gpu": gpu,
        "hf_cache": str(Path.home() / ".cache" / "huggingface"),
    }


def load_fixture(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


# ---------- measurement primitives ----------


def _time_metric(
    metric: str,
    expected_key: str | None,
    refs: list[str],
    hyps: list[str],
    extra_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Benchmark one metric. Returns a speed record or a skip record.

    Runs in-process. Caller is responsible for teardown between metrics.
    """
    from radeval.metrics._registry import get_metric_class

    extra_kwargs = extra_kwargs or {}

    # --- Load / instantiate. `cached_init_s` is measured here.
    # VRAM baseline captured IMMEDIATELY BEFORE cls() so we accommodate
    # residual CUDA context from prior work.
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        vram_base = torch.cuda.memory_allocated()
    else:
        vram_base = 0

    try:
        cls = get_metric_class(metric)
    except Exception as exc:  # pragma: no cover — registry shouldn't fail
        return skip_record(metric, expected_key, f"registry-error:{type(exc).__name__}:{exc}")

    t0 = time.perf_counter()
    try:
        scorer = cls(**extra_kwargs)
    except EnvironmentError as exc:
        # LLM-based metrics raise EnvironmentError when the required API
        # key isn't set. Give it a distinct skip reason so the snapshot
        # clearly shows "would be benchmarked if OPENAI_API_KEY were set"
        # vs. genuine load failures.
        return skip_record(metric, expected_key, f"no-api-key:{str(exc)[:120]}")
    except (FileNotFoundError, ImportError) as exc:
        return skip_record(metric, expected_key, f"load-error:{type(exc).__name__}:{str(exc)[:120]}")
    except Exception as exc:
        # Hub/network/etc — be broad at load time but labeled.
        return skip_record(metric, expected_key, f"load-error:{type(exc).__name__}:{str(exc)[:120]}")
    if on_gpu:
        torch.cuda.synchronize()
    cached_init_s = time.perf_counter() - t0

    # --- Warm-up call (doubles as key-map validation).
    try:
        result = scorer.compute(refs=refs, hyps=hyps, per_sample=True)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        msg = str(exc)
        reason = "OOM" if "out of memory" in msg.lower() else f"runtime-error:{msg[:120]}"
        _teardown(scorer, on_gpu)
        return skip_record(metric, expected_key, reason)
    except Exception as exc:
        _teardown(scorer, on_gpu)
        return skip_record(metric, expected_key, f"compute-error:{type(exc).__name__}:{str(exc)[:120]}")

    observed_keys = list(result.keys())
    chosen_key = validate_key(metric, expected_key, observed_keys)
    if chosen_key is None:
        _teardown(scorer, on_gpu)
        return skip_record(
            metric, expected_key,
            f"key-drift:observed={observed_keys}",
        )

    # --- Warm-batch timing: 3 calls, median.
    batch_times: list[float] = []
    for _ in range(3):
        if on_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            scorer.compute(refs=refs, hyps=hyps, per_sample=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc)
            reason = "OOM" if "out of memory" in msg.lower() else f"runtime-error:{msg[:120]}"
            _teardown(scorer, on_gpu)
            return skip_record(metric, chosen_key, reason)
        if on_gpu:
            torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t0)

    warm_median = statistics.median(batch_times)
    n = len(refs)
    per_sample_ms = (warm_median * 1000.0) / n if n else 0.0

    # --- VRAM delta.
    vram_mb: float | None = None
    if on_gpu:
        peak = torch.cuda.max_memory_allocated()
        vram_mb = max(0.0, (peak - vram_base) / (1024 * 1024))

    _teardown(scorer, on_gpu)

    return speed_record(
        metric=metric,
        key=chosen_key,
        cached_init_s=cached_init_s,
        warm_batch_s_median=warm_median,
        warm_per_sample_ms=per_sample_ms,
        peak_vram_mb_approx=vram_mb,
    )


def _teardown(scorer: Any, on_gpu: bool) -> None:
    """Aggressive cleanup between metrics."""
    try:
        del scorer
    except Exception:
        pass
    gc.collect()
    if on_gpu:
        torch.cuda.empty_cache()


def _score_one_pair(
    metric: str,
    expected_key: str | None,
    ref: str,
    hyp: str,
    extra_kwargs: dict | None = None,
) -> float | None:
    """Score a single (ref, hyp) pair and return the scalar. None on skip."""
    from radeval.metrics._registry import get_metric_class

    extra_kwargs = extra_kwargs or {}

    try:
        cls = get_metric_class(metric)
        scorer = cls(**extra_kwargs)
    except Exception:
        return None
    try:
        result = scorer.compute(refs=[ref], hyps=[hyp], per_sample=True)
    except Exception:
        _teardown(scorer, torch.cuda.is_available())
        return None
    chosen_key = validate_key(metric, expected_key, list(result.keys()))
    if chosen_key is None:
        _teardown(scorer, torch.cuda.is_available())
        return None
    values = result[chosen_key]
    scalar = values[0] if values else None
    if hasattr(scalar, "item"):
        scalar = scalar.item()
    _teardown(scorer, torch.cuda.is_available())
    if scalar is None or (isinstance(scalar, float) and math.isnan(scalar)):
        return None
    return float(scalar)


# ---------- main orchestration ----------


def _fake_speed_record(metric: str, key: str | None) -> dict[str, Any]:
    """Dummy speed record used by --dry-run. Exercises the pipeline without
    actually loading any models. Numbers are placeholders."""
    return speed_record(
        metric=metric, key=key,
        cached_init_s=0.01, warm_batch_s_median=0.001,
        warm_per_sample_ms=0.05,
        peak_vram_mb_approx=0.0 if torch.cuda.is_available() else None,
    )


def _fake_score(metric: str) -> float:
    """Dummy per-sample score used by --dry-run."""
    return 0.5


def run_benchmark(output_path: Path, dry_run: bool = False) -> None:
    # Load fixtures.
    bench_dir = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "fixtures"
    speed_rows = load_fixture(bench_dir / "speed_workload.json")
    divergence_rows = load_fixture(bench_dir / "divergence_examples.json")

    refs = [r["ref"] for r in speed_rows]
    hyps = [r["hyp"] for r in speed_rows]

    snapshot = {
        "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "env": env_block() if not dry_run else {"dry_run": True},
        "workload": {
            "n_samples": len(speed_rows),
            "fixture": "docs/benchmarks/fixtures/speed_workload.json",
        },
        "speed": [],
        "divergence_fixture": "docs/benchmarks/fixtures/divergence_examples.json",
        "divergence": [],
    }

    # E1: speed. radcliq last (plan rule); heavy LLM metrics at the end
    # so their cost isn't blocking the interesting fast rows.
    for metric, key, extra in METRIC_PLAN:
        sys.stderr.write(f"[speed] {metric}\n")
        sys.stderr.flush()
        if dry_run:
            row = _fake_speed_record(metric, key)
        else:
            row = _time_metric(metric, key, refs, hyps, extra_kwargs=extra)
        snapshot["speed"].append(row)

    # E2: divergence. Score each pair with each gallery metric.
    for entry in divergence_rows:
        scores: dict[str, float | None] = {}
        for metric, key, extra in GALLERY_METRICS:
            sys.stderr.write(f"[divergence row {entry['id']}] {metric}\n")
            sys.stderr.flush()
            if dry_run:
                scores[metric] = _fake_score(metric)
            else:
                scores[metric] = _score_one_pair(
                    metric, key, entry["ref"], entry["hyp"],
                    extra_kwargs=extra,
                )
        snapshot["divergence"].append(divergence_row(entry, scores))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    sys.stderr.write(f"\nSnapshot written to {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the JSON snapshot.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Bypass model loading and scoring; emit a snapshot with "
             "placeholder numbers. Exercises the data-loading / iteration / "
             "JSON-writing path only. Used by tests.",
    )
    args = parser.parse_args()
    try:
        run_benchmark(args.output, dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Generate the regression snapshot for the current RadEval + transformers stack.

This captures the expected outputs of the public, non-API metrics on a fixed
(REFS, HYPS) pair across all three output modes. `tests/test_regression.py`
loads the saved JSON and asserts bit-equality.

Usage (requires GPU + full model downloads):

    python tests/generate_regression_snapshot.py

Outputs:
    tests/fixtures/expected_scores_cpu.json  (bleu + rouge, no downloads)
    tests/fixtures/expected_scores_gpu.json  (all v2.x public local metrics)

⚠️ Regenerate this snapshot intentionally — it pins numeric outputs against a
specific (transformers, torch, model-checkpoint) combination. Regenerate when:
  • You bump `transformers` to a new major version (e.g. 5 → 6)
  • An underlying model checkpoint on the Hub is re-uploaded
  • A metric's internal implementation changes in a way that intentionally
    shifts outputs.

The current snapshot is anchored on:
  * transformers 5.6.2
  * torch 2.9.1+cu128
  * RadEval 2.1.0 (vendored bert_score + vendored radgraph 0.1.18 patched
    for transformers v5)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fixtures.regression_samples import REFS, HYPS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radeval import RadEval


def _convert_for_json(obj):
    """Convert numpy/torch types to Python natives for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    return obj


def _run_tier(metrics: list[str]) -> dict:
    """Run a metric list across the three output modes."""
    results = {}
    for mode in ("default", "per_sample", "detailed"):
        kwargs = {"metrics": metrics, "show_progress": False}
        if mode == "per_sample":
            kwargs["per_sample"] = True
        elif mode == "detailed":
            kwargs["detailed"] = True
        evaluator = RadEval(**kwargs)
        results[mode] = evaluator(refs=REFS, hyps=HYPS)
    return results


# Public metrics on RadEval 2.1 + transformers 5.x. The vendored radgraph
# (radeval/metrics/radgraph/_vendor/) enables the radgraph-family metrics
# again on transformers 5.x.
CPU_METRICS = ["bleu", "rouge"]
GPU_METRICS = [
    "bleu", "rouge",
    "bertscore", "radeval_bertscore",
    "f1chexbert",
    "srrbert",
    "ratescore",
    "temporal",
    "radgraph", "radgraph_radcliq", "radcliq",
]


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "fixtures")

    print("Running CPU tier:", CPU_METRICS)
    cpu_snapshot = _convert_for_json(_run_tier(CPU_METRICS))
    cpu_path = os.path.join(outdir, "expected_scores_cpu.json")
    with open(cpu_path, "w") as f:
        json.dump(cpu_snapshot, f, indent=4)
    print(f"Saved CPU snapshot to {cpu_path}")

    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU available — skipping GPU snapshot")
            sys.exit(0)
    except ImportError:
        print("No torch — skipping GPU snapshot")
        sys.exit(0)

    print("Running GPU tier:", GPU_METRICS)
    gpu_snapshot = _convert_for_json(_run_tier(GPU_METRICS))
    gpu_path = os.path.join(outdir, "expected_scores_gpu.json")
    with open(gpu_path, "w") as f:
        json.dump(gpu_snapshot, f, indent=4)
    print(f"Saved GPU snapshot to {gpu_path}")

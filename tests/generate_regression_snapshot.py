"""Run ONCE on the OLD codebase to capture expected scores.

Usage: python tests/generate_regression_snapshot.py
Output: tests/fixtures/expected_scores_cpu.json
        tests/fixtures/expected_scores_gpu.json (if GPU available)
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fixtures.regression_samples import REFS, HYPS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from RadEval import RadEval


def run_tier(metrics_flags, modes=("default", "per_sample", "detailed")):
    results = {}
    for mode in modes:
        kwargs = dict(metrics_flags)
        if mode == "per_sample":
            kwargs["do_per_sample"] = True
        elif mode == "detailed":
            kwargs["do_details"] = True
        evaluator = RadEval(**kwargs)
        results[mode] = evaluator(refs=REFS, hyps=HYPS)
    return results


def convert_for_json(obj):
    """Convert numpy/torch types to Python native for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "fixtures")

    # CPU tier (always runnable)
    print("Running CPU tier (bleu, rouge)...")
    cpu_flags = {"do_bleu": True, "do_rouge": True}
    cpu_snapshot = convert_for_json(run_tier(cpu_flags))
    cpu_path = os.path.join(outdir, "expected_scores_cpu.json")
    with open(cpu_path, "w") as f:
        json.dump(cpu_snapshot, f, indent=4)
    print(f"Saved CPU snapshot to {cpu_path}")

    # GPU tier (requires GPU + model downloads)
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU available — skipping GPU snapshot")
            sys.exit(0)
    except ImportError:
        print("No torch — skipping GPU snapshot")
        sys.exit(0)

    print("Running GPU tier (all local metrics)...")
    gpu_flags = {
        "do_bleu": True, "do_rouge": True, "do_bertscore": True,
        "do_radeval_bertscore": True, "do_f1chexbert": True,
        "do_srrbert": True, "do_ratescore": True, "do_radcliq": True,
        "do_temporal": True, "do_radgraph": True, "do_radgraph_radcliq": True,
    }
    gpu_snapshot = convert_for_json(run_tier(gpu_flags))
    gpu_path = os.path.join(outdir, "expected_scores_gpu.json")
    with open(gpu_path, "w") as f:
        json.dump(gpu_snapshot, f, indent=4)
    print(f"Saved GPU snapshot to {gpu_path}")

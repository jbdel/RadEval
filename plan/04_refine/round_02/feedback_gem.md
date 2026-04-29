# Verdict
revise_minor

# Critical Issues
- severity: medium
- issue: Contradiction in workload generation strategy.
- why it matters: Section E1 states the workload is a "Static 20-row `docs/benchmarks/fixtures/speed_workload.json` with hand-written `{ref, hyp}` pairs... no perturbation logic inside the script." However, the "Decisions & alternatives considered" table states the exact opposite: "Perturb `synthetic_reports.json` ground_truth strings to make hyps for E1... Deterministic perturbations... achieve this cheaply." This creates ambiguity for implementation.
- recommended fix: Stick to the static, hand-written fixture described in E1. It adheres better to KISS and keeps the benchmarking script strictly focused on measurement rather than data synthesis.

- severity: medium
- issue: Unsafe mathematical transform suggested for RadCliQ inversion.
- why it matters: The plan suggests `lambda x: 1.0 / (1.0 + x)` to invert RadCliQ. RadCliQ is a linear regression composite of other metrics. If its output range is not strictly bounded to `[0, \infty)` (e.g., if it can be negative, which is possible for standardized regression outputs), this transform risks division by zero or asymptotes.
- recommended fix: Remove the `1.0 / (1.0 + x)` suggestion. Stick exclusively to `lambda x: -x`, which is universally safe for any monotonically decreasing requirement regardless of the metric's bounds.

# Overengineering Flags
- component: `--isolate <metric>` subprocess orchestration
- why unnecessary: The plan acknowledges this is a "YAGNI fallback" but still designs it into the CLI and script architecture. Having the Python script spawn its own subprocesses to manage memory leaks violates KISS and adds orchestration complexity.
- simpler alternative: Provide a simple `--metric <name>` argument that runs the benchmark for *only* that metric. If memory isolation is needed, the caller can trivially orchestrate it via bash: `for m in metrics; do python scripts/bench_rewards.py --metric $m; done`.

# Assumptions to Test
- Verify the theoretical bounds of RadCliQ-v1 before suggesting any bounded inversion transforms.
- Verify that `torch.cuda.max_memory_allocated()` captures the full VRAM footprint for all metrics (i.e., ensure no metric relies on a non-PyTorch backend like ONNX Runtime or raw CUDA allocations that would bypass PyTorch's memory allocator).

# Recommended Revisions
- Resolve the contradiction regarding `speed_workload.json` generation by removing the perturbation logic from the "Decisions" table and committing to the static fixture.
- Drop the `--isolate` subprocess logic from the script design; replace it with a standard `--metric` filter.
- Update the documentation outline to only recommend `lambda x: -x` for RadCliQ inversion.

# Confidence
high

# Verdict
revise_major

# Critical Issues
- severity: high
- issue: The plan claims “every practical public metric for per-step RL rewards” and “13+ metrics usable as RL rewards” based on a subjective latency threshold and one probed environment.
- why it matters: This overstates evidence. “Practical” and “usable” depend on hardware, batch size, and training setup; a single-machine snapshot does not justify broad eligibility claims.
- recommended fix: Narrow the language to “benchmarked public metrics selected as plausible RL rewards in this environment” and explicitly label exclusions/inclusions as judgment calls for this snapshot.

- severity: high
- issue: The speed benchmark mixes incomparable conditions: cached init is measured after a prewarm pass, while “first-download” is optional/anecdotal and may or may not exist per metric.
- why it matters: Readers may infer a fair cold/warm comparison, but the methodology does not produce controlled cold-start numbers across metrics.
- recommended fix: Either drop first-download entirely from the plan/doc, or define a controlled cold-start procedure in isolated environments. Keep the main artifact to warm-cache init + warm batch latency if simplicity is the goal.

- severity: high
- issue: The divergence gallery is explicitly curated to produce disagreement, and the plan proposes swapping examples if the hypothesis does not hold.
- why it matters: That is selection on the outcome. It supports an illustrative doc, but not a strong comparative claim about metric behavior.
- recommended fix: State more plainly that the gallery is hand-picked examples demonstrating possible disagreement, not evidence of prevalence or average superiority. Avoid language like “this is the differentiator” unless softened.

- severity: medium
- issue: The VRAM measurement method is weakly justified and likely contaminated by allocator/caching behavior, especially in-process.
- why it matters: Peak VRAM numbers may look precise while being unreliable across sequential metric runs.
- recommended fix: Either remove VRAM from the primary table, or mark it as approximate and collect it only in isolated runs for heavyweight GPU metrics.

- severity: medium
- issue: The plan relies on verified key mappings and instantiation facts from a probe, but the script itself embeds a static key map and does not validate drift beyond runtime failure.
- why it matters: The doc is pinned, but the script is also presented as reproducible. Silent drift will degrade usability.
- recommended fix: Add a minimal startup validation that the configured key exists for each metric before timing, and fail that metric clearly if not.

- severity: medium
- issue: Runtime budget and isolation strategy are underspecified if `radcliq` or other metrics interfere materially.
- why it matters: “Use `--isolate` if needed” is reactive; without a defined trigger, results may be inconsistent.
- recommended fix: Define a simple rule: benchmark all metrics in-process first, but always isolate designated heavyweight metrics (e.g. `radcliq`) if GPU measurements are reported.

# Overengineering Flags
- component: Optional first-download capture
- why unnecessary: It adds methodological complexity without producing a controlled, comparable artifact.
- simpler alternative: Omit first-download from both script and doc; report only warm-cache init and warm batch latency.

- component: Optional `--isolate` subprocess mode
- why unnecessary: As written, it adds branching behavior and merge logic for a one-shot doc script.
- simpler alternative: Either keep everything in-process and drop VRAM, or always isolate only the one known heavyweight metric if needed.

- component: Peak VRAM delta in the main benchmark
- why unnecessary: It is likely noisy and not central to the stated doc goal of making reward tradeoffs concrete.
- simpler alternative: Focus the table on init time and warm batch latency; mention memory qualitatively if observed.

- component: Extensive environment metadata including `trl`
- why unnecessary: `trl` is not part of the benchmark execution path described.
- simpler alternative: Record only versions that materially affect metric loading/scoring.

# Assumptions to Test
- The selected “practical” metrics are actually runnable within a useful RL-loop budget on the target hardware.
- In-process teardown is sufficient to prevent cross-metric contamination of timing and memory measurements.
- The curated divergence examples produce the intended disagreement without excessive hand-tuning.
- `torch.cuda.max_memory_allocated` is informative enough for the intended audience if VRAM is retained.
- `radcliq` direction/inversion guidance is compatible with the actual reward API behavior for scalar transforms.

# Recommended Revisions
- Reframe scope claims to avoid universal language like “every practical” and “usable as RL rewards.”
- Simplify E1 to one controlled methodology: warm-cache init + warm batch latency only.
- Either remove VRAM from the main artifact or explicitly downgrade it to approximate/secondary.
- Tighten the divergence-gallery wording so it is clearly illustrative and not evidence of general superiority.
- Add a minimal validation step in the script for configured metric keys before benchmarking.
- Predefine whether heavyweight metrics are always isolated or never isolated, rather than leaving it ad hoc.

# Confidence
high

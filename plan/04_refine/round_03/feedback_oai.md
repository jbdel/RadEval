# Verdict
revise_major

# Critical Issues
- severity: high
- issue: The plan claims “every in-scope metric” and “13+ metrics usable as RL rewards,” but the inclusion rule is explicitly subjective (“author’s judgment”) and excludes some public local metrics on practicality grounds.
- why it matters: This overstates what the evidence supports and risks misleading readers into treating the table as comprehensive rather than curated for one machine/workflow.
- recommended fix: Tighten the framing everywhere to “benchmarked subset of public metrics judged plausible for this setup,” and remove broad repo-level claims like “every in-scope metric” / “13+ usable as RL rewards” unless you enumerate and justify them consistently.

- severity: high
- issue: The benchmark methodology does not adequately control for cross-metric contamination, yet the resulting numbers are presented as comparable. Only `radcliq` is isolated; other model-backed metrics may still inherit allocator/cache/process state.
- why it matters: Warm latency and especially VRAM deltas may not be comparable across metrics if measured in one long-lived process.
- recommended fix: Either isolate all model-backed metrics in subprocesses, or explicitly downgrade the claim of comparability and drop/soften VRAM reporting for in-process rows. At minimum, validate one or two metrics both in-process and isolated to estimate contamination.

- severity: medium
- issue: The plan says no CI/tests because this is “one-shot,” but the script contains nontrivial logic: key validation, subprocess merge path, skip handling, OOM handling, JSON schema generation.
- why it matters: “One-shot” does not remove the need to validate behavior that the docs will depend on. A broken script undermines the snapshot and reproducibility claim.
- recommended fix: Add minimal tests for the pure logic only: key-map/schema/output merge/skip-record formatting. Avoid testing measured numbers.

- severity: medium
- issue: The plan mixes verified facts with speculative doc conclusions, especially around `radcliq` cost (“≈3× the per-sample cost of its heaviest component”) and preferred usage guidance.
- why it matters: Those claims are stronger than the presented evidence until measurements exist.
- recommended fix: Remove quantitative comparative claims from the plan/doc outline until the snapshot exists; phrase them as placeholders to be filled from measured output.

- severity: medium
- issue: Error handling for OOM via broad `RuntimeError` matching is underspecified and may mask unrelated failures as OOM.
- why it matters: Misclassification weakens the benchmark’s diagnostic value and can hide real adapter/script bugs.
- recommended fix: Narrow the OOM detection logic and record unexpected `RuntimeError`s distinctly unless the message clearly matches known CUDA OOM patterns.

- severity: medium
- issue: The divergence gallery is intentionally curated to show disagreement, but the plan also uses it to support reward-selection guidance.
- why it matters: Existence proofs are fine, but they are weak evidence for stronger recommendations like “BLEU is not enough” or “prefer X reward” without clearer separation.
- recommended fix: Keep the gallery strictly illustrative and ensure reward recommendations are explicitly grounded in external literature or the speed table, not in the curated examples.

# Overengineering Flags
- component: Startup key-map probing plus hardcoded key-map literal
- why unnecessary: This duplicates source-of-truth information and adds drift surface for a one-shot docs script.
- simpler alternative: Hardcode the selected metric calls only, and fail fast if instantiation/scoring raises. If key drift is a concern, validate only the configured metrics/keys without building a broader “fact sheet” mechanism.

- component: Date-stamped snapshot naming to “add more later”
- why unnecessary: The plan also says this is not a living contract and not intended for repeated reruns.
- simpler alternative: Use a single fixed snapshot filename for this doc, or justify multiple snapshots as an actual current need.

- component: Three-commit structure
- why unnecessary: Splitting a small docs-focused change into three commits may add review overhead without much benefit.
- simpler alternative: Keep script+fixtures separate from docs if desired, but avoid over-optimizing commit granularity unless reviewers asked for it.

# Assumptions to Test
- In-process measurements for non-`radcliq` metrics are close enough to isolated-process measurements to support comparison.
- `torch.cuda.max_memory_allocated` is informative enough for the included metrics despite possible non-torch allocations.
- The selected 20-sample workload is large enough to reduce timing noise for fast metrics.
- `prewarm` actually removes the dominant download/init variance for all included metrics.
- The curated divergence examples produce the intended disagreement patterns across the chosen metrics.

# Recommended Revisions
- Reframe the benchmark as a curated subset for one setup, not a comprehensive statement about all RL-usable rewards.
- Add one minimal validation pass comparing isolated vs in-process measurement for at least a representative fast metric and a representative model-backed metric.
- Add lightweight tests for script logic that does not depend on benchmark numbers.
- Remove placeholder quantitative claims from the doc outline until backed by actual measurements.
- Either drop VRAM from the main comparison table or label non-`radcliq` VRAM rows more cautiously unless isolation/validation is added.
- Tighten the connection between the illustrative gallery and the stronger reward-selection recommendations.

# Confidence
high

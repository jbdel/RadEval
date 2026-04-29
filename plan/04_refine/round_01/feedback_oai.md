# Verdict
revise_minor

# Critical Issues
- severity: medium
- issue: The plan claims “every RL-eligible metric” but excludes `green` based on practicality rather than a clearly stated eligibility rule.
- why it matters: This overstates completeness. A reader or reviewer could reasonably interpret “every RL-eligible metric” as exhaustive, then find a public reward-capable metric omitted.
- recommended fix: Narrow the claim to “every practical public metric for per-step RL rewards” or add a short explicit inclusion rule that excludes very-large local LLM metrics from the benchmark table.

- severity: medium
- issue: Cold-start timing is not well controlled because it mixes model initialization with possible network download and uses a heuristic “cache_hit” concept.
- why it matters: The resulting numbers are hard to interpret and compare across metrics; “cold-start” becomes partly a network benchmark.
- recommended fix: Split initialization timing into at least two labeled modes if practical (“cached init” vs “first download”), or explicitly benchmark only cached initialization and mention first-download as anecdotal/non-comparable.

- severity: medium
- issue: The divergence gallery is intentionally curated to produce disagreement, but the plan does not include a check against cherry-picking or misleading interpretation.
- why it matters: The doc risks sounding stronger than the evidence supports; curated examples demonstrate possibility, not prevalence.
- recommended fix: Add one sentence in the doc that the gallery is illustrative, not a representative sample, and avoid broad claims from it.

- severity: low
- issue: Writing benchmark fixtures under `tests/fixtures/` without tests blurs the purpose of that directory.
- why it matters: It may create confusion about whether these are test assets or benchmark/doc assets.
- recommended fix: Prefer a docs/benchmarks or scripts/fixtures location unless there is a strong existing convention for non-test fixtures under `tests/fixtures`.

# Overengineering Flags
- component: Subprocess-per-metric orchestration
- why unnecessary: For a one-shot script, recursive subprocess management plus timeout handling and append semantics adds complexity.
- simpler alternative: Use a simple in-process loop first; only isolate specific problematic metrics in subprocesses if interference is observed.

- component: HF cache-hit detection
- why unnecessary: “Plausibly matching entry” is heuristic and may not be reliable enough to justify the added logic.
- simpler alternative: Record the HF cache path and state plainly whether the run was performed after manual cache warm-up; omit per-metric cache-hit booleans.

- component: Appending speed and divergence into one JSON via multiple command modes
- why unnecessary: Append/update behavior increases script complexity for little value in a one-shot workflow.
- simpler alternative: Generate the full snapshot in one `all` mode, or emit separate files and cite both if needed.

# Assumptions to Test
- `radcliq` completes within the stated timeout on the target hardware.
- `torch.cuda.max_memory_allocated` is meaningful for these metrics and not badly misleading due to non-torch allocations or subprocess boundaries.
- The chosen perturbations for E1 do not materially affect latency in a way that makes cross-metric comparisons noisy or misleading.
- The curated divergence examples actually produce the intended score ordering across the selected metrics.
- `make_reward_fn("radcliq", score_transform=...)` is a supported and documented usage pattern, not just an inferred convenience.

# Recommended Revisions
- Tighten the scope language so “all metrics” does not overclaim beyond the actual inclusion rule.
- Reframe cold-start measurement to avoid conflating initialization with network download, or label the result more carefully.
- Add an explicit caveat that the divergence gallery is illustrative and curated, not representative.
- Simplify the script surface area: prefer one-shot generation and drop heuristic cache-hit logic unless it proves necessary.
- Reconsider fixture placement for divergence examples to keep benchmark/doc data separate from test data.

# Confidence
high

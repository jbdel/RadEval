# Verdict
revise_minor

# Critical Issues
- severity: medium
- issue: Contradiction regarding `radcliq` isolation strategy.
- why it matters: The plan states in "Implementation notes" that `radcliq` runs *last* in the in-process loop to avoid subprocess orchestration. However, the "Doc outline" and "Decisions" table explicitly state `radcliq` is isolated to its own subprocess. This creates confusion for implementation.
- recommended fix: Standardize on running `radcliq` *last* in the in-process loop. It avoids subprocess complexity and achieves the goal of preventing its large footprint from contaminating subsequent metrics. Update the doc outline and decisions table to reflect this.

# Overengineering Flags
- component: Implementation-time measurement gate
- why unnecessary: The plan proposes a formal meta-experiment (comparing in-process vs. subprocess execution), complete with thresholds (~10% latency, ~50MB VRAM) and recording the result in the JSON snapshot. For a one-shot script where VRAM is explicitly documented as "approximate" and "directional guidance", this level of validation is excessive and expands scope.
- simpler alternative: Trust the in-process loop with explicit teardown (`del`, `gc.collect()`, `torch.cuda.empty_cache()`). Drop the measurement gate and the `--metric` flag (if it was only added to support this gate).

- component: `prewarm` subcommand
- why unnecessary: Adding a dedicated CLI command just to download models adds unnecessary script complexity for a one-shot developer tool.
- simpler alternative: Instruct the user (in the reproduction steps) to simply run the script twice: once to download models to the HF cache (ignoring the output), and a second time to generate the clean timing snapshot. 

# Assumptions to Test
- `make_reward_fn` single-key guard: Double-check that `make_reward_fn("f1radbert_ct")` and `make_reward_fn("radcliq")` truly do not require a `key=` argument when the underlying metric returns a dictionary with exactly one key. If the adapter wraps the output in a way that triggers the guard, the script will fail.

# Recommended Revisions
- Resolve the `radcliq` subprocess vs. in-process contradiction in favor of in-process (run last).
- Remove the "Implementation-time measurement gate" and the associated `comparability_check` field in the JSON output.
- Remove the `prewarm` subcommand and update the reproduction instructions in the markdown to run the script twice for clean timings.

# Confidence
high

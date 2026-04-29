# Verdict
revise_minor

# Critical Issues
- severity: medium
- issue: The plan is internally inconsistent about `radcliq` isolation: some sections say in-process with `radcliq` last, while later sections and the doc outline say `radcliq` is measured in its own subprocess / always isolated.
- why it matters: This makes the implementation and resulting numbers ambiguous, and weakens confidence in the measurement methodology.
- recommended fix: Pick one approach and use it consistently everywhere. Given the stated KISS preference, either keep `radcliq` last plus the comparability gate, or always isolate `radcliq`; do not describe both as the default.

- severity: medium
- issue: The plan claims “no CI coverage” and “no changes to tests,” but later adds `tests/test_bench_rewards_logic.py`.
- why it matters: This is a scope contradiction and suggests the testing stance is not fully thought through.
- recommended fix: Clarify that there will be no integration/measurement CI, but there will be minimal unit tests for pure logic/schema helpers.

- severity: medium
- issue: The benchmark inclusion rule (“plausibly run fast enough for per-step use”) is partly circular because inclusion is justified by the measurements the benchmark itself will produce.
- why it matters: It risks looking post-hoc and subjective rather than a clean, predeclared scope.
- recommended fix: State a simpler pre-measurement inclusion rule based on public/local availability and intended RL relevance, then let the measured table speak for practicality.

- severity: low
- issue: The JSON examples are inconsistent with the stated schema/metadata decisions (`trl` appears in example env despite being explicitly excluded; `cold_start_s`/`peak_vram_mb` names differ from `cached_init_s`/`peak_vram_mb_approx`; fixture path differs).
- why it matters: Small schema drift creates avoidable implementation confusion and review churn.
- recommended fix: Normalize the example JSON to the exact intended field names and paths.

# Overengineering Flags
- component: comparability gate with explicit thresholds and snapshot field
- why unnecessary: For a one-shot docs benchmark, the gate adds process complexity and another artifact to explain, especially if the final decision is simply “use in-process unless obviously broken.”
- simpler alternative: Either always isolate only `radcliq`, or run in-process for all metrics and mention VRAM/latency are approximate without formalizing a gate.

- component: Extensive key-map fact sheet plus runtime validation plus skip-on-drift behavior
- why unnecessary: The plan is pinned to one release and already uses an explicit metric list. Full drift-handling logic may be more machinery than needed for a snapshot script.
- simpler alternative: Keep an explicit metric→kwargs map in the script and fail fast if a configured metric no longer works, since this is a one-shot pinned benchmark.

- component: Date-stamped snapshot naming “so we can add more later”
- why unnecessary: The plan repeatedly says this is not a living contract and not intended for repeated reruns.
- simpler alternative: Use one fixed snapshot filename for this doc, unless there is already an established pattern of keeping multiple dated benchmark snapshots.

# Assumptions to Test
- `radcliq` runtime on the target GPU is acceptable enough to include in the same one-shot run.
- `torch.cuda.max_memory_allocated` is informative for the included metrics and not misleadingly low for metrics using substantial non-torch allocations.
- The curated divergence examples actually produce the intended disagreement patterns across the chosen five metrics.
- All included public metrics instantiate successfully in the stated environment without extra undocumented setup.

# Recommended Revisions
- Resolve the `radcliq` measurement strategy contradiction and update all sections to match.
- Reconcile the testing story: explicitly allow minimal unit tests for script logic while keeping measurement out of CI.
- Replace the subjective/circular inclusion rule with a simpler fixed scope rule.
- Normalize the JSON schema examples and field names to the planned output.
- Consider simplifying failure handling for this one-shot script: explicit metric config, clear skip reasons, and less adaptive logic.
- Tighten the doc claims around “best-correlated-with-radiologist-preferences” to ensure they are clearly attributed to prior studies, not this benchmark.

# Confidence
high

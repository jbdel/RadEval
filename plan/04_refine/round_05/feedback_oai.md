# Verdict
revise_major

# Critical Issues
- severity: high
- issue: The plan contradicts itself on tests and deliverables.
- why it matters: Early sections say “No changes to tests” and “No CI coverage,” but the commit plan adds `tests/test_bench_rewards_logic.py`. This makes scope unclear and weakens confidence that the plan is settled.
- recommended fix: Decide explicitly whether minimal unit tests are in scope. If yes, update non-goals/deliverables to say “no integration/measurement tests, only small logic tests.” If no, remove the test commit.

- severity: high
- issue: The documented CLI and JSON schema are internally inconsistent.
- why it matters: The script is described as “one command, no subcommands,” but the reproduction snippet uses `python scripts/bench_rewards.py all ...`. The JSON examples also conflict (`cached_init_s` vs `cold_start_s`, `peak_vram_mb_approx` vs `peak_vram_mb`, workload fixture path mismatch, env includes/excludes `trl` inconsistently).
- recommended fix: Normalize the interface and schema in one place and make every example match it exactly.

- severity: high
- issue: The claim that all included metrics are RL-reward candidates is stronger than the evidence presented.
- why it matters: “Locally runnable” and “not 7B+/API” is not sufficient to establish RL suitability. Some included metrics may still be impractical due to latency, batching behavior, CPU-only execution, or hidden dependencies. The plan partly acknowledges this, but the framing still overclaims.
- recommended fix: Reframe inclusion as “benchmarked public local metrics” rather than “RL-eligible,” and let the measured table support practical reward selection.

- severity: medium
- issue: The in-process benchmarking methodology is accepted without any validation despite known contamination risks.
- why it matters: The plan explicitly drops the comparability check, but still intends readers to compare rows. Teardown plus `radcliq`-last may be enough, but that is an assumption, not evidence.
- recommended fix: Add one minimal validation: rerun 1-2 representative metrics twice (early and late) or compare one metric in a fresh process vs in-process. Keep it narrow; no full subsystem needed.

- severity: medium
- issue: The speed workload rationale is weakly justified.
- why it matters: “20 samples is large enough that noise is acceptable” is asserted, not demonstrated. For fast metrics, 3 timed runs on a tiny batch may still be noisy relative to Python overhead.
- recommended fix: Add a simple fallback rule such as increasing repeat count for very fast metrics or reporting that sub-millisecond rows are approximate. Minimum change: state that fast-metric timings are coarse and primarily rank-order guidance.

- severity: medium
- issue: The plan relies on “verified key-map literals” that may drift, while also saying the doc is a snapshot.
- why it matters: For the one-shot snapshot this is acceptable, but for the script it creates fragility if anyone reruns later. The current mitigation records skips, but the plan still presents the literal map as authoritative.
- recommended fix: Present the key map as snapshot-specific and ensure the script degrades cleanly with a clear skip reason. Avoid stronger wording like “verified” in the durable docs unless tied to the snapshot date.

- severity: low
- issue: The doc is intended to be short, but the planned content is dense and risks exceeding the stated skimmability goal.
- why it matters: A page with a full metric table, caveats, divergence gallery, decision guide, and reproduction notes may not stay under “~1.5 screens.”
- recommended fix: Prioritize the two artifacts and move secondary caveats/reproduction detail into collapsible or lower-page sections if the docs system supports it, or trim prose.

# Overengineering Flags
- component: Date-stamped snapshot accumulation strategy
- why unnecessary: The plan says this is a one-shot snapshot and not a living contract, yet introduces a naming convention to support multiple future snapshots.
- simpler alternative: Use a single snapshot filename for this page unless there is a concrete need for multiple committed snapshots now.

- component: Extensive key-map fact sheet in the plan/doc
- why unnecessary: Much of this is implementation detail for the script, not essential reader-facing documentation.
- simpler alternative: Keep the literal map in the script and mention only the few user-relevant `key=` examples in the doc.

- component: Three-commit structure with reviewer segmentation rationale
- why unnecessary: This is process detail not clearly justified by the task itself.
- simpler alternative: Keep commits small, but don’t over-specify the exact split unless required by team norms.

# Assumptions to Test
- In-process teardown is sufficient for row-to-row timing comparability.
- Three timed warm runs on a 20-sample batch produce stable enough latency estimates, especially for very fast metrics.
- `radcliq` completes within an acceptable runtime on the target machine.
- `torch.cuda.max_memory_allocated` is informative enough for the included GPU-backed metrics.
- The curated divergence examples actually produce the intended disagreement pattern across the chosen five metrics.

# Recommended Revisions
- Resolve all internal inconsistencies: tests in/out of scope, CLI syntax, JSON field names, fixture paths, and whether `trl` is recorded.
- Reframe the inclusion language from “RL-eligible” to “benchmarked public local metrics,” with practicality determined by measured results.
- Add one minimal methodology check for in-process contamination rather than a full measurement gate.
- Calibrate the timing claims for very fast metrics; either increase repeats selectively or label those rows as coarse.
- Trim reader-facing documentation to the essentials and move implementation-specific key-map detail out of the main doc.

# Confidence
high

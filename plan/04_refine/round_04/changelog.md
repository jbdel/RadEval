# Plan Revision Changelog — Round 04

## Input
- previous plan: round_03/revised.md
- critic feedback: round_04/feedback_oai.md (revise_minor), round_04/feedback_gem.md (revise_minor)
- current revision: round_04/revised.md → flows into round_05/revised.md

Both critics agree (`revise_minor` with high confidence) — the remaining items are all simplifications and consistency fixes.

## Accepted Critiques

### Both critics: `radcliq` internal contradiction
- **OpenAI medium + Gemini medium.** Round 3 introduced "`radcliq` runs last" in the implementation notes but left stale "subprocess isolation" language in the doc outline and the decisions table. Round 5 sweeps all remaining subprocess references for `radcliq` and commits to the single approach: **in-process, sort `radcliq` last.** Drop the `--metric` flag entirely.

### Both critics: drop the measurement gate (overengineering)
- Gemini calls it "a formal meta-experiment" and "excessive for a one-shot script." OpenAI calls it "another artifact to explain." Both are right — the gate is more process than the underlying purpose (one doc page) justifies. Round 5 removes the gate, the `comparability_check` JSON field, and the threshold rules. In their place: a one-line caveat in the doc that all measurements are in-process with `radcliq`-last and explicit teardown, and that VRAM is approximate.

### Both critics: drop the `prewarm` subcommand (Gemini), simplify CLI (OpenAI)
- Gemini: "run the script twice" is simpler. Accepted. Round 5 removes `prewarm` as a subcommand — the `all` command is the only command, and reproduction docs instruct the user to run it twice (first run populates HF cache; second run produces the clean snapshot). Single-command CLI.

### OpenAI medium: circular inclusion rule
- "Plausibly fast enough ... as verified by the measurements in this document" is circular. Accepted. Round 5 uses a simpler pre-measurement rule: **include every public metric in the registry that is (a) locally runnable and (b) not an API metric and (c) not a 7B+ LLM metric.** Let the measured table speak for practicality.

### OpenAI low: JSON schema drift in examples
- Accepted. Round 5 normalizes all JSON examples to use `cached_init_s`, `warm_batch_s_median`, `peak_vram_mb_approx`, `docs/benchmarks/fixtures/speed_workload.json`, and removes `trl` from any example env block.

### OpenAI medium: test contradiction
- Accepted. Round 5 reframes explicitly: "no measurement-based tests, no CI coverage of the benchmark run itself; but a small `test_bench_rewards_logic.py` with ~30 LOC of pure-logic tests (schema, skip-record format, key-map lookup) for defense against regressions in the script's glue code."

### OpenAI overengineering: extensive key-map fact sheet + runtime validation + skip-on-drift
- Partial accept. Round 5 keeps the key-map literal (it's just a lookup table, minimal cost) but simplifies the drift behavior: if a metric's `per_sample=True` output doesn't contain the configured key, record `"skipped": "key-drift"` and continue. No elaborate "fact sheet validation mechanism" — just a plain dict lookup inside the warm-up call.

### OpenAI low: tighten "best-correlated with radiologists" attribution
- Accepted. Round 5 ensures every mention of RadCliQ quality is cited to the paper (the RadCliQ paper, `rajpurkarlab/CXR-Report-Metric`, and the RadEval paper), not to this benchmark.

## Rejected Critiques

- **OpenAI overengineering: drop date-stamped snapshot naming.** Partial rejection (reaffirmed from round 3). The date stamp is 0 extra LOC and preserves history. Keeping.
- **OpenAI: simplify commit structure.** Partial rejection. The three-commit split keeps script+fixtures / doc+snapshot / cross-links cleanly separable. Standard. Keeping.

## Uncertain / Deferred
- If during implementation the 20-sample benchmark produces misleading numbers on a specific metric (e.g., `radcliq` blows past a reasonable runtime), we may surface that in the doc's "Takeaways" but we won't add a separate escalation path. One-shot discipline.

## Major Plan Changes (Round 05)
- Sweep all residual `radcliq`-subprocess language → everywhere says "last, in-process."
- Drop the implementation-time measurement gate and `comparability_check` JSON field.
- Drop the `prewarm` subcommand; reproduction instructions say "run twice."
- Drop the `--metric` flag (only existed for the gate).
- Simplify the inclusion rule: (a) public + locally runnable + (b) not API + (c) not 7B+ LLM. No "plausibly" wording.
- Normalize all JSON examples to final field names.
- Clarify the testing story: no measurement/CI; ~30 LOC pure-logic tests only.
- Tighten RadCliQ-correlation attribution to external papers.

## Net Effect
- scope: reduced (one CLI command; no gate; no subprocess orchestration; no prewarm)
- confidence: increased — both critics converged on `revise_minor` with actionable, minor simplifications. Round 5 is the finalization round.

## Stopping intent
Round 5 is intended to be final. Both critics now agree (`revise_minor`) and disagree on nothing substantive. If the round-5 critics still object, the disagreement will be over wording not methodology, and the materiality-threshold rule lets us stop.

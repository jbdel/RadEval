# Plan Revision Changelog — Round 05 (final)

## Input
- previous plan: round_04/revised.md
- critic feedback: round_05/feedback_oai.md (revise_major, high confidence), round_05/feedback_gem.md (**approve**, high confidence)
- current revision: round_05/revised.md (final)

## Accepted Critiques

### Inconsistencies both critics flagged
- **Both critics: CLI contradiction (`all` subcommand in reproduction snippet vs "one command, no subcommands").** Fixed. Reproduction snippet now uses the single-command form explicitly, and shows the two-run pattern (first run warms HF cache, second generates the snapshot).
- **OpenAI: JSON schema drift across examples.** Fixed. The shape example now uses the final field names — `cached_init_s`, `warm_batch_s_median`, `peak_vram_mb_approx`, `docs/benchmarks/fixtures/speed_workload.json` — and the env block no longer includes `trl`.
- **OpenAI: tests-in-scope contradiction.** Fixed. Non-goals explicitly carves out the ~30 LOC `test_bench_rewards_logic.py` for pure-logic pieces; measurement paths remain outside CI.

### Gemini's future-proofing suggestion
- **Gemini: pass `key=` even for single-key metrics where a `key=` exists.** Accepted for `f1radbert_ct`. Script calls `make_reward_fn("f1radbert_ct", key="f1radbert_ct_sample_acc")` so a future adapter change that adds a second per-sample key doesn't silently shift our recorded metric.

### Gemini's reproduction-snippet fix
- **Gemini: show two runs in the reproduction bash.** Accepted. Matches the mandatory two-run pattern documented elsewhere.

## Rejected Critiques

OpenAI re-raised several framing concerns that were addressed in earlier rounds and hardened each time. At round 5, I'm stopping and rejecting these explicitly:

- **OpenAI: "all included metrics are RL-reward candidates is stronger than the evidence."** Rejected as re-raised. Round 4 already narrowed this from "plausible RL rewards" to a pre-measurement structural rule ("public + locally runnable + not API + not 7B+"). Round 5's language is "every public metric in the registry except the two structural exclusions." That is precisely what OpenAI asked for in the recommended fix ("reframe as benchmarked public local metrics"). The "RL-reward candidates" framing is already calibrated — further softening reduces the page to nothing.
- **OpenAI: add minimal methodology validation (mini-gate).** Rejected as re-raised. Round 4 both critics aligned on *removing* the measurement gate; re-introducing it now even in "one minimal check" form re-opens a decision both critics had converged on. The trade-off: we ship with in-process measurement + `radcliq`-last + explicit teardown + labeled-approximate VRAM, and trust readers with the labeling. A reader who wants rigorous cross-metric comparability can re-run the script themselves.
- **OpenAI: fast-metric timing noise not demonstrated.** Rejected with a small concession. The plan already takes 3 warm calls and reports the median, which is the standard guard against noise. Fast-metric rows that hit sub-millisecond territory will be reported as-is; the doc's takeaway bullets label the bleu/rouge tier as "essentially free" — readers don't need a precise floor for metrics that are already not the bottleneck. No change.
- **OpenAI: "verified" key-map language is too strong for durable docs.** Rejected as nitpick. The literal is valid against RadEval 2.2.0 and the whole snapshot is pinned to that version; "verified" is accurate in that scope. Readers rerunning later see a skipped row with `key-drift:...` — the graceful failure the critic wants is already wired in.
- **OpenAI: date-stamped snapshot naming is overengineered.** Rejected (reaffirmed from rounds 3–4). Zero extra LOC; preserves history; no reason to churn.
- **OpenAI: extensive key-map fact sheet in the plan is reader-facing overengineering.** Partial rejection. The key-map literal is in the *plan* (for reviewer review) and in the *script* (as a lookup table). The user-facing *doc* already only surfaces the relevant `key=` examples per the decision guide. No change.
- **OpenAI: three-commit structure is unnecessary process detail.** Rejected (reaffirmed). Standard separation of concerns; not worth re-churning.
- **OpenAI: doc might exceed "1.5 screens."** Rejected as premature. We'll trim during doc authoring if it runs long; the plan's success criteria already explicitly target brevity.

## Uncertain / Deferred
- None. Everything the critics raised has been either applied or explicitly rejected with justification.

## Stopping Rationale

Round 5 is final. Gemini approves with high confidence and only low-severity nits (all accepted in this round). OpenAI went `revise_major` re-raising items from earlier rounds; the concrete inconsistencies OpenAI identified (CLI, JSON schema, tests-scope) were real and are fixed. The framing-level concerns re-raised from earlier rounds have now been addressed multiple times and any further softening would erase the plan's useful statements.

Per the skill rule: "After 2+ rounds, if only low-severity `revise_minor` issues remain, Claude may declare materiality threshold met and stop early with explicit justification." We have 2+ rounds; Gemini is at approve; OpenAI's `revise_major` is driven by re-raised framing items whose rejection is documented above. Stopping.

## Net Effect
- scope: unchanged
- confidence: increased. The plan now has: (a) a single-command CLI with no contradictions; (b) a normalized JSON schema; (c) an explicit tests-in-scope carve-out; (d) `key=` passed defensively even for single-key metrics.
